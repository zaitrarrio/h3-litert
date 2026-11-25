// The video VAE's host side (python vae_host.py; h3-rs `vae.rs`; Diffusers
// `AutoencoderKLMiniMaxH3`): spatial tiles and their stitch, the temporal chunk
// plan and frame blend, the posterior, pixel and latent normalisation. Pure
// Swift on `[C, T, H, W]` tensors; graphs are called through closures so the
// tests drive it with fakes.

import Foundation

public enum VaeHost {
    public static let tile = 256
    public static let overlap = 64

    /// `_split_tiles`: starts, lengths, overlaps along one pixel axis. `ratio` is the
    /// granularity the slack is handed out in (16 px, one latent cell).
    public static func splitTiles(length: Int, tile: Int, minOverlap: Int, ratio: Int) -> (starts: [Int], lengths: [Int], overlaps: [Int]) {
        if tile >= length {
            return ([0], [length], [])
        }
        var n = (length + tile - 1) / tile
        while tile * n - minOverlap * (n - 1) - length < 0 {
            n += 1
        }
        var overlaps = [Int](repeating: minOverlap, count: n - 1)
        let remaining = tile * n - overlaps.reduce(0, +) - length
        for i in 0..<(remaining / ratio) {
            overlaps[i % (n - 1)] += ratio
        }
        var starts = [0]
        for i in 0..<(n - 1) {
            starts.append(starts[starts.count - 1] + tile - overlaps[i])
        }
        return (starts, [Int](repeating: tile, count: n), overlaps)
    }

    /// Linear cross-fade of `a`'s last `extent` with `b`'s first `extent` along `axis`, then the rest of `b`.
    public static func blend(_ a: Tensor, _ b: Tensor, extent: Int, axis: Int) -> Tensor {
        precondition(a.rank == b.rank, "blend ranks \(a.shape) and \(b.shape)")
        let ax = b.normalized(axis: axis)
        for d in 0..<b.rank where d != ax {
            precondition(a.shape[d] == b.shape[d], "blend shapes \(a.shape) and \(b.shape) along axis \(ax)")
        }
        let e = min(a.shape[ax], b.shape[ax], extent)
        if e == 0 { return b }
        let (outer, nb, inner) = b.strides(axis: ax)
        let na = a.shape[ax]
        var out = b.data
        let ad = a.data, bd = b.data
        let ef = Float(e)
        for i in 0..<e {
            let wb = Float(i) / ef
            let wa = Float(1.0) - wb
            for o in 0..<outer {
                let aBase = (o * na + (na - e + i)) * inner
                let bBase = (o * nb + i) * inner
                for k in 0..<inner {
                    out[bBase + k] = ad[aBase + k] * wa + bd[bBase + k] * wb
                }
            }
        }
        return Tensor(shape: b.shape, data: out)
    }

    /// Blend neighbouring tiles over their overlaps and crop them to the canvas.
    public static func stitchTiles(_ tiles: [[Tensor]], hOverlaps: [Int], wOverlaps: [Int]) -> Tensor {
        var rows: [Tensor] = []
        for (i, row) in tiles.enumerated() {
            var outRow: [Tensor] = []
            for (j, t) in row.enumerated() {
                var tile = t
                if i > 0 {
                    tile = blend(tiles[i - 1][j], tile, extent: hOverlaps[i - 1], axis: -2)
                }
                if j > 0 {
                    tile = blend(row[j - 1], tile, extent: wOverlaps[j - 1], axis: -1)
                }
                if i < tiles.count - 1 {
                    tile = tile.slice(axis: -2, 0..<(tile.shape[tile.rank - 2] - hOverlaps[i]))
                }
                if j < row.count - 1 {
                    tile = tile.slice(axis: -1, 0..<(tile.shape[tile.rank - 1] - wOverlaps[j]))
                }
                outRow.append(tile)
            }
            rows.append(Tensor.concat(outRow, axis: -1))
        }
        return Tensor.concat(rows, axis: -2)
    }

    /// Run `run` on tiles of `x [C, T, H, W]` laid out in pixel space (`H · ratioIn` pixels), stitch in
    /// the output's units (`ratioOut` pixels per output cell).
    public static func tiled(_ x: Tensor, ratioIn: Int, ratioOut: Int, tilePx: Int, overlapPx: Int, run: (Tensor) throws -> Tensor) throws -> Tensor {
        let hDim = x.shape[x.rank - 2], wDim = x.shape[x.rank - 1]
        let hp = hDim * ratioIn, wp = wDim * ratioIn
        let (ys, yl, yo) = splitTiles(length: hp, tile: tilePx, minOverlap: overlapPx, ratio: 16)
        let (xs, xl, xo) = splitTiles(length: wp, tile: tilePx, minOverlap: overlapPx, ratio: 16)
        var rows: [[Tensor]] = []
        for (y0, yn) in zip(ys, yl) {
            var row: [Tensor] = []
            for (x0, xn) in zip(xs, xl) {
                let t = x.slice(axis: -2, (y0 / ratioIn)..<min((y0 + yn) / ratioIn, hDim))
                    .slice(axis: -1, (x0 / ratioIn)..<min((x0 + xn) / ratioIn, wDim))
                row.append(try run(t))
            }
            rows.append(row)
        }
        if rows.count == 1 && rows[0].count == 1 {
            return rows[0][0]
        }
        return stitchTiles(rows, hOverlaps: yo.map { $0 / ratioOut }, wOverlaps: xo.map { $0 / ratioOut })
    }

    /// `(pad latent frames, chunks, output frames)` of `_decode`.
    public static func temporalPlan(_ cfg: VideoVaeConfig, latentFrames: Int) throws -> (pad: Int, chunks: Int, frames: Int) {
        if latentFrames == 0 {
            throw H3Error.shape("no latent frames to decode")
        }
        let drop = cfg.tokenDrop, chunk = cfg.tokensChunk, tr = cfg.temporalRatio
        let n = latentFrames + drop
        var pad = (chunk - n % chunk) % chunk
        var chunks = (n + pad) / chunk - (drop > 0 ? 1 : 0)
        if chunks < 1 {
            pad += chunk
            chunks = 1
        }
        var frames = chunks * (chunk * tr - cfg.framePrePadding)
        if drop > 0 {
            frames += cfg.frameOverlap
        }
        if pad > 0 {
            let intra = cfg.clipLength % tr
            var padFrames = 0
            for off in 0..<pad {
                padFrames += (intra != 0 && (latentFrames + off) % chunk == 0) ? intra : tr
            }
            frames -= padFrames
        }
        if frames <= 0 {
            throw H3Error.shape("the decode plan yields no frames for \(latentFrames) latent frames")
        }
        return (pad, chunks, frames)
    }

    /// Denormalised latents `[lz, T, h, w]` → pixels `[3, F, H, W]` in `[0, 1]`; `decodeClip` decodes one
    /// `[lz, 7, h, w]` (tiled by the caller or not) into ImageNet-normalised `[3, 28, 16h, 16w]`.
    /// `onChunk` sees each block of finished frames; with `collect` false the returned tensor is
    /// empty (`[3, 0, H, W]`) so a long clip is not held twice.
    public static func decode(
        _ cfg: VideoVaeConfig, z zIn: Tensor, decodeClip: (Tensor) throws -> Tensor, onChunk: ((Tensor) -> Void)? = nil, collect: Bool = true
    ) throws -> Tensor {
        precondition(zIn.rank == 4, "decode wants [lz, T, h, w], got \(zIn.shape)")
        let (pad, chunks, total) = try temporalPlan(cfg, latentFrames: zIn.shape[1])
        var z = zIn
        if pad > 0 {
            z = Tensor.concat([z, z.repeated(axis: 1, index: z.shape[1] - 1, times: pad)], axis: 1)
        }
        let chunkTok = cfg.tokensChunk, tr = cfg.temporalRatio, pre = cfg.framePrePadding
        let chunkFrames = chunkTok * tr
        var out: [Tensor] = []
        var written = 0
        var overlap: Tensor? = nil
        var height = 0, width = 0
        for ci in 0..<chunks {
            let start = ci * chunkTok
            let n = min(chunkTok + cfg.tokenOverlap, z.shape[1] - start)
            let clip = try decodeClip(z.slice(axis: 1, start..<(start + n)))  // [3, 4n, H, W]
            guard clip.rank == 4, clip.shape[0] == 3 else {
                throw H3Error.shape("the decoder returned \(clip.shape), expected [3, frames, H, W]")
            }
            height = clip.shape[2]
            width = clip.shape[3]
            let frames = clip.shape[1]
            var chunk = clip.slice(axis: 1, min(pre, frames)..<min(chunkFrames, frames))
            if let ov = overlap {
                chunk = blend(ov, chunk, extent: cfg.frameOverlap, axis: 1)
            }
            overlap = (cfg.tokenDrop > 0 && frames > chunkFrames + pre) ? clip.slice(axis: 1, (chunkFrames + pre)..<frames) : nil
            let keep = min(chunk.shape[1], total - written)
            if keep > 0 {
                let px = denormalizePixels(chunk.slice(axis: 1, 0..<keep))
                if collect { out.append(px) }
                written += keep
                onChunk?(px)
            }
        }
        if let ov = overlap, written < total {
            let keep = min(total - written, ov.shape[1])
            let px = denormalizePixels(ov.slice(axis: 1, 0..<keep))
            if collect { out.append(px) }
            written += keep
            onChunk?(px)
        }
        if written != total {
            throw H3Error.shape("decoded \(written) frames, the plan expected \(total)")
        }
        if !collect {
            return Tensor.zeros([3, 0, height, width])
        }
        return Tensor.concat(out, axis: 1)
    }

    /// `clip(x · std + mean, 0, 1)` per colour channel (`[3, …]`).
    public static func denormalizePixels(_ x: Tensor) -> Tensor {
        var out = x.data
        let s = x.rowStride
        for c in 0..<3 {
            let sc = H3Latents.pixelStd[c], m = H3Latents.pixelMean[c]
            for i in (c * s)..<((c + 1) * s) {
                out[i] = min(max(out[i] * sc + m, 0.0), 1.0)
            }
        }
        return Tensor(shape: x.shape, data: out)
    }

    /// `(x − mean) / std` per colour channel.
    public static func normalizePixels(_ x: Tensor) -> Tensor {
        var out = x.data
        let s = x.rowStride
        for c in 0..<3 {
            let sc = H3Latents.pixelStd[c], m = H3Latents.pixelMean[c]
            for i in (c * s)..<((c + 1) * s) {
                out[i] = (out[i] - m) / sc
            }
        }
        return Tensor(shape: x.shape, data: out)
    }

    public static func normalizeLatents(_ cfg: VideoVaeConfig, _ z: Tensor) -> Tensor {
        precondition(z.rows == cfg.latentChannels, "normalizeLatents wants \(cfg.latentChannels) channels, got \(z.shape)")
        var out = z.data
        let s = z.rowStride
        for c in 0..<z.rows {
            let m = cfg.latentsMean[c], sd = cfg.latentsStd[c]
            for i in (c * s)..<((c + 1) * s) {
                out[i] = (out[i] - m) / sd
            }
        }
        return Tensor(shape: z.shape, data: out)
    }

    public static func denormalizeLatents(_ cfg: VideoVaeConfig, _ z: Tensor) -> Tensor {
        precondition(z.rows == cfg.latentChannels, "denormalizeLatents wants \(cfg.latentChannels) channels, got \(z.shape)")
        return z.channelAffine(scale: cfg.latentsStd, shift: cfg.latentsMean)
    }

    /// `mean + exp(logvar/2)·ε` from `[2·lz, …]` moments (logvar clamped to [−30, 20]).
    public static func posteriorSample(moments: Tensor, noise: Tensor) -> Tensor {
        let lz = moments.rows / 2
        let mean = moments.rows(0..<lz)
        let logvar = moments.rows(lz..<(2 * lz))
        precondition(noise.shape == mean.shape, "posterior noise \(noise.shape) for moments \(moments.shape)")
        var out = [Float](repeating: 0, count: mean.count)
        for i in 0..<out.count {
            let lv = min(max(logvar.data[i], -30.0), 20.0)
            out[i] = mean.data[i] + exp(0.5 * lv) * noise.data[i]
        }
        return Tensor(shape: mean.shape, data: out)
    }

    /// ImageNet-normalised pixels `[3, F, H, W]` → moments `[2·lz, T, h, w]` (`_encode`: 17-frame clips, the
    /// trailing `token_drop` latent frames dropped; a single frame goes through alone).
    public static func encode(_ cfg: VideoVaeConfig, pixels: Tensor, encodeClip: (Tensor) throws -> Tensor) throws -> Tensor {
        precondition(pixels.rank == 4, "encode wants [3, F, H, W], got \(pixels.shape)")
        let f = pixels.shape[1]
        if f == 1 {
            return try encodeClip(pixels)
        }
        let cl = cfg.clipLength
        var px = pixels
        if f % cl != 0 {
            px = Tensor.concat([px, px.repeated(axis: 1, index: f - 1, times: (cl - f % cl) % cl)], axis: 1)
        }
        var parts: [Tensor] = []
        for i in 0..<(px.shape[1] / cl) {
            parts.append(try encodeClip(px.slice(axis: 1, (i * cl)..<((i + 1) * cl))))
        }
        var moments = Tensor.concat(parts, axis: 1)
        if cfg.tokenDrop > 0 {
            moments = moments.slice(axis: 1, 0..<(moments.shape[1] - cfg.tokenDrop))
        }
        return moments
    }
}
