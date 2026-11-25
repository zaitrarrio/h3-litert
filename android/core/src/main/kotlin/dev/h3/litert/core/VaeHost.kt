package dev.h3.litert.core

import kotlin.math.exp

/**
 * The video VAE's host side (`vae_host.py`; h3-rs `vae.rs`; Diffusers `AutoencoderKLMiniMaxH3`):
 * spatial tiles and their stitch, the temporal chunk plan and frame blend, the posterior, pixel and
 * latent normalisation. Graphs are called through callbacks so the pipeline and the tests share it.
 *
 * Every tensor here is `[C, T, H, W]` (pixels `[3, F, H, W]`, latents `[lz, T, h, w]`).
 */
public object VaeHost {
    public const val TILE: Int = 256
    public const val OVERLAP: Int = 64

    public class Tiles(public val starts: IntArray, public val lengths: IntArray, public val overlaps: IntArray)

    /** `_split_tiles`: starts, lengths, overlaps along one pixel axis. */
    public fun splitTiles(length: Int, tile: Int, minOverlap: Int, ratio: Int): Tiles {
        if (tile >= length) return Tiles(intArrayOf(0), intArrayOf(length), IntArray(0))
        var n = (length + tile - 1) / tile
        while (tile * n - minOverlap * (n - 1) - length < 0) n++
        val overlaps = IntArray(n - 1) { minOverlap }
        val remaining = tile * n - overlaps.sum() - length
        for (i in 0 until remaining / ratio) overlaps[i % (n - 1)] += ratio
        val starts = IntArray(n)
        for (i in 0 until n - 1) starts[i + 1] = starts[i] + tile - overlaps[i]
        return Tiles(starts, IntArray(n) { tile }, overlaps)
    }

    // ---- axis helpers on [C, T, H, W] -----------------------------------------------------------

    private fun outerInner(shape: IntArray, axis: Int): Pair<Int, Int> {
        var outer = 1
        for (i in 0 until axis) outer *= shape[i]
        var inner = 1
        for (i in axis + 1 until shape.size) inner *= shape[i]
        return outer to inner
    }

    /** `x[..., from:to, ...]` along `axis`. */
    public fun sliceAxis(x: Tensor, axis: Int, from: Int, to: Int): Tensor {
        require(axis in x.shape.indices && from in 0..to && to <= x.shape[axis]) {
            "sliceAxis $axis [$from, $to) outside ${x.shapeString()}"
        }
        val (outer, inner) = outerInner(x.shape, axis)
        val len = x.shape[axis]
        val n = to - from
        val out = FloatArray(outer * n * inner)
        for (o in 0 until outer) {
            System.arraycopy(x.data, (o * len + from) * inner, out, o * n * inner, n * inner)
        }
        val shape = x.shape.copyOf()
        shape[axis] = n
        return Tensor(out, shape)
    }

    /** Concatenate along `axis`; the other dimensions must agree. */
    public fun concatAxis(parts: List<Tensor>, axis: Int): Tensor {
        require(parts.isNotEmpty()) { "nothing to concatenate" }
        if (parts.size == 1) return parts[0]
        val ref = parts[0].shape
        var total = 0
        for (p in parts) {
            require(p.rank == ref.size && p.shape.indices.all { it == axis || p.shape[it] == ref[it] }) {
                "cannot concatenate ${p.shapeString()} with ${parts[0].shapeString()} along $axis"
            }
            total += p.shape[axis]
        }
        val (outer, inner) = outerInner(ref, axis)
        val out = FloatArray(outer * total * inner)
        for (o in 0 until outer) {
            var at = o * total * inner
            for (p in parts) {
                val len = p.shape[axis]
                System.arraycopy(p.data, o * len * inner, out, at, len * inner)
                at += len * inner
            }
        }
        val shape = ref.copyOf()
        shape[axis] = total
        return Tensor(out, shape)
    }

    /** Repeat the last slice along `axis` `n` times (`np.repeat(x[..., -1:], n)` appended). */
    public fun padRepeatLast(x: Tensor, axis: Int, n: Int): Tensor {
        if (n == 0) return x
        val last = sliceAxis(x, axis, x.shape[axis] - 1, x.shape[axis])
        return concatAxis(listOf(x) + List(n) { last }, axis)
    }

    /**
     * Linear cross-fade of `a`'s last `extent` with `b`'s first `extent` along `axis`, then the rest
     * of `b`. Weights `w_j = j / extent` in f32; `a·(1 − w) + b·w`.
     */
    public fun blend(a: Tensor, b: Tensor, extent: Int, axis: Int): Tensor {
        val e = minOf(a.shape[axis], b.shape[axis], extent)
        if (e == 0) return b
        require(a.rank == b.rank && a.shape.indices.all { it == axis || a.shape[it] == b.shape[it] }) {
            "blend: ${a.shapeString()} vs ${b.shapeString()} along $axis"
        }
        val (outer, inner) = outerInner(b.shape, axis)
        val la = a.shape[axis]
        val lb = b.shape[axis]
        val out = FloatArray(b.size)
        val ef = e.toFloat()
        for (o in 0 until outer) {
            for (j in 0 until lb) {
                val dst = (o * lb + j) * inner
                if (j < e) {
                    val w = j.toFloat() / ef
                    val wa = 1.0f - w
                    val sa = (o * la + la - e + j) * inner
                    for (i in 0 until inner) out[dst + i] = a.data[sa + i] * wa + b.data[dst + i] * w
                } else {
                    System.arraycopy(b.data, dst, out, dst, inner)
                }
            }
        }
        return Tensor(out, b.shape.copyOf())
    }

    /** Stitch a grid of tiles (`[C, T, h, w]` each) with cross-faded overlaps: rows along H, columns along W. */
    public fun stitchTiles(tiles: List<List<Tensor>>, hOverlaps: IntArray, wOverlaps: IntArray): Tensor {
        val rows = ArrayList<Tensor>(tiles.size)
        for ((i, row) in tiles.withIndex()) {
            val outRow = ArrayList<Tensor>(row.size)
            for ((j, tile0) in row.withIndex()) {
                var tile = tile0
                if (i > 0) tile = blend(tiles[i - 1][j], tile, hOverlaps[i - 1], 2)
                if (j > 0) tile = blend(row[j - 1], tile, wOverlaps[j - 1], 3)
                if (i < tiles.size - 1) tile = sliceAxis(tile, 2, 0, tile.shape[2] - hOverlaps[i])
                if (j < row.size - 1) tile = sliceAxis(tile, 3, 0, tile.shape[3] - wOverlaps[j])
                outRow += tile
            }
            rows += concatAxis(outRow, 3)
        }
        return concatAxis(rows, 2)
    }

    /**
     * Run `run` on tiles of `x [C, T, H, W]` laid out in pixel space (`H·ratioIn` pixels) and
     * stitch in the output's units (`ratioOut` pixels per output cell).
     */
    public fun tiled(x: Tensor, ratioIn: Int, ratioOut: Int, tilePx: Int, overlapPx: Int, run: (Tensor) -> Tensor): Tensor {
        require(x.rank == 4) { "tiled wants [C, T, H, W], got ${x.shapeString()}" }
        val hp = x.shape[2] * ratioIn
        val wp = x.shape[3] * ratioIn
        val ys = splitTiles(hp, tilePx, overlapPx, 16)
        val xs = splitTiles(wp, tilePx, overlapPx, 16)
        val rows = ArrayList<List<Tensor>>()
        for (yi in ys.starts.indices) {
            val y0 = ys.starts[yi] / ratioIn
            val y1 = (ys.starts[yi] + ys.lengths[yi]) / ratioIn
            val row = ArrayList<Tensor>()
            for (xi in xs.starts.indices) {
                val x0 = xs.starts[xi] / ratioIn
                val x1 = (xs.starts[xi] + xs.lengths[xi]) / ratioIn
                row += run(sliceAxis(sliceAxis(x, 2, y0, y1), 3, x0, x1))
            }
            rows += row
        }
        if (rows.size == 1 && rows[0].size == 1) return rows[0][0]
        return stitchTiles(rows, IntArray(ys.overlaps.size) { ys.overlaps[it] / ratioOut }, IntArray(xs.overlaps.size) { xs.overlaps[it] / ratioOut })
    }

    // ---- temporal plan -------------------------------------------------------------------------

    public data class TemporalPlan(val pad: Int, val chunks: Int, val frames: Int)

    /** `(pad latent frames, chunks, output frames)` of `_decode`. */
    public fun temporalPlan(cfg: VideoVaeConfig, latentFrames: Int): TemporalPlan {
        if (latentFrames == 0) throw IllegalArgumentException("no latent frames to decode")
        val drop = cfg.tokenDrop
        val chunk = cfg.tokensChunk()
        val tr = cfg.temporalRatio()
        val n = latentFrames + drop
        var pad = (chunk - n % chunk) % chunk
        var chunks = (n + pad) / chunk - (if (drop > 0) 1 else 0)
        if (chunks < 1) {
            pad += chunk
            chunks = 1
        }
        var frames = chunks * (chunk * tr - cfg.framePrePadding())
        if (drop > 0) frames += cfg.frameOverlap()
        if (pad > 0) {
            val intra = cfg.clipLength % tr
            var padFrames = 0
            for (off in 0 until pad) padFrames += if (intra != 0 && (latentFrames + off) % chunk == 0) intra else tr
            frames -= padFrames
        }
        if (frames <= 0) throw IllegalArgumentException("the decode plan yields no frames for $latentFrames latent frames")
        return TemporalPlan(pad, chunks, frames)
    }

    /**
     * Denormalised latents `[lz, T, h, w]` → pixels `[3, F, H, W]` in `[0, 1]`; `decodeClip` decodes
     * one `[lz, 7, h, w]` into ImageNet-normalised `[3, 28, 16h, 16w]`. `onChunk` receives each
     * denormalised piece as it is finished.
     */
    public fun decode(cfg: VideoVaeConfig, z0: Tensor, decodeClip: (Tensor) -> Tensor, onChunk: ((Tensor) -> Unit)? = null): Tensor {
        require(z0.rank == 4) { "decode wants [lz, T, h, w], got ${z0.shapeString()}" }
        val plan = temporalPlan(cfg, z0.shape[1])
        val z = padRepeatLast(z0, 1, plan.pad)
        val chunkTok = cfg.tokensChunk()
        val tr = cfg.temporalRatio()
        val pre = cfg.framePrePadding()
        val chunkFrames = chunkTok * tr
        val out = ArrayList<Tensor>()
        var written = 0
        var overlap: Tensor? = null
        for (ci in 0 until plan.chunks) {
            val start = ci * chunkTok
            val n = minOf(chunkTok + cfg.tokenOverlap(), z.shape[1] - start)
            val clip = decodeClip(sliceAxis(z, 1, start, start + n)) // [3, 4n, H, W]
            var chunk = sliceAxis(clip, 1, pre, chunkFrames)
            overlap?.let { chunk = blend(it, chunk, cfg.frameOverlap(), 1) }
            overlap = if (cfg.tokenDrop > 0 && clip.shape[1] > chunkFrames + pre) sliceAxis(clip, 1, chunkFrames + pre, clip.shape[1]) else null
            val keep = minOf(chunk.shape[1], plan.frames - written)
            if (keep > 0) {
                val px = denormalizePixels(sliceAxis(chunk, 1, 0, keep))
                out += px
                written += keep
                onChunk?.invoke(px)
            }
        }
        val ov = overlap
        if (ov != null && written < plan.frames) {
            val keep = minOf(plan.frames - written, ov.shape[1])
            val px = denormalizePixels(sliceAxis(ov, 1, 0, keep))
            out += px
            written += keep
            onChunk?.invoke(px)
        }
        if (written != plan.frames) throw IllegalStateException("decoded $written frames, the plan expected ${plan.frames}")
        return concatAxis(out, 1)
    }

    // ---- normalisation -------------------------------------------------------------------------

    private fun perChannel(x: Tensor, mean: FloatArray, std: FloatArray, forward: Boolean, clamp: Boolean): Tensor {
        require(x.rank == 4 && x.shape[0] == mean.size) { "wants [${mean.size}, T, H, W], got ${x.shapeString()}" }
        val plane = x.rowSize
        val out = FloatArray(x.size)
        for (c in 0 until x.shape[0]) {
            val m = mean[c]
            val s = std[c]
            val base = c * plane
            if (forward) {
                for (i in 0 until plane) out[base + i] = (x.data[base + i] - m) / s
            } else {
                for (i in 0 until plane) {
                    var v = x.data[base + i] * s + m
                    if (clamp) v = if (v < 0.0f) 0.0f else if (v > 1.0f) 1.0f else v
                    out[base + i] = v
                }
            }
        }
        return Tensor(out, x.shape.copyOf())
    }

    /** `clip(x·std + mean, 0, 1)` with the ImageNet statistics. */
    public fun denormalizePixels(x: Tensor): Tensor = perChannel(x, PIXEL_MEAN, PIXEL_STD, forward = false, clamp = true)

    /** `(x − mean) / std` with the ImageNet statistics. */
    public fun normalizePixels(x: Tensor): Tensor = perChannel(x, PIXEL_MEAN, PIXEL_STD, forward = true, clamp = false)

    private fun f32(v: List<Double>) = FloatArray(v.size) { v[it].toFloat() }

    public fun normalizeLatents(cfg: VideoVaeConfig, z: Tensor): Tensor =
        perChannel(z, f32(cfg.latentsMean), f32(cfg.latentsStd), forward = true, clamp = false)

    public fun denormalizeLatents(cfg: VideoVaeConfig, z: Tensor): Tensor =
        perChannel(z, f32(cfg.latentsMean), f32(cfg.latentsStd), forward = false, clamp = false)

    /** `mean + exp(logvar/2)·ε` from `[2·lz, …]` moments (logvar clamped to `[−30, 20]`). */
    public fun posteriorSample(moments: Tensor, noise: Tensor): Tensor {
        val lz = moments.shape[0] / 2
        val plane = moments.rowSize
        require(noise.size == lz * plane) { "posteriorSample: noise ${noise.shapeString()} for moments ${moments.shapeString()}" }
        val out = FloatArray(lz * plane)
        for (i in out.indices) {
            var lv = moments.data[lz * plane + i]
            if (lv < -30.0f) lv = -30.0f else if (lv > 20.0f) lv = 20.0f
            out[i] = moments.data[i] + exp(0.5f * lv) * noise.data[i]
        }
        val shape = moments.shape.copyOf()
        shape[0] = lz
        return Tensor(out, shape)
    }

    /**
     * ImageNet-normalised pixels `[3, F, H, W]` → moments `[2·lz, T, h, w]` (`_encode`: 17-frame
     * clips, the trailing `token_drop` latent frames dropped; a single frame goes through alone).
     */
    public fun encode(cfg: VideoVaeConfig, pixels: Tensor, encodeClip: (Tensor) -> Tensor): Tensor {
        val f = pixels.shape[1]
        if (f == 1) return encodeClip(pixels)
        val cl = cfg.clipLength
        val padded = if (f % cl != 0) padRepeatLast(pixels, 1, (cl - f % cl) % cl) else pixels
        val clips = padded.shape[1] / cl
        var moments = concatAxis(List(clips) { i -> encodeClip(sliceAxis(padded, 1, i * cl, (i + 1) * cl)) }, 1)
        if (cfg.tokenDrop > 0) moments = sliceAxis(moments, 1, 0, moments.shape[1] - cfg.tokenDrop)
        return moments
    }
}
