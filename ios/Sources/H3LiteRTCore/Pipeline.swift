// Prompt embeddings → packed layout → denoising loop → decoded video and
// audio, driving the graphs exactly as docs/graphs.md's host loop says
// (python pipeline.py). T2VA and FL2VA; Euler, renoise and PDD samplers.

import Foundation

/// The additive key mask of a padded text row.
public let maskPad: Float = -1e9

/// A keyframe of an FL2VA request: `anchor` "first" or "last", pixels `[3, 1, H, W]` in `[0, 1]` at the canvas size.
public struct Keyframe {
    public var anchor: String
    public var pixels: Tensor

    public init(anchor: String, pixels: Tensor) {
        self.anchor = anchor
        self.pixels = pixels
    }
}

/// What a generation asks for.
public struct Request {
    /// `[rows, text_dim]` prompt embeddings.
    public var embeds: Tensor
    /// One tag per prompt row: text (1) or vision (0).
    public var textTags: [Int]
    public var geometry: Geometry
    public var seed: UInt64
    public var keyframes: [Keyframe]
    /// TAEH3 instead of the VAE decoder.
    public var preview: Bool

    public init(embeds: Tensor, textTags: [Int], geometry: Geometry, seed: UInt64 = 7, keyframes: [Keyframe] = [], preview: Bool = false) {
        self.embeds = embeds
        self.textTags = textTags
        self.geometry = geometry
        self.seed = seed
        self.keyframes = keyframes
        self.preview = preview
    }
}

/// What the pipeline reports while it runs. Times are seconds (the Python
/// reference reports `ms`; `ms = seconds · 1000`).
public enum PipelineEvent {
    case prepared(seconds: Double, tokens: Int)
    case step(index: Int, total: Int, seconds: Double)
    case frames(Frames)
    case audio(Audio)
}

public typealias OnEvent = (PipelineEvent) -> Void

public struct Timings {
    public var prepareSeconds: Double = 0
    public var stepSeconds: [Double] = []
    public var decodeSeconds: Double = 0
    public var audioSeconds: Double = 0

    public init() {}
}

/// Everything `prepare` builds once per request: the padded layout, the refined
/// text, the row buffers of both modalities and the per-row tables.
public struct Prepared {
    public var layout: Layout
    /// `[T, d]`: the text bucket's rows (the padded rows are zeros after masking, never read back).
    public var text: Tensor
    public var nText: Int
    /// `T − n_text` padded text rows.
    public var pad: Int
    /// `[condition rows + target rows, P]`.
    public var videoRows: Tensor
    /// `[2·A, A_c]`.
    public var audioRows: Tensor
    /// `[S, r/2]` over the padded sequence.
    public var cos: Tensor
    public var sin: Tensor
    /// `[S]`: 0 for a real row, −1e9 for a padded text row.
    public var mask: Tensor
    /// Packed indices after the text padding shift.
    public var videoIdx: [Int]
    public var audioIdx: [Int]
    /// Packed tags after the shift (padded rows are text).
    public var tags: [Int]

    /// `S`: rows of the padded packed sequence.
    public var rows: Int { layout.sequenceLength + pad }
}

public final class Pipeline {
    public let store: GraphStore
    public let manifest: Manifest
    public let cfg: H3Config
    public let adaln: AdalnTables

    public init(store: GraphStore, adaln: AdalnTables) {
        self.store = store
        self.manifest = store.manifest
        self.cfg = store.manifest.cfg
        self.adaln = adaln
    }

    // MARK: helpers

    func one(_ outs: [Tensor], of name: String) throws -> Tensor {
        guard outs.count == 1 else {
            throw H3Error.graph("\(name): expected 1 output, got \(outs.count)")
        }
        return outs[0]
    }

    /// Run a per-row graph over `rows` arrays (each `[nReal, …]`) in chunks of `chunk` (the last one
    /// zero-padded) with the same `fixed` inputs every call; outputs concatenated and trimmed to `nReal`.
    func rows(_ graph: Graph, chunk: Int, rows: [String: Tensor], fixed: [String: Tensor], real nReal: Int) throws -> [Tensor] {
        guard nReal > 0 else {
            throw H3Error.shape("\(graph.name): no rows to run")
        }
        for (k, v) in rows where v.rows != nReal {
            throw H3Error.shape("\(graph.name): row input `\(k)` has \(v.rows) rows, expected \(nReal)")
        }
        var outs: [[Tensor]] = []
        var r = 0
        while r < nReal {
            let n = min(chunk, nReal - r)
            var inputs = fixed
            for (k, v) in rows {
                var blk = v.rows(r..<(r + n))
                if n < chunk {
                    blk = Tensor.concatRows([blk, Tensor.zeros([chunk - n] + Array(v.shape.dropFirst()))])
                }
                inputs[k] = blk
            }
            let o = try graph.run(inputs)
            if let first = outs.first, first.count != o.count {
                throw H3Error.graph("\(graph.name): output count changed between calls")
            }
            outs.append(o)
            r += chunk
        }
        let first = outs[0]
        return (0..<first.count).map { i in
            Tensor.concatRows(outs.map { $0[i] }).rows(0..<nReal)
        }
    }

    // MARK: prompt + layout

    /// The VAE encoder graphs with the manifest's tiling, for `prepare` (a closure that throws
    /// when called if the export has no encoders).
    public func encoderTileRunner() -> (Tensor) throws -> Tensor {
        let m = manifest
        let ratio = m.vae.spatialRatio
        return { [store] x in
            guard let vv = m.videoVae else {
                throw H3Error.manifest("this export has no video VAE (needed to encode keyframes)")
            }
            let name = x.shape[1] == 1 ? vv.encoderImage : vv.encoderClip
            guard let file = name else {
                throw H3Error.manifest("this export has no VAE encoder graphs (exported with --no-encoders)")
            }
            let graph = try store.fixed(file)
            return try VaeHost.tiled(x, ratioIn: 1, ratioOut: ratio, tilePx: vv.tile, overlapPx: vv.overlap) { t in
                let outs = try graph.run(["x": t])
                guard outs.count == 1 else {
                    throw H3Error.graph("\(file): expected 1 output (moments), got \(outs.count)")
                }
                return outs[0]
            }
        }
    }

    /// Condition rows of one keyframe: encode, sample the posterior (seed 42), f16 round, normalise, patchify.
    public func encodeKeyframe(pixels: Tensor, encodeClip: (Tensor) throws -> Tensor) throws -> Tensor {
        let vc = manifest.vae
        guard pixels.rank == 4, pixels.shape[0] == 3 else {
            throw H3Error.shape("keyframe pixels are \(pixels.shape), expected [3, 1, H, W]")
        }
        let moments = try VaeHost.encode(vc, pixels: VaeHost.normalizePixels(pixels), encodeClip: encodeClip)
        guard moments.rank == 4, moments.rows == 2 * vc.latentChannels else {
            throw H3Error.shape("encoder moments are \(moments.shape), expected [\(2 * vc.latentChannels), T, h, w]")
        }
        let lz = moments.rows / 2
        var rng = Rng(seed: keyframeEncodeSeed)
        let noise = rng.normal(shape: [lz] + Array(moments.shape.dropFirst()))
        var z = VaeHost.posteriorSample(moments: moments, noise: noise)
        z = VaeHost.normalizeLatents(vc, z.map { Half.round($0) })
        return patchRows(z, patch: cfg.patch)
    }

    public func prepare(_ req: Request, encodeClip: (Tensor) throws -> Tensor) throws -> Prepared {
        let c = cfg, m = manifest
        let ratio = m.vae.spatialRatio
        let g = req.geometry
        try g.validate(spatialRatio: ratio)
        let (lt, lh, lw) = try g.latent(spatialRatio: ratio)
        let na = g.audioLatents
        let nText = req.textTags.count
        guard req.embeds.shape == [nText, c.textDim] else {
            throw H3Error.shape("prompt embeds are \(req.embeds.shape), expected [\(nText), \(c.textDim)]")
        }
        let bucket = try m.textRefineFor(rows: nText)
        let T = bucket.rows
        let pad = T - nText
        var emb = Tensor.zeros([T, c.textDim])
        if nText > 0 { emb.setRows(0..<nText, req.embeds) }
        var tmask = Tensor.zeros([T])
        for i in nText..<T { tmask.data[i] = maskPad }
        let text = try one(try store.fixed(bucket.file).run(["embeds": emb, "mask": tmask]), of: bucket.file)
        guard text.shape == [T, c.hidden] else {
            throw H3Error.shape("\(bucket.file): text is \(text.shape), expected [\(T), \(c.hidden)]")
        }
        let anchors = req.keyframes.map { $0.anchor }
        let layout = try buildLayout(textTags: req.textTags, lt: lt, lh: lh, lw: lw, na: na, patch: c.patch, anchors: anchors)
        var rng = Rng(seed: req.seed)
        var cond = Tensor.zeros([0, c.videoPatchDim])
        if !req.keyframes.isEmpty {
            let parts = try req.keyframes.map { try encodeKeyframe(pixels: $0.pixels, encodeClip: encodeClip) }
            cond = Tensor.concatRows(parts)
            let noise = rng.normal(cond.count)
            // `np.float32(0.999)` and `np.float32(1.0 - 0.999)`: the second is formed in f64 first.
            let aug = keyframeNoiseAug
            let inv = Float(1.0 - 0.999)
            for i in 0..<cond.count {
                cond.data[i] = aug * cond.data[i] + inv * noise[i]
            }
        }
        let noise = rng.normal(shape: [c.inChannels, lt, lh, lw])
        let videoRows = Tensor.concatRows([cond, patchRows(noise, patch: c.patch)])
        let audioRows = rng.normal(shape: [na * audioChannels, c.audioInChannels])
        func shift(_ i: Int) -> Int { i < nText ? i : i + pad }
        let sPadded = layout.sequenceLength + pad
        var pos = [Float](repeating: 0, count: sPadded * 3)
        var tags = [Int](repeating: textTag, count: sPadded)
        for (i, p) in layout.positionIds.enumerated() {
            let j = shift(i)
            pos[3 * j] = Float(p.t)
            pos[3 * j + 1] = Float(p.h)
            pos[3 * j + 2] = Float(p.w)
            tags[j] = layout.tokenTags[i]
        }
        let (cos, sin) = Rope.h3Rope(positions: pos, ropeFreqDim: c.ropeFreqDim, theta: c.ropeTheta)
        var mask = Tensor.zeros([sPadded])
        for i in nText..<T { mask.data[i] = maskPad }
        return Prepared(
            layout: layout, text: text, nText: nText, pad: pad, videoRows: videoRows, audioRows: audioRows,
            cos: cos, sin: sin, mask: mask, videoIdx: layout.videoIndices.map(shift), audioIdx: layout.audioIndices.map(shift), tags: tags
        )
    }

    // MARK: one forward

    /// Head group `g` of a `[S, H, hd]` tensor as `[S, G, hd]`.
    static func headGroup(_ x: Tensor, group g: Int, size G: Int, heads H: Int, headDim hd: Int) -> Tensor {
        let s = x.rows
        let w = G * hd
        var out = [Float](repeating: 0, count: s * w)
        let src = x.data
        for row in 0..<s {
            let from = (row * H + g * G) * hd
            out.replaceSubrange((row * w)..<((row + 1) * w), with: src[from..<(from + w)])
        }
        return Tensor(shape: [s, G, hd], data: out)
    }

    /// Write `og [S, G·hd]` into columns `gG·hd …` of `o [S, I]`.
    static func scatterHeads(_ og: Tensor, into o: inout Tensor, group g: Int, size G: Int, headDim hd: Int) {
        let s = o.rows, inner = o.rowStride
        let w = G * hd
        precondition(og.shape == [s, w], "attention output \(og.shape), expected [\(s), \(w)]")
        let src = og.data
        for row in 0..<s {
            let to = row * inner + g * w
            o.data.replaceSubrange(to..<(to + w), with: src[(row * w)..<((row + 1) * w)])
        }
    }

    /// One DiT forward of the padded sequence at `step`: `(v_video [rows, P], v_audio [2A, A_c])`.
    public func forward(_ p: Prepared, step: Manifest.SamplerStep, attention attn: Manifest.AttentionEntry) throws -> (video: Tensor, audio: Tensor) {
        let c = cfg, m = manifest, ad = adaln
        let C = m.chunk, G = m.headGroup, d = c.hidden, inner = c.dim, hd = c.headDim, H = c.heads
        let s = p.rows
        guard attn.tokens == s, attn.heads == G else {
            throw H3Error.shape("\(attn.file) is for \(attn.tokens) rows and \(attn.heads) heads; this request has \(s) rows and head groups of \(G)")
        }
        guard ad.layers == m.graphs.blocks.count, ad.hidden == d else {
            throw H3Error.shape("the AdaLN table is [\(ad.count), \(ad.layers), 3, 6, \(ad.hidden)], the manifest has \(m.graphs.blocks.count) blocks of hidden \(d)")
        }
        let (unique, inv) = rowTimesteps(
            layout: p.layout, videoT: step.tVideo, audioT: step.tAudio,
            condVideoT: max(step.tVideo, m.sampler.condVideoT), condAudioT: m.sampler.condAudioT
        )
        let uOf = try unique.map { try ad.indexOf($0) }
        func shift(_ i: Int) -> Int { i < p.nText ? i : i + p.pad }
        var tIdx = [Int](repeating: 0, count: s)
        for (i, ti) in inv.enumerated() {
            tIdx[shift(i)] = ti
        }
        for i in p.nText..<(p.nText + p.pad) {
            tIdx[i] = inv[0]
        }
        var h = Tensor.zeros([s, d])
        h.setRows(0..<(p.nText + p.pad), p.text)
        let embV = try store.fixed(m.graphs.embedVideo), embA = try store.fixed(m.graphs.embedAudio)
        for r in runs(p.videoIdx) {
            let out = try rows(embV, chunk: C, rows: ["rows": p.videoRows.rows(r.offset..<(r.offset + r.rows))], fixed: [:], real: r.rows)
            h.setRows(r.range, try one(out, of: m.graphs.embedVideo))
        }
        for r in runs(p.audioIdx) {
            let out = try rows(embA, chunk: C, rows: ["rows": p.audioRows.rows(r.offset..<(r.offset + r.rows))], fixed: [:], real: r.rows)
            h.setRows(r.range, try one(out, of: m.graphs.embedAudio))
        }
        let mruns = modRuns(tIdx: tIdx, tags: p.tags)
        let attnGraph = try store.fixed(attn.file)
        let cq = attn.qChunk
        var q = Tensor.zeros([s, H, hd])
        var k = Tensor.zeros([s, H, hd])
        var v = Tensor.zeros([s, H, hd])
        var o = Tensor.zeros([s, inner])
        for (l, b) in m.graphs.blocks.enumerated() {
            let gq = try store.block(b.qkv)
            for mr in mruns {
                let mods = ad.mods(u: uOf[mr.tIdx], layer: l, tag: mr.tag)
                let outs = try rows(
                    gq, chunk: C, rows: ["h": h.rows(mr.range), "cos": p.cos.rows(mr.range), "sin": p.sin.rows(mr.range)],
                    fixed: ["mods": mods], real: mr.rows
                )
                guard outs.count == 3 else {
                    throw H3Error.graph("\(b.qkv): expected 3 outputs (q, k, v), got \(outs.count)")
                }
                q.setRows(mr.range, outs[0])
                k.setRows(mr.range, outs[1])
                v.setRows(mr.range, outs[2])
            }
            for g in 0..<(H / G) {
                let qg = Pipeline.headGroup(q, group: g, size: G, heads: H, headDim: hd)
                let kg = Pipeline.headGroup(k, group: g, size: G, heads: H, headDim: hd)
                let vg = Pipeline.headGroup(v, group: g, size: G, heads: H, headDim: hd)
                let og = try one(try rows(attnGraph, chunk: cq, rows: ["q": qg], fixed: ["k": kg, "v": vg, "mask": p.mask], real: s), of: attn.file)
                Pipeline.scatterHeads(og, into: &o, group: g, size: G, headDim: hd)
            }
            let gp = try store.block(b.post)
            for mr in mruns {
                let mods = ad.mods(u: uOf[mr.tIdx], layer: l, tag: mr.tag)
                let hOut = try one(try rows(gp, chunk: C, rows: ["h": h.rows(mr.range), "o": o.rows(mr.range)], fixed: ["mods": mods], real: mr.rows), of: b.post)
                h.setRows(mr.range, hOut)
            }
        }
        var extraV: [String: Tensor] = [:], extraA: [String: Tensor] = [:]
        if m.sampler.rule == .pdd {
            guard let wv = step.pddVideoWeights, let wa = step.pddAudioWeights else {
                throw H3Error.manifest("a pdd step carries no pdd_video_weights / pdd_audio_weights")
            }
            extraV["weights"] = Tensor(shape: [wv.count], data: wv.map { Float($0) })
            extraA["weights"] = Tensor(shape: [wa.count], data: wa.map { Float($0) })
        }
        var vOut = Tensor.zeros([p.videoIdx.count, c.videoPatchDim])
        var aOut = Tensor.zeros([p.audioIdx.count, c.audioInChannels])
        func heads(_ file: String, idx: [Int], extra: [String: Tensor], into out: inout Tensor) throws {
            let graph = try store.fixed(file)
            for r in runs(idx) {
                // A run of one modality may still span two timesteps (condition rows vs target rows).
                var start = r.start
                while start < r.start + r.rows {
                    let ti = tIdx[start]
                    var end = start
                    while end < r.start + r.rows && tIdx[end] == ti {
                        end += 1
                    }
                    let n = end - start
                    let off = r.offset + (start - r.start)
                    var fixed = extra
                    fixed["mods"] = ad.headMods(u: uOf[ti])
                    let res = try one(try rows(graph, chunk: C, rows: ["h": h.rows(start..<end)], fixed: fixed, real: n), of: file)
                    out.setRows(off..<(off + n), res)
                    start = end
                }
            }
        }
        try heads(m.graphs.headVideo, idx: p.videoIdx, extra: extraV, into: &vOut)
        try heads(m.graphs.headAudio, idx: p.audioIdx, extra: extraA, into: &aOut)
        return (vOut, aOut)
    }

    // MARK: denoise

    /// The target video latent `[C, T, H, W]` (normalised) and the target audio rows `[2A, A_c]`.
    public func denoise(_ req: Request, onEvent on: OnEvent? = nil, timings: inout Timings) throws -> (latent: Tensor, audioRows: Tensor) {
        let c = cfg, m = manifest
        let t0 = Date()
        var p = try prepare(req, encodeClip: encoderTileRunner())
        let attn = try m.attentionFor(tokens: p.rows)
        timings.prepareSeconds = Date().timeIntervalSince(t0)
        on?(.prepared(seconds: timings.prepareSeconds, tokens: p.layout.sequenceLength))
        let rule = m.sampler.rule
        let steps = m.sampler.steps
        guard let first = steps.first else {
            throw H3Error.manifest("the sampler has no steps")
        }
        let offV = p.layout.numConditionVideoRows
        let nVideo = p.videoRows.rows
        if rule == .pdd {
            let sv = Float(first.sigmaVideo), sa = Float(first.sigmaAudio)
            let stride = p.videoRows.rowStride
            for i in (offV * stride)..<(nVideo * stride) {
                p.videoRows.data[i] *= sv
            }
            p.audioRows = p.audioRows.scaled(sa)
        }
        var renoise = Rng(seed: req.seed &+ 1)
        for (i, st) in steps.enumerated() {
            let ts = Date()
            let (vv, aa) = try forward(p, step: st, attention: attn)
            let tgt = p.videoRows.rows(offV..<nVideo)
            var nz: Tensor? = nil
            if Sampler.needsNoise(rule: rule, next: st.sigmaVideoNext) {
                nz = renoise.normal(shape: tgt.shape)
            }
            let newTgt = try Sampler.flowStep(x: tgt, v: vv.rows(offV..<nVideo), sigma: st.sigmaVideo, next: st.sigmaVideoNext, rule: rule, noise: nz)
            p.videoRows.setRows(offV..<nVideo, newTgt)
            nz = nil
            if Sampler.needsNoise(rule: rule, next: st.sigmaAudioNext) {
                nz = renoise.normal(shape: p.audioRows.shape)
            }
            p.audioRows = try Sampler.flowStep(x: p.audioRows, v: aa, sigma: st.sigmaAudio, next: st.sigmaAudioNext, rule: rule, noise: nz)
            let secs = Date().timeIntervalSince(ts)
            timings.stepSeconds.append(secs)
            on?(.step(index: i, total: steps.count, seconds: secs))
        }
        let (lt, lh, lw) = try req.geometry.latent(spatialRatio: m.vae.spatialRatio)
        let latent = unpatchRows(p.videoRows.rows(offV..<nVideo), c: c.inChannels, t: lt, h: lh, w: lw, patch: c.patch)
        return (latent, p.audioRows)
    }

    public func denoise(_ req: Request, onEvent on: OnEvent? = nil) throws -> (latent: Tensor, audioRows: Tensor) {
        var tm = Timings()
        return try denoise(req, onEvent: on, timings: &tm)
    }

    // MARK: decode

    /// Normalised latent `[C, T, h, w]` → the clip, through the VAE decoder (tiles, chunks) or TAEH3 (`preview`).
    public func decodeVideo(latent: Tensor, preview: Bool = false, onEvent on: OnEvent? = nil, timings: inout Timings) throws -> Video {
        let t0 = Date()
        defer { timings.decodeSeconds = Date().timeIntervalSince(t0) }
        guard latent.rank == 4 else { throw H3Error.shape("latent is \(latent.shape), expected [C, T, h, w]") }
        if preview {
            return try decodeTaeh3(latent: latent, onEvent: on)
        }
        let m = manifest
        guard let vv = m.videoVae else {
            throw H3Error.manifest("this export has no video VAE decoder")
        }
        let dec = vv.decoder
        let ratio = m.vae.spatialRatio
        let gIn = try store.fixed(dec.input), gOut = try store.fixed(dec.output)
        let blocks = try dec.blocks.map { try store.fixed($0) }
        let clipTile: (Tensor) throws -> Tensor = { z in
            guard Array(z.shape.dropFirst()) == [dec.latentT, dec.latentH, dec.latentW] else {
                throw H3Error.shape("decoder tile \(Array(z.shape.dropFirst())) but the export has graphs for [\(dec.latentT), \(dec.latentH), \(dec.latentW)]")
            }
            let outs = try gIn.run(["z": z])
            guard outs.count == 3 else {
                throw H3Error.graph("\(dec.input): expected 3 outputs (h, cos, sin), got \(outs.count)")
            }
            var h = outs[0]
            let cos = outs[1], sin = outs[2]
            for (i, b) in blocks.enumerated() {
                h = try self.one(try b.run(["h": h, "cos": cos, "sin": sin]), of: dec.blocks[i])
            }
            return try self.one(try gOut.run(["h": h]), of: dec.output)
        }
        var rgba: [UInt8] = []
        var total = 0
        var width = 0, height = 0
        _ = try VaeHost.decode(
            m.vae, z: VaeHost.denormalizeLatents(m.vae, latent),
            decodeClip: { z in try VaeHost.tiled(z, ratioIn: ratio, ratioOut: 1, tilePx: vv.tile, overlapPx: vv.overlap, run: clipTile) },
            onChunk: { px in
                let conv = Pixels.rgbaFromChannelMajor(px)
                width = conv.width
                height = conv.height
                rgba.append(contentsOf: conv.rgba)
                on?(.frames(Frames(first: total, count: conv.frames, width: conv.width, height: conv.height, rgba: conv.rgba)))
                total += conv.frames
            },
            collect: false
        )
        return Video(frames: total, width: width, height: height, rgba: rgba)
    }

    public func decodeVideo(latent: Tensor, preview: Bool = false, onEvent on: OnEvent? = nil) throws -> Video {
        var tm = Timings()
        return try decodeVideo(latent: latent, preview: preview, onEvent: on, timings: &tm)
    }

    /// TAEH3: chunks of `frames` latent frames, nine memories carried, raw frame `i` kept when `i mod 20 ≥ 3`.
    func decodeTaeh3(latent: Tensor, onEvent on: OnEvent?) throws -> Video {
        let m = manifest
        guard let tc = m.taeh3 else {
            throw H3Error.manifest("this export has no TAEH3 graph")
        }
        let c = latent.shape[0], t = latent.shape[1], h = latent.shape[2], w = latent.shape[3]
        let entry = try m.taeh3For(latentH: h, latentW: w)
        let graph = try store.fixed(entry.file)
        let f = entry.frames
        var mems: [Tensor] = []
        for s in 0..<3 {
            for _ in 0..<tc.memBlocks {
                mems.append(Tensor.zeros([1, tc.widths[s], h << s, w << s]))
            }
        }
        let up = tc.tUpscale
        let plane = h * w
        var rgba: [UInt8] = []
        var rawIndex = 0, total = 0
        var width = 0, height = 0
        var f0 = 0
        while f0 < t {
            let n = min(f, t - f0)
            var z = Tensor.zeros([f, c, h, w])
            for fi in 0..<n {
                for ch in 0..<c {
                    let src = (ch * t + f0 + fi) * plane
                    let dst = (fi * c + ch) * plane
                    z.data.replaceSubrange(dst..<(dst + plane), with: latent.data[src..<(src + plane)])
                }
            }
            var inputs: [String: Tensor] = ["z": z]
            for (i, mem) in mems.enumerated() {
                inputs["mem_\(i)"] = mem
            }
            let res = try graph.run(inputs)
            guard res.count == 1 + mems.count else {
                throw H3Error.graph("\(entry.file): expected \(1 + mems.count) outputs (rgb + \(mems.count) memories), got \(res.count)")
            }
            let rgbAll = res[0]
            guard rgbAll.rank == 4, rgbAll.shape[1] == 3, rgbAll.rows >= n * up else {
                throw H3Error.shape("\(entry.file): rgb is \(rgbAll.shape), expected [\(f * up), 3, H, W]")
            }
            let rgb = rgbAll.rows(0..<(n * up))
            mems = Array(res[1...])
            let keep = (0..<rgb.rows).filter { TaeH3Config.keepsRawFrame(rawIndex + $0) }
            rawIndex += rgb.rows
            if !keep.isEmpty {
                let kept = Tensor.concatRows(keep.map { rgb.rows($0..<($0 + 1)) })
                let conv = Pixels.rgbaFromFrameMajor(kept)
                width = conv.width
                height = conv.height
                rgba.append(contentsOf: conv.rgba)
                on?(.frames(Frames(first: total, count: conv.frames, width: conv.width, height: conv.height, rgba: conv.rgba)))
                total += conv.frames
            }
            f0 += f
        }
        return Video(frames: total, width: width, height: height, rgba: rgba)
    }

    /// Normalised target audio rows `[2A, A_c]` → stereo `[2, 800·A]` (two mono decodes).
    public func decodeAudio(audioRows: Tensor, latents na: Int, timings: inout Timings) throws -> Audio {
        let t0 = Date()
        defer { timings.audioSeconds = Date().timeIntervalSince(t0) }
        let ac = manifest.audioVae
        let lc = ac.latentChannels
        guard audioRows.shape == [audioChannels * na, lc] else {
            throw H3Error.shape("audio rows are \(audioRows.shape), expected [\(audioChannels * na), \(lc)]")
        }
        let entry = try manifest.audioFor(latents: na)
        let g = try store.fixed(entry.decoder)
        var data: [Float] = []
        var samples = 0
        for ch in 0..<audioChannels {
            var z = [Float](repeating: 0, count: lc * na)
            for i in 0..<na {
                for k in 0..<lc {
                    z[k * na + i] = audioRows.data[(ch * na + i) * lc + k] * ac.latentsStd[k] + ac.latentsMean[k]
                }
            }
            let wave = try one(try g.run(["z": Tensor(shape: [lc, na], data: z)]), of: entry.decoder)
            if ch == 0 {
                samples = wave.count
            } else if wave.count != samples {
                throw H3Error.shape("\(entry.decoder): channel \(ch) has \(wave.count) samples, channel 0 had \(samples)")
            }
            data.append(contentsOf: wave.data)
        }
        return Audio(sampleRate: ac.sampleRate, channels: audioChannels, samples: samples, data: data)
    }

    public func decodeAudio(audioRows: Tensor, latents na: Int) throws -> Audio {
        var tm = Timings()
        return try decodeAudio(audioRows: audioRows, latents: na, timings: &tm)
    }

    // MARK: generate

    public func generate(_ req: Request, onEvent on: OnEvent? = nil) throws -> (video: Video, audio: Audio, timings: Timings) {
        var tm = Timings()
        let (latent, audioRows) = try denoise(req, onEvent: on, timings: &tm)
        let video = try decodeVideo(latent: latent, preview: req.preview, onEvent: on, timings: &tm)
        let audio = try decodeAudio(audioRows: audioRows, latents: req.geometry.audioLatents, timings: &tm)
        on?(.audio(audio))
        return (video, audio, tm)
    }
}
