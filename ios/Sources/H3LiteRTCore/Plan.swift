// Memory plan (docs/design.md "Tiers"): what a generation holds at once,
// before anything is allocated: resident blocks × block bytes + the AdaLN
// table + `q/k/v/o/h` at `S × 7168 / 5376` f32 + the attention score chunk +
// the decoder's peak. Hosts call it before `generate` to refuse what will not
// fit: a phone tier does not exist for a 33 B model, and `plan` says so.

import Foundation

public let bytesPerWeight: [String: Double] = ["f32": 4.0, "f16": 2.0, "int8": 1.0, "int4": 0.5]

public struct Plan: CustomStringConvertible {
    public var tokens: Int
    public var textBucket: Int
    public var chunk: Int
    public var qChunk: Int
    public var headGroup: Int
    public var blockBytes: Int
    public var weightsTotal: Int
    public var weightsResident: Int
    public var adalnBytes: Int
    public var activations: Int
    public var attentionScratch: Int
    public var decodePeak: Int
    public var outputBytes: Int
    public var peak: Int
    public var budget: Int
    public var fits: Bool
    public var notes: [String]

    public var description: String {
        func mib(_ b: Int) -> String { String(format: "%9.1f", Double(b) / Double(1 << 20)) }
        var lines = [
            "rows              \(tokens) (text bucket \(textBucket))",
            "weights (all)     \(mib(weightsTotal)) MiB",
            "weights resident  \(mib(weightsResident)) MiB (\(mib(blockBytes)) MiB per block)",
            "AdaLN table       \(mib(adalnBytes)) MiB",
            "DiT activations   \(mib(activations)) MiB (row chunk \(chunk))",
            "attention scratch \(mib(attentionScratch)) MiB (query chunk \(qChunk), \(headGroup) heads)",
            "decode peak       \(mib(decodePeak)) MiB",
            "frames + audio    \(mib(outputBytes)) MiB",
            "peak              \(mib(peak)) MiB of \(String(format: "%.0f", Double(budget) / Double(1 << 20))) MiB  fits: \(fits)",
        ]
        lines.append(contentsOf: notes.map { "note: \($0)" })
        return lines.joined(separator: "\n")
    }

    /// The plan of a manifest's export for one request shape: `textRows` prompt rows, `keyframes`
    /// anchors (0, 1 or 2), the VAE decoder or TAEH3 (`preview`), against a byte budget.
    public static func make(
        manifest m: Manifest, geometry g: Geometry, textRows: Int, keyframes: Int = 0, preview: Bool = false,
        budgetBytes: Int, residentBlocks: Int = 2
    ) throws -> Plan {
        let c = m.cfg
        guard let wb = bytesPerWeight[m.weights] else {
            throw H3Error.manifest("unknown weight format `\(m.weights)` (f32, f16, int8, int4)")
        }
        let ratio = m.vae.spatialRatio
        try g.validate(spatialRatio: ratio)
        let (lt, lh, lw) = try g.latent(spatialRatio: ratio)
        let na = g.audioLatents
        let bucket = try m.textRefineFor(rows: textRows)
        let anchors = Array(["first", "last"].prefix(max(0, min(keyframes, 2))))
        let layout = try buildLayout(textTags: [Int](repeating: textTag, count: textRows), lt: lt, lh: lh, lw: lw, na: na, patch: c.patch, anchors: anchors)
        let s = layout.sequenceLength + (bucket.rows - textRows)
        let attn = try m.attentionFor(tokens: s)
        let d = c.hidden, inner = c.dim, hd = c.headDim, G = m.headGroup, cq = attn.qChunk
        let a = 4  // f32
        var notes: [String] = []

        // Weights: the manifest's byte counts when it has them, the dense parameter count otherwise.
        let blockParams = 3 * d * inner + inner * d + 3 * d * c.ffnDim + 2 * d
        let blockEstimate = Int(Double(blockParams) * wb)
        let blockBytes = m.graphs.blocks.compactMap { $0.bytes }.max() ?? blockEstimate
        func fileBytes(_ f: String) -> Int { m.bytes(of: f) ?? 0 }
        let fixedFiles = [bucket.file, m.graphs.embedVideo, m.graphs.embedAudio, m.graphs.headVideo, m.graphs.headAudio, attn.file]
        var fixedBytes = fixedFiles.reduce(0) { $0 + fileBytes($1) }
        if m.files == nil {
            notes.append("the manifest has no `files` map; fixed graph sizes are not counted")
        }
        let weightsTotal = m.files?.values.reduce(0) { $0 + $1.bytes } ?? (m.graphs.blocks.count * blockBytes)
        let adalnBytes = m.bytes(of: m.adaln.file) ?? (m.adaln.timesteps.count * (c.layers * 3 * 6 * d + 2 * d) * 2)

        // DiT activations: h [S, d]; q, k, v [S, H, hd]; o [S, I]; cos / sin [S, r/2]; mask; the text bucket; the row buffers.
        let nVideo = layout.videoIndices.count, nAudio = layout.audioIndices.count
        let activations = (s * (d + 3 * inner + inner) + 2 * s * (c.rotaryDim / 2) + s + bucket.rows * d
            + nVideo * c.videoPatchDim + nAudio * c.audioInChannels + m.chunk * (d + inner + 3 * inner)) * a
        // One attention call: the gathered k, v [S, G, hd], the query chunk, the [Cq, G, S] scores, the [S, G·hd] output.
        let attentionScratch = (2 * s * G * hd + cq * G * hd + G * cq * s + s * G * hd) * a

        // Decode.
        let px = lh * lw
        let decodePeak: Int
        if preview {
            guard let tae = m.taeh3 else { throw H3Error.manifest("this export has no TAEH3 graph for a preview") }
            let entry = try m.taeh3For(latentH: lh, latentW: lw)
            let f = min(entry.frames, lt)
            let nf = tae.widths
            let s0 = f * nf[0] * px
            let s1 = f * tae.tgrow[0] * nf[1] * 4 * px
            let s2 = f * tae.tgrow[0] * tae.tgrow[1] * nf[2] * 16 * px
            let out = f * tae.tUpscale * 3 * (tae.spatialScale * tae.spatialScale) * px
            decodePeak = max(max(5 * s0, 5 * s1), max(5 * s2, s2 + 2 * out)) * a
            fixedBytes += fileBytes(entry.file)
        } else {
            guard let vv = m.videoVae else { throw H3Error.manifest("this export has no video VAE decoder") }
            let dec = vv.decoder
            let r = dec.rows, dim = m.vae.decoderDim, ffn = dim * m.vae.decoderFfnMult, heads = m.vae.decoderHeads
            let rot = m.vae.decoderRotaryDim
            let tileRows = (r * dim * 5 + r * rot + heads * r * r + 2 * r * ffn) * a
            let tr = m.vae.temporalRatio
            let pixTile = 3 * dec.latentT * tr * (ratio * dec.latentH) * (ratio * dec.latentW) * a
            let (ys, _, _) = VaeHost.splitTiles(length: g.height, tile: vv.tile, minOverlap: vv.overlap, ratio: 16)
            let (xs, _, _) = VaeHost.splitTiles(length: g.width, tile: vv.tile, minOverlap: vv.overlap, ratio: 16)
            let chunkPixels = 3 * dec.latentT * tr * g.height * g.width * a
            let latentBytes = c.inChannels * lt * px * a
            decodePeak = tileRows + ys.count * xs.count * pixTile + 2 * chunkPixels + latentBytes
            let decFiles = [dec.input, dec.output] + dec.blocks
            fixedBytes += decFiles.reduce(0) { $0 + fileBytes($1) }
        }
        let audioEntry = try m.audioFor(latents: na)
        fixedBytes += fileBytes(audioEntry.decoder)
        let outputBytes = g.frames * g.height * g.width * 4 + 2 * m.audioVae.hop * na * a
        let weightsResident = residentBlocks * blockBytes + fixedBytes

        let denoisePeak = weightsResident + adalnBytes + activations + attentionScratch
        let decodeTotal = weightsResident + decodePeak + outputBytes
        let peak = max(denoisePeak, decodeTotal)
        if residentBlocks < c.layers {
            notes.append("\(residentBlocks) of \(c.layers) block graphs resident; the rest stream from storage each step")
        }
        if bucket.rows > textRows {
            notes.append("\(textRows) prompt rows padded to the \(bucket.rows)-row text bucket (\(bucket.rows - textRows) masked rows)")
        }
        if s % m.chunk != 0 {
            notes.append("runs are chunked by \(m.chunk) rows; the last chunk of each run is zero-padded")
        }
        if weightsTotal > budgetBytes {
            notes.append("the export alone (\(String(format: "%.1f", Double(weightsTotal) / Double(1 << 30))) GiB) exceeds the budget: this device cannot hold the model")
        }
        return Plan(
            tokens: s, textBucket: bucket.rows, chunk: m.chunk, qChunk: cq, headGroup: G, blockBytes: blockBytes,
            weightsTotal: weightsTotal, weightsResident: weightsResident, adalnBytes: adalnBytes, activations: activations,
            attentionScratch: attentionScratch, decodePeak: decodePeak, outputBytes: outputBytes, peak: peak, budget: budgetBytes,
            fits: peak <= budgetBytes, notes: notes
        )
    }
}
