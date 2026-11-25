// The pipeline's orchestration over a fake `GraphStore` that knows the
// contract's shapes (docs/graphs.md) and records every call, on the tiny
// manifest: 6 prompt rows tagged [1, 1, 0, 1, 1, 1] → text bucket 8 (2 padded
// rows), 74 audio rows, 42 video rows: S = 124; chunk 16; q_chunk 32; head
// groups of 1 of 2 heads; 2 Euler steps; the VAE decoder at one 7-frame chunk
// of 171 rows; the audio VAE at 37 latents; TAEH3 at 5 frames.

import Foundation
import XCTest
@testable import H3LiteRTCore

/// A graph that checks its inputs against the contract and returns outputs of
/// the contract's shapes, filled with `fill`.
final class FakeGraph: Graph {
    struct Call {
        var inputs: [String: Tensor]
        func shape(_ k: String) -> [Int]? { inputs[k]?.shape }
    }

    let name: String
    let inputShapes: [String: [Int]]
    let outputShapes: [[Int]]
    var fill: (Int, [String: Tensor]) -> Float  // (output index, inputs) → value
    private(set) var calls: [Call] = []

    init(name: String, inputs: [String: [Int]], outputs: [[Int]], fill: @escaping (Int, [String: Tensor]) -> Float = { _, _ in 0 }) {
        self.name = name
        self.inputShapes = inputs
        self.outputShapes = outputs
        self.fill = fill
    }

    func run(_ inputs: [String: Tensor]) throws -> [Tensor] {
        guard Set(inputs.keys) == Set(inputShapes.keys) else {
            throw H3Error.graph("\(name): inputs \(inputShapes.keys.sorted()), got \(inputs.keys.sorted())")
        }
        for (k, t) in inputs {
            guard t.shape == inputShapes[k]! else {
                throw H3Error.shape("\(name): input `\(k)` is \(t.shape), the graph wants \(inputShapes[k]!)")
            }
        }
        calls.append(Call(inputs: inputs))
        return outputShapes.enumerated().map { (i, shape) in
            Tensor(shape: shape, data: [Float](repeating: fill(i, inputs), count: shape.reduce(1, *)))
        }
    }
}

/// Every graph of the tiny manifest with the contract's shapes.
final class FakeStore: GraphStore {
    let manifest: Manifest
    private(set) var graphs: [String: FakeGraph] = [:]
    private(set) var blockOpens: [String] = []
    let cache = ResidentGraphCache(capacity: 1)

    init(manifest m: Manifest) {
        self.manifest = m
        let c = m.cfg
        let d = c.hidden, H = c.heads, hd = c.headDim, inner = c.dim, C = m.chunk, G = m.headGroup
        let P = c.videoPatchDim, Ac = c.audioInChannels, half = c.rotaryDim / 2
        func add(_ g: FakeGraph) { graphs[g.name] = g }
        for t in m.graphs.textRefine {
            add(FakeGraph(name: t.file, inputs: ["embeds": [t.rows, c.textDim], "mask": [t.rows]], outputs: [[t.rows, d]]))
        }
        add(FakeGraph(name: m.graphs.embedVideo, inputs: ["rows": [C, P]], outputs: [[C, d]]))
        add(FakeGraph(name: m.graphs.embedAudio, inputs: ["rows": [C, Ac]], outputs: [[C, d]]))
        add(FakeGraph(name: m.graphs.headVideo, inputs: ["h": [C, d], "mods": [2, d]], outputs: [[C, P]]))
        add(FakeGraph(name: m.graphs.headAudio, inputs: ["h": [C, d], "mods": [2, d]], outputs: [[C, Ac]]))
        for b in m.graphs.blocks {
            add(FakeGraph(name: b.qkv, inputs: ["h": [C, d], "mods": [6, d], "cos": [C, half], "sin": [C, half]], outputs: [[C, H, hd], [C, H, hd], [C, H, hd]]))
            add(FakeGraph(name: b.post, inputs: ["h": [C, d], "o": [C, inner], "mods": [6, d]], outputs: [[C, d]]))
        }
        for a in m.attention {
            add(FakeGraph(name: a.file, inputs: ["q": [a.qChunk, G, hd], "k": [a.tokens, G, hd], "v": [a.tokens, G, hd], "mask": [a.tokens]], outputs: [[a.qChunk, G * hd]]))
        }
        let vv = m.videoVae!
        let dec = vv.decoder
        let vc = m.vae
        let R = dec.rows, dd = vc.decoderDim, dHalf = vc.decoderRotaryDim / 2, ratio = vc.spatialRatio, tr = vc.temporalRatio
        add(FakeGraph(name: dec.input, inputs: ["z": [vc.latentChannels, dec.latentT, dec.latentH, dec.latentW]], outputs: [[R, dd], [R, dHalf], [R, dHalf]]))
        for b in dec.blocks {
            add(FakeGraph(name: b, inputs: ["h": [R, dd], "cos": [R, dHalf], "sin": [R, dHalf]], outputs: [[R, dd]]))
        }
        add(FakeGraph(name: dec.output, inputs: ["h": [R, dd]], outputs: [[3, dec.latentT * tr, ratio * dec.latentH, ratio * dec.latentW]]))
        let pxH = ratio * dec.latentH, pxW = ratio * dec.latentW
        add(FakeGraph(name: vv.encoderClip!, inputs: ["x": [3, vc.clipLength, pxH, pxW]], outputs: [[2 * vc.latentChannels, vc.tokensChunk, dec.latentH, dec.latentW]]))
        add(FakeGraph(name: vv.encoderImage!, inputs: ["x": [3, 1, pxH, pxW]], outputs: [[2 * vc.latentChannels, 1, dec.latentH, dec.latentW]]))
        let ac = m.audioVae
        for a in m.audio {
            add(FakeGraph(name: a.decoder, inputs: ["z": [ac.latentChannels, a.latents]], outputs: [[ac.hop * a.latents]]))
            if let e = a.encoder {
                add(FakeGraph(name: e, inputs: ["wave": [ac.hop * a.latents]], outputs: [[ac.latentChannels, a.latents]]))
            }
        }
        let tae = m.taeh3!
        for t in m.taeh3Graphs {
            var taeIn: [String: [Int]] = ["z": [t.frames, tae.latentChannels, t.latentH, t.latentW]]
            var taeOut: [[Int]] = [[tae.tUpscale * t.frames, 3, tae.spatialScale * t.latentH, tae.spatialScale * t.latentW]]
            for i in 0..<(3 * tae.memBlocks) {
                let s = i / tae.memBlocks
                let shape = [1, tae.widths[s], t.latentH << s, t.latentW << s]
                taeIn["mem_\(i)"] = shape
                taeOut.append(shape)
            }
            add(FakeGraph(name: t.file, inputs: taeIn, outputs: taeOut))
        }
    }

    func fixed(_ file: String) throws -> Graph {
        guard let g = graphs[file] else { throw H3Error.io("no graph \(file)") }
        return g
    }

    func block(_ file: String) throws -> Graph {
        try cache.get(file) {
            guard let g = graphs[file] else { throw H3Error.io("no graph \(file)") }
            blockOpens.append(file)
            return g
        }
    }

    func dropBlocks() { cache.clear() }

    func calls(_ file: String) -> [FakeGraph.Call] { graphs[file]?.calls ?? [] }
}

final class PipelineTests: XCTestCase {
    let tags = [1, 1, 0, 1, 1, 1]
    let maskRows = 124

    func makeStore() throws -> FakeStore {
        FakeStore(manifest: try Manifest(json: tinyManifestData()))
    }

    func makePipeline(_ store: FakeStore) throws -> Pipeline {
        Pipeline(store: store, adaln: try AdalnTables(url: tinyAdalnURL()))
    }

    func request(seed: UInt64 = 7, keyframes: [Keyframe] = [], preview: Bool = false) -> Request {
        Request(embeds: Tensor.zeros([6, 24]), textTags: tags, geometry: .tiny, seed: seed, keyframes: keyframes, preview: preview)
    }

    func testDenoiseDrivesTheGraphsAsTheContractSays() throws {
        let store = try makeStore()
        let pipe = try makePipeline(store)
        let m = store.manifest
        var events: [PipelineEvent] = []
        var tm = Timings()
        let (latent, audioRows) = try pipe.denoise(request(), onEvent: { events.append($0) }, timings: &tm)
        XCTAssertEqual(latent.shape, [4, 7, 4, 6])
        XCTAssertEqual(audioRows.shape, [74, 6])
        XCTAssertEqual(tm.stepSeconds.count, 2)

        // Once per prompt: the 8-row bucket with the 2 padded rows masked; the 16-row bucket untouched.
        let refine = store.calls("text_refine_8.tflite")
        XCTAssertEqual(refine.count, 1)
        XCTAssertEqual(refine[0].shape("embeds"), [8, 24])
        XCTAssertEqual(refine[0].inputs["mask"]!.data, [0, 0, 0, 0, 0, 0, -1e9, -1e9])
        XCTAssertEqual(store.calls("text_refine_16.tflite").count, 0)
        // Per step: video rows 42 in chunks of 16 → 3 embed calls; audio rows 74 → 5.
        XCTAssertEqual(store.calls(m.graphs.embedVideo).count, 2 * 3)
        XCTAssertEqual(store.calls(m.graphs.embedAudio).count, 2 * 5)
        // Modulation runs: text [1,1] | vision [0] | text [1,1,1] + 2 pad rows | audio 74 | video 42
        // → 1 + 1 + 1 + 5 + 3 = 11 qkv calls per block per step, post the same.
        for b in m.graphs.blocks {
            XCTAssertEqual(store.calls(b.qkv).count, 2 * 11, b.qkv)
            XCTAssertEqual(store.calls(b.post).count, 2 * 11, b.post)
            for call in store.calls(b.qkv) {
                XCTAssertEqual(call.shape("h"), [16, 32])
                XCTAssertEqual(call.shape("cos"), [16, 6])
                XCTAssertEqual(call.shape("mods"), [6, 32])
            }
            for call in store.calls(b.post) {
                XCTAssertEqual(call.shape("o"), [16, 32])
            }
        }
        // Attention: 2 head groups × 4 query chunks per block per step; keys, values and the mask are
        // exactly S = 124 long with −1e9 at the padded text rows 6 and 7.
        let attn = store.calls("attn_s124_q32_h1.tflite")
        XCTAssertEqual(attn.count, 2 * 2 * 4 * 2)
        for call in attn {
            XCTAssertEqual(call.shape("q"), [32, 1, 16])
            XCTAssertEqual(call.shape("k"), [124, 1, 16])
            XCTAssertEqual(call.shape("v"), [124, 1, 16])
            let mask = call.inputs["mask"]!
            XCTAssertEqual(mask.shape, [124])
            for i in 0..<124 {
                XCTAssertEqual(mask.data[i], (i == 6 || i == 7) ? -1e9 : 0, "mask[\(i)]")
            }
        }
        for a in m.attention where a.tokens != 124 {
            XCTAssertEqual(store.calls(a.file).count, 0, a.file)
        }
        // Heads: one video run (42 rows → 3 calls) and one audio run (74 → 5) per step, each with a [2, d] row.
        XCTAssertEqual(store.calls(m.graphs.headVideo).count, 2 * 3)
        XCTAssertEqual(store.calls(m.graphs.headAudio).count, 2 * 5)
        XCTAssertEqual(store.calls(m.graphs.headVideo)[0].shape("mods"), [2, 32])
        // Events: prepared (the unpadded 122 rows), then one step per sampler step.
        XCTAssertEqual(events.count, 3)
        if case .prepared(_, let tokens) = events[0] {
            XCTAssertEqual(tokens, 122)
        } else {
            XCTFail("first event is not prepared")
        }
        for i in 0..<2 {
            if case .step(let index, let total, _) = events[i + 1] {
                XCTAssertEqual(index, i)
                XCTAssertEqual(total, 2)
            } else {
                XCTFail("event \(i + 1) is not a step")
            }
        }
        // Block graphs open in order; the LRU store (capacity 1) re-opens them every step.
        XCTAssertEqual(store.blockOpens, [
            "block_00_qkv.tflite", "block_00_post.tflite", "block_01_qkv.tflite", "block_01_post.tflite",
            "block_00_qkv.tflite", "block_00_post.tflite", "block_01_qkv.tflite", "block_01_post.tflite",
        ])
    }

    func testPreparedLayoutShiftsPastThePaddedText() throws {
        let store = try makeStore()
        let pipe = try makePipeline(store)
        let p = try pipe.prepare(request(), encodeClip: pipe.encoderTileRunner())
        XCTAssertEqual(p.nText, 6)
        XCTAssertEqual(p.pad, 2)
        XCTAssertEqual(p.rows, 124)
        XCTAssertEqual(p.layout.sequenceLength, 122)
        XCTAssertEqual(p.text.shape, [8, 32])
        XCTAssertEqual(p.videoRows.shape, [42, 16])
        XCTAssertEqual(p.audioRows.shape, [74, 6])
        XCTAssertEqual(p.cos.shape, [124, 6])
        XCTAssertEqual(p.mask.shape, [124])
        XCTAssertEqual(p.audioIdx.first, 8)
        XCTAssertEqual(p.audioIdx.last, 81)
        XCTAssertEqual(p.videoIdx.first, 82)
        XCTAssertEqual(p.videoIdx.last, 123)
        XCTAssertEqual(Array(p.tags[0..<8]), [1, 1, 0, 1, 1, 1, 1, 1])
        XCTAssertEqual(p.tags[8], audioTag)
        XCTAssertEqual(p.tags[82], videoTag)
        // Padded rows sit at position 0: cos 1, sin 0.
        XCTAssert(p.cos.row(6).data.allSatisfy { $0 == 1.0 })
        XCTAssert(p.sin.row(7).data.allSatisfy { $0 == 0.0 })
        // The first audio row's clock is n_text = 6 (not 8): the layout is built with the real rows.
        XCTAssertEqual(p.layout.positionIds[p.layout.audioIndices[0]].t, 6.0)
        // The noise order: the video latent first, then the audio rows, from one Rng(seed).
        var rng = Rng(seed: 7)
        let video = patchRows(rng.normal(shape: [4, 7, 4, 6]), patch: [1, 2, 2])
        XCTAssertEqual(p.videoRows, video)
        XCTAssertEqual(p.audioRows, rng.normal(shape: [74, 6]))
    }

    func testNoiseOrderIsLatentThenAudioThenRenoise() throws {
        // With zero velocity from the fake heads, the rows are a pure function of the noise stream and
        // the sampler: the same flow steps applied to the same draws give the same bits.
        let store = try makeStore()
        let pipe = try makePipeline(store)
        let (latent, audioRows) = try pipe.denoise(request())
        var rng = Rng(seed: 7)
        var video = patchRows(rng.normal(shape: [4, 7, 4, 6]), patch: [1, 2, 2])
        var audio = rng.normal(shape: [74, 6])
        for st in store.manifest.sampler.steps {
            video = try Sampler.flowStep(x: video, v: Tensor.zeros(video.shape), sigma: st.sigmaVideo, next: st.sigmaVideoNext, rule: .euler, noise: nil)
            audio = try Sampler.flowStep(x: audio, v: Tensor.zeros(audio.shape), sigma: st.sigmaAudio, next: st.sigmaAudioNext, rule: .euler, noise: nil)
        }
        XCTAssertEqual(latent, unpatchRows(video, c: 4, t: 7, h: 4, w: 6, patch: [1, 2, 2]))
        XCTAssertEqual(audioRows, audio)
        let (other, _) = try pipe.denoise(request(seed: 8))
        XCTAssertNotEqual(other, latent)
        let (again, _) = try pipe.denoise(request(seed: 7))
        XCTAssertEqual(again, latent)
    }

    func testVelocityFlowsThroughTheHeads() throws {
        let store = try makeStore()
        store.graphs["head_video.tflite"]!.fill = { _, _ in 1.0 }
        store.graphs["head_audio.tflite"]!.fill = { _, _ in -1.0 }
        let pipe = try makePipeline(store)
        let m = store.manifest
        let p = try pipe.prepare(request(), encodeClip: pipe.encoderTileRunner())
        let (v, a) = try pipe.forward(p, step: m.sampler.steps[0], attention: try m.attentionFor(tokens: 124))
        XCTAssertEqual(v.shape, [42, 16])
        XCTAssert(v.data.allSatisfy { $0 == 1.0 })
        XCTAssertEqual(a.shape, [74, 6])
        XCTAssert(a.data.allSatisfy { $0 == -1.0 })
        // The AdaLN rows of this step: video rows at t_video (table row 0), audio rows at t_audio (row 1).
        let ad = pipe.adaln
        let qkv = store.calls("block_00_qkv.tflite")
        XCTAssertEqual(qkv.count, 11)
        XCTAssertEqual(qkv[0].inputs["mods"], ad.mods(u: 0, layer: 0, tag: textTag))
        XCTAssertEqual(qkv[1].inputs["mods"], ad.mods(u: 0, layer: 0, tag: videoTag))
        XCTAssertEqual(qkv[3].inputs["mods"], ad.mods(u: 1, layer: 0, tag: audioTag))
        XCTAssertEqual(qkv[8].inputs["mods"], ad.mods(u: 0, layer: 0, tag: videoTag))
        XCTAssertEqual(store.calls("block_01_post.tflite")[10].inputs["mods"], ad.mods(u: 0, layer: 1, tag: videoTag))
        XCTAssertEqual(store.calls("head_video.tflite")[0].inputs["mods"], ad.headMods(u: 0))
        XCTAssertEqual(store.calls("head_audio.tflite")[0].inputs["mods"], ad.headMods(u: 1))
        // A wrong attention graph is refused before anything runs.
        XCTAssertThrowsError(try pipe.forward(p, step: m.sampler.steps[0], attention: try m.attentionFor(tokens: 130)))
        // Prompt rows must match the tags and the text width.
        XCTAssertThrowsError(try pipe.prepare(Request(embeds: Tensor.zeros([6, 24]), textTags: [1, 1, 1], geometry: .tiny), encodeClip: pipe.encoderTileRunner()))
        XCTAssertThrowsError(try pipe.prepare(Request(embeds: Tensor.zeros([6, 25]), textTags: tags, geometry: .tiny), encodeClip: pipe.encoderTileRunner()))
    }

    func testFl2vaKeyframeAddsConditionRows() throws {
        let store = try makeStore()
        let pipe = try makePipeline(store)
        let m = store.manifest
        let px = Tensor(shape: [3, 1, 64, 96], data: [Float](repeating: 0.5, count: 3 * 64 * 96))
        let kf = Keyframe(anchor: "first", pixels: px)
        var events: [PipelineEvent] = []
        let (latent, audioRows) = try pipe.denoise(request(keyframes: [kf]), onEvent: { events.append($0) })
        XCTAssertEqual(latent.shape, [4, 7, 4, 6])
        XCTAssertEqual(audioRows.shape, [74, 6])
        // The keyframe went through the image encoder once, at the canvas size (one 96x64 tile).
        let enc = store.calls("vae_enc_image.tflite")
        XCTAssertEqual(enc.count, 1)
        XCTAssertEqual(enc[0].shape("x"), [3, 1, 64, 96])
        XCTAssertEqual(store.calls("vae_enc_clip.tflite").count, 0)
        // 6 condition rows ahead of the audio: S = 6 + 2 + 6 + 74 + 42 = 130, the 130-row attention graph.
        if case .prepared(_, let tokens) = events[0] {
            XCTAssertEqual(tokens, 128)
        } else {
            XCTFail("first event is not prepared")
        }
        let attn = store.calls("attn_s130_q32_h1.tflite")
        XCTAssertEqual(attn.count, 2 * 2 * 5 * 2)  // 2 groups × ceil(130 / 32) chunks × 2 blocks × 2 steps
        XCTAssertEqual(store.calls("attn_s124_q32_h1.tflite").count, 0)
        for i in 0..<130 {
            XCTAssertEqual(attn[0].inputs["mask"]!.data[i], (i == 6 || i == 7) ? -1e9 : 0)
        }
        // Runs: text (3 runs) | condition video at 0.999 | audio | video → 1 + 1 + 1 + 1 + 5 + 3 = 12 qkv calls.
        XCTAssertEqual(store.calls("block_00_qkv.tflite").count, 2 * 12)
        XCTAssertEqual(store.calls(m.graphs.embedVideo).count, 2 * (1 + 3))
        // The video head sees the condition run (0.999 → table row 4) and the target run (row 0) apart.
        let heads = store.calls(m.graphs.headVideo)
        XCTAssertEqual(heads.count, 2 * (1 + 3))
        XCTAssertEqual(heads[0].inputs["mods"], pipe.adaln.headMods(u: 4))
        XCTAssertEqual(heads[1].inputs["mods"], pipe.adaln.headMods(u: 0))
        // Condition rows: posterior noise (seed 42) through f16, normalised, patched, then 0.999-augmented
        // with the request generator's first draws; the target noise follows.
        let p = try pipe.prepare(request(keyframes: [kf]), encodeClip: pipe.encoderTileRunner())
        XCTAssertEqual(p.layout.numConditionVideoRows, 6)
        XCTAssertEqual(p.videoRows.shape, [48, 16])
        var enc42 = Rng(seed: 42)
        let z = VaeHost.posteriorSample(moments: Tensor.zeros([8, 1, 4, 6]), noise: enc42.normal(shape: [4, 1, 4, 6]))
        let cond = patchRows(VaeHost.normalizeLatents(m.vae, z.map { Half.round($0) }), patch: [1, 2, 2])
        var rng = Rng(seed: 7)
        let aug = rng.normal(6 * 16)
        for i in 0..<(6 * 16) {
            XCTAssertEqual(p.videoRows.data[i], Float(0.999) * cond.data[i] + Float(1.0 - 0.999) * aug[i])
        }
        XCTAssertEqual(p.videoRows.rows(6..<48), patchRows(rng.normal(shape: [4, 7, 4, 6]), patch: [1, 2, 2]))
        // Condition rows are never written by a step.
        XCTAssertEqual(try pipe.prepare(request(keyframes: [kf]), encodeClip: pipe.encoderTileRunner()).videoRows.rows(0..<6), p.videoRows.rows(0..<6))
        XCTAssertThrowsError(try pipe.denoise(request(keyframes: [Keyframe(anchor: "middle", pixels: px)])))
        XCTAssertThrowsError(try pipe.denoise(request(keyframes: [Keyframe(anchor: "first", pixels: Tensor.zeros([3, 1, 32, 32]))])))
    }

    func testDecodeVideoRunsTheDecoderOverOneChunk() throws {
        let store = try makeStore()
        let m = store.manifest
        store.graphs["vae_dec_out.tflite"]!.fill = { _, _ in 0.5 }
        let pipe = try makePipeline(store)
        var events: [PipelineEvent] = []
        var tm = Timings()
        let latent = Tensor(shape: [4, 7, 4, 6], data: (0..<(4 * 7 * 4 * 6)).map { Float($0) * 0.01 })
        let video = try pipe.decodeVideo(latent: latent, onEvent: { events.append($0) }, timings: &tm)
        // One 7-frame chunk, one 96x64 tile: dec_in once, both blocks once, dec_out once.
        XCTAssertEqual(store.calls("vae_dec_in.tflite").count, 1)
        XCTAssertEqual(store.calls("vae_dec_in.tflite")[0].shape("z"), [4, 7, 4, 6])
        for b in m.videoVae!.decoder.blocks {
            XCTAssertEqual(store.calls(b).count, 1, b)
            XCTAssertEqual(store.calls(b)[0].shape("cos"), [171, 6])
        }
        XCTAssertEqual(store.calls("vae_dec_out.tflite").count, 1)
        // The decoder sees denormalised latents: z·std + mean per channel.
        let z = store.calls("vae_dec_in.tflite")[0].inputs["z"]!
        XCTAssertEqual(z.data[0], latent.data[0] * 1.1 + 0.1, accuracy: 1e-6)
        XCTAssertEqual(z.data[168], latent.data[168] * 0.9 - 0.2, accuracy: 1e-6)
        // 22 frames: 17 from the chunk, 5 from the trailing overlap, in two events.
        XCTAssertEqual(video.frames, 22)
        XCTAssertEqual(video.width, 96)
        XCTAssertEqual(video.height, 64)
        XCTAssertEqual(video.rgba.count, 22 * 64 * 96 * 4)
        // 0.5 in ImageNet space on the red channel: 0.5·0.229 + 0.485 → 153.
        XCTAssertEqual(video.rgba[0], 153)
        XCTAssertEqual(video.rgba[1], 145)
        XCTAssertEqual(video.rgba[2], 132)
        XCTAssertEqual(video.rgba[3], 255)
        XCTAssertEqual(video.rgb.count, 22 * 64 * 96 * 3)
        XCTAssertEqual(events.count, 2)
        if case .frames(let f) = events[0] {
            XCTAssertEqual(f.first, 0)
            XCTAssertEqual(f.count, 17)
            XCTAssertEqual(f.width, 96)
            XCTAssertEqual(f.height, 64)
            XCTAssertEqual(f.frame(16).count, 64 * 96 * 4)
        } else {
            XCTFail("first event is not frames")
        }
        if case .frames(let f) = events[1] {
            XCTAssertEqual(f.first, 17)
            XCTAssertEqual(f.count, 5)
        } else {
            XCTFail("second event is not frames")
        }
        XCTAssertGreaterThanOrEqual(tm.decodeSeconds, 0)
        XCTAssertThrowsError(try pipe.decodeVideo(latent: Tensor.zeros([4, 7, 8, 6]))) { e in
            XCTAssert("\(e)".contains("decoder tile"), "\(e)")
        }
    }

    func testTaeh3PreviewKeepsTheContractsFrames() throws {
        let store = try makeStore()
        let tae = store.graphs["taeh3_f5_4x6.tflite"]!
        var callNo = 0
        tae.fill = { i, inputs in
            if i == 0 { return 0.5 }
            // mem_0 of the second call must be the first call's memory output (1.0); the first call sees zeros.
            callNo = Int(inputs["mem_0"]!.data[0]) + 1
            return Float(callNo)
        }
        let pipe = try makePipeline(store)
        var events: [PipelineEvent] = []
        let latent = Tensor(shape: [4, 7, 4, 6], data: (0..<(4 * 7 * 4 * 6)).map { Float($0) })
        let video = try pipe.decodeVideo(latent: latent, preview: true, onEvent: { events.append($0) })
        // 7 latent frames in chunks of 5 → 2 calls; 28 raw frames, those ≥ 3 (mod 20) kept → 17 + 5 = 22.
        XCTAssertEqual(tae.calls.count, 2)
        XCTAssertEqual(callNo, 2)
        XCTAssertEqual(video.frames, 22)
        XCTAssertEqual(video.width, 96)
        XCTAssertEqual(video.height, 64)
        XCTAssertEqual(video.rgba[0], 128)
        XCTAssertEqual(video.rgba[3], 255)
        XCTAssertEqual(events.count, 2)
        if case .frames(let f) = events[0] {
            XCTAssertEqual(f.first, 0)
            XCTAssertEqual(f.count, 17)
        } else {
            XCTFail("first event is not frames")
        }
        if case .frames(let f) = events[1] {
            XCTAssertEqual(f.first, 17)
            XCTAssertEqual(f.count, 5)
        } else {
            XCTFail("second event is not frames")
        }
        // z is frame-major and zero-padded: z[fi, ch] = latent[ch, f0 + fi].
        let plane = 24
        let z0 = tae.calls[0].inputs["z"]!, z1 = tae.calls[1].inputs["z"]!
        XCTAssertEqual(z0.shape, [5, 4, 4, 6])
        for fi in 0..<5 {
            for ch in 0..<4 {
                XCTAssertEqual(z0.data[(fi * 4 + ch) * plane], latent.data[(ch * 7 + fi) * plane])
            }
        }
        for ch in 0..<4 {
            XCTAssertEqual(z1.data[ch * plane], latent.data[(ch * 7 + 5) * plane])
            XCTAssertEqual(z1.data[(1 * 4 + ch) * plane], latent.data[(ch * 7 + 6) * plane])
        }
        XCTAssert(z1.data[(2 * 4) * plane..<(5 * 4) * plane].allSatisfy { $0 == 0 })
        XCTAssertThrowsError(try pipe.decodeVideo(latent: Tensor.zeros([4, 7, 8, 6]), preview: true)) { e in
            XCTAssert("\(e)".contains("8x6"), "\(e)")
        }
    }

    func testDecodeAudioIsTwoMonoDecodes() throws {
        let store = try makeStore()
        store.graphs["avae_dec_37.tflite"]!.fill = { _, inputs in inputs["z"]!.data[0] }
        let pipe = try makePipeline(store)
        var rows = Tensor.zeros([74, 6])
        for i in 0..<rows.count { rows.data[i] = Float(i) * 0.001 }
        var tm = Timings()
        let audio = try pipe.decodeAudio(audioRows: rows, latents: 37, timings: &tm)
        XCTAssertEqual(audio.sampleRate, 320)
        XCTAssertEqual(audio.channels, 2)
        XCTAssertEqual(audio.samples, 370)
        XCTAssertEqual(audio.tensor.shape, [2, 370])
        let calls = store.calls("avae_dec_37.tflite")
        XCTAssertEqual(calls.count, 2)
        // z[k, i] = rows[ch·37 + i, k]·std[k] + mean[k]: denormalised and transposed to [lc, A].
        let z0 = calls[0].inputs["z"]!, z1 = calls[1].inputs["z"]!
        XCTAssertEqual(z0.shape, [6, 37])
        XCTAssertEqual(z0.data[1 * 37 + 2], rows.data[2 * 6 + 1] * 1.5 + 0.1, accuracy: 1e-6)
        XCTAssertEqual(z1.data[0 * 37 + 0], rows.data[37 * 6 + 0] * 1.0 + 0.0, accuracy: 1e-6)
        XCTAssertEqual(audio.channel(0).first, z0.data[0])
        XCTAssertEqual(audio.channel(1).first, z1.data[0])
        XCTAssertThrowsError(try pipe.decodeAudio(audioRows: rows, latents: 40)) { e in
            XCTAssert("\(e)".contains("40"), "\(e)")
        }
        XCTAssertThrowsError(try pipe.decodeAudio(audioRows: Tensor.zeros([74, 5]), latents: 37))
    }

    func testGenerateReportsEverythingInOrder() throws {
        let store = try makeStore()
        let pipe = try makePipeline(store)
        var kinds: [String] = []
        let (video, audio, tm) = try pipe.generate(request()) { ev in
            switch ev {
            case .prepared: kinds.append("prepared")
            case .step: kinds.append("step")
            case .frames: kinds.append("frames")
            case .audio: kinds.append("audio")
            }
        }
        XCTAssertEqual(kinds, ["prepared", "step", "step", "frames", "frames", "audio"])
        XCTAssertEqual(video.frames, 22)
        XCTAssertEqual(audio.samples, 370)
        XCTAssertEqual(tm.stepSeconds.count, 2)
        XCTAssertGreaterThanOrEqual(tm.audioSeconds, 0)
        var previewKinds: [String] = []
        let (pv, _, _) = try pipe.generate(request(preview: true)) { ev in
            if case .frames = ev { previewKinds.append("frames") }
        }
        XCTAssertEqual(pv.frames, 22)
        XCTAssertEqual(previewKinds.count, 2)
        XCTAssertEqual(store.calls("taeh3_f5_4x6.tflite").count, 2)
    }

    func testMissingGraphsAreErrors() throws {
        let store = try makeStore()
        let pipe = try makePipeline(store)
        // 160x64x22: latent (7, 4, 10), 70 video rows, S = 152: no attention graph.
        XCTAssertThrowsError(try pipe.denoise(Request(embeds: Tensor.zeros([6, 24]), textTags: tags, geometry: Geometry(width: 160, height: 64, frames: 22)))) { e in
            XCTAssert("\(e)".contains("152 rows"), "\(e)")
        }
        XCTAssertThrowsError(try pipe.denoise(Request(embeds: Tensor.zeros([6, 24]), textTags: tags, geometry: Geometry(width: 100, height: 64, frames: 22))))
        XCTAssertThrowsError(try pipe.denoise(Request(embeds: Tensor.zeros([6, 24]), textTags: tags, geometry: Geometry(width: 96, height: 64, frames: 21))))
        // 17 prompt rows exceed the largest text bucket.
        XCTAssertThrowsError(try pipe.denoise(Request(embeds: Tensor.zeros([17, 24]), textTags: [Int](repeating: 1, count: 17), geometry: .tiny))) { e in
            XCTAssert("\(e)".contains("17 rows"), "\(e)")
        }
        // A 9-row prompt takes the 16-row bucket and the 132-row attention graph.
        let (latent, _) = try pipe.denoise(Request(embeds: Tensor.zeros([9, 24]), textTags: [Int](repeating: 1, count: 9), geometry: .tiny))
        XCTAssertEqual(latent.shape, [4, 7, 4, 6])
        XCTAssertEqual(store.calls("text_refine_16.tflite").count, 1)
        XCTAssertEqual(store.calls("text_refine_16.tflite")[0].inputs["mask"]!.data.filter { $0 == -1e9 }.count, 7)
        XCTAssertEqual(store.calls("attn_s132_q32_h1.tflite").count, 2 * 2 * 5 * 2)
    }

    func testResidentGraphCacheIsLRU() throws {
        let cache = ResidentGraphCache(capacity: 2)
        var opens: [String] = []
        func get(_ f: String) throws -> Graph {
            try cache.get(f) {
                opens.append(f)
                return FakeGraph(name: f, inputs: [:], outputs: [])
            }
        }
        _ = try get("a")
        _ = try get("b")
        _ = try get("a")  // a is now the most recent
        _ = try get("c")  // evicts b
        XCTAssertEqual(cache.residentFiles, ["a", "c"])
        _ = try get("b")  // evicts a
        XCTAssertEqual(cache.residentFiles, ["c", "b"])
        XCTAssertEqual(opens, ["a", "b", "c", "b"])
        cache.clear()
        XCTAssertEqual(cache.residentFiles, [])
    }
}
