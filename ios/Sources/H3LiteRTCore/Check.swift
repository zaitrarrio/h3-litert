// The runtime self-test (`h3-litert check`): run the tiny fixture through this
// loop on the given store and compare with `expected.safetensors`.

import Foundation

public struct CheckResult: CustomStringConvertible {
    public var weights: String
    public var latentRel: Double
    public var audioRowsRel: Double
    public var frameMaxDiff: Int
    public var waveformRel: Double
    public var tolerance: Double
    public var frameTolerance: Int
    public var seconds: Double

    public var ok: Bool {
        latentRel <= tolerance && audioRowsRel <= tolerance && frameMaxDiff <= frameTolerance && waveformRel <= 10 * tolerance
    }

    public var description: String {
        func e(_ v: Double) -> String { String(format: "%.2e", v) }
        return "[\(weights)] video latent rel \(e(latentRel)), audio rows rel \(e(audioRowsRel)), frames max |Δ| \(frameMaxDiff)/255, "
            + "waveform rel \(e(waveformRel)) (tol \(String(format: "%.0e", tolerance))) in \(String(format: "%.1f", seconds)) s: \(ok ? "OK" : "FAIL")"
    }
}

public enum FixtureCheck {
    /// `rel = max|Δ| / max|expected|`.
    public static func rel(_ got: [Float], _ want: [Float]) -> Double {
        var maxDiff: Float = 0, maxAbs: Float = 0
        for i in 0..<want.count {
            maxDiff = max(maxDiff, abs(got[i] - want[i]))
            maxAbs = max(maxAbs, abs(want[i]))
        }
        return Double(maxDiff) / Double(max(maxAbs, 1e-6))
    }

    /// `directory` holds the export (`manifest.json`, the graphs, `adaln.safetensors`) and
    /// `expected.safetensors` (prompt_embeds, text_tags, latent, audio_rows, frames, audio; metadata seed).
    /// The f32 fixture is judged at `tolerance` (1e-3) and 2/255; f16 / int8 / int4 exports at 2e-2 and 16/255.
    public static func run(store: GraphStore, directory: URL, tolerance: Double = 1e-3, onEvent: OnEvent? = nil) throws -> CheckResult {
        let m = store.manifest
        let adaln = try AdalnTables(url: directory.appendingPathComponent(m.adaln.file))
        let pipe = Pipeline(store: store, adaln: adaln)
        let exp = try SafeTensors(url: directory.appendingPathComponent("expected.safetensors"))
        guard let size = m.sizes.first else { throw H3Error.manifest("manifest lists no sizes") }
        let seed = UInt64(exp.metadata["seed"] ?? "7") ?? 7
        let embeds = try exp.float32("prompt_embeds")
        let tags: [Int]
        if exp.contains("text_tags") {
            tags = try exp.float32("text_tags").data.map { Int($0) }
        } else {
            tags = [Int](repeating: textTag, count: embeds.rows)
        }
        let req = Request(embeds: embeds, textTags: tags, geometry: size, seed: seed)
        let t0 = Date()
        let (latent, audioRows) = try pipe.denoise(req, onEvent: onEvent)
        let wantLatent = try exp.float32("latent")
        guard wantLatent.shape == latent.shape else {
            throw H3Error.shape("latent is \(latent.shape), expected \(wantLatent.shape)")
        }
        let wantRows = try exp.float32("audio_rows")
        guard wantRows.shape == audioRows.shape else {
            throw H3Error.shape("audio rows are \(audioRows.shape), expected \(wantRows.shape)")
        }
        let latentRel = rel(latent.data, wantLatent.data)
        let rowsRel = rel(audioRows.data, wantRows.data)
        let video = try pipe.decodeVideo(latent: latent, onEvent: onEvent)
        let (fshape, frames) = try exp.uint8("frames")
        guard fshape == [video.frames, video.height, video.width, 3] else {
            throw H3Error.shape("frames are [\(video.frames), \(video.height), \(video.width), 3], expected \(fshape)")
        }
        let got = video.rgb
        var fdiff = 0
        for i in 0..<frames.count {
            fdiff = max(fdiff, abs(Int(got[i]) - Int(frames[i])))
        }
        let audio = try pipe.decodeAudio(audioRows: audioRows, latents: size.audioLatents)
        let wantAudio = try exp.float32("audio")
        guard wantAudio.shape == [audio.channels, audio.samples] else {
            throw H3Error.shape("audio is [\(audio.channels), \(audio.samples)], expected \(wantAudio.shape)")
        }
        let waveRel = rel(audio.data, wantAudio.data)
        let f32 = m.weights == "f32"
        return CheckResult(
            weights: m.weights, latentRel: latentRel, audioRowsRel: rowsRel, frameMaxDiff: fdiff, waveformRel: waveRel,
            tolerance: f32 ? tolerance : max(tolerance, 2e-2), frameTolerance: f32 ? 2 : 16, seconds: Date().timeIntervalSince(t0)
        )
    }
}
