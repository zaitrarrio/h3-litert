// `manifest.json` of an export (docs/graphs.md). Decoding validates the
// format; lookups by rows / latents / tile throw with the mismatch, never fall back.

import Foundation

public let manifestFormat = "h3-litert/1"

public struct Manifest: Codable, Equatable {
    /// One step of the precomputed schedule: clean times, sigmas and (PDD) fused-head weights.
    public struct SamplerStep: Codable, Equatable {
        public var tVideo: Double
        public var tAudio: Double
        public var sigmaVideo: Double
        public var sigmaVideoNext: Double
        public var sigmaAudio: Double
        public var sigmaAudioNext: Double
        public var pddVideoWeights: [Double]?
        public var pddAudioWeights: [Double]?

        enum CodingKeys: String, CodingKey {
            case tVideo = "t_video"
            case tAudio = "t_audio"
            case sigmaVideo = "sigma_video"
            case sigmaVideoNext = "sigma_video_next"
            case sigmaAudio = "sigma_audio"
            case sigmaAudioNext = "sigma_audio_next"
            case pddVideoWeights = "pdd_video_weights"
            case pddAudioWeights = "pdd_audio_weights"
        }

        public init(
            tVideo: Double, tAudio: Double, sigmaVideo: Double, sigmaVideoNext: Double, sigmaAudio: Double, sigmaAudioNext: Double,
            pddVideoWeights: [Double]? = nil, pddAudioWeights: [Double]? = nil
        ) {
            self.tVideo = tVideo
            self.tAudio = tAudio
            self.sigmaVideo = sigmaVideo
            self.sigmaVideoNext = sigmaVideoNext
            self.sigmaAudio = sigmaAudio
            self.sigmaAudioNext = sigmaAudioNext
            self.pddVideoWeights = pddVideoWeights
            self.pddAudioWeights = pddAudioWeights
        }
    }

    public struct SamplerPlan: Codable, Equatable {
        public var rule: SamplerRule
        public var shifts: Shifts
        public var steps: [SamplerStep]
        public var condVideoT: Double
        public var condAudioT: Double

        enum CodingKeys: String, CodingKey {
            case rule, shifts, steps
            case condVideoT = "cond_video_t"
            case condAudioT = "cond_audio_t"
        }

        public init(rule: SamplerRule, shifts: Shifts, steps: [SamplerStep], condVideoT: Double, condAudioT: Double) {
            self.rule = rule
            self.shifts = shifts
            self.steps = steps
            self.condVideoT = condVideoT
            self.condAudioT = condAudioT
        }
    }

    public struct AdalnInfo: Codable, Equatable {
        public var file: String
        public var timesteps: [Double]
    }

    public struct TextRefineEntry: Codable, Equatable {
        public var rows: Int
        public var file: String

        public init(rows: Int, file: String) {
            self.rows = rows
            self.file = file
        }
    }

    public struct BlockGraphs: Codable, Equatable {
        public var qkv: String
        public var post: String
        public var bytes: Int?

        public init(qkv: String, post: String, bytes: Int? = nil) {
            self.qkv = qkv
            self.post = post
            self.bytes = bytes
        }
    }

    public struct Graphs: Codable, Equatable {
        public var textRefine: [TextRefineEntry]
        public var embedVideo: String
        public var embedAudio: String
        public var headVideo: String
        public var headAudio: String
        public var blocks: [BlockGraphs]

        enum CodingKeys: String, CodingKey {
            case blocks
            case textRefine = "text_refine"
            case embedVideo = "embed_video"
            case embedAudio = "embed_audio"
            case headVideo = "head_video"
            case headAudio = "head_audio"
        }

        public init(textRefine: [TextRefineEntry], embedVideo: String, embedAudio: String, headVideo: String, headAudio: String, blocks: [BlockGraphs]) {
            self.textRefine = textRefine
            self.embedVideo = embedVideo
            self.embedAudio = embedAudio
            self.headVideo = headVideo
            self.headAudio = headAudio
            self.blocks = blocks
        }
    }

    public struct AttentionEntry: Codable, Equatable {
        public var tokens: Int
        public var qChunk: Int
        public var heads: Int
        public var file: String

        enum CodingKeys: String, CodingKey {
            case tokens, heads, file
            case qChunk = "q_chunk"
        }

        public init(tokens: Int, qChunk: Int, heads: Int, file: String) {
            self.tokens = tokens
            self.qChunk = qChunk
            self.heads = heads
            self.file = file
        }
    }

    /// The ViT decoder's graphs at one `(latent_t, latent_h, latent_w)` tile: `R = latent_t·latent_h·latent_w + register_tokens + 1` rows.
    public struct VaeDecoder: Codable, Equatable {
        public var latentT: Int
        public var latentH: Int
        public var latentW: Int
        public var rows: Int
        public var input: String
        public var blocks: [String]
        public var output: String

        enum CodingKeys: String, CodingKey {
            case rows, blocks
            case latentT = "latent_t"
            case latentH = "latent_h"
            case latentW = "latent_w"
            case input = "in"
            case output = "out"
        }

        public init(latentT: Int, latentH: Int, latentW: Int, rows: Int, input: String, blocks: [String], output: String) {
            self.latentT = latentT
            self.latentH = latentH
            self.latentW = latentW
            self.rows = rows
            self.input = input
            self.blocks = blocks
            self.output = output
        }
    }

    public struct VideoVae: Codable, Equatable {
        public var decoder: VaeDecoder
        public var encoderClip: String?
        public var encoderImage: String?
        public var tile: Int
        public var overlap: Int

        enum CodingKeys: String, CodingKey {
            case decoder, tile, overlap
            case encoderClip = "encoder_clip"
            case encoderImage = "encoder_image"
        }

        public init(decoder: VaeDecoder, encoderClip: String?, encoderImage: String?, tile: Int, overlap: Int) {
            self.decoder = decoder
            self.encoderClip = encoderClip
            self.encoderImage = encoderImage
            self.tile = tile
            self.overlap = overlap
        }
    }

    public struct AudioEntry: Codable, Equatable {
        public var latents: Int
        public var decoder: String
        public var encoder: String?

        public init(latents: Int, decoder: String, encoder: String? = nil) {
            self.latents = latents
            self.decoder = decoder
            self.encoder = encoder
        }
    }

    public struct TaeEntry: Codable, Equatable {
        public var frames: Int
        public var latentH: Int
        public var latentW: Int
        public var file: String

        enum CodingKeys: String, CodingKey {
            case frames, file
            case latentH = "latent_h"
            case latentW = "latent_w"
        }

        public init(frames: Int, latentH: Int, latentW: Int, file: String) {
            self.frames = frames
            self.latentH = latentH
            self.latentW = latentW
            self.file = file
        }
    }

    public struct FileInfo: Codable, Equatable {
        public var bytes: Int
        public var sha256: String?
    }

    public var format: String
    public var model: String
    public var variant: String?
    public var config: H3Config
    public var vae: VideoVaeConfig
    public var audioVae: AudioVaeConfig
    public var taeh3: TaeH3Config?
    public var sampler: SamplerPlan
    public var adaln: AdalnInfo
    public var weights: String
    public var chunk: Int
    public var headGroup: Int
    public var graphs: Graphs
    public var attention: [AttentionEntry]
    public var videoVae: VideoVae?
    public var audio: [AudioEntry]
    public var taeh3Graphs: [TaeEntry]
    public var sizes: [Geometry]
    public var files: [String: FileInfo]?

    enum CodingKeys: String, CodingKey {
        case format, model, variant, config, vae, taeh3, sampler, adaln, weights, chunk, graphs, attention, audio, sizes, files
        case audioVae = "audio_vae"
        case headGroup = "head_group"
        case videoVae = "video_vae"
        case taeh3Graphs = "taeh3_graphs"
    }

    /// Decode and validate `manifest.json` bytes.
    public init(json: Data, source: String = "manifest.json") throws {
        let decoded: Manifest
        do {
            decoded = try JSONDecoder().decode(Manifest.self, from: json)
        } catch {
            throw H3Error.manifest("\(source): \(error)")
        }
        guard decoded.format == manifestFormat else {
            throw H3Error.manifest("\(source): format \"\(decoded.format)\", expected \"\(manifestFormat)\"")
        }
        guard decoded.config.heads % max(decoded.headGroup, 1) == 0, decoded.headGroup > 0 else {
            throw H3Error.manifest("\(source): head_group \(decoded.headGroup) must divide heads \(decoded.config.heads)")
        }
        guard decoded.config.patch.count == 3 else {
            throw H3Error.manifest("\(source): config.patch \(decoded.config.patch) is not (pt, ph, pw)")
        }
        self = decoded
    }

    /// Read `<directory>/manifest.json`.
    public init(directory: URL) throws {
        let url = directory.appendingPathComponent("manifest.json")
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw H3Error.io("cannot read \(url.path): \(error)")
        }
        try self.init(json: data, source: url.path)
    }

    public var cfg: H3Config { config }

    /// The attention graph of a packed sequence of exactly `tokens` rows (text padded to its bucket).
    public func attentionFor(tokens: Int) throws -> AttentionEntry {
        if let a = attention.first(where: { $0.tokens == tokens }) {
            return a
        }
        let have = Array(Set(attention.map { $0.tokens })).sorted()
        throw H3Error.noAttentionGraph(tokens: tokens, have: have)
    }

    /// The smallest text bucket that holds `rows` prompt rows.
    public func textRefineFor(rows: Int) throws -> TextRefineEntry {
        if let t = graphs.textRefine.sorted(by: { $0.rows < $1.rows }).first(where: { $0.rows >= rows }) {
            return t
        }
        throw H3Error.noTextRefineGraph(rows: rows, largest: graphs.textRefine.map { $0.rows }.max() ?? 0)
    }

    public func audioFor(latents: Int) throws -> AudioEntry {
        if let a = audio.first(where: { $0.latents == latents }) {
            return a
        }
        throw H3Error.noAudioGraph(latents: latents, have: audio.map { $0.latents }.sorted())
    }

    public func taeh3For(latentH h: Int, latentW w: Int) throws -> TaeEntry {
        if let t = taeh3Graphs.first(where: { $0.latentH == h && $0.latentW == w }) {
            return t
        }
        let have = Array(Set(taeh3Graphs.map { "\($0.latentH)x\($0.latentW)" })).sorted()
        throw H3Error.noTaeGraph(latentH: h, latentW: w, have: have)
    }

    /// Every graph file the manifest names, in a stable order (the AdaLN table is not a graph).
    public var graphFiles: [String] {
        var out = graphs.textRefine.map { $0.file }
        out.append(contentsOf: [graphs.embedVideo, graphs.embedAudio, graphs.headVideo, graphs.headAudio])
        for b in graphs.blocks {
            out.append(contentsOf: [b.qkv, b.post])
        }
        out.append(contentsOf: attention.map { $0.file })
        if let vv = videoVae {
            out.append(vv.decoder.input)
            out.append(contentsOf: vv.decoder.blocks)
            out.append(vv.decoder.output)
            if let e = vv.encoderClip { out.append(e) }
            if let e = vv.encoderImage { out.append(e) }
        }
        for a in audio {
            out.append(a.decoder)
            if let e = a.encoder { out.append(e) }
        }
        out.append(contentsOf: taeh3Graphs.map { $0.file })
        return out
    }

    /// Every file of the export, graphs and the AdaLN table.
    public var allFiles: [String] { graphFiles + [adaln.file] }

    /// The size of a file from the manifest's `files` map, if recorded.
    public func bytes(of file: String) -> Int? {
        files?[file]?.bytes
    }
}
