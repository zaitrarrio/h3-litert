// Architecture constants (python/h3_litert/config.py; h3-core `config.rs`):
// the DiT, the video VAE, the audio VAE and TAEH3. Each decodes from the
// manifest's JSON with the Python field names.

import Foundation

public let modSlots = 6
public let modalities = 3
/// Modality tags of packed rows.
public let videoTag = 0
public let textTag = 1
public let audioTag = 2

/// The H3 Omni-Transformer. Batch is always 1.
public struct H3Config: Codable, Equatable, Sendable {
    public var hidden: Int
    public var heads: Int
    public var headDim: Int
    public var layers: Int
    public var refinerLayers: Int
    public var ffnDim: Int
    public var inChannels: Int
    public var audioInChannels: Int
    /// `(pt, ph, pw)`, always `[1, 2, 2]` for H3.
    public var patch: [Int]
    public var textDim: Int
    public var freqDim: Int
    public var timeEmbedHidden: Int
    public var timeEmbedDim: Int
    public var ropeFreqDim: Int
    public var ropeTheta: Double
    public var normEps: Double
    public var qkNormEps: Double
    public var finalNormEps: Double
    public var adalnRank: Int?
    public var pddSteps: Int?
    public var keptBlockIndices: [Int]?

    enum CodingKeys: String, CodingKey {
        case hidden, heads, layers, patch
        case headDim = "head_dim"
        case refinerLayers = "refiner_layers"
        case ffnDim = "ffn_dim"
        case inChannels = "in_channels"
        case audioInChannels = "audio_in_channels"
        case textDim = "text_dim"
        case freqDim = "freq_dim"
        case timeEmbedHidden = "time_embed_hidden"
        case timeEmbedDim = "time_embed_dim"
        case ropeFreqDim = "rope_freq_dim"
        case ropeTheta = "rope_theta"
        case normEps = "norm_eps"
        case qkNormEps = "qk_norm_eps"
        case finalNormEps = "final_norm_eps"
        case adalnRank = "adaln_rank"
        case pddSteps = "pdd_steps"
        case keptBlockIndices = "kept_block_indices"
    }

    public init(
        hidden: Int, heads: Int, headDim: Int, layers: Int, refinerLayers: Int, ffnDim: Int,
        inChannels: Int, audioInChannels: Int, patch: [Int], textDim: Int, freqDim: Int,
        timeEmbedHidden: Int, timeEmbedDim: Int, ropeFreqDim: Int, ropeTheta: Double,
        normEps: Double, qkNormEps: Double, finalNormEps: Double,
        adalnRank: Int? = nil, pddSteps: Int? = nil, keptBlockIndices: [Int]? = nil
    ) {
        self.hidden = hidden
        self.heads = heads
        self.headDim = headDim
        self.layers = layers
        self.refinerLayers = refinerLayers
        self.ffnDim = ffnDim
        self.inChannels = inChannels
        self.audioInChannels = audioInChannels
        self.patch = patch
        self.textDim = textDim
        self.freqDim = freqDim
        self.timeEmbedHidden = timeEmbedHidden
        self.timeEmbedDim = timeEmbedDim
        self.ropeFreqDim = ropeFreqDim
        self.ropeTheta = ropeTheta
        self.normEps = normEps
        self.qkNormEps = qkNormEps
        self.finalNormEps = finalNormEps
        self.adalnRank = adalnRank
        self.pddSteps = pddSteps
        self.keptBlockIndices = keptBlockIndices
    }

    /// `I = heads · head_dim` (7168: larger than `hidden`).
    public var dim: Int { heads * headDim }

    /// `r = 6 · rope_freq_dim` rotated channels per head (96 of 128).
    public var rotaryDim: Int { 6 * ropeFreqDim }

    /// `P = in_channels · pt · ph · pw` (96).
    public var videoPatchDim: Int { inChannels * patch.reduce(1, *) }

    public var adalnIn: Int { adalnRank ?? timeEmbedDim }

    /// `MiniMaxAI/MiniMax-H3` and the FastH3 students that keep its shape.
    public static let h3 = H3Config(
        hidden: 5376, heads: 56, headDim: 128, layers: 50, refinerLayers: 2, ffnDim: 14336,
        inChannels: 24, audioInChannels: 32, patch: [1, 2, 2], textDim: 5120, freqDim: 256,
        timeEmbedHidden: 5376, timeEmbedDim: 2688, ropeFreqDim: 16, ropeTheta: 10000.0,
        normEps: 1e-5, qkNormEps: 1e-5, finalNormEps: 1e-5
    )

    public static let fasth3Trim = H3Config(
        hidden: 5376, heads: 56, headDim: 128, layers: 42, refinerLayers: 2, ffnDim: 14336,
        inChannels: 24, audioInChannels: 32, patch: [1, 2, 2], textDim: 5120, freqDim: 256,
        timeEmbedHidden: 5376, timeEmbedDim: 2688, ropeFreqDim: 16, ropeTheta: 10000.0,
        normEps: 1e-5, qkNormEps: 1e-5, finalNormEps: 1e-5, adalnRank: 16,
        keptBlockIndices: [
            0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 14, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        ]
    )

    public static let fasth3Omni = H3Config(
        hidden: 5376, heads: 56, headDim: 128, layers: 50, refinerLayers: 2, ffnDim: 14336,
        inChannels: 24, audioInChannels: 32, patch: [1, 2, 2], textDim: 5120, freqDim: 256,
        timeEmbedHidden: 5376, timeEmbedDim: 2688, ropeFreqDim: 16, ropeTheta: 10000.0,
        normEps: 1e-5, qkNormEps: 1e-5, finalNormEps: 1e-5, pddSteps: 32
    )

    /// h3-core's tiny: head_dim 16 with rope_freq_dim 2 rotates 12 of 16.
    public static let tiny = H3Config(
        hidden: 32, heads: 2, headDim: 16, layers: 2, refinerLayers: 1, ffnDim: 48,
        inChannels: 4, audioInChannels: 6, patch: [1, 2, 2], textDim: 24, freqDim: 16,
        timeEmbedHidden: 32, timeEmbedDim: 24, ropeFreqDim: 2, ropeTheta: 10000.0,
        normEps: 1e-5, qkNormEps: 1e-5, finalNormEps: 1e-5
    )
}

/// H3's latent statistics and the ImageNet pixel normalisation of the ViT decoder.
public enum H3Latents {
    public static let mean: [Float] = [
        0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949,
        -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736,
        0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
        0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
        -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
    ]
    public static let std: [Float] = [
        1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833,
        2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114,
        1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
        0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
        3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
    ]
    public static let pixelMean: [Float] = [0.485, 0.456, 0.406]
    public static let pixelStd: [Float] = [0.229, 0.224, 0.225]
}

/// The video VAE: a causal 3-D CNN encoder and a ViT decoder (`AutoencoderKLMiniMaxH3`).
public struct VideoVaeConfig: Codable, Equatable, Sendable {
    public var inChannels: Int
    public var latentChannels: Int
    public var blockOutChannels: [Int]
    public var layersPerBlock: Int
    public var spatialDown: [Int]
    public var temporalDown: [Int]
    public var normGroups: Int
    public var normEps: Double
    public var decoderLayers: Int
    public var decoderHeads: Int
    public var decoderHeadDim: Int
    public var decoderRegisterTokens: Int
    public var decoderFfnMult: Int
    public var decoderRopeTheta: Double
    public var decoderRopeDimRatio: Double
    public var decoderNormEps: Double
    public var clipLength: Int
    public var tokenDrop: Int
    public var latentsMean: [Float]
    public var latentsStd: [Float]

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case latentChannels = "latent_channels"
        case blockOutChannels = "block_out_channels"
        case layersPerBlock = "layers_per_block"
        case spatialDown = "spatial_down"
        case temporalDown = "temporal_down"
        case normGroups = "norm_groups"
        case normEps = "norm_eps"
        case decoderLayers = "decoder_layers"
        case decoderHeads = "decoder_heads"
        case decoderHeadDim = "decoder_head_dim"
        case decoderRegisterTokens = "decoder_register_tokens"
        case decoderFfnMult = "decoder_ffn_mult"
        case decoderRopeTheta = "decoder_rope_theta"
        case decoderRopeDimRatio = "decoder_rope_dim_ratio"
        case decoderNormEps = "decoder_norm_eps"
        case clipLength = "clip_length"
        case tokenDrop = "token_drop"
        case latentsMean = "latents_mean"
        case latentsStd = "latents_std"
    }

    public init(
        inChannels: Int, latentChannels: Int, blockOutChannels: [Int], layersPerBlock: Int,
        spatialDown: [Int], temporalDown: [Int], normGroups: Int, normEps: Double,
        decoderLayers: Int, decoderHeads: Int, decoderHeadDim: Int, decoderRegisterTokens: Int,
        decoderFfnMult: Int, decoderRopeTheta: Double, decoderRopeDimRatio: Double, decoderNormEps: Double,
        clipLength: Int, tokenDrop: Int, latentsMean: [Float], latentsStd: [Float]
    ) {
        self.inChannels = inChannels
        self.latentChannels = latentChannels
        self.blockOutChannels = blockOutChannels
        self.layersPerBlock = layersPerBlock
        self.spatialDown = spatialDown
        self.temporalDown = temporalDown
        self.normGroups = normGroups
        self.normEps = normEps
        self.decoderLayers = decoderLayers
        self.decoderHeads = decoderHeads
        self.decoderHeadDim = decoderHeadDim
        self.decoderRegisterTokens = decoderRegisterTokens
        self.decoderFfnMult = decoderFfnMult
        self.decoderRopeTheta = decoderRopeTheta
        self.decoderRopeDimRatio = decoderRopeDimRatio
        self.decoderNormEps = decoderNormEps
        self.clipLength = clipLength
        self.tokenDrop = tokenDrop
        self.latentsMean = latentsMean
        self.latentsStd = latentsStd
    }

    /// Pixels per latent cell (16).
    public var spatialRatio: Int { spatialDown.reduce(1, *) }

    /// Frames per latent frame (4).
    public var temporalRatio: Int { temporalDown.reduce(1, *) }

    public var decoderDim: Int { decoderHeads * decoderHeadDim }

    public var decoderRotaryDim: Int { Int(Double(decoderHeadDim) * decoderRopeDimRatio) }

    /// Frames the decoder emits before the clip's first (3): `(tr − clip_length mod tr) mod tr`.
    public var framePrePadding: Int {
        let tr = temporalRatio
        return (tr - clipLength % tr) % tr
    }

    /// Latent frames of one decoder chunk (5): `ceil(clip_length / tr)`.
    public var tokensChunk: Int { (clipLength + temporalRatio - 1) / temporalRatio }

    /// Latent frames a chunk overlaps the next (2).
    public var tokenOverlap: Int {
        let c = tokensChunk
        return (c - tokenDrop % c) % c
    }

    /// Decoded frames blended between chunks (5).
    public var frameOverlap: Int { max(tokenOverlap * temporalRatio - framePrePadding, 0) }

    /// Decoder rows per `(latent_t, latent_h, latent_w)` tile: tokens + register tokens + the cls row.
    public func decoderRows(latentT t: Int, latentH h: Int, latentW w: Int) -> Int {
        t * h * w + decoderRegisterTokens + 1
    }

    public static let h3 = VideoVaeConfig(
        inChannels: 3, latentChannels: 24, blockOutChannels: [128, 256, 256, 512, 512, 1024], layersPerBlock: 2,
        spatialDown: [2, 2, 2, 2, 1, 1], temporalDown: [1, 2, 2, 1, 1, 1], normGroups: 32, normEps: 1e-6,
        decoderLayers: 36, decoderHeads: 32, decoderHeadDim: 64, decoderRegisterTokens: 4, decoderFfnMult: 4,
        decoderRopeTheta: 100.0, decoderRopeDimRatio: 0.75, decoderNormEps: 1e-5, clipLength: 17, tokenDrop: 3,
        latentsMean: H3Latents.mean, latentsStd: H3Latents.std
    )

    public static var fasth3Trim: VideoVaeConfig {
        var c = h3
        c.decoderLayers = 26
        return c
    }

    public static let tiny = VideoVaeConfig(
        inChannels: 3, latentChannels: 4, blockOutChannels: [8, 8, 16, 16, 16, 16], layersPerBlock: 1,
        spatialDown: [2, 2, 2, 2, 1, 1], temporalDown: [1, 2, 2, 1, 1, 1], normGroups: 4, normEps: 1e-6,
        decoderLayers: 2, decoderHeads: 2, decoderHeadDim: 16, decoderRegisterTokens: 2, decoderFfnMult: 2,
        decoderRopeTheta: 100.0, decoderRopeDimRatio: 0.75, decoderNormEps: 1e-5, clipLength: 17, tokenDrop: 3,
        latentsMean: [0.1, -0.2, 0.3, 0.0], latentsStd: [1.1, 0.9, 1.3, 1.0]
    )
}

/// The audio VAE (DAC encoder, BigVGAN decoder; `AutoencoderKLMiniMaxH3Audio`).
public struct AudioVaeConfig: Codable, Equatable, Sendable {
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int
    public var latentChannels: Int
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var decoderKernels: [Int]
    public var heads: Int
    public var resblockKernels: [Int]
    public var resblockDilations: [[Int]]
    public var sampleRate: Int
    public var latentsMean: [Float]
    public var latentsStd: [Float]

    enum CodingKeys: String, CodingKey {
        case heads
        case encoderDim = "encoder_dim"
        case encoderRates = "encoder_rates"
        case latentDim = "latent_dim"
        case latentChannels = "latent_channels"
        case decoderDim = "decoder_dim"
        case decoderRates = "decoder_rates"
        case decoderKernels = "decoder_kernels"
        case resblockKernels = "resblock_kernels"
        case resblockDilations = "resblock_dilations"
        case sampleRate = "sample_rate"
        case latentsMean = "latents_mean"
        case latentsStd = "latents_std"
    }

    public init(
        encoderDim: Int, encoderRates: [Int], latentDim: Int, latentChannels: Int, decoderDim: Int,
        decoderRates: [Int], decoderKernels: [Int], heads: Int, resblockKernels: [Int], resblockDilations: [[Int]],
        sampleRate: Int, latentsMean: [Float], latentsStd: [Float]
    ) {
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        self.latentDim = latentDim
        self.latentChannels = latentChannels
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates
        self.decoderKernels = decoderKernels
        self.heads = heads
        self.resblockKernels = resblockKernels
        self.resblockDilations = resblockDilations
        self.sampleRate = sampleRate
        self.latentsMean = latentsMean
        self.latentsStd = latentsStd
    }

    /// Samples per latent (800 at 32 kHz: 40 latents per second).
    public var hop: Int { encoderRates.reduce(1, *) }

    public static let h3 = AudioVaeConfig(
        encoderDim: 64, encoderRates: [2, 4, 4, 5, 5], latentDim: 2048, latentChannels: 32, decoderDim: 1024,
        decoderRates: [5, 5, 2, 2, 2, 2, 2], decoderKernels: [9, 9, 4, 4, 4, 4, 4], heads: 8, resblockKernels: [3, 7, 11],
        resblockDilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]], sampleRate: 32000,
        latentsMean: [
            -0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373, 0.08179686214561671,
            -0.35782641352446604, 0.040623809960919084, -0.01552534501956604, -0.223362481667332, 0.1821006842509091,
            0.2941778783780663, -0.07901167601970885, -0.056815072777201, -0.3699028221860095, -0.31616315591624855,
            0.5905951377425391, -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
            -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344, 0.02868032184767538,
            -0.21217779266454084, -0.1678263169941987, 0.3221287889040614, -0.1223055851554907, 0.4356604928128464,
            -0.0502599202236253, 0.3979258376211797,
        ],
        latentsStd: [
            1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828, 1.6390226546605453,
            2.7788298348882177, 1.7659090095747236, 1.6199757612137327, 2.6336525640336896, 1.8539356672817833,
            2.5056497896915633, 1.811019237886178, 1.9579657790720237, 1.6685498243529284, 1.4922469314453364,
            3.298670198067373, 1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
            1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912, 1.8318420762504692,
            1.5540637421583379, 1.9344930328968526, 1.599198216109855, 1.718045989838149, 1.6307219190837705,
            1.8661226051202384, 1.5613768203168363,
        ]
    )

    public static let tiny = AudioVaeConfig(
        encoderDim: 4, encoderRates: [2, 5], latentDim: 24, latentChannels: 6, decoderDim: 8, decoderRates: [5, 2],
        decoderKernels: [9, 4], heads: 2, resblockKernels: [3, 7], resblockDilations: [[1, 3], [1, 3]], sampleRate: 320,
        latentsMean: [0.0, 0.1, -0.1, 0.2, 0.05, -0.3], latentsStd: [1.0, 1.5, 0.8, 1.2, 0.9, 1.1]
    )
}

/// TAEH3: fastwan-litert's TAEHV with 24 latent channels and a ×2 pixel-shuffle head.
public struct TaeH3Config: Codable, Equatable, Sendable {
    public var latentChannels: Int
    public var widths: [Int]
    public var tgrow: [Int]
    public var memBlocks: Int
    public var patch: Int

    enum CodingKeys: String, CodingKey {
        case widths, tgrow, patch
        case latentChannels = "latent_channels"
        case memBlocks = "mem_blocks"
    }

    public init(latentChannels: Int, widths: [Int], tgrow: [Int], memBlocks: Int, patch: Int) {
        self.latentChannels = latentChannels
        self.widths = widths
        self.tgrow = tgrow
        self.memBlocks = memBlocks
        self.patch = patch
    }

    public var tUpscale: Int { tgrow.reduce(1, *) }

    public var spatialScale: Int { 8 * patch }

    public var outChannels: Int { 3 * patch * patch }

    /// H3's 17 frames per 5 latents are the raw frames `>= 3 (mod 20)`.
    public static func keepsRawFrame(_ rawIndex: Int) -> Bool { rawIndex % 20 >= 3 }

    public static let taeh3 = TaeH3Config(latentChannels: 24, widths: [256, 128, 64, 64], tgrow: [1, 2, 2], memBlocks: 3, patch: 2)

    public static let tiny = TaeH3Config(latentChannels: 4, widths: [8, 6, 4, 4], tgrow: [1, 2, 2], memBlocks: 3, patch: 2)
}
