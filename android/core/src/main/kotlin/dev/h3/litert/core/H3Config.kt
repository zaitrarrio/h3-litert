package dev.h3.litert.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** Modality tags of a packed row (`config.py`). */
public const val VIDEO_TAG: Int = 0
public const val TEXT_TAG: Int = 1
public const val AUDIO_TAG: Int = 2
public const val MOD_SLOTS: Int = 6
public const val MODALITIES: Int = 3

/**
 * Architecture constants of the H3 Omni-Transformer (`config.py`; h3-core `config.rs`). Batch is
 * always 1; there is no cross-attention and one weight set for every modality.
 */
@Serializable
public data class H3Config(
    val hidden: Int,
    val heads: Int,
    @SerialName("head_dim") val headDim: Int,
    val layers: Int,
    @SerialName("refiner_layers") val refinerLayers: Int,
    @SerialName("ffn_dim") val ffnDim: Int,
    @SerialName("in_channels") val inChannels: Int,
    @SerialName("audio_in_channels") val audioInChannels: Int,
    val patch: List<Int>,
    @SerialName("text_dim") val textDim: Int,
    @SerialName("freq_dim") val freqDim: Int,
    @SerialName("time_embed_hidden") val timeEmbedHidden: Int,
    @SerialName("time_embed_dim") val timeEmbedDim: Int,
    @SerialName("rope_freq_dim") val ropeFreqDim: Int,
    @SerialName("rope_theta") val ropeTheta: Double = 10000.0,
    @SerialName("norm_eps") val normEps: Double = 1e-5,
    @SerialName("qk_norm_eps") val qkNormEps: Double = 1e-5,
    @SerialName("final_norm_eps") val finalNormEps: Double = 1e-5,
    @SerialName("adaln_rank") val adalnRank: Int? = null,
    @SerialName("pdd_steps") val pddSteps: Int? = null,
    @SerialName("kept_block_indices") val keptBlockIndices: List<Int>? = null,
) {
    init {
        require(patch.size == 3) { "patch must be (t, h, w), got $patch" }
        require(rotaryDim() <= headDim && rotaryDim() % 2 == 0) {
            "rotary width ${rotaryDim()} must be even and <= head_dim $headDim"
        }
    }

    /** `heads · head_dim` (7168: larger than `hidden`). */
    val dim: Int get() = heads * headDim

    /** Rotated channels per head: `6 · rope_freq_dim` (96 of 128). */
    public fun rotaryDim(): Int = 6 * ropeFreqDim

    /** `P = in_channels · pt · ph · pw` (96). */
    public fun videoPatchDim(): Int = inChannels * patch[0] * patch[1] * patch[2]

    public fun adalnIn(): Int = adalnRank ?: timeEmbedDim

    public companion object {
        public fun h3(): H3Config = H3Config(
            hidden = 5376, heads = 56, headDim = 128, layers = 50, refinerLayers = 2, ffnDim = 14336,
            inChannels = 24, audioInChannels = 32, patch = listOf(1, 2, 2), textDim = 5120, freqDim = 256,
            timeEmbedHidden = 5376, timeEmbedDim = 2688, ropeFreqDim = 16,
        )

        public fun fasth3Trim(): H3Config = h3().copy(
            layers = 42, adalnRank = 16,
            keptBlockIndices = listOf(
                0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 14, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            ),
        )

        public fun fasth3Omni(): H3Config = h3().copy(pddSteps = 32)

        /** h3-core's tiny: head_dim 16 with rope_freq_dim 2 rotates 12 of 16. */
        public fun tiny(): H3Config = H3Config(
            hidden = 32, heads = 2, headDim = 16, layers = 2, refinerLayers = 1, ffnDim = 48,
            inChannels = 4, audioInChannels = 6, patch = listOf(1, 2, 2), textDim = 24, freqDim = 16,
            timeEmbedHidden = 32, timeEmbedDim = 24, ropeFreqDim = 2,
        )
    }
}

public val H3_LATENTS_MEAN: List<Double> = listOf(
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949,
    -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736,
    0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
public val H3_LATENTS_STD: List<Double> = listOf(
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833,
    2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114,
    1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)
public val PIXEL_MEAN: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f)
public val PIXEL_STD: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f)

/** `AutoencoderKLMiniMaxH3`: the causal 3-D CNN encoder and the ViT decoder (`config.py`). */
@Serializable
public data class VideoVaeConfig(
    @SerialName("in_channels") val inChannels: Int,
    @SerialName("latent_channels") val latentChannels: Int,
    @SerialName("block_out_channels") val blockOutChannels: List<Int>,
    @SerialName("layers_per_block") val layersPerBlock: Int,
    @SerialName("spatial_down") val spatialDown: List<Int>,
    @SerialName("temporal_down") val temporalDown: List<Int>,
    @SerialName("norm_groups") val normGroups: Int,
    @SerialName("norm_eps") val normEps: Double,
    @SerialName("decoder_layers") val decoderLayers: Int,
    @SerialName("decoder_heads") val decoderHeads: Int,
    @SerialName("decoder_head_dim") val decoderHeadDim: Int,
    @SerialName("decoder_register_tokens") val decoderRegisterTokens: Int,
    @SerialName("decoder_ffn_mult") val decoderFfnMult: Int,
    @SerialName("decoder_rope_theta") val decoderRopeTheta: Double,
    @SerialName("decoder_rope_dim_ratio") val decoderRopeDimRatio: Double,
    @SerialName("decoder_norm_eps") val decoderNormEps: Double,
    @SerialName("clip_length") val clipLength: Int,
    @SerialName("token_drop") val tokenDrop: Int,
    @SerialName("latents_mean") val latentsMean: List<Double>,
    @SerialName("latents_std") val latentsStd: List<Double>,
) {
    init {
        require(latentsMean.size == latentChannels && latentsStd.size == latentChannels) {
            "latents_mean / latents_std need $latentChannels entries"
        }
    }

    public fun spatialRatio(): Int = spatialDown.fold(1) { a, b -> a * b }

    public fun temporalRatio(): Int = temporalDown.fold(1) { a, b -> a * b }

    public fun decoderDim(): Int = decoderHeads * decoderHeadDim

    public fun decoderRotaryDim(): Int = (decoderHeadDim * decoderRopeDimRatio).toInt()

    /** Frames the decoder emits before the first real one: `(tr − 17 mod tr) mod tr` = 3. */
    public fun framePrePadding(): Int {
        val tr = temporalRatio()
        return (tr - clipLength % tr) % tr
    }

    /** Latent frames per temporal chunk: `ceil(17 / 4)` = 5. */
    public fun tokensChunk(): Int = (clipLength + temporalRatio() - 1) / temporalRatio()

    /** Latent frames carried into the next chunk: 2. */
    public fun tokenOverlap(): Int {
        val c = tokensChunk()
        return (c - tokenDrop % c) % c
    }

    /** Pixel frames cross-faded between chunks: 5. */
    public fun frameOverlap(): Int = maxOf(tokenOverlap() * temporalRatio() - framePrePadding(), 0)

    /** Rows of one decoder tile: `latent_t · latent_h · latent_w + register tokens + 1`. */
    public fun decoderRows(latentT: Int, latentH: Int, latentW: Int): Int =
        latentT * latentH * latentW + decoderRegisterTokens + 1

    public companion object {
        public fun h3(): VideoVaeConfig = VideoVaeConfig(
            3, 24, listOf(128, 256, 256, 512, 512, 1024), 2, listOf(2, 2, 2, 2, 1, 1), listOf(1, 2, 2, 1, 1, 1), 32, 1e-6,
            36, 32, 64, 4, 4, 100.0, 0.75, 1e-5, 17, 3, H3_LATENTS_MEAN, H3_LATENTS_STD,
        )

        public fun fasth3Trim(): VideoVaeConfig = h3().copy(decoderLayers = 26)

        public fun tiny(): VideoVaeConfig = VideoVaeConfig(
            3, 4, listOf(8, 8, 16, 16, 16, 16), 1, listOf(2, 2, 2, 2, 1, 1), listOf(1, 2, 2, 1, 1, 1), 4, 1e-6,
            2, 2, 16, 2, 2, 100.0, 0.75, 1e-5, 17, 3, listOf(0.1, -0.2, 0.3, 0.0), listOf(1.1, 0.9, 1.3, 1.0),
        )
    }
}

public val AUDIO_LATENTS_MEAN: List<Double> = listOf(
    -0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373, 0.08179686214561671,
    -0.35782641352446604, 0.040623809960919084, -0.01552534501956604, -0.223362481667332, 0.1821006842509091,
    0.2941778783780663, -0.07901167601970885, -0.056815072777201, -0.3699028221860095, -0.31616315591624855,
    0.5905951377425391, -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
    -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344, 0.02868032184767538,
    -0.21217779266454084, -0.1678263169941987, 0.3221287889040614, -0.1223055851554907, 0.4356604928128464,
    -0.0502599202236253, 0.3979258376211797,
)
public val AUDIO_LATENTS_STD: List<Double> = listOf(
    1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828, 1.6390226546605453,
    2.7788298348882177, 1.7659090095747236, 1.6199757612137327, 2.6336525640336896, 1.8539356672817833,
    2.5056497896915633, 1.811019237886178, 1.9579657790720237, 1.6685498243529284, 1.4922469314453364,
    3.298670198067373, 1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
    1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912, 1.8318420762504692,
    1.5540637421583379, 1.9344930328968526, 1.599198216109855, 1.718045989838149, 1.6307219190837705,
    1.8661226051202384, 1.5613768203168363,
)

/** `AutoencoderKLMiniMaxH3Audio`: DAC encoder, BigVGAN decoder, 32 kHz, 800 samples per latent. */
@Serializable
public data class AudioVaeConfig(
    @SerialName("encoder_dim") val encoderDim: Int,
    @SerialName("encoder_rates") val encoderRates: List<Int>,
    @SerialName("latent_dim") val latentDim: Int,
    @SerialName("latent_channels") val latentChannels: Int,
    @SerialName("decoder_dim") val decoderDim: Int,
    @SerialName("decoder_rates") val decoderRates: List<Int>,
    @SerialName("decoder_kernels") val decoderKernels: List<Int>,
    val heads: Int,
    @SerialName("resblock_kernels") val resblockKernels: List<Int>,
    @SerialName("resblock_dilations") val resblockDilations: List<List<Int>>,
    @SerialName("sample_rate") val sampleRate: Int,
    @SerialName("latents_mean") val latentsMean: List<Double>,
    @SerialName("latents_std") val latentsStd: List<Double>,
) {
    init {
        require(latentsMean.size == latentChannels && latentsStd.size == latentChannels) {
            "audio latents_mean / latents_std need $latentChannels entries"
        }
    }

    /** Samples per audio latent (800 at 32 kHz / 40 Hz). */
    public fun hop(): Int = encoderRates.fold(1) { a, b -> a * b }

    public companion object {
        public fun h3(): AudioVaeConfig = AudioVaeConfig(
            64, listOf(2, 4, 4, 5, 5), 2048, 32, 1024, listOf(5, 5, 2, 2, 2, 2, 2), listOf(9, 9, 4, 4, 4, 4, 4), 8, listOf(3, 7, 11),
            listOf(listOf(1, 3, 5), listOf(1, 3, 5), listOf(1, 3, 5)), 32000, AUDIO_LATENTS_MEAN, AUDIO_LATENTS_STD,
        )

        public fun tiny(): AudioVaeConfig = AudioVaeConfig(
            4, listOf(2, 5), 24, 6, 8, listOf(5, 2), listOf(9, 4), 2, listOf(3, 7), listOf(listOf(1, 3), listOf(1, 3)), 320,
            listOf(0.0, 0.1, -0.1, 0.2, 0.05, -0.3), listOf(1.0, 1.5, 0.8, 1.2, 0.9, 1.1),
        )
    }
}

/** TAEH3: fastwan-litert's TAEHV with 24 latent channels and a ×2 pixel-shuffle head. */
@Serializable
public data class TaeH3Config(
    @SerialName("latent_channels") val latentChannels: Int,
    val widths: List<Int>,
    val tgrow: List<Int>,
    @SerialName("mem_blocks") val memBlocks: Int,
    val patch: Int,
) {
    init {
        require(widths.size == 4) { "taeh3.widths must have 4 entries, got $widths" }
        require(tgrow.size == 3) { "taeh3.tgrow must have 3 entries, got $tgrow" }
    }

    public fun tUpscale(): Int = tgrow.fold(1) { a, b -> a * b }

    public fun spatialScale(): Int = 8 * patch

    public fun outChannels(): Int = 3 * patch * patch

    public companion object {
        /** H3's 17 frames per 5 latents are the raw frames `>= 3 (mod 20)`. */
        public fun keepsRawFrame(rawIndex: Int): Boolean = rawIndex % 20 >= 3

        public fun taeh3(): TaeH3Config = TaeH3Config(24, listOf(256, 128, 64, 64), listOf(1, 2, 2), 3, 2)

        public fun tiny(): TaeH3Config = TaeH3Config(4, listOf(8, 6, 4, 4), listOf(1, 2, 2), 3, 2)
    }
}
