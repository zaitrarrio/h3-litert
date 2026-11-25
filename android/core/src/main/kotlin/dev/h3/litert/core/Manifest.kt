package dev.h3.litert.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.jsonObject
import java.io.File
import kotlin.math.abs

public const val MANIFEST_FORMAT: String = "h3-litert/1"
public const val MANIFEST_FILE: String = "manifest.json"

/** One step of the precomputed schedule (`sampler.steps`); every value is the exporter's f32. */
@Serializable
public data class SamplerStep(
    @SerialName("t_video") val tVideo: Float,
    @SerialName("t_audio") val tAudio: Float,
    @SerialName("sigma_video") val sigmaVideo: Float,
    @SerialName("sigma_video_next") val sigmaVideoNext: Float,
    @SerialName("sigma_audio") val sigmaAudio: Float,
    @SerialName("sigma_audio_next") val sigmaAudioNext: Float,
    @SerialName("pdd_video_weights") val pddVideoWeights: List<Float>? = null,
    @SerialName("pdd_audio_weights") val pddAudioWeights: List<Float>? = null,
)

@Serializable
public data class SamplerShifts(val video: Double, val audio: Double)

@Serializable
public data class SamplerSpec(
    /** `euler`, `renoise` or `pdd`. */
    val rule: String,
    val shifts: SamplerShifts? = null,
    val steps: List<SamplerStep>,
    @SerialName("cond_video_t") val condVideoT: Float = 0.999f,
    @SerialName("cond_audio_t") val condAudioT: Float = 1.0f,
) {
    init {
        require(rule == "euler" || rule == "renoise" || rule == "pdd") { "sampler rule `$rule` is not euler, renoise or pdd" }
        require(steps.isNotEmpty()) { "sampler.steps is empty" }
    }
}

@Serializable
public data class AdalnEntry(val file: String, val timesteps: List<Float>)

@Serializable
public data class TextRefineEntry(val rows: Int, val file: String)

@Serializable
public data class BlockFiles(val qkv: String, val post: String, val bytes: Long = 0)

@Serializable
public data class GraphFiles(
    @SerialName("text_refine") val textRefine: List<TextRefineEntry>,
    @SerialName("embed_video") val embedVideo: String,
    @SerialName("embed_audio") val embedAudio: String,
    @SerialName("head_video") val headVideo: String,
    @SerialName("head_audio") val headAudio: String,
    val blocks: List<BlockFiles>,
)

@Serializable
public data class AttentionEntry(
    val tokens: Int,
    @SerialName("q_chunk") val qChunk: Int,
    val heads: Int,
    val file: String,
)

@Serializable
public data class DecoderGraphs(
    @SerialName("latent_t") val latentT: Int,
    @SerialName("latent_h") val latentH: Int,
    @SerialName("latent_w") val latentW: Int,
    val rows: Int,
    @SerialName("in") val inFile: String,
    val blocks: List<String>,
    val out: String,
)

@Serializable
public data class VideoVaeGraphs(
    val decoder: DecoderGraphs,
    @SerialName("encoder_clip") val encoderClip: String? = null,
    @SerialName("encoder_image") val encoderImage: String? = null,
    val tile: Int = VaeHost.TILE,
    val overlap: Int = VaeHost.OVERLAP,
)

@Serializable
public data class AudioEntry(val latents: Int, val decoder: String, val encoder: String? = null)

@Serializable
public data class TaeH3Entry(
    val frames: Int,
    @SerialName("latent_h") val latentH: Int,
    @SerialName("latent_w") val latentW: Int,
    val file: String,
)

@Serializable
public data class FileEntry(val bytes: Long, val sha256: String = "")

/**
 * `manifest.json` (`docs/graphs.md`): the contract between the exporter and this runtime.
 * Parse with [parse] or [load]; both refuse any other `format` with the mismatch in the message.
 */
@Serializable
public data class Manifest(
    val format: String,
    val model: String = "",
    val variant: String = "",
    val config: H3Config,
    val vae: VideoVaeConfig,
    @SerialName("audio_vae") val audioVae: AudioVaeConfig,
    val taeh3: TaeH3Config? = null,
    val sampler: SamplerSpec,
    val adaln: AdalnEntry,
    val weights: String,
    val chunk: Int,
    @SerialName("head_group") val headGroup: Int,
    val graphs: GraphFiles,
    val attention: List<AttentionEntry>,
    @SerialName("video_vae") val videoVae: VideoVaeGraphs? = null,
    val audio: List<AudioEntry> = emptyList(),
    @SerialName("taeh3_graphs") val taeh3Graphs: List<TaeH3Entry> = emptyList(),
    val sizes: List<Geometry> = emptyList(),
    val files: Map<String, FileEntry> = emptyMap(),
    /** The directory the graph files live in, when the manifest was read from one. */
    @Transient val directory: File? = null,
) {
    init {
        if (format != MANIFEST_FORMAT) throw IllegalArgumentException("manifest format \"$format\", expected \"$MANIFEST_FORMAT\"")
        require(chunk > 0) { "manifest chunk must be positive, got $chunk" }
        require(headGroup > 0 && config.heads % headGroup == 0) { "head_group $headGroup must divide heads ${config.heads}" }
        require(config.layers == graphs.blocks.size) {
            "manifest config.layers is ${config.layers} but graphs.blocks lists ${graphs.blocks.size} blocks"
        }
        require(graphs.textRefine.isNotEmpty()) { "manifest lists no text_refine graphs" }
        if (sampler.rule == "pdd") {
            require(config.pddSteps != null) { "a pdd sampler needs config.pdd_steps" }
            for ((i, s) in sampler.steps.withIndex()) {
                require(s.pddVideoWeights?.size == config.pddSteps && s.pddAudioWeights?.size == config.pddSteps) {
                    "sampler.steps[$i] needs pdd_video_weights / pdd_audio_weights of ${config.pddSteps}"
                }
            }
        }
        // Every time a step can ask the AdaLN table for must be in the table.
        for ((i, s) in sampler.steps.withIndex()) {
            for (t in listOf(s.tVideo, s.tAudio, maxOf(s.tVideo, sampler.condVideoT), sampler.condAudioT)) {
                if (adaln.timesteps.none { abs(it - t) <= ADALN_ATOL }) {
                    throw IllegalArgumentException("sampler.steps[$i] uses timestep $t, which adaln.timesteps ${adaln.timesteps} does not cover")
                }
            }
        }
    }

    val cfg: H3Config get() = config

    public fun attentionFor(tokens: Int): AttentionEntry =
        attention.firstOrNull { it.tokens == tokens } ?: throw IllegalArgumentException(
            "this export has no attention graph for $tokens rows (it has ${attention.map { it.tokens }.distinct().sorted()}); re-export with --size",
        )

    /** The smallest text bucket that holds `rows` prompt rows. */
    public fun textRefineFor(rows: Int): TextRefineEntry =
        graphs.textRefine.sortedBy { it.rows }.firstOrNull { it.rows >= rows } ?: throw IllegalArgumentException(
            "the prompt has $rows rows; this export's largest text_refine bucket is ${graphs.textRefine.maxOf { it.rows }}",
        )

    public fun audioFor(latents: Int): AudioEntry =
        audio.firstOrNull { it.latents == latents } ?: throw IllegalArgumentException(
            "this export has no audio VAE graphs for $latents latents (it has ${audio.map { it.latents }.sorted()})",
        )

    public fun taeh3For(latentH: Int, latentW: Int): TaeH3Entry =
        taeh3Graphs.firstOrNull { it.latentH == latentH && it.latentW == latentW } ?: throw IllegalArgumentException(
            "this export has no TAEH3 graph for latent ${latentH}x$latentW (it has ${
                taeh3Graphs.map { "${it.latentH}x${it.latentW}" }.distinct().sorted()
            })",
        )

    /** Every graph file this manifest names, in load order. */
    public fun graphFiles(): List<String> = buildList {
        for (t in graphs.textRefine) add(t.file)
        add(graphs.embedVideo); add(graphs.embedAudio); add(graphs.headVideo); add(graphs.headAudio)
        for (b in graphs.blocks) { add(b.qkv); add(b.post) }
        for (a in attention) add(a.file)
        videoVae?.let { v ->
            add(v.decoder.inFile); addAll(v.decoder.blocks); add(v.decoder.out)
            v.encoderClip?.let { add(it) }
            v.encoderImage?.let { add(it) }
        }
        for (a in audio) { add(a.decoder); a.encoder?.let { add(it) } }
        for (t in taeh3Graphs) add(t.file)
    }

    /** Absolute path of a graph file; needs a [directory]. */
    public fun path(file: String): File {
        val dir = directory ?: throw IllegalStateException("manifest was not loaded from a directory; cannot resolve `$file`")
        return File(dir, file)
    }

    /** The memory plan of a clip on this export. */
    public fun plan(g: Geometry, textRows: Int, keyframes: Int, budgetBytes: Long, residentBlocks: Int = 2, preview: Boolean = false): Plan =
        Plan.compute(this, g, textRows, keyframes, budgetBytes, residentBlocks, preview)

    public companion object {
        public const val ADALN_ATOL: Float = 1e-6f

        private val json = Json { ignoreUnknownKeys = true; isLenient = true }

        /** Parse manifest text; `directory` is where its graph files live (optional). */
        public fun parse(text: String, directory: File? = null): Manifest {
            val root = try {
                json.parseToJsonElement(text).jsonObject
            } catch (e: Exception) {
                throw IllegalArgumentException("manifest is not a JSON object: ${e.message}", e)
            }
            val fmt = (root["format"] as? JsonPrimitive)?.contentOrNull
            if (fmt != MANIFEST_FORMAT) {
                throw IllegalArgumentException("manifest format ${fmt?.let { "\"$it\"" } ?: "missing"}, expected \"$MANIFEST_FORMAT\"")
            }
            val m = try {
                json.decodeFromJsonElement<Manifest>(root)
            } catch (e: IllegalArgumentException) {
                throw e
            } catch (e: Exception) {
                throw IllegalArgumentException("manifest does not follow $MANIFEST_FORMAT: ${e.message}", e)
            }
            return if (directory != null) m.copy(directory = directory) else m
        }

        /** Read `<dir>/manifest.json`. */
        public fun load(directory: File): Manifest {
            val f = File(directory, MANIFEST_FILE)
            if (!f.isFile) throw IllegalArgumentException("no $MANIFEST_FILE in $directory")
            return parse(f.readText(), directory)
        }
    }
}
