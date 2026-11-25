package dev.h3.litert.app

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.litert.Accelerator
import dev.h3.litert.core.AdalnTables
import dev.h3.litert.core.Event
import dev.h3.litert.core.Geometry
import dev.h3.litert.core.Keyframe
import dev.h3.litert.core.Manifest
import dev.h3.litert.core.Pipeline
import dev.h3.litert.core.Plan
import dev.h3.litert.core.Request
import dev.h3.litert.core.SafeTensors
import dev.h3.litert.core.TEXT_TAG
import dev.h3.litert.core.Tensor
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import java.nio.ByteBuffer
import java.util.Locale

/** A prompt gallery entry: `prompt_embeds [rows, text_dim]` and `text_tags [rows]` in one safetensors file. */
data class PromptFile(val file: File, val rows: Int, val textDim: Int, val visionRows: Int) {
    val name: String get() = file.name
}

/** Everything the screen shows. Immutable; the ViewModel replaces it. */
data class UiState(
    val graphsDir: File,
    val galleryDir: File,
    val manifest: Manifest? = null,
    val manifestError: String? = null,
    val gallery: List<PromptFile> = emptyList(),
    val prompt: PromptFile? = null,
    val sizes: List<Geometry> = emptyList(),
    val size: Geometry? = null,
    val seed: String = "7",
    val useGpu: Boolean = true,
    val preview: Boolean = false,
    val firstKeyframe: Uri? = null,
    val lastKeyframe: Uri? = null,
    val budgetBytes: Long = 0,
    val plan: Plan? = null,
    val planError: String? = null,
    val running: Boolean = false,
    val status: String = "",
    val stepMs: List<Double> = emptyList(),
    val prepareMs: Double? = null,
    val decodeMs: Double? = null,
    val audioMs: Double? = null,
    val frames: List<Bitmap> = emptyList(),
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val audio: Array<FloatArray>? = null,
    val sampleRate: Int = 0,
    val playing: Boolean = false,
    val log: List<String> = emptyList(),
    val error: String? = null,
) {
    val keyframes: Int get() = (if (firstKeyframe != null) 1 else 0) + (if (lastKeyframe != null) 1 else 0)
    val planFits: Boolean get() = plan?.fits == true

    /** The plan is enforced: a clip that does not fit the free memory cannot be started. */
    val canGenerate: Boolean
        get() = manifest != null && prompt != null && size != null && !running && seed.toLongOrNull() != null && planFits
}

/**
 * Runs `Pipeline.generate` on a background dispatcher and streams its events into [state].
 *
 * Files: `<external files>/graphs/` holds `manifest.json`, `adaln.safetensors` and the `.tflite`
 * graphs; `<external files>/gallery/` holds `*.safetensors` prompts (`prompt_embeds [rows, 5120]`
 * and `text_tags [rows]`, from `h3-litert encode` or an `h3 encode` gallery).
 */
class GenerateViewModel(app: Application) : AndroidViewModel(app) {
    private val root: File = app.getExternalFilesDir(null) ?: app.filesDir
    private val _state = MutableStateFlow(UiState(File(root, "graphs"), File(root, "gallery")))
    val state: StateFlow<UiState> = _state

    private var job: Job? = null
    private val player = AudioPlayer()

    init {
        refresh()
    }

    /** Re-read the manifest and the gallery (after an `adb push`). */
    fun refresh() {
        val s = _state.value
        s.graphsDir.mkdirs()
        s.galleryDir.mkdirs()
        val (manifest, err) = try {
            Manifest.load(s.graphsDir) to null
        } catch (e: Exception) {
            null to (e.message ?: e.toString())
        }
        val gallery = (s.galleryDir.listFiles { f -> f.isFile && f.name.endsWith(".safetensors") } ?: emptyArray())
            .sortedBy { it.name.lowercase(Locale.ROOT) }
            .mapNotNull { f -> runCatching { describePrompt(f) }.onFailure { log("${f.name}: ${it.message}") }.getOrNull() }
        val sizes = manifest?.sizes ?: emptyList()
        val size = s.size?.takeIf { it in sizes } ?: sizes.firstOrNull()
        _state.update {
            it.copy(
                manifest = manifest,
                manifestError = err,
                gallery = gallery,
                prompt = gallery.firstOrNull { p -> p.file == it.prompt?.file } ?: gallery.firstOrNull(),
                sizes = sizes,
                size = size,
                budgetBytes = availableMemory(),
                error = null,
            )
        }
        replan()
    }

    private fun describePrompt(f: File): PromptFile {
        val st = SafeTensors.read(f)
        val key = if ("prompt_embeds" in st) "prompt_embeds" else st.names.first()
        val shape = st.shape(key)
        val (rows, dim) = when {
            shape.size == 2 -> shape[0] to shape[1]
            shape.size == 3 && shape[0] == 1 -> shape[1] to shape[2]
            else -> throw IllegalArgumentException("$key is ${shape.toList()}, expected [rows, text_dim]")
        }
        val vision = if ("text_tags" in st) st.tensor("text_tags").data.count { it.toInt() != TEXT_TAG } else 0
        return PromptFile(f, rows, dim, vision)
    }

    fun selectPrompt(p: PromptFile) {
        _state.update { it.copy(prompt = p) }
        replan()
    }

    fun selectSize(g: Geometry) {
        _state.update { it.copy(size = g) }
        replan()
    }

    fun setSeed(seed: String) = _state.update { it.copy(seed = seed.filter { c -> c.isDigit() || c == '-' }.take(20)) }

    fun setUseGpu(on: Boolean) = _state.update { it.copy(useGpu = on) }

    fun setPreview(on: Boolean) {
        _state.update { it.copy(preview = on) }
        replan()
    }

    /** `anchor` is "first" or "last"; `uri` null clears it. */
    fun setKeyframe(anchor: String, uri: Uri?) {
        _state.update { if (anchor == "first") it.copy(firstKeyframe = uri) else it.copy(lastKeyframe = uri) }
        replan()
    }

    private fun replan() {
        val s = _state.value
        val m = s.manifest
        val g = s.size
        if (m == null || g == null) {
            _state.update { it.copy(plan = null, planError = null) }
            return
        }
        try {
            val p = m.plan(g, s.prompt?.rows ?: 1, s.keyframes, s.budgetBytes, residentBlocks = RESIDENT_BLOCKS, preview = s.preview)
            _state.update { it.copy(plan = p, planError = null) }
        } catch (e: Exception) {
            _state.update { it.copy(plan = null, planError = e.message ?: e.toString()) }
        }
    }

    private fun availableMemory(): Long {
        val am = getApplication<Application>().getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        return mi.availMem
    }

    fun generate() {
        val s = _state.value
        val manifest = s.manifest ?: return
        val prompt = s.prompt ?: return
        val size = s.size ?: return
        val seed = s.seed.toLongOrNull() ?: return
        if (!s.planFits) return
        job?.cancel()
        player.stop()
        _state.update {
            it.copy(
                running = true, status = "loading prompt and tables", stepMs = emptyList(), prepareMs = null, decodeMs = null, audioMs = null,
                frames = emptyList(), audio = null, playing = false, error = null, log = emptyList(),
            )
        }
        val app = getApplication<Application>()
        job = viewModelScope.launch(Dispatchers.Default) {
            val t0 = System.nanoTime()
            try {
                val (embeds, tags) = loadPrompt(prompt.file, manifest)
                val keyframes = buildList {
                    s.firstKeyframe?.let { add(Keyframe("first", Keyframes.load(app.contentResolver, it, size.width, size.height))) }
                    s.lastKeyframe?.let { add(Keyframe("last", Keyframes.load(app.contentResolver, it, size.width, size.height))) }
                }
                val req = Request(embeds, tags, size, seed, keyframes, s.preview)
                val adaln = AdalnTables.load(manifest.path(manifest.adaln.file))
                val cacheDir = File(app.cacheDir, "litert-gpu").apply { mkdirs() }
                LiteRtGraphStore(
                    manifest,
                    accelerator = if (s.useGpu) Accelerator.GPU else Accelerator.CPU,
                    cacheDir = cacheDir,
                    resident = RESIDENT_BLOCKS,
                    onFallback = { file, e -> log("$file: GPU compile failed, using CPU (${e.message})") },
                ).use { store ->
                    val pipeline = Pipeline(store, manifest, adaln)
                    log(
                        "${manifest.model} (${manifest.variant}) ${manifest.weights}, chunk ${manifest.chunk}, head group ${manifest.headGroup}, " +
                            "${manifest.graphs.blocks.size} blocks, ${manifest.sampler.steps.size} ${manifest.sampler.rule} steps; " +
                            "$size, seed $seed, ${prompt.rows} prompt rows, ${keyframes.size} keyframes${if (s.preview) ", TAEH3 preview" else ""}",
                    )
                    _state.update { it.copy(status = "opening graphs") }
                    val gen = pipeline.generate(req) { ev ->
                        ensureActive()
                        when (ev) {
                            is Event.Prepared -> _state.update {
                                it.copy(status = "prepared (${ev.tokens} tokens, ${ev.rows} rows)", prepareMs = ev.ms)
                            }
                            is Event.Step -> _state.update {
                                it.copy(status = "step ${ev.index + 1}/${ev.total}", stepMs = it.stepMs + ev.ms)
                            }
                            is Event.Frames -> {
                                val bitmaps = List(ev.count) { i -> toBitmap(ev.frame(i), ev.width, ev.height) }
                                _state.update {
                                    it.copy(
                                        status = "decoded ${ev.first + ev.count} frames", frames = it.frames + bitmaps,
                                        frameWidth = ev.width, frameHeight = ev.height,
                                    )
                                }
                            }
                            is Event.Audio -> _state.update {
                                it.copy(status = "audio decoded (${ev.samples} samples @ ${ev.sampleRate} Hz)", audio = ev.stereo, sampleRate = ev.sampleRate)
                            }
                        }
                    }
                    val total = (System.nanoTime() - t0) / 1e6
                    _state.update {
                        it.copy(
                            running = false, decodeMs = gen.timings.decodeS * 1000.0, audioMs = gen.timings.audioS * 1000.0,
                            status = "${gen.video.frames} frames ${gen.video.width}x${gen.video.height} + ${gen.audio[0].size} audio samples in ${fmtMs(total)}",
                        )
                    }
                    log("timings: ${gen.timings}")
                    playAudio()
                }
            } catch (e: CancellationException) {
                _state.update { it.copy(running = false, status = "cancelled") }
            } catch (e: Throwable) {
                _state.update { it.copy(running = false, status = "failed", error = e.message ?: e.toString()) }
            }
        }
    }

    fun cancel() {
        job?.cancel()
    }

    fun playAudio() {
        val s = _state.value
        val audio = s.audio ?: return
        try {
            player.play(audio, s.sampleRate)
            if (player.playbackRate != s.sampleRate) log("audio: ${s.sampleRate} Hz is below what AudioTrack plays; upsampled to ${player.playbackRate} Hz by repetition")
            _state.update { it.copy(playing = true) }
        } catch (e: Exception) {
            log("audio playback failed: ${e.message}")
        }
    }

    fun stopAudio() {
        player.stop()
        _state.update { it.copy(playing = false) }
    }

    override fun onCleared() {
        player.stop()
        super.onCleared()
    }

    /** `prompt_embeds [rows, text_dim]` (a leading batch axis of 1 is tolerated) and `text_tags [rows]` (all text when absent). */
    private fun loadPrompt(file: File, manifest: Manifest): Pair<Tensor, List<Int>> {
        val st = SafeTensors.read(file)
        val key = if ("prompt_embeds" in st) "prompt_embeds" else st.names.first()
        var t = st.tensor(key)
        if (t.rank == 3 && t.shape[0] == 1) t = t.reshape(t.shape[1], t.shape[2])
        val c = manifest.cfg
        if (!(t.rank == 2 && t.shape[1] == c.textDim)) {
            throw IllegalArgumentException("${file.name}: $key is ${t.shapeString()}, expected [rows, ${c.textDim}]")
        }
        val tags = if ("text_tags" in st) st.tensor("text_tags").data.map { it.toInt() } else List(t.rows) { TEXT_TAG }
        if (tags.size != t.rows) throw IllegalArgumentException("${file.name}: text_tags has ${tags.size} entries for ${t.rows} rows")
        return t to tags
    }

    private fun log(line: String) = _state.update { it.copy(log = (it.log + line).takeLast(200)) }

    private fun fmtMs(ms: Double) = if (ms >= 1000) String.format(Locale.ROOT, "%.1f s", ms / 1000) else String.format(Locale.ROOT, "%.0f ms", ms)

    companion object {
        /** Block graphs kept open at once (`LiteRtGraphStore`'s LRU and the plan's residency). */
        const val RESIDENT_BLOCKS = 4

        /** `[H, W, 4]` RGBA bytes → an `ARGB_8888` bitmap (its memory order is R, G, B, A). */
        fun toBitmap(rgba: ByteArray, width: Int, height: Int): Bitmap {
            val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bmp.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
            return bmp
        }
    }
}
