package dev.h3.litert.core

/** A keyframe for FL2VA: `anchor` is "first" or "last", `pixels` `[3, 1, H, W]` in `[0, 1]` at the canvas size. */
public class Keyframe(public val anchor: String, public val pixels: Tensor) {
    init {
        require(anchor == "first" || anchor == "last") { "keyframe anchor `$anchor` is not first or last" }
        require(pixels.rank == 4 && pixels.shape[0] == 3 && pixels.shape[1] == 1) { "keyframe pixels must be [3, 1, H, W], got ${pixels.shapeString()}" }
    }
}

/** What a generation asks for (`request.py`). */
public class Request(
    /** `[rows, text_dim]` prompt embeddings. */
    public val embeds: Tensor,
    /** One tag per prompt row: text (1) or vision (0). */
    public val textTags: List<Int>,
    public val geometry: Geometry,
    public val seed: Long = 7,
    public val keyframes: List<Keyframe> = emptyList(),
    /** TAEH3 instead of the VAE decoder. */
    public val preview: Boolean = false,
)

/** What the pipeline reports while it runs. */
public sealed class Event {
    /** Text refined, layout and noise ready; `tokens` is the unpadded packed length, `rows` the attention `S`. */
    public data class Prepared(val ms: Double, val tokens: Int, val rows: Int) : Event()

    /** One denoising step finished. */
    public data class Step(val index: Int, val total: Int, val ms: Double) : Event()

    /**
     * Frames `[first, first + count)` decoded by one chunk, as `[count, height, width, 4]` RGBA bytes
     * (alpha 255), ready for an `ARGB_8888` `Bitmap`.
     */
    public class Frames(
        public val first: Int,
        public val count: Int,
        public val width: Int,
        public val height: Int,
        public val rgba: ByteArray,
    ) : Event() {
        public fun frame(i: Int): ByteArray {
            val n = width * height * 4
            return rgba.copyOfRange(i * n, (i + 1) * n)
        }

        override fun toString(): String = "Frames(first=$first, count=$count, ${width}x$height)"
    }

    /** The decoded stereo waveform in `[−1, 1]`: `stereo[channel][sample]`. */
    public class Audio(public val stereo: Array<FloatArray>, public val sampleRate: Int) : Event() {
        public val samples: Int get() = stereo[0].size

        override fun toString(): String = "Audio(${stereo.size}x$samples @ $sampleRate Hz)"
    }
}

public class Timings {
    public var prepareS: Double = 0.0
    public val stepsS: MutableList<Double> = mutableListOf()
    public var decodeS: Double = 0.0
    public var audioS: Double = 0.0

    override fun toString(): String =
        "prepare ${fmt(prepareS)} s; steps ${stepsS.joinToString(" / ") { fmt(it) }} s; decode ${fmt(decodeS)} s; audio ${fmt(audioS)} s"

    private fun fmt(v: Double) = String.format(java.util.Locale.ROOT, "%.3f", v)
}

/** A decoded clip: `[frames, height, width, 4]` RGBA bytes. */
public class Video(public val frames: Int, public val width: Int, public val height: Int, public val rgba: ByteArray) {
    public fun frame(i: Int): ByteArray {
        val n = width * height * 4
        return rgba.copyOfRange(i * n, (i + 1) * n)
    }
}

/** Everything `generate` produces. */
public class Generation(public val video: Video, public val audio: Array<FloatArray>, public val sampleRate: Int, public val timings: Timings)

/** Encodes one ImageNet-normalised clip or frame `[3, F, H, W]` into moments `[2·lz, T, h, w]` (`encode_clip`). */
public fun interface ClipEncoder {
    public fun encode(pixels: Tensor): Tensor
}

/**
 * `encoder_tile_runner`: the keyframe encoder over the manifest's `vae_enc_image` / `vae_enc_clip`
 * graphs with its tiling (pixel tiles of `tile`, `overlap`; moments stitched at the VAE ratio).
 */
public class KeyframeEncoder(private val store: GraphStore, private val manifest: Manifest = store.manifest) : ClipEncoder {
    override fun encode(pixels: Tensor): Tensor {
        val vv = manifest.videoVae ?: throw IllegalArgumentException("this export has no video VAE graphs")
        val name = (if (pixels.shape[1] == 1) vv.encoderImage else vv.encoderClip)
            ?: throw IllegalArgumentException("this export has no VAE encoder graphs (exported with --no-encoders)")
        val ratio = manifest.vae.spatialRatio()
        return VaeHost.tiled(pixels, 1, ratio, vv.tile, vv.overlap) { t -> store.fixed(name).run(mapOf("x" to t)).single() }
    }
}

/** The prepared state of one request: the layout after the text padding shift, the row buffers, the tables. */
public class Prepared(
    public val layout: Layout,
    /** `[T, d]`: the refined text bucket, pad rows included. */
    public val text: Tensor,
    public val nText: Int,
    public val pad: Int,
    /** `[cond + target video rows, P]`; the target rows are the state the sampler advances. */
    public var videoRows: Tensor,
    /** `[2A, A_c]`. */
    public var audioRows: Tensor,
    public val rope: RopeTable,
    /** `[S]`: 0 for a real row, −1e9 for a padded text row. */
    public val mask: Tensor,
    /** Packed indices after the text padding shift. */
    public val videoIdx: IntArray,
    public val audioIdx: IntArray,
    /** Packed tags after the shift (pad rows are text). */
    public val tags: IntArray,
) {
    /** The attention graph's `S`. */
    public val rows: Int get() = layout.sequenceLength + pad
}

/**
 * Prompt embeddings → packed layout → denoising loop → decoded video and audio, driving the graphs
 * exactly as `docs/graphs.md`'s host loop and `pipeline.py` do. T2VA and FL2VA; Euler, renoise
 * and PDD samplers.
 */
public class Pipeline(
    public val store: GraphStore,
    public val manifest: Manifest = store.manifest,
    public val adaln: AdalnTables = AdalnTables.load(manifest.path(manifest.adaln.file)),
) {
    public val cfg: H3Config get() = manifest.cfg

    init {
        adaln.validate(cfg, manifest.adaln.timesteps)
    }

    // ---- helpers ------------------------------------------------------------------------------

    /**
     * Run a per-row graph over row chunks of `perRow` (zero-padded to `chunk`), with `fixed` bound
     * unchanged to every call; returns the outputs with the padded rows dropped.
     */
    internal fun runRows(graph: Graph, chunk: Int, nReal: Int, perRow: Map<String, Tensor>, fixed: Map<String, Tensor>): List<Tensor> {
        val padded = perRow.mapValues { Host.padRows(it.value, chunk).tensor }
        val n = padded.values.first().rows
        for ((k, v) in padded) require(v.rows == n) { "`$k` has ${v.rows} rows, expected $n" }
        val outs = ArrayList<List<Tensor>>(n / chunk)
        var r = 0
        while (r < n) {
            val inputs = HashMap<String, Tensor>(padded.size + fixed.size)
            for ((k, v) in padded) inputs[k] = v.sliceRows(r, r + chunk)
            inputs.putAll(fixed)
            outs += graph.run(inputs)
            r += chunk
        }
        val nOut = outs[0].size
        return List(nOut) { i -> Tensor.concatRows(outs.map { it[i] }).sliceRows(0, nReal) }
    }

    private fun copyRows(src: Tensor, srcRow: Int, dst: Tensor, dstRow: Int, n: Int) {
        val rs = dst.rowSize
        require(src.rowSize == rs) { "copyRows: row size ${src.rowSize} vs $rs" }
        System.arraycopy(src.data, srcRow * rs, dst.data, dstRow * rs, n * rs)
    }

    private fun secs(since: Long): Double = (System.nanoTime() - since) / 1e9

    // ---- prompt + layout ------------------------------------------------------------------------

    public fun prepare(req: Request, encoder: ClipEncoder? = null): Prepared {
        val c = cfg
        val m = manifest
        val ratio = m.vae.spatialRatio()
        val g = req.geometry
        g.validate(ratio)
        val lat = g.latent(ratio)
        val na = g.audioLatents()
        val nText = req.textTags.size
        if (!(req.embeds.rank == 2 && req.embeds.shape[0] == nText && req.embeds.shape[1] == c.textDim)) {
            throw IllegalArgumentException("prompt embeds are ${req.embeds.shapeString()}, expected [$nText, ${c.textDim}]")
        }
        val bucket = m.textRefineFor(nText)
        val t = bucket.rows
        val pad = t - nText
        val emb = Tensor.zeros(t, c.textDim)
        System.arraycopy(req.embeds.data, 0, emb.data, 0, req.embeds.size)
        val tmask = Tensor.zeros(t)
        for (i in nText until t) tmask.data[i] = MASK_PAD
        val text = store.fixed(bucket.file).run(mapOf("embeds" to emb, "mask" to tmask)).single()
        require(text.rank == 2 && text.shape[0] == t && text.shape[1] == c.hidden) {
            "${bucket.file} returned ${text.shapeString()}, expected [$t, ${c.hidden}]"
        }
        val anchors = req.keyframes.map { it.anchor }
        val layout = Packing.buildLayout(req.textTags, lat.t, lat.h, lat.w, na, c.patch, anchors)
        val rng = Rng(req.seed)
        var cond = Tensor.zeros(0, c.videoPatchDim())
        if (req.keyframes.isNotEmpty()) {
            val enc = encoder ?: throw IllegalArgumentException("keyframes need a ClipEncoder (KeyframeEncoder over the VAE encoder graphs)")
            cond = Tensor.concatRows(req.keyframes.map { encodeKeyframe(it.pixels, enc) })
            val noise = rng.normal(cond.size)
            val aug = Schedules.KEYFRAME_NOISE_AUG
            val oneMinus = 1.0f - Schedules.KEYFRAME_NOISE_AUG
            for (i in cond.data.indices) cond.data[i] = aug * cond.data[i] + oneMinus * noise[i]
        }
        val noise = rng.normal(c.inChannels, lat.t, lat.h, lat.w)
        val videoRows = Tensor.concatRows(listOf(cond, Packing.patchRows(noise, c.patch)))
        val audioRows = rng.normal(na * 2, c.audioInChannels)
        fun shift(i: Int) = if (i < nText) i else i + pad
        val sPadded = layout.sequenceLength + pad
        val pos = Positions(sPadded)
        val tags = IntArray(sPadded) { TEXT_TAG }
        for (i in 0 until layout.sequenceLength) {
            val j = shift(i)
            pos.set(j, layout.positions.t(i), layout.positions.h(i), layout.positions.w(i))
            tags[j] = layout.tokenTags[i]
        }
        val rope = Rope.h3Rope(pos, c.ropeFreqDim, c.ropeTheta)
        val mask = Tensor.zeros(sPadded)
        for (i in nText until t) mask.data[i] = MASK_PAD
        return Prepared(
            layout, text, nText, pad, videoRows, audioRows, rope, mask,
            IntArray(layout.videoIndices.size) { shift(layout.videoIndices[it]) },
            IntArray(layout.audioIndices.size) { shift(layout.audioIndices[it]) },
            tags,
        )
    }

    /** Pixels `[3, 1, H, W]` in `[0, 1]` → normalised, f16-rounded latent patch rows (the reference's seed 42 posterior). */
    internal fun encodeKeyframe(pixels: Tensor, encoder: ClipEncoder): Tensor {
        val vc = manifest.vae
        val moments = VaeHost.encode(vc, VaeHost.normalizePixels(pixels), encoder::encode)
        val lz = moments.shape[0] / 2
        val rng = Rng(KEYFRAME_ENCODE_SEED)
        val noise = Tensor(rng.normal(lz * moments.rowSize), intArrayOf(lz) + moments.shape.copyOfRange(1, moments.shape.size))
        val z = VaeHost.posteriorSample(moments, noise)
        val zn = VaeHost.normalizeLatents(vc, Host.roundThroughHalf(z))
        return Packing.patchRows(zn, cfg.patch)
    }

    // ---- one forward ----------------------------------------------------------------------------

    /** `(v_video [video rows, P], v_audio [audio rows, A_c])` of one step. */
    public fun forward(p: Prepared, step: SamplerStep, attn: AttentionEntry): Pair<Tensor, Tensor> {
        val c = cfg
        val m = manifest
        val chunk = m.chunk
        val groupHeads = m.headGroup
        val d = c.hidden
        val inner = c.dim
        val hd = c.headDim
        val s = p.rows
        require(attn.tokens == s) { "attention graph ${attn.file} is for ${attn.tokens} rows, the request has $s" }
        require(attn.heads == groupHeads) { "attention graph ${attn.file} takes ${attn.heads} heads, the manifest's head_group is $groupHeads" }

        val rt = Packing.rowTimesteps(p.layout, step.tVideo, step.tAudio, maxOf(step.tVideo, m.sampler.condVideoT), m.sampler.condAudioT)
        val uOf = IntArray(rt.unique.size) { adaln.indexOf(rt.unique[it]) }
        val tIdx = IntArray(s)
        for (i in 0 until p.layout.sequenceLength) tIdx[if (i < p.nText) i else i + p.pad] = rt.index[i]
        for (i in p.nText until p.nText + p.pad) tIdx[i] = rt.index[0]

        val h = Tensor.zeros(s, d)
        copyRows(p.text, 0, h, 0, p.nText + p.pad)
        val embV = store.fixed(m.graphs.embedVideo)
        val embA = store.fixed(m.graphs.embedAudio)
        for (r in Packing.runs(p.videoIdx)) {
            val out = runRows(embV, chunk, r.rows, mapOf("rows" to p.videoRows.sliceRows(r.offset, r.offset + r.rows)), emptyMap()).single()
            copyRows(out, 0, h, r.start, r.rows)
        }
        for (r in Packing.runs(p.audioIdx)) {
            val out = runRows(embA, chunk, r.rows, mapOf("rows" to p.audioRows.sliceRows(r.offset, r.offset + r.rows)), emptyMap()).single()
            copyRows(out, 0, h, r.start, r.rows)
        }

        val mruns = Packing.modRuns(tIdx, p.tags)
        val attnGraph = store.fixed(attn.file)
        val cq = attn.qChunk
        val q = Tensor.zeros(s, c.heads, hd)
        val k = Tensor.zeros(s, c.heads, hd)
        val v = Tensor.zeros(s, c.heads, hd)
        val o = Tensor.zeros(s, inner)
        for ((l, b) in m.graphs.blocks.withIndex()) {
            val gq = store.block(b.qkv)
            for (run in mruns) {
                val mods = adaln.mods(uOf[run.tIdx], l, run.tag)
                val end = run.start + run.rows
                val outs = runRows(
                    gq, chunk, run.rows,
                    mapOf("h" to h.sliceRows(run.start, end), "cos" to p.rope.cos.sliceRows(run.start, end), "sin" to p.rope.sin.sliceRows(run.start, end)),
                    mapOf("mods" to mods),
                )
                require(outs.size == 3) { "${b.qkv} returned ${outs.size} outputs, expected q, k, v" }
                copyRows(outs[0], 0, q, run.start, run.rows)
                copyRows(outs[1], 0, k, run.start, run.rows)
                copyRows(outs[2], 0, v, run.start, run.rows)
            }
            for (g in 0 until c.heads / groupHeads) {
                val qg = headGroup(q, g, groupHeads)
                val kg = headGroup(k, g, groupHeads)
                val vg = headGroup(v, g, groupHeads)
                val og = runRows(attnGraph, cq, s, mapOf("q" to qg), mapOf("k" to kg, "v" to vg, "mask" to p.mask)).single()
                require(og.rank == 2 && og.shape[0] == s && og.shape[1] == groupHeads * hd) {
                    "${attn.file} returned ${og.shapeString()}, expected [$s, ${groupHeads * hd}]"
                }
                // o[:, g·G·hd … (g+1)·G·hd) = og
                val w = groupHeads * hd
                for (row in 0 until s) System.arraycopy(og.data, row * w, o.data, row * inner + g * w, w)
            }
            val gp = store.block(b.post)
            for (run in mruns) {
                val mods = adaln.mods(uOf[run.tIdx], l, run.tag)
                val end = run.start + run.rows
                val out = runRows(gp, chunk, run.rows, mapOf("h" to h.sliceRows(run.start, end), "o" to o.sliceRows(run.start, end)), mapOf("mods" to mods)).single()
                copyRows(out, 0, h, run.start, run.rows)
            }
        }

        val pdd = m.sampler.rule == "pdd"
        val vOut = Tensor.zeros(p.videoIdx.size, c.videoPatchDim())
        val aOut = Tensor.zeros(p.audioIdx.size, c.audioInChannels)
        for ((graphName, idx, out, weights) in listOf(
            Head(m.graphs.headVideo, p.videoIdx, vOut, step.pddVideoWeights),
            Head(m.graphs.headAudio, p.audioIdx, aOut, step.pddAudioWeights),
        )) {
            val graph = store.fixed(graphName)
            val extra = if (pdd) {
                val w = weights ?: throw IllegalArgumentException("pdd step without fused-head weights")
                mapOf("weights" to Tensor(w.toFloatArray(), intArrayOf(w.size)))
            } else emptyMap()
            for (r in Packing.runs(idx)) {
                // a run of one modality may still span two timesteps (condition rows vs target rows)
                var start = r.start
                while (start < r.start + r.rows) {
                    val ti = tIdx[start]
                    var end = start
                    while (end < r.start + r.rows && tIdx[end] == ti) end++
                    val n = end - start
                    val off = r.offset + (start - r.start)
                    val res = runRows(graph, chunk, n, mapOf("h" to h.sliceRows(start, end)), mapOf("mods" to adaln.headMods(uOf[ti])) + extra).single()
                    copyRows(res, 0, out, off, n)
                    start = end
                }
            }
        }
        return vOut to aOut
    }

    private data class Head(val graph: String, val idx: IntArray, val out: Tensor, val weights: List<Float>?)

    /** `x[:, gG:(g+1)G, :]` of an `[S, H, hd]` tensor, contiguous `[S, G, hd]`. */
    private fun headGroup(x: Tensor, g: Int, groupHeads: Int): Tensor {
        val s = x.shape[0]
        val heads = x.shape[1]
        val hd = x.shape[2]
        if (groupHeads == heads) return x
        val out = FloatArray(s * groupHeads * hd)
        val w = groupHeads * hd
        for (row in 0 until s) System.arraycopy(x.data, (row * heads + g * groupHeads) * hd, out, row * w, w)
        return Tensor(out, intArrayOf(s, groupHeads, hd))
    }

    // ---- denoise --------------------------------------------------------------------------------

    public class Denoised(public val latent: Tensor, public val audioRows: Tensor)

    /** The normalised video latent `[C, T, h, w]` and the target audio rows `[2A, A_c]`. */
    public fun denoise(req: Request, on: ((Event) -> Unit)? = null, timings: Timings? = null, encoder: ClipEncoder? = null): Denoised {
        val c = cfg
        val m = manifest
        val tm = timings ?: Timings()
        val t0 = System.nanoTime()
        val enc = encoder ?: if (req.keyframes.isNotEmpty()) KeyframeEncoder(store, m) else null
        val p = prepare(req, enc)
        val attn = m.attentionFor(p.rows)
        tm.prepareS = secs(t0)
        on?.invoke(Event.Prepared(tm.prepareS * 1000.0, p.layout.sequenceLength, p.rows))
        val rule = m.sampler.rule
        val steps = m.sampler.steps
        val offV = p.layout.numConditionVideoRows
        if (rule == "pdd") {
            val sv = steps[0].sigmaVideo
            val rs = p.videoRows.rowSize
            for (i in offV * rs until p.videoRows.size) p.videoRows.data[i] *= sv
            val sa = steps[0].sigmaAudio
            for (i in p.audioRows.data.indices) p.audioRows.data[i] *= sa
        }
        val renoise = Rng(req.seed + 1)
        for ((i, st) in steps.withIndex()) {
            val ts = System.nanoTime()
            val (vv, va) = forward(p, st, attn)
            val tgt = p.videoRows.sliceRows(offV, p.videoRows.rows)
            val nzV = if (Sampler.needsNoise(rule, st.sigmaVideoNext)) rng(renoise, tgt) else null
            val newV = Sampler.flowStep(tgt, vv.sliceRows(offV, vv.rows), st.sigmaVideo, st.sigmaVideoNext, rule, nzV)
            copyRows(newV, 0, p.videoRows, offV, newV.rows)
            val nzA = if (Sampler.needsNoise(rule, st.sigmaAudioNext)) rng(renoise, p.audioRows) else null
            p.audioRows = Sampler.flowStep(p.audioRows, va, st.sigmaAudio, st.sigmaAudioNext, rule, nzA)
            tm.stepsS += secs(ts)
            on?.invoke(Event.Step(i, steps.size, tm.stepsS.last() * 1000.0))
        }
        val lat = req.geometry.latent(m.vae.spatialRatio())
        val latent = Packing.unpatchRows(p.videoRows.sliceRows(offV, p.videoRows.rows), c.inChannels, lat.t, lat.h, lat.w, c.patch)
        return Denoised(latent, p.audioRows)
    }

    private fun rng(r: Rng, like: Tensor): Tensor = Tensor(r.normal(like.size), like.shape.copyOf())

    // ---- decode ---------------------------------------------------------------------------------

    /** Normalised latent `[C, T, h, w]` → RGBA frames, through the VAE decoder graphs or TAEH3 (`preview`). */
    public fun decodeVideo(latent: Tensor, preview: Boolean = false, on: ((Event) -> Unit)? = null, timings: Timings? = null): Video {
        val tm = timings ?: Timings()
        val t0 = System.nanoTime()
        require(latent.rank == 4) { "latent must be [C, T, h, w], got ${latent.shapeString()}" }
        val parts = ArrayList<ByteArray>()
        var total = 0
        var width = 0
        var height = 0
        fun emit(rgba: ByteArray, count: Int, w: Int, h: Int) {
            width = w
            height = h
            parts += rgba
            on?.invoke(Event.Frames(total, count, w, h, rgba))
            total += count
        }
        if (preview) {
            decodeTaeh3(latent) { rgba, count, w, h -> emit(rgba, count, w, h) }
        } else {
            val m = manifest
            val vv = m.videoVae ?: throw IllegalArgumentException("this export has no video VAE decoder")
            val dec = vv.decoder
            val ratio = m.vae.spatialRatio()
            val gIn = store.fixed(dec.inFile)
            val gOut = store.fixed(dec.out)
            val blocks = dec.blocks.map { store.fixed(it) }
            val clipTile: (Tensor) -> Tensor = { z ->
                if (!(z.shape[1] == dec.latentT && z.shape[2] == dec.latentH && z.shape[3] == dec.latentW)) {
                    throw IllegalArgumentException(
                        "decoder tile [${z.shape[1]}, ${z.shape[2]}, ${z.shape[3]}] but the export has graphs for [${dec.latentT}, ${dec.latentH}, ${dec.latentW}]",
                    )
                }
                val first = gIn.run(mapOf("z" to z))
                require(first.size == 3) { "${dec.inFile} returned ${first.size} outputs, expected h, cos, sin" }
                var hRows = first[0]
                val cos = first[1]
                val sin = first[2]
                for (b in blocks) hRows = b.run(mapOf("h" to hRows, "cos" to cos, "sin" to sin)).single()
                gOut.run(mapOf("h" to hRows)).single()
            }
            VaeHost.decode(m.vae, VaeHost.denormalizeLatents(m.vae, latent), { z -> VaeHost.tiled(z, ratio, 1, vv.tile, vv.overlap, clipTile) }) { px ->
                emit(Host.toRgba8(px), px.shape[1], px.shape[3], px.shape[2])
            }
        }
        tm.decodeS = secs(t0)
        val all = ByteArray(parts.sumOf { it.size })
        var o = 0
        for (p in parts) { System.arraycopy(p, 0, all, o, p.size); o += p.size }
        return Video(total, width, height, all)
    }

    private fun decodeTaeh3(latent: Tensor, emit: (ByteArray, Int, Int, Int) -> Unit) {
        val m = manifest
        val tc = m.taeh3 ?: throw IllegalArgumentException("this export has no TAEH3 graph")
        val c = latent.shape[0]
        val t = latent.shape[1]
        val h = latent.shape[2]
        val w = latent.shape[3]
        val entry = m.taeh3For(h, w)
        val graph = store.fixed(entry.file)
        val f = entry.frames
        val nmem = 3 * tc.memBlocks
        var mems: List<Tensor> = List(nmem) { i ->
            val s = i / tc.memBlocks
            Tensor.zeros(1, tc.widths[s], h shl s, w shl s)
        }
        val up = tc.tUpscale()
        val plane = h * w
        var rawIndex = 0
        var f0 = 0
        while (f0 < t) {
            val n = minOf(f, t - f0)
            // z[frame, channel, y, x] = latent[channel, f0 + frame, y, x]; padded frames stay zero.
            val z = Tensor.zeros(f, c, h, w)
            for (fr in 0 until n) for (ch in 0 until c) {
                System.arraycopy(latent.data, (ch * t + f0 + fr) * plane, z.data, (fr * c + ch) * plane, plane)
            }
            val inputs = HashMap<String, Tensor>(nmem + 1)
            inputs["z"] = z
            for (i in 0 until nmem) inputs["mem_$i"] = mems[i]
            val out = graph.run(inputs)
            require(out.size == nmem + 1) { "${entry.file} returned ${out.size} outputs, expected rgb + $nmem memories" }
            val rgbAll = out[0]
            require(rgbAll.rank == 4 && rgbAll.shape[0] == f * up && rgbAll.shape[1] == 3) {
                "${entry.file} rgb is ${rgbAll.shapeString()}, expected [${f * up}, 3, H, W]"
            }
            mems = out.subList(1, out.size)
            val real = n * up
            val keep = (0 until real).filter { TaeH3Config.keepsRawFrame(rawIndex + it) }
            rawIndex += real
            if (keep.isNotEmpty()) {
                val frames = Tensor.concatRows(keep.map { rgbAll.sliceRows(it, it + 1) })
                emit(Host.framesToRgba8(frames), frames.shape[0], frames.shape[3], frames.shape[2])
            }
            f0 += f
        }
    }

    /** Normalised target audio rows `[2A, A_c]` → stereo `[2][800·A]` in `[−1, 1]`. */
    public fun decodeAudio(audioRows: Tensor, na: Int, timings: Timings? = null): Array<FloatArray> {
        val tm = timings ?: Timings()
        val t0 = System.nanoTime()
        val ac = manifest.audioVae
        val entry = manifest.audioFor(na)
        val g = store.fixed(entry.decoder)
        val ch = ac.latentChannels
        require(audioRows.rank == 2 && audioRows.shape[0] == 2 * na && audioRows.shape[1] == ch) {
            "audio rows are ${audioRows.shapeString()}, expected [${2 * na}, $ch]"
        }
        val out = Array(2) { channel ->
            // z[c, a] = rows[channel·A + a, c]·std[c] + mean[c]
            val z = Tensor.zeros(ch, na)
            for (cc in 0 until ch) {
                val std = ac.latentsStd[cc].toFloat()
                val mean = ac.latentsMean[cc].toFloat()
                for (a in 0 until na) z.data[cc * na + a] = audioRows.data[(channel * na + a) * ch + cc] * std + mean
            }
            val wave = g.run(mapOf("z" to z)).single()
            require(wave.size == ac.hop() * na) { "${entry.decoder} returned ${wave.shapeString()}, expected [${ac.hop() * na}]" }
            wave.data
        }
        tm.audioS = secs(t0)
        return out
    }

    public fun generate(req: Request, on: ((Event) -> Unit)? = null): Generation {
        val tm = Timings()
        val den = denoise(req, on, tm)
        val video = decodeVideo(den.latent, req.preview, on, tm)
        val audio = decodeAudio(den.audioRows, req.geometry.audioLatents(), tm)
        on?.invoke(Event.Audio(audio, manifest.audioVae.sampleRate))
        return Generation(video, audio, manifest.audioVae.sampleRate, tm)
    }

    public companion object {
        public const val MASK_PAD: Float = -1e9f
        public const val KEYFRAME_ENCODE_SEED: Long = 42
    }
}
