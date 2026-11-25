package dev.h3.litert.core

/**
 * A GraphStore whose graphs check every input name and shape against `docs/graphs.md` for the
 * manifest's config, record the calls, and return contract-shaped outputs.
 *
 * Row graphs return 1.0 for rows whose primary input row is non-zero and NaN for all-zero rows, so
 * a padded row that leaks into a real output poisons the latent. The attention fake poisons every
 * query when an unmasked key row is all-zero (a padded row the host forgot to mask). The TAEH3 fake
 * returns 1.0 for frames of real latent frames and 0.25 for padded ones; the VAE decoder fake
 * returns a constant in normalised pixel space.
 */
class FakeGraphStore(manifest: Manifest, resident: Int = 4) : GraphStore(manifest, resident) {
    class Call(val graph: String, val inputs: Map<String, List<Int>>, val mask: FloatArray? = null, val modsFirst: Float? = null)

    val calls = mutableListOf<Call>()
    val opened = mutableListOf<String>()
    val closed = mutableListOf<String>()

    fun calls(graph: String) = calls.filter { it.graph == graph }
    fun count(graph: String) = calls.count { it.graph == graph }
    fun names() = calls.map { it.graph }

    override fun open(file: String): Graph {
        opened += file
        return FakeGraph(file)
    }

    private inner class FakeGraph(override val name: String) : Graph {
        private val c = manifest.cfg
        private val d = c.hidden
        private val C = manifest.chunk
        private val G = manifest.headGroup

        override fun close() { closed += name }

        private fun expect(inputs: Map<String, Tensor>, vararg spec: Pair<String, List<Int>>) {
            val want = spec.toMap()
            if (inputs.keys != want.keys) throw IllegalArgumentException("$name: inputs ${want.keys.sorted()}, got ${inputs.keys.sorted()}")
            for ((k, shape) in want) {
                val got = inputs[k]!!.shape.toList()
                if (got != shape) throw IllegalArgumentException("$name: input `$k` is $got, expected $shape")
            }
        }

        private fun rowMarker(t: Tensor): FloatArray {
            val rs = t.rowSize
            return FloatArray(t.rows) { r ->
                var zero = true
                for (i in 0 until rs) if (t.data[r * rs + i] != 0f) { zero = false; break }
                if (zero) Float.NaN else 1f
            }
        }

        private fun rowsOf(marker: FloatArray, vararg tail: Int): Tensor {
            val rs = Tensor.numel(tail)
            val out = FloatArray(marker.size * rs)
            for (r in marker.indices) java.util.Arrays.fill(out, r * rs, (r + 1) * rs, marker[r])
            return Tensor(out, intArrayOf(marker.size) + tail)
        }

        private fun const(v: Float, vararg shape: Int) = Tensor(FloatArray(Tensor.numel(shape)) { v }, shape)

        override fun run(inputs: Map<String, Tensor>): List<Tensor> {
            calls += Call(name, inputs.mapValues { it.value.shape.toList() }, inputs["mask"]?.data?.copyOf(), inputs["mods"]?.data?.get(0))
            val g = manifest.graphs
            val H = c.heads
            val hd = c.headDim
            val half = c.rotaryDim() / 2
            val vae = manifest.vae
            val av = manifest.audioVae
            val vv = manifest.videoVae
            return when {
                g.textRefine.any { it.file == name } -> {
                    val t = g.textRefine.first { it.file == name }.rows
                    expect(inputs, "embeds" to listOf(t, c.textDim), "mask" to listOf(t))
                    listOf(const(1f, t, d))
                }
                name == g.embedVideo -> {
                    expect(inputs, "rows" to listOf(C, c.videoPatchDim()))
                    listOf(rowsOf(rowMarker(inputs["rows"]!!), d))
                }
                name == g.embedAudio -> {
                    expect(inputs, "rows" to listOf(C, c.audioInChannels))
                    listOf(rowsOf(rowMarker(inputs["rows"]!!), d))
                }
                g.blocks.any { it.qkv == name } -> {
                    expect(inputs, "h" to listOf(C, d), "mods" to listOf(6, d), "cos" to listOf(C, half), "sin" to listOf(C, half))
                    val m = rowMarker(inputs["h"]!!)
                    listOf(rowsOf(m, H, hd), rowsOf(m, H, hd), rowsOf(m, H, hd))
                }
                name.startsWith("attn_") -> {
                    val a = manifest.attention.first { it.file == name }
                    expect(inputs, "q" to listOf(a.qChunk, G, hd), "k" to listOf(a.tokens, G, hd), "v" to listOf(a.tokens, G, hd), "mask" to listOf(a.tokens))
                    val m = rowMarker(inputs["q"]!!)
                    val mask = inputs["mask"]!!.data
                    val km = rowMarker(inputs["k"]!!)
                    val vm = rowMarker(inputs["v"]!!)
                    // every query attends to every unmasked key: a zero (padded) key that is not masked poisons every row
                    val kBad = mask.indices.any { mask[it] == 0f && (km[it].isNaN() || vm[it].isNaN()) }
                    listOf(rowsOf(if (kBad) FloatArray(m.size) { Float.NaN } else m, G * hd))
                }
                g.blocks.any { it.post == name } -> {
                    expect(inputs, "h" to listOf(C, d), "o" to listOf(C, c.dim), "mods" to listOf(6, d))
                    val mh = rowMarker(inputs["h"]!!)
                    val mo = rowMarker(inputs["o"]!!)
                    listOf(rowsOf(FloatArray(C) { mh[it] + mo[it] }, d))
                }
                name == g.headVideo || name == g.headAudio -> {
                    val spec = mutableListOf("h" to listOf(C, d), "mods" to listOf(2, d))
                    if (c.pddSteps != null) spec += "weights" to listOf(c.pddSteps!!)
                    expect(inputs, *spec.toTypedArray())
                    listOf(rowsOf(rowMarker(inputs["h"]!!), if (name == g.headVideo) c.videoPatchDim() else c.audioInChannels))
                }
                vv != null && name == vv.decoder.inFile -> {
                    val dec = vv.decoder
                    expect(inputs, "z" to listOf(vae.latentChannels, dec.latentT, dec.latentH, dec.latentW))
                    val r = dec.rows
                    val rh = vae.decoderRotaryDim() / 2
                    listOf(const(1f, r, vae.decoderDim()), const(1f, r, rh), const(0f, r, rh))
                }
                vv != null && name in vv.decoder.blocks -> {
                    val dec = vv.decoder
                    val rh = vae.decoderRotaryDim() / 2
                    expect(inputs, "h" to listOf(dec.rows, vae.decoderDim()), "cos" to listOf(dec.rows, rh), "sin" to listOf(dec.rows, rh))
                    listOf(inputs["h"]!!)
                }
                vv != null && name == vv.decoder.out -> {
                    val dec = vv.decoder
                    expect(inputs, "h" to listOf(dec.rows, vae.decoderDim()))
                    val ratio = vae.spatialRatio()
                    // 0.0 in ImageNet-normalised space → the channel means (0.485, 0.456, 0.406) after denormalisation
                    listOf(const(0f, 3, vae.temporalRatio() * dec.latentT, ratio * dec.latentH, ratio * dec.latentW))
                }
                vv != null && (name == vv.encoderClip || name == vv.encoderImage) -> {
                    val dec = vv.decoder
                    val ratio = vae.spatialRatio()
                    val frames = if (name == vv.encoderImage) 1 else vae.clipLength
                    expect(inputs, "x" to listOf(3, frames, ratio * dec.latentH, ratio * dec.latentW))
                    val t = if (frames == 1) 1 else vae.tokensChunk()
                    // mean 0.5, logvar 0
                    val mom = Tensor.zeros(2 * vae.latentChannels, t, dec.latentH, dec.latentW)
                    java.util.Arrays.fill(mom.data, 0, mom.size / 2, 0.5f)
                    listOf(mom)
                }
                manifest.audio.any { it.decoder == name } -> {
                    val a = manifest.audio.first { it.decoder == name }
                    expect(inputs, "z" to listOf(av.latentChannels, a.latents))
                    listOf(const(0.1f, av.hop() * a.latents))
                }
                manifest.audio.any { it.encoder == name } -> {
                    val a = manifest.audio.first { it.encoder == name }
                    expect(inputs, "wave" to listOf(av.hop() * a.latents))
                    listOf(const(0f, av.latentChannels, a.latents))
                }
                name.startsWith("taeh3_") -> {
                    val e = manifest.taeh3Graphs.first { it.file == name }
                    val tae = manifest.taeh3!!
                    val spec = mutableListOf("z" to listOf(e.frames, tae.latentChannels, e.latentH, e.latentW))
                    for (i in 0 until 3 * tae.memBlocks) {
                        val s = i / tae.memBlocks
                        spec += "mem_$i" to listOf(1, tae.widths[s], e.latentH shl s, e.latentW shl s)
                    }
                    expect(inputs, *spec.toTypedArray())
                    val up = tae.tUpscale()
                    val ph = tae.spatialScale() * e.latentH
                    val pw = tae.spatialScale() * e.latentW
                    val z = inputs["z"]!!
                    val perFrame = tae.latentChannels * e.latentH * e.latentW
                    val rgb = Tensor.zeros(e.frames * up, 3, ph, pw)
                    for (fr in 0 until e.frames) {
                        var zero = true
                        for (i in 0 until perFrame) if (z.data[fr * perFrame + i] != 0f) { zero = false; break }
                        val v = if (zero) 0.25f else 1f
                        java.util.Arrays.fill(rgb.data, fr * up * 3 * ph * pw, (fr + 1) * up * 3 * ph * pw, v)
                    }
                    val mems = (0 until 3 * tae.memBlocks).map { i ->
                        val s = i / tae.memBlocks
                        const(2f, 1, tae.widths[s], e.latentH shl s, e.latentW shl s)
                    }
                    listOf(rgb) + mems
                }
                else -> throw IllegalArgumentException("fake store has no graph `$name`")
            }
        }
    }
}
