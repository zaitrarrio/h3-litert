package dev.h3.litert.core

/**
 * Memory plan (`docs/design.md` "Tiers"): what a generation holds at once, before anything is
 * allocated. Hosts show it and refuse what will not fit. It is a model, not a measurement: the
 * LiteRT runtime's own scratch (the ViT decoder's `R × R` attention scores, kernel workspaces) is
 * estimated, not read.
 */
public data class Plan(
    /** Packed rows after the text padding: the attention graph's `S`. */
    val rows: Int,
    val chunk: Int,
    val qChunk: Int,
    val headGroup: Int,
    val weightsTotal: Long,
    val weightsResident: Long,
    val adalnBytes: Long,
    val activations: Long,
    val attentionScratch: Long,
    val decodePeak: Long,
    val peak: Long,
    val budget: Long,
    val fits: Boolean,
    val notes: List<String> = emptyList(),
) {
    override fun toString(): String {
        fun mib(b: Long): String = String.format(java.util.Locale.ROOT, "%9.1f MiB", b / (1 shl 20).toDouble())
        val lines = mutableListOf(
            "rows              $rows",
            "weights (all)     ${mib(weightsTotal)}",
            "weights resident  ${mib(weightsResident)}",
            "AdaLN tables      ${mib(adalnBytes)}",
            "DiT activations   ${mib(activations)} (row chunk $chunk)",
            "attention scratch ${mib(attentionScratch)} (query chunk $qChunk, $headGroup heads)",
            "decode peak       ${mib(decodePeak)}",
            "peak              ${mib(peak)} of ${String.format(java.util.Locale.ROOT, "%.0f", budget / (1 shl 20).toDouble())} MiB  fits: $fits",
        )
        lines += notes.map { "note: $it" }
        return lines.joinToString("\n")
    }

    public companion object {
        public fun compute(
            m: Manifest,
            g: Geometry,
            textRows: Int,
            keyframes: Int,
            budgetBytes: Long,
            residentBlocks: Int = 2,
            preview: Boolean = false,
        ): Plan {
            val c = m.cfg
            val ratio = m.vae.spatialRatio()
            g.validate(ratio)
            val lat = g.latent(ratio)
            val na = g.audioLatents()
            val bucket = m.textRefineFor(textRows)
            val anchors = listOf("first", "last").take(keyframes)
            val s = Packing.buildLayout(List(bucket.rows) { TEXT_TAG }, lat.t, lat.h, lat.w, na, c.patch, anchors).sequenceLength
            val attn = m.attentionFor(s)
            val a = 4L
            val d = c.hidden.toLong()
            val inner = c.dim.toLong()
            val sl = s.toLong()

            val blockFiles = m.graphs.blocks.flatMap { listOf(it.qkv, it.post) }.toSet()
            val blockBytes = m.graphs.blocks.maxOfOrNull { b ->
                if (b.bytes > 0) b.bytes else (m.files[b.qkv]?.bytes ?: 0L) + (m.files[b.post]?.bytes ?: 0L)
            } ?: 0L
            val fixedBytes = m.files.filterKeys { it !in blockFiles && it != m.adaln.file }.values.sumOf { it.bytes }
            val weightsTotal = m.graphs.blocks.sumOf { b -> if (b.bytes > 0) b.bytes else (m.files[b.qkv]?.bytes ?: 0L) + (m.files[b.post]?.bytes ?: 0L) } + fixedBytes
            val resident = minOf(maxOf(residentBlocks, 1), c.layers)
            val weightsResident = resident * blockBytes + fixedBytes
            // The f16 table as shipped, plus the f32 rows one step decodes (≤ 4 unique times × 3 tags).
            val adalnBytes = (m.files[m.adaln.file]?.bytes ?: (m.adaln.timesteps.size.toLong() * c.layers * MODALITIES * MOD_SLOTS * d * 2)) +
                4L * c.layers * MODALITIES * MOD_SLOTS * d * a
            // h [S, d]; q, k, v, o [S, I]; the text bucket; rope cos/sin; the mask; the modality row buffers.
            val activations = (sl * d + 4 * sl * inner + bucket.rows.toLong() * d + 2 * sl * 3 * c.ropeFreqDim + sl +
                (sl * c.videoPatchDim() + sl * c.audioInChannels)) * a
            // One head group's k/v copies, the score chunk and its output.
            val attentionScratch = (2 * sl * attn.heads * c.headDim + attn.heads.toLong() * attn.qChunk * sl + attn.qChunk.toLong() * attn.heads * c.headDim) * a

            val notes = mutableListOf<String>()
            val decodePeak: Long = if (preview) {
                val tae = m.taeh3 ?: throw IllegalArgumentException("this export has no TAEH3 config for a preview plan")
                val entry = m.taeh3For(lat.h, lat.w)
                val f = minOf(entry.frames, lat.t).toLong()
                val px = lat.h.toLong() * lat.w
                val nf = tae.widths
                val s0 = f * nf[0] * px
                val s1 = f * tae.tgrow[0] * nf[1] * 4 * px
                val s2 = f * tae.tgrow[0] * tae.tgrow[1] * nf[2] * 16 * px
                val out = f * tae.tUpscale() * nf[3] * 64 * px
                maxOf(5 * s0, 5 * s1, 5 * s2, s2 + 2 * out) * a + rgbaBytes(g)
            } else {
                val vv = m.videoVae ?: throw IllegalArgumentException("this export has no video VAE decoder")
                val r = vv.decoder.rows.toLong()
                val dd = m.vae.decoderDim().toLong()
                val heads = m.vae.decoderHeads.toLong()
                val ffn = dd * m.vae.decoderFfnMult
                // h, q, k, v, o rows plus the dense score matrix or the FFN hidden, whichever is larger.
                val blockScratch = (5 * r * dd + maxOf(heads * r * r, 2 * r * ffn)) * a
                val tileFrames = 4L * vv.decoder.latentT
                val tilePx = (vv.decoder.latentH.toLong() * ratio) * (vv.decoder.latentW.toLong() * ratio)
                val tileOut = 3 * tileFrames * tilePx * a
                // The stitched chunk (28 frames at the canvas), the previous overlap and the finished RGBA frames.
                val canvas = g.width.toLong() * g.height
                val chunkOut = 3 * tileFrames * canvas * a
                blockScratch + tileOut + 2 * chunkOut + rgbaBytes(g)
            }
            val peak = weightsResident + adalnBytes + maxOf(activations + attentionScratch, decodePeak)
            if (resident < c.layers) notes += "$resident of ${c.layers} block graphs resident; the rest stream from storage"
            if (sl % m.chunk != 0L) notes += "$s rows run in chunks of ${m.chunk} per modulation run; the last chunk of each run is padded"
            if (weightsTotal > budgetBytes) {
                notes += "the ${m.weights} weights alone (${weightsTotal / (1 shl 20)} MiB) exceed the ${budgetBytes / (1 shl 20)} MiB budget: " +
                    "a phone cannot hold this model, only the tiny self-test export runs (docs/design.md \"Tiers\")"
            }
            return Plan(
                s, m.chunk, attn.qChunk, attn.heads, weightsTotal, weightsResident, adalnBytes, activations, attentionScratch,
                decodePeak, peak, budgetBytes, peak <= budgetBytes, notes,
            )
        }

        private fun rgbaBytes(g: Geometry): Long = g.frames.toLong() * g.width * g.height * 4
    }
}
