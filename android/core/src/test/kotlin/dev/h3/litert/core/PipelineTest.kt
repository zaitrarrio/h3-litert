package dev.h3.litert.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The host loop of `docs/graphs.md` over the tiny manifest with a fake GraphStore: the tiny T2VA
 * request (6 prompt rows tagged `[1, 1, 0, 1, 1, 1]` → text bucket 8, pad 2; S = 124; C = 16;
 * Cq = 32; head groups of 1 of 2 heads), the call sequence per block, the modulation runs, the
 * attention mask, the decoders' chunking and the event stream.
 */
class PipelineTest {
    private val manifest = Fixture.manifest()

    private fun store() = FakeGraphStore(manifest)

    @Test
    fun tinyRequestShapes() {
        val req = Fixture.request()
        assertEquals(listOf(6, 24), req.embeds.shape.toList())
        assertEquals(listOf(1, 1, 0, 1, 1, 1), req.textTags)
        assertEquals(Dims3(7, 4, 6), req.geometry.latent(16))
        assertEquals(37, req.geometry.audioLatents())
        val l = Packing.buildLayout(req.textTags, 7, 4, 6, 37, listOf(1, 2, 2), emptyList())
        assertEquals(122, l.sequenceLength)
        assertEquals(6, l.audioIndices[0])
        assertEquals(80, l.videoIndices[0])
    }

    @Test
    fun prepareRefinesPaddedTextAndShiftsTheLayout() {
        val s = store()
        val pipe = Fixture.pipeline(s)
        val p = pipe.prepare(Fixture.request())
        assertEquals(listOf("text_refine_8.tflite"), s.names())
        val call = s.calls[0]
        assertEquals(listOf(8, 24), call.inputs["embeds"])
        assertEquals(listOf(8), call.inputs["mask"])
        assertEquals(listOf(0f, 0f, 0f, 0f, 0f, 0f, -1e9f, -1e9f), call.mask!!.toList())
        assertEquals(6, p.nText)
        assertEquals(2, p.pad)
        assertEquals(124, p.rows)
        assertEquals(listOf(8, 32), p.text.shape.toList())
        assertEquals(8, p.audioIdx[0])
        assertEquals(82, p.videoIdx[0])
        assertEquals(74, p.audioIdx.size)
        assertEquals(42, p.videoIdx.size)
        // the mask marks the two pad rows, nothing else
        assertEquals(listOf(6, 7), p.mask.data.indices.filter { p.mask.data[it] != 0f })
        assertEquals(-1e9f, p.mask.data[6])
        // tags after the shift: text, text, vision, text, text, text, pad(text), pad(text), audio…, video…
        assertEquals(listOf(1, 1, 0, 1, 1, 1, 1, 1, 2), p.tags.toList().take(9))
        assertEquals(VIDEO_TAG, p.tags[82])
        // rope rows follow the shifted positions: pad rows sit at the origin (cos 1), audio starts at t = 6
        assertEquals(listOf(124, 6), p.rope.cos.shape.toList())
        for (j in 0 until 6) assertEquals(1f, p.rope.cos.data[6 * 6 + j])
        assertEquals(kotlin.math.cos(6.0).toFloat(), p.rope.cos.data[8 * 6 + 0])
        // noise order: the video latent [4, 7, 4, 6] first, then the audio [74, 6]
        val rng = Rng(7L)
        val video = rng.normal(4, 7, 4, 6)
        assertTrue(p.videoRows.data.contentEquals(Packing.patchRows(video, listOf(1, 2, 2)).data))
        assertTrue(p.audioRows.data.contentEquals(rng.normal(74 * 6)))
        assertEquals(listOf(42, 16), p.videoRows.shape.toList())
        assertEquals(listOf(74, 6), p.audioRows.shape.toList())
    }

    @Test
    fun forwardWalksRunsHeadGroupsAndQueryChunks() {
        val s = store()
        val pipe = Fixture.pipeline(s)
        val p = pipe.prepare(Fixture.request())
        s.calls.clear()
        val (vv, va) = pipe.forward(p, manifest.sampler.steps[0], manifest.attentionFor(124))
        assertEquals(listOf(42, 16), vv.shape.toList())
        assertEquals(listOf(74, 6), va.shape.toList())
        assertTrue(vv.data.all { it == 1f } && va.data.all { it == 1f }, "padded rows leaked into the heads' output")

        // modulation runs of the tiny T2VA layout: text(2) | vision(1) | text + pad(5) | audio(74) | video(42)
        // → per-row chunks of 16: 1 + 1 + 1 + 5 + 3 = 11 qkv / post calls per block
        for (b in listOf("00", "01")) {
            assertEquals(11, s.count("block_${b}_qkv.tflite"), "qkv calls of block $b")
            assertEquals(11, s.count("block_${b}_post.tflite"), "post calls of block $b")
        }
        // 2 head groups × 4 query chunks (124 rows in chunks of 32) per block
        assertEquals(16, s.count("attn_s124_q32_h1.tflite"))
        assertEquals(3, s.count("embed_video.tflite"))
        assertEquals(5, s.count("embed_audio.tflite"))
        assertEquals(3, s.count("head_video.tflite"))
        assertEquals(5, s.count("head_audio.tflite"))

        val expected = List(3) { "embed_video.tflite" } + List(5) { "embed_audio.tflite" } +
            listOf("00", "01").flatMap { b ->
                List(11) { "block_${b}_qkv.tflite" } + List(8) { "attn_s124_q32_h1.tflite" } + List(11) { "block_${b}_post.tflite" }
            } + List(3) { "head_video.tflite" } + List(5) { "head_audio.tflite" }
        assertEquals(expected, s.names())

        // every per-row call sees C rows; the modulation table is [6, d]; rope rows are [C, r/2]
        for (c in s.calls("block_00_qkv.tflite")) {
            assertEquals(listOf(16, 32), c.inputs["h"])
            assertEquals(listOf(6, 32), c.inputs["mods"])
            assertEquals(listOf(16, 6), c.inputs["cos"])
            assertEquals(listOf(16, 6), c.inputs["sin"])
        }
        // the modulation entry changes between runs: text / vision / text / audio / video rows of block 0
        val mods = s.calls("block_00_qkv.tflite").map { it.modsFirst!! }
        val text = Fixture.adaln().mods(Fixture.adaln().indexOf(manifest.sampler.steps[0].tVideo), 0, TEXT_TAG).data[0]
        val vision = Fixture.adaln().mods(Fixture.adaln().indexOf(manifest.sampler.steps[0].tVideo), 0, VIDEO_TAG).data[0]
        val audio = Fixture.adaln().mods(Fixture.adaln().indexOf(manifest.sampler.steps[0].tAudio), 0, AUDIO_TAG).data[0]
        assertEquals(listOf(text, vision, text) + List(5) { audio } + List(3) { vision }, mods)
        // attention: q chunks of 32, keys and values at the full S with the mask of length 124, −1e9 at rows 6..7
        for (c in s.calls("attn_s124_q32_h1.tflite")) {
            assertEquals(listOf(32, 1, 16), c.inputs["q"])
            assertEquals(listOf(124, 1, 16), c.inputs["k"])
            assertEquals(listOf(124, 1, 16), c.inputs["v"])
            assertEquals(listOf(124), c.inputs["mask"])
            val m = c.mask!!
            assertEquals(listOf(6, 7), m.indices.filter { m[it] != 0f })
            assertEquals(-1e9f, m[6])
        }
        for (c in s.calls("block_01_post.tflite")) {
            assertEquals(listOf(16, 32), c.inputs["o"])
            assertEquals(listOf(6, 32), c.inputs["mods"])
        }
        for (c in s.calls("head_audio.tflite")) assertEquals(listOf(2, 32), c.inputs["mods"])
        val headMods = Fixture.adaln().headMods(Fixture.adaln().indexOf(manifest.sampler.steps[0].tAudio)).data[0]
        assertTrue(s.calls("head_audio.tflite").all { it.modsFirst == headMods })
    }

    @Test
    fun denoiseRunsTwoStepsAndDrawsNoNoiseForEuler() {
        val s = store()
        val pipe = Fixture.pipeline(s)
        val events = mutableListOf<Event>()
        val tm = Timings()
        val den = pipe.denoise(Fixture.request(), { events += it }, tm)
        assertEquals(listOf(4, 7, 4, 6), den.latent.shape.toList())
        assertEquals(listOf(74, 6), den.audioRows.shape.toList())
        assertTrue(den.latent.data.all { it.isFinite() } && den.audioRows.data.all { it.isFinite() })
        val prepared = events[0] as Event.Prepared
        assertEquals(122, prepared.tokens)
        assertEquals(124, prepared.rows)
        val steps = events.filterIsInstance<Event.Step>()
        assertEquals(listOf(0, 1), steps.map { it.index })
        assertEquals(listOf(2, 2), steps.map { it.total })
        assertEquals(2, tm.stepsS.size)
        assertEquals(1, s.count("text_refine_8.tflite"))
        assertEquals(2 * 11, s.count("block_00_qkv.tflite"))
        assertEquals(2 * 16, s.count("attn_s124_q32_h1.tflite"))
        assertEquals(2 * 3, s.count("head_video.tflite"))

        // With every velocity 1.0 the latent is a pure function of the noise and the Euler ladder.
        val rng = Rng(7L)
        var x = Packing.patchRows(rng.normal(4, 7, 4, 6), listOf(1, 2, 2))
        var a = rng.normal(74, 6)
        val ones = { t: Tensor -> Tensor(FloatArray(t.size) { 1f }, t.shape) }
        for (st in manifest.sampler.steps) {
            x = Sampler.flowStep(x, ones(x), st.sigmaVideo, st.sigmaVideoNext, "euler", null)
            a = Sampler.flowStep(a, ones(a), st.sigmaAudio, st.sigmaAudioNext, "euler", null)
        }
        assertTrue(den.latent.data.contentEquals(Packing.unpatchRows(x, 4, 7, 4, 6, listOf(1, 2, 2)).data))
        assertTrue(den.audioRows.data.contentEquals(a.data))
        // deterministic per seed, different across seeds
        assertTrue(den.latent.data.contentEquals(Fixture.pipeline(store()).denoise(Fixture.request()).latent.data))
        assertTrue(!den.latent.data.contentEquals(Fixture.pipeline(store()).denoise(Fixture.request(seed = 8)).latent.data))
    }

    @Test
    fun vaeDecodeRunsOneChunkOfOneTile() {
        val s = store()
        val pipe = Fixture.pipeline(s)
        val events = mutableListOf<Event>()
        val latent = Fixture.expected().tensor("latent")
        val video = pipe.decodeVideo(latent, false, { events += it })
        assertEquals(1, s.count("vae_dec_in.tflite"))
        assertEquals(listOf(4, 7, 4, 6), s.calls("vae_dec_in.tflite")[0].inputs["z"])
        assertEquals(1, s.count("vae_dec_block_00.tflite"))
        assertEquals(1, s.count("vae_dec_block_01.tflite"))
        assertEquals(listOf(171, 32), s.calls("vae_dec_block_00.tflite")[0].inputs["h"])
        assertEquals(listOf(171, 6), s.calls("vae_dec_block_00.tflite")[0].inputs["cos"])
        assertEquals(1, s.count("vae_dec_out.tflite"))
        assertEquals(
            listOf("vae_dec_in.tflite", "vae_dec_block_00.tflite", "vae_dec_block_01.tflite", "vae_dec_out.tflite"),
            s.names(),
        )
        val frames = events.filterIsInstance<Event.Frames>()
        assertEquals(listOf(0 to 17, 17 to 5), frames.map { it.first to it.count })
        assertEquals(22, video.frames)
        assertEquals(96, video.width)
        assertEquals(64, video.height)
        assertEquals(22 * 64 * 96 * 4, video.rgba.size)
        // the fake decoder emits 0 in normalised space → the ImageNet means: (124, 116, 104, 255)
        assertEquals(listOf(124, 116, 104, 255), video.frame(21).toList().take(4).map { it.toInt() and 0xFF })
        assertEquals(listOf(22, 64, 96, 3), Fixture.expected().shape("frames").toList())
    }

    @Test
    fun audioDecodesEachChannel() {
        val s = store()
        val pipe = Fixture.pipeline(s)
        val rows = Fixture.expected().tensor("audio_rows")
        val tm = Timings()
        val audio = pipe.decodeAudio(rows, 37, tm)
        assertEquals(2, s.count("avae_dec_37.tflite"))
        assertTrue(s.calls("avae_dec_37.tflite").all { it.inputs["z"] == listOf(6, 37) })
        assertEquals(2, audio.size)
        assertEquals(370, audio[0].size)
        assertEquals(370, audio[1].size)
        assertTrue(audio[0].all { it == 0.1f })
        assertEquals(listOf(2, 370), Fixture.expected().shape("audio").toList())
        val bad = assertFailsWith<IllegalArgumentException> { pipe.decodeAudio(rows, 36) }
        assertTrue("36" in bad.message!!, bad.message)
    }

    @Test
    fun taeh3PreviewKeepsFramesByTheModRule() {
        val s = store()
        val pipe = Fixture.pipeline(s)
        val events = mutableListOf<Event>()
        val video = pipe.decodeVideo(Fixture.expected().tensor("latent"), true, { events += it })
        assertEquals(2, s.count("taeh3_f5_4x6.tflite"))
        val calls = s.calls("taeh3_f5_4x6.tflite")
        assertEquals(listOf(5, 4, 4, 6), calls[0].inputs["z"])
        assertEquals(listOf(1, 8, 4, 6), calls[0].inputs["mem_0"])
        assertEquals(listOf(1, 6, 8, 12), calls[0].inputs["mem_3"])
        assertEquals(listOf(1, 4, 16, 24), calls[0].inputs["mem_8"])
        // raw frames 0..27: kept 3..19 (17) from the first call, 23..27 (5) from the second (2 real + 3 padded latents)
        val frames = events.filterIsInstance<Event.Frames>()
        assertEquals(listOf(0 to 17, 17 to 5), frames.map { it.first to it.count })
        assertEquals(22, video.frames)
        assertEquals(96, video.width)
        assertEquals(64, video.height)
        // real frames decode to 1.0 → 255; the padded latent frames' 0.25 output (64) was dropped by the rule
        assertTrue(video.rgba.all { (it.toInt() and 0xFF) == 255 })
    }

    @Test
    fun generateEndToEndEmitsAudioLast() {
        val s = store()
        val pipe = Fixture.pipeline(s)
        val events = mutableListOf<Event>()
        val g = pipe.generate(Fixture.request()) { events += it }
        val kinds = events.map { it::class.simpleName }
        assertEquals(listOf("Prepared", "Step", "Step", "Frames", "Frames", "Audio"), kinds)
        assertEquals(22, g.video.frames)
        assertEquals(320, g.sampleRate)
        assertEquals(370, g.audio[0].size)
        assertEquals(2, g.timings.stepsS.size)
        assertTrue(g.timings.decodeS >= 0 && g.timings.audioS >= 0)
        val audio = events.last() as Event.Audio
        assertEquals(370, audio.samples)
        assertEquals(320, audio.sampleRate)
    }

    @Test
    fun fl2vaEncodesTheKeyframeAndAddsAConditionRun() {
        val s = store()
        val pipe = Fixture.pipeline(s)
        val px = Tensor(Rng(99L).normal(3 * 64 * 96).map { (it * 0.2f + 0.5f).coerceIn(0f, 1f) }.toFloatArray(), intArrayOf(3, 1, 64, 96))
        val req = Fixture.request(keyframes = listOf(Keyframe("first", px)))
        val events = mutableListOf<Event>()
        val den = pipe.denoise(req, { events += it })
        assertEquals(1, s.count("vae_enc_image.tflite"))
        assertEquals(listOf(3, 1, 64, 96), s.calls("vae_enc_image.tflite")[0].inputs["x"])
        assertEquals(0, s.count("vae_enc_clip.tflite"))
        // the layout: text 6 | pad 2 | cond 6 | audio 74 | video 42 → S = 130
        assertEquals(130, (events[0] as Event.Prepared).rows)
        assertEquals(128, (events[0] as Event.Prepared).tokens)
        assertEquals(2 * 2 * 2 * 5, s.count("attn_s130_q32_h1.tflite")) // 2 steps × 2 blocks × 2 groups × 5 query chunks of 32
        // mod runs: text(2) | vision(1) | text+pad(5) | cond video at 0.999 (6) | audio(74) | video(42) → 12 per block per step
        assertEquals(2 * 12, s.count("block_00_qkv.tflite"))
        // heads: the condition run (one call at the 0.999 entry) then the 42 target rows (3 calls), per step
        assertEquals(2 * 4, s.count("head_video.tflite"))
        assertEquals(2 * 5, s.count("head_audio.tflite"))
        assertEquals(listOf(4, 7, 4, 6), den.latent.shape.toList())
        assertTrue(den.latent.data.all { it.isFinite() })

        // The condition rows are the keyframe's rows: posterior (seed 42) → f16 → normalised → patch rows, then
        // 0.999·rows + 0.001·noise, where that noise is drawn before the video latent noise.
        val p = pipe.prepare(req, KeyframeEncoder(s))
        assertEquals(6, p.layout.numConditionVideoRows)
        assertEquals(listOf(48, 16), p.videoRows.shape.toList())
        val moments = Tensor.zeros(8, 1, 4, 6).also { java.util.Arrays.fill(it.data, 0, it.size / 2, 0.5f) }
        val z = VaeHost.posteriorSample(moments, Tensor(Rng(42L).normal(4 * 24), intArrayOf(4, 1, 4, 6)))
        val rows = Packing.patchRows(VaeHost.normalizeLatents(manifest.vae, Host.roundThroughHalf(z)), listOf(1, 2, 2))
        val rng = Rng(7L)
        val aug = rng.normal(6 * 16)
        for (i in 0 until 6 * 16) assertEquals(0.999f * rows.data[i] + (1f - 0.999f) * aug[i], p.videoRows.data[i])
        val video = rng.normal(4, 7, 4, 6)
        assertTrue(p.videoRows.sliceRows(6, 48).data.contentEquals(Packing.patchRows(video, listOf(1, 2, 2)).data))
        // condition rows are never written by the step
        val after = Fixture.pipeline(store()).denoise(req)
        assertEquals(listOf(4, 7, 4, 6), after.latent.shape.toList())
        // a second keyframe at "last" → S = 136
        val two = Fixture.request(keyframes = listOf(Keyframe("first", px), Keyframe("last", px)))
        val ev2 = mutableListOf<Event>()
        Fixture.pipeline(store()).denoise(two, { ev2 += it })
        assertEquals(136, (ev2[0] as Event.Prepared).rows)
    }

    @Test
    fun refusesWrongPromptGeometryAndMissingEncoder() {
        val pipe = Fixture.pipeline(store())
        val e = Fixture.expected()
        val bad = assertFailsWith<IllegalArgumentException> {
            pipe.prepare(Request(Tensor.zeros(6, 23), List(6) { TEXT_TAG }, Geometry.TINY))
        }
        assertTrue("[6, 23]" in bad.message!! && "[6, 24]" in bad.message!!, bad.message)
        val tags = assertFailsWith<IllegalArgumentException> {
            pipe.prepare(Request(e.tensor("prompt_embeds"), listOf(1, 1, 2, 1, 1, 1), Geometry.TINY))
        }
        assertTrue("text or vision" in tags.message!!, tags.message)
        val frames = assertFailsWith<IllegalArgumentException> { pipe.denoise(Fixture.request().let { Request(it.embeds, it.textTags, Geometry(96, 64, 21)) }) }
        assertTrue("17n + 5" in frames.message!!, frames.message)
        val size = assertFailsWith<IllegalArgumentException> { pipe.denoise(Fixture.request().let { Request(it.embeds, it.textTags, Geometry(160, 64, 22)) }) }
        assertTrue("no attention graph for 152 rows" in size.message!!, size.message)
        val long = assertFailsWith<IllegalArgumentException> {
            pipe.prepare(Request(Tensor.zeros(17, 24), List(17) { TEXT_TAG }, Geometry.TINY))
        }
        assertTrue("17 rows" in long.message!!, long.message)
        val noEnc = assertFailsWith<IllegalArgumentException> {
            pipe.prepare(Fixture.request(keyframes = listOf(Keyframe("first", Tensor.zeros(3, 1, 64, 96)))), null)
        }
        assertTrue("ClipEncoder" in noEnc.message!!, noEnc.message)
    }

    @Test
    fun runRowsDropsPaddedRowsAndNeverStraddles() {
        val s = store()
        val pipe = Fixture.pipeline(s)
        val rows = Tensor(FloatArray(42 * 16) { 1f }, intArrayOf(42, 16))
        val out = pipe.runRows(s.fixed("embed_video.tflite"), 16, 42, mapOf("rows" to rows), emptyMap())
        assertEquals(1, out.size)
        assertEquals(listOf(42, 32), out[0].shape.toList())
        assertTrue(out[0].data.all { it == 1f })
        assertEquals(3, s.count("embed_video.tflite"))
        assertTrue(s.calls("embed_video.tflite").all { it.inputs["rows"] == listOf(16, 16) })
    }
}
