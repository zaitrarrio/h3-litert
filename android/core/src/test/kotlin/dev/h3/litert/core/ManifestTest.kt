package dev.h3.litert.core

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** `manifest.json`, `adaln.safetensors` and `expected.safetensors` of the tiny fixture, and the lookups of `runtime.py`. */
class ManifestTest {
    @Test
    fun parsesTheTinyManifest() {
        val m = Fixture.manifest()
        assertEquals("tiny", m.model)
        assertEquals("tiny", m.variant)
        assertEquals(H3Config.tiny(), m.cfg)
        assertEquals(VideoVaeConfig.tiny(), m.vae)
        assertEquals(AudioVaeConfig.tiny(), m.audioVae)
        assertEquals(TaeH3Config.tiny(), m.taeh3)
        assertEquals("f32", m.weights)
        assertEquals(16, m.chunk)
        assertEquals(1, m.headGroup)
        assertEquals("euler", m.sampler.rule)
        assertEquals(SamplerShifts(10.0, 3.0), m.sampler.shifts)
        assertEquals(2, m.sampler.steps.size)
        // the exporter's ladder: rungs 999 / 500 at shifts 10 / 3, in f32
        val v = Schedule.fromRungs(listOf(999, 500), 10.0)
        val a = Schedule.fromRungs(listOf(999, 500), 3.0)
        for (i in 0 until 2) {
            val s = m.sampler.steps[i]
            assertEquals(v.sigmas[i], s.sigmaVideo)
            assertEquals(v.sigmas[i + 1], s.sigmaVideoNext)
            assertEquals(a.sigmas[i], s.sigmaAudio)
            assertEquals(a.sigmas[i + 1], s.sigmaAudioNext)
            assertEquals(v.timestep(i), s.tVideo)
            assertEquals(a.timestep(i), s.tAudio)
        }
        assertEquals(0.999f, m.sampler.condVideoT)
        assertEquals(1.0f, m.sampler.condAudioT)
        assertEquals("adaln.safetensors", m.adaln.file)
        assertEquals(6, m.adaln.timesteps.size)
        assertEquals(listOf(8, 16), m.graphs.textRefine.map { it.rows })
        assertEquals(2, m.graphs.blocks.size)
        assertEquals("block_01_post.tflite", m.graphs.blocks[1].post)
        assertEquals(54368L, m.graphs.blocks[1].bytes)
        assertEquals(listOf(124, 130, 132, 136, 138, 144), m.attention.map { it.tokens }.sorted())
        val vv = assertNotNull(m.videoVae)
        assertEquals(DecoderGraphs(7, 4, 6, 171, "vae_dec_in.tflite", listOf("vae_dec_block_00.tflite", "vae_dec_block_01.tflite"), "vae_dec_out.tflite"), vv.decoder)
        assertEquals(256, vv.tile)
        assertEquals(64, vv.overlap)
        assertEquals("vae_enc_image.tflite", vv.encoderImage)
        assertEquals(listOf(AudioEntry(37, "avae_dec_37.tflite", "avae_enc_37.tflite")), m.audio)
        assertEquals(listOf(TaeH3Entry(5, 4, 6, "taeh3_f5_4x6.tflite")), m.taeh3Graphs)
        assertEquals(listOf(Geometry.TINY), m.sizes)
        assertEquals(26, m.files.size)
        assertEquals(25, m.graphFiles().toSet().size)
        assertEquals(m.files.keys - "adaln.safetensors", m.graphFiles().toSet())
    }

    @Test
    fun refusesOtherFormatsAndBrokenContracts() {
        val text = Fixture.text("manifest.json")
        val e = assertFailsWith<IllegalArgumentException> { Manifest.parse(text.replace("h3-litert/1", "h3-litert/2")) }
        assertTrue("h3-litert/2" in e.message!! && "h3-litert/1" in e.message!!, e.message)
        val missing = assertFailsWith<IllegalArgumentException> { Manifest.parse("""{"model": "x"}""") }
        assertTrue("missing" in missing.message!!, missing.message)
        // a step whose time the AdaLN table does not cover is an error
        val bad = assertFailsWith<IllegalArgumentException> { Manifest.parse(text.replace("\"t_audio\": 0.25", "\"t_audio\": 0.26")) }
        assertTrue("0.26" in bad.message!!, bad.message)
        // head_group must divide heads
        val hg = assertFailsWith<IllegalArgumentException> { Manifest.parse(text.replace("\"head_group\": 1", "\"head_group\": 3")) }
        assertTrue("head_group" in hg.message!!, hg.message)
    }

    @Test
    fun lookupsNameTheMismatch() {
        val m = Fixture.manifest()
        assertEquals("attn_s124_q32_h1.tflite", m.attentionFor(124).file)
        val a = assertFailsWith<IllegalArgumentException> { m.attentionFor(38396) }
        assertTrue("38396" in a.message!! && "124" in a.message!!, a.message)
        assertEquals(8, m.textRefineFor(6).rows)
        assertEquals(8, m.textRefineFor(8).rows)
        assertEquals(16, m.textRefineFor(9).rows)
        val t = assertFailsWith<IllegalArgumentException> { m.textRefineFor(17) }
        assertTrue("17 rows" in t.message!! && "16" in t.message!!, t.message)
        assertEquals("avae_dec_37.tflite", m.audioFor(37).decoder)
        val au = assertFailsWith<IllegalArgumentException> { m.audioFor(207) }
        assertTrue("207" in au.message!! && "[37]" in au.message!!, au.message)
        assertEquals("taeh3_f5_4x6.tflite", m.taeh3For(4, 6).file)
        val tae = assertFailsWith<IllegalArgumentException> { m.taeh3For(48, 84) }
        assertTrue("48x84" in tae.message!! && "4x6" in tae.message!!, tae.message)
    }

    @Test
    fun adalnTablesDecodeRows() {
        val m = Fixture.manifest()
        val t = Fixture.adaln()
        assertEquals(6, t.entries)
        assertEquals(2, t.layers)
        assertEquals(32, t.hidden)
        t.validate(m.cfg, m.adaln.timesteps)
        for ((i, ts) in m.adaln.timesteps.withIndex()) assertEquals(i, t.indexOf(ts))
        assertEquals(4, t.indexOf(0.999f))
        val e = assertFailsWith<IllegalArgumentException> { t.indexOf(0.5f) }
        assertTrue("0.5" in e.message!!, e.message)
        val mods = t.mods(2, 1, AUDIO_TAG)
        assertEquals(listOf(6, 32), mods.shape.toList())
        assertTrue(mods.data.all { it.isFinite() })
        // the same row through the generic reader
        val st = SafeTensors.read(Fixture.bytes("adaln.safetensors"))
        val all = st.tensor("blocks")
        assertEquals(listOf(6, 2, 3, 6, 32), all.shape.toList())
        val off = ((2 * 2 + 1) * 3 + AUDIO_TAG) * 6 * 32
        assertTrue(mods.data.contentEquals(all.data.copyOfRange(off, off + 6 * 32)))
        val head = t.headMods(5)
        assertEquals(listOf(2, 32), head.shape.toList())
        assertTrue(head.data.contentEquals(st.tensor("head").data.copyOfRange(5 * 64, 6 * 64)))
        val wrong = assertFailsWith<IllegalArgumentException> { t.validate(H3Config.h3(), m.adaln.timesteps) }
        assertTrue("50" in wrong.message!!, wrong.message)
    }

    @Test
    fun expectedFixtureShapes() {
        val e = Fixture.expected()
        assertEquals(listOf(6, 24), e.shape("prompt_embeds").toList())
        assertEquals(listOf(1f, 1f, 0f, 1f, 1f, 1f), e.tensor("text_tags").data.toList())
        assertEquals(listOf(4, 7, 4, 6), e.shape("latent").toList())
        assertEquals(listOf(74, 6), e.shape("audio_rows").toList())
        assertEquals(listOf(22, 64, 96, 3), e.shape("frames").toList())
        assertEquals("U8", e.entry("frames").dtype)
        assertEquals(22 * 64 * 96 * 3, e.bytes("frames").size)
        assertEquals(listOf(2, 370), e.shape("audio").toList())
        assertEquals("7", e.metadata["seed"])
        assertEquals("96x64x22", e.metadata["geometry"])
        // the prompt is Rng(1234)'s first 6·24 normals
        assertTrue(e.tensor("prompt_embeds").data.contentEquals(Rng(1234L).normal(6 * 24)))
        val r = assertFailsWith<IllegalArgumentException> { e.tensor("frames") }
        assertTrue("U8" in r.message!!, r.message)
    }

    @Test
    fun planOfTheTinyClip() {
        val m = Fixture.manifest()
        val p = m.plan(Geometry.TINY, 6, 0, 64L shl 20)
        assertEquals(124, p.rows)
        assertEquals(16, p.chunk)
        assertEquals(32, p.qChunk)
        assertEquals(1, p.headGroup)
        assertTrue(p.fits, p.toString())
        assertTrue(p.weightsTotal > 0 && p.weightsResident <= p.weightsTotal)
        assertEquals(130, m.plan(Geometry.TINY, 6, 1, 64L shl 20).rows)
        assertEquals(136, m.plan(Geometry.TINY, 6, 2, 64L shl 20).rows)
        assertEquals(132, m.plan(Geometry.TINY, 12, 0, 64L shl 20).rows)
        val preview = m.plan(Geometry.TINY, 6, 0, 64L shl 20, preview = true)
        assertTrue(preview.fits)
        // a budget below the weights names the tier rule
        val tiny = m.plan(Geometry.TINY, 6, 0, 1024)
        assertTrue(!tiny.fits && tiny.notes.any { "a phone cannot hold this model" in it }, tiny.toString())
        assertTrue("fits: true" in p.toString())
    }

    @Test
    fun pathNeedsADirectory() {
        val m = Fixture.manifest()
        assertFailsWith<IllegalStateException> { m.path("head_video.tflite") }
        val withDir = Fixture.manifest(java.io.File("/graphs"))
        assertEquals(java.io.File("/graphs/head_video.tflite"), withDir.path("head_video.tflite"))
    }

    @Test
    fun blockGraphsAreLruResident() {
        val store = FakeGraphStore(Fixture.manifest(), resident = 2)
        store.block("block_00_qkv.tflite")
        store.block("block_00_post.tflite")
        store.block("block_00_qkv.tflite") // touch: now most recent
        store.block("block_01_qkv.tflite") // evicts block_00_post
        assertEquals(listOf("block_00_post.tflite"), store.closed)
        assertEquals(listOf("block_00_qkv.tflite", "block_01_qkv.tflite"), store.residentBlocks())
        store.fixed("head_video.tflite")
        store.close()
        assertEquals(setOf("block_00_qkv.tflite", "block_00_post.tflite", "block_01_qkv.tflite", "head_video.tflite"), store.closed.toSet())
    }

    @Test
    fun scheduleOfTheManifestMatchesTheTablesTimes() {
        val m = Fixture.manifest()
        val t = Fixture.adaln()
        for (s in m.sampler.steps) {
            for (x in listOf(s.tVideo, s.tAudio, maxOf(s.tVideo, m.sampler.condVideoT), m.sampler.condAudioT)) {
                assertTrue(abs(t.timesteps[t.indexOf(x)] - x) <= 1e-6f)
            }
        }
    }
}
