package dev.h3.litert.core

import kotlin.math.abs
import kotlin.math.cos
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** `python/tests/test_host.py`: the host math against the values h3-rs's tests pin down. */
class HostMathTest {
    private fun bits(x: Float): Int = x.toRawBits()

    @Test
    fun v2Ladder() {
        val s = Schedule.fromRungs(Schedules.FASTH3_V2_RUNGS, 10.0)
        assertEquals(8, s.steps)
        assertTrue(abs(s.sigmas[0] - Schedules.shiftSigma(0.999f, 10.0f)) < 1e-7)
        assertTrue(abs(s.sigmas[4] - 10.0 * 0.5 / 5.5) < 1e-6)
        assertEquals(0.0f, s.sigmas.last())
        assertTrue(abs(s.timestep(4) - (1.0 - 10.0 * 0.5 / 5.5)) < 1e-6)
        val a = Schedule.fromRungs(Schedules.FASTH3_V2_RUNGS, 3.0)
        assertTrue(a.sigmas[4] > 0.74 && a.sigmas[4] < 0.76)
    }

    @Test
    fun uniformAndPika() {
        val s = Schedule.uniform(50, 12.0)
        assertEquals(49, s.steps)
        assertEquals(1.0f, s.sigmas[0])
        val p = Schedule.fromRungs(Schedules.PIKA_KNOTS, 12.0, "renoise")
        assertEquals(6, p.steps)
        assertEquals(1.0f, p.sigmas[0])
        assertTrue(abs(p.sigmas[5] - Schedules.shiftSigma(0.04f, 12.0f)) < 1e-7)
    }

    @Test
    fun plug4GridMatchesDiffusersBits() {
        val want = mapOf(
            12.0 to listOf(0x3F800000, 0x3F7914C2, 0x3F6C4EC5, 0x3F4CCCCD, 0),
            3.0 to listOf(0x3F800000, 0x3F666666, 0x3F400000, 0x3F000000, 0),
        )
        for ((shift, b) in want) {
            val s = Schedule.fromRungs(Schedules.PLUG4_KNOTS, shift, "renoise")
            assertEquals(b, s.sigmas.map { bits(it) }, "shift $shift")
            assertEquals(s.sigmas, Schedule.uniform(5, shift).sigmas)
            for (i in 0 until 4) assertEquals(s.sigmas[i], 1.0f - s.timestep(i))
        }
    }

    @Test
    fun pddOmniPlan() {
        val p = PddPlan.of(listOf(0, 4, 8, 12, 16, 20, 24, 28, 32), Shifts.H3)
        assertEquals(32, p.fineGrid.size - 1)
        assertEquals(8, p.numSteps())
        assertEquals(0 to 4, p.block(0))
        assertTrue(abs(p.videoSigmas[0] - 0.999) < 1e-6)
        assertEquals(0.0f, p.videoSigmas.last())
        assertTrue(p.videoWeights.all { it < 0 })
        assertTrue(abs(PddPlan.fusedWeights(p.videoWeights, 0, 4).sum() - 1.0) < 1e-6)
        assertTrue(abs(p.videoWeights.subList(4, 8).sum() - (p.videoSigmas[2] - p.videoSigmas[1])) < 1e-6)
        val (v, a) = p.schedules()
        assertEquals(8, v.steps)
        assertEquals(8, a.steps)
    }

    @Test
    fun frameAndLatentCounts() {
        assertEquals(listOf(124, 243, 362), listOf(120, 240, 360).map { Packing.alignNumFrames(it) })
        assertEquals(37 to 107, Packing.videoLatentFrames(124) to Packing.videoLatentFrames(362))
        assertEquals(207 to 405, Packing.audioLatents(124) to Packing.audioLatents(243))
        assertEquals(124, Packing.resolveNumFrames(5 * 24))
        assertFailsWith<IllegalArgumentException> { Packing.videoLatentFrames(120) }
        assertFailsWith<IllegalArgumentException> { Packing.resolveNumFrames(1000) }
        assertEquals(Dims3(7, 4, 6), Geometry.TINY.latent(16))
        assertEquals(37, Geometry.TINY.audioLatents())
        assertEquals(Geometry(832, 480, 124), Geometry.parse("832x480x124"))
    }

    @Test
    fun canvasRules() {
        assertEquals(768 to 1344, Packing.canvasSize(16.0, 9.0))
        assertEquals(1344 to 768, Packing.canvasSize(9.0, 16.0))
        assertEquals(768 to 768, Packing.canvasSize(1.0, 1.0))
        val (h, w) = Packing.canvasSize(21.0, 9.0)
        assertTrue(h * w <= 768 * 1344 + 32 * 32 && h % 32 == 0 && w % 32 == 0 && w > h)
        assertFailsWith<IllegalArgumentException> { Packing.canvasSize(5.0, 1.0) }
    }

    @Test
    fun t2vaLayout768p() {
        val (t, h, w) = Geometry.H3_768P.latent(16)
        assertEquals(Dims3(37, 48, 84), Dims3(t, h, w))
        val l = Packing.buildLayout(List(10) { TEXT_TAG }, t, h, w, Geometry.H3_768P.audioLatents(), listOf(1, 2, 2), emptyList())
        val video = 37 * 24 * 42
        assertEquals(10 + 2 * 207 + video, l.sequenceLength)
        assertEquals(414, l.audioIndices.size)
        assertEquals(listOf(3.0, 0.0, 0.0), l.positions.get(3).toList())
        val a0 = l.positions.get(l.audioIndices[0])
        assertEquals(10.0, a0[0])
        assertEquals(0.0, a0[1])
        val aR = l.positions.get(l.audioIndices[207])
        assertEquals(a0[0], aR[0])
        assertTrue(aR[2] > a0[2])
        val v0 = l.positions.t(l.videoIndices[0])
        val v1 = l.positions.t(l.videoIndices[24 * 42])
        val v2 = l.positions.t(l.videoIndices[2 * 24 * 42])
        assertEquals(10.0, v0)
        assertTrue(abs(v1 - (10 + 5.0 / 3)) < 1e-9)
        assertTrue(abs(v2 - (10 + 5.0 / 3 + 4 * 5.0 / 3)) < 1e-9)
    }

    @Test
    fun fl2vaAnchorsAndRowTimesteps() {
        val l = Packing.buildLayout(listOf(TEXT_TAG, TEXT_TAG, VIDEO_TAG), 7, 4, 6, 20, listOf(1, 2, 2), listOf("first", "last"))
        assertEquals(12, l.numConditionVideoRows)
        assertEquals(3.0, l.positions.t(l.videoIndices[0]))
        assertTrue(abs(l.positions.t(l.videoIndices[6]) - (3.0 + 5.0 / 3.0 * 22.0 - 5.0 / 3.0)) < 1e-9)
        val rt = Packing.rowTimesteps(l, 0.3f, 0.6f, 0.999f, 1.0f)
        assertEquals(listOf(0.3f, 0.6f, 0.999f), rt.unique.toList())
        assertEquals(2, rt.index[l.videoIndices[0]])
        assertEquals(1, rt.index[l.audioIndices[0]])
        assertEquals(0, rt.index[0])
        val mr = Packing.modRuns(rt.index, l.tokenTags)
        // text(0,text) | vision placeholder(0,video) | cond video (2, video) | audio (1, audio) | video (0, video)
        assertEquals(
            listOf(Triple(2, 0, 1), Triple(1, 0, 0), Triple(12, 2, 0), Triple(40, 1, 2), Triple(42, 0, 0)),
            mr.map { Triple(it.rows, it.tIdx, it.tag) },
        )
        assertEquals(listOf(0, 2, 3, 15, 55), mr.map { it.start })
    }

    @Test
    fun ref2vaLayoutOrdersSegments() {
        val refs = listOf(RefGeom("image", 0, 4, 4), RefGeom("video", 2, 4, 6, 3), RefGeom("audio", audioLatents = 5))
        val l = Packing.buildRef2vaLayout(List(4) { TEXT_TAG }, refs, 7, 4, 6, 20, listOf(1, 2, 2))
        assertEquals(4 + 12, l.numConditionVideoRows)
        assertEquals(16, l.numConditionAudioRows)
        assertEquals(listOf("image", "audio", "video", "audio"), l.referenceSegments.map { it.kind })
        assertEquals(4 + 4 + 6 + 12 + 10 + 40 + 42, l.sequenceLength)
        assertEquals(l.sequenceLength, l.videoIndices.size + l.audioIndices.size + l.textIndices.size)
        assertFailsWith<IllegalArgumentException> { Packing.buildRef2vaLayout(List(4) { TEXT_TAG }, emptyList(), 7, 4, 6, 20, listOf(1, 2, 2)) }
    }

    @Test
    fun runsAndPatchRows() {
        assertEquals(listOf(Run(3, 3, 0), Run(9, 2, 3)), Packing.runs(intArrayOf(3, 4, 5, 9, 10)))
        val z = Rng(0L).normal(4, 3, 4, 6)
        val rows = Packing.patchRows(z, listOf(1, 2, 2))
        assertEquals(listOf(3 * 2 * 3, 16), rows.shape.toList())
        // feature c·4 + kh·2 + kw of row 0 is z[c, 0, kh, kw]
        assertEquals(z.data[((1 * 3 + 0) * 4 + 1) * 6 + 0], rows.data[1 * 4 + 1 * 2 + 0])
        assertTrue(Packing.unpatchRows(rows, 4, 3, 4, 6, listOf(1, 2, 2)).data.contentEquals(z.data))
    }

    @Test
    fun ropeTables() {
        val t = Rope.h3Rope(listOf(Triple(0.0, 0.0, 0.0), Triple(1.0, 2.0, 3.0)), 16, 10000.0)
        assertEquals(listOf(2, 48), t.cos.shape.toList())
        for (i in 0 until 48) assertEquals(1.0f, t.cos.data[i])
        for (i in 0 until 48) assertEquals(0.0f, t.sin.data[i])
        assertTrue(abs(t.cos.data[48 + 0] - cos(1.0)) < 1e-6)
        assertTrue(abs(t.cos.data[48 + 16] - cos(2.0)) < 1e-6)
        assertTrue(abs(t.cos.data[48 + 32] - cos(3.0)) < 1e-6)
        // the tiny model rotates 12 of 16: 6 angles per row
        val tiny = Rope.h3Rope(listOf(Triple(5.0, 1.5, -2.0)), 2, 10000.0)
        assertEquals(listOf(1, 6), tiny.cos.shape.toList())
        assertTrue(abs(tiny.cos.data[1] - cos(5.0 * 0.01)) < 1e-6)
    }

    @Test
    fun configsAndTaeRule() {
        assertEquals(96, H3Config.h3().rotaryDim())
        assertEquals(12, H3Config.tiny().rotaryDim())
        assertEquals(7168, H3Config.h3().dim)
        assertEquals(96, H3Config.h3().videoPatchDim())
        assertEquals(42, H3Config.fasth3Trim().layers)
        assertEquals(32, H3Config.fasth3Omni().pddSteps)
        val v = VideoVaeConfig.h3()
        assertEquals(
            listOf(16, 4, 3, 5, 2, 5),
            listOf(v.spatialRatio(), v.temporalRatio(), v.framePrePadding(), v.tokensChunk(), v.tokenOverlap(), v.frameOverlap()),
        )
        assertEquals(1797, v.decoderRows(7, 16, 16))
        assertEquals(171, VideoVaeConfig.tiny().decoderRows(7, 4, 6))
        assertEquals(800, AudioVaeConfig.h3().hop())
        assertEquals(10, AudioVaeConfig.tiny().hop())
        fun kept(n: Int) = (0 until n * 4).count { TaeH3Config.keepsRawFrame(it) }
        assertEquals(listOf(17, 124, 5), listOf(kept(5), kept(37), kept(2)))
        assertEquals(16, TaeH3Config.taeh3().spatialScale())
    }

    @Test
    fun splitMix64AndNormals() {
        val r = Rng(0x9E3779B97F4A7C15uL)
        assertEquals(0xE220A8397B1DCDAFuL, r.nextU64())
        assertEquals(0x6E789E6AA1B965F4uL, r.nextU64())
        // the same stream regardless of how it is drawn
        val a = Rng(7L).normal(10)
        val b = Rng(7L)
        val first = b.normal(4)
        val second = b.normal(6)
        assertTrue(a.contentEquals(first + second))
        val n = Rng(1L).normal(200_000)
        val mean = n.sumOf { it.toDouble() } / n.size
        val variance = n.sumOf { it.toDouble() * it.toDouble() } / n.size - mean * mean
        assertTrue(abs(mean) < 0.01 && abs(variance - 1.0) < 0.02, "mean $mean var $variance")
    }

    @Test
    fun flowStepRules() {
        val x = Tensor(floatArrayOf(1f, -2f), intArrayOf(2))
        val v = Tensor(floatArrayOf(0.5f, 0.25f), intArrayOf(2))
        // euler: r = σ'/σ; r·x + (1 − r)·(x + σ·v)
        val e = Sampler.flowStep(x, v, 0.8f, 0.4f, "euler", null)
        val r = 0.4f / 0.8f
        assertEquals(r * 1f + (1f - r) * (1f + 0.8f * 0.5f), e.data[0])
        assertEquals(r * -2f + (1f - r) * (-2f + 0.8f * 0.25f), e.data[1])
        // renoise to σ' = 0 is x0
        val last = Sampler.flowStep(x, v, 0.5f, 0f, "renoise", null)
        assertEquals(1f + 0.5f * 0.5f, last.data[0])
        val noise = Tensor(floatArrayOf(3f, 3f), intArrayOf(2))
        val rn = Sampler.flowStep(x, v, 0.5f, 0.25f, "renoise", noise)
        assertEquals(0.75f * (1f + 0.25f) + 0.25f * 3f, rn.data[0])
        assertTrue(Sampler.needsNoise("renoise", 0.3f) && !Sampler.needsNoise("renoise", 0f) && !Sampler.needsNoise("euler", 0.3f))
        assertFailsWith<IllegalArgumentException> { Sampler.flowStep(x, v, 0.5f, 0.25f, "renoise", null) }
    }

    @Test
    fun halfRoundTrip() {
        for (f in floatArrayOf(0f, -0f, 1f, -1f, 0.1f, 3.14159f, 65504f, 1e-5f, 6.1e-5f, 5.96e-8f, 1234.5678f)) {
            val h = Host.floatToHalf(f)
            val back = SafeTensors.halfToFloat(h)
            val tol = maxOf(abs(f) * 1e-3f, 1e-7f)
            assertTrue(abs(back - f) <= tol, "$f -> $back")
        }
        // round to nearest even on the half-way mantissa bit
        assertEquals(1.0f, SafeTensors.halfToFloat(Host.floatToHalf(1.0f + 1f / 4096f)))
        assertEquals(1.0f + 2f / 2048f, SafeTensors.halfToFloat(Host.floatToHalf(1.0f + 3f / 4096f)))
        assertEquals(Float.POSITIVE_INFINITY, SafeTensors.halfToFloat(Host.floatToHalf(1e6f)))
        assertEquals(1.0f, Host.roundThroughHalf(Tensor(floatArrayOf(1.0001f), intArrayOf(1))).data[0])
    }

    @Test
    fun rgbaQuantisation() {
        val px = Tensor(floatArrayOf(0f, 0.5f, 1.2f, -0.1f, 0.25f, 0.75f), intArrayOf(3, 1, 1, 2)) // [3, F, H, W]
        val rgba = Host.toRgba8(px)
        // pixel 0: R 0, G 1.2 → 255, B 0.25 → 64; pixel 1: R 0.5 → 128, G −0.1 → 0, B 0.75 → 191
        assertEquals(listOf(0, 255, 64, 255, 128, 0, 191, 255), rgba.map { it.toInt() and 0xFF })
        assertEquals(6, Host.toRgb8(px).size)
    }
}
