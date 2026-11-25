package dev.h3.litert.core

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** `vae_host.py` against h3-rs `vae.rs`'s tests: tiles, stitch, the temporal plan and the chunk blend. */
class VaeHostTest {
    @Test
    fun splitTiles768() {
        val t = VaeHost.splitTiles(768, 256, 64, 16)
        assertEquals(listOf(0, 160, 336, 512), t.starts.toList())
        assertEquals(List(4) { 256 }, t.lengths.toList())
        assertEquals(listOf(96, 80, 80), t.overlaps.toList())
        // last tile ends exactly at the edge
        assertEquals(768, t.starts.last() + t.lengths.last())
        val one = VaeHost.splitTiles(64, 256, 64, 16)
        assertEquals(listOf(0), one.starts.toList())
        assertEquals(listOf(64), one.lengths.toList())
        assertEquals(0, one.overlaps.size)
        val w = VaeHost.splitTiles(1344, 256, 64, 16)
        assertEquals(1344, w.starts.last() + w.lengths.last())
        assertTrue(w.overlaps.all { it >= 64 && it % 16 == 0 })
    }

    @Test
    fun temporalPlans() {
        val h3 = VideoVaeConfig.h3()
        assertEquals(VaeHost.TemporalPlan(0, 7, 124), VaeHost.temporalPlan(h3, 37))
        assertEquals(VaeHost.TemporalPlan(0, 1, 22), VaeHost.temporalPlan(h3, 7))
        assertEquals(VaeHost.TemporalPlan(5, 1, 5), VaeHost.temporalPlan(h3, 2))
        assertEquals(VaeHost.TemporalPlan(0, 1, 22), VaeHost.temporalPlan(VideoVaeConfig.tiny(), 7))
        assertFailsWith<IllegalArgumentException> { VaeHost.temporalPlan(h3, 0) }
    }

    @Test
    fun blendIsALinearCrossFade() {
        // [1, 1, 1, W]: a = 1s (W 4), b = 3s (W 6), extent 4 → weights 0, .25, .5, .75 then the rest of b
        val a = Tensor(FloatArray(4) { 1f }, intArrayOf(1, 1, 1, 4))
        val b = Tensor(FloatArray(6) { 3f }, intArrayOf(1, 1, 1, 6))
        val out = VaeHost.blend(a, b, 4, 3)
        assertEquals(listOf(1, 1, 1, 6), out.shape.toList())
        assertEquals(listOf(1f, 1.5f, 2f, 2.5f, 3f, 3f), out.data.toList())
        // extent larger than both is clipped; 0 returns b
        assertEquals(b.data.toList(), VaeHost.blend(a, b, 0, 3).data.toList())
        assertEquals(6, VaeHost.blend(a, b, 10, 3).size)
        // along frames (axis 1)
        val fa = Tensor(FloatArray(2 * 3) { 0f }, intArrayOf(1, 2, 1, 3))
        val fb = Tensor(FloatArray(2 * 3) { 2f }, intArrayOf(1, 2, 1, 3))
        val fo = VaeHost.blend(fa, fb, 2, 1)
        assertEquals(listOf(0f, 0f, 0f, 1f, 1f, 1f), fo.data.toList())
    }

    @Test
    fun stitchTrimsOverlapsAndKeepsTheCanvas() {
        // two 4-px tiles overlapping by 2 along W → a 6-px canvas; the left tile is 0, the right 2
        val left = Tensor(FloatArray(4) { 0f }, intArrayOf(1, 1, 1, 4))
        val right = Tensor(FloatArray(4) { 2f }, intArrayOf(1, 1, 1, 4))
        val out = VaeHost.stitchTiles(listOf(listOf(left, right)), IntArray(0), intArrayOf(2))
        assertEquals(listOf(1, 1, 1, 6), out.shape.toList())
        // left keeps 2 px; the right tile's first two px fade from the left tile (0) to itself (2): 0, 1; then 2, 2
        assertEquals(listOf(0f, 0f, 0f, 1f, 2f, 2f), out.data.toList())
        // a 2×2 grid along both axes
        val tiles = listOf(listOf(left, right), listOf(right, left))
        val grid = VaeHost.stitchTiles(tiles, intArrayOf(0), intArrayOf(2))
        assertEquals(listOf(1, 1, 2, 6), grid.shape.toList())
    }

    @Test
    fun tiledRunsEveryTileOnceAndStitchesInOutputUnits() {
        // a 768×1344 latent canvas at ratio 16: 4 × 7 tiles of 16×16 latents (1344 px needs 7 tiles of 256 with
        // >= 64 px overlap), stitched in pixels (ratio_out 1)
        assertEquals(7, VaeHost.splitTiles(1344, 256, 64, 16).starts.size)
        val z = Tensor.zeros(1, 1, 48, 84)
        val seen = mutableListOf<List<Int>>()
        val out = VaeHost.tiled(z, 16, 1, 256, 64) { t ->
            seen += t.shape.toList()
            Tensor.zeros(1, 1, 16 * t.shape[2], 16 * t.shape[3])
        }
        assertEquals(4 * 7, seen.size)
        assertTrue(seen.all { it == listOf(1, 1, 16, 16) })
        assertEquals(listOf(1, 1, 768, 1344), out.shape.toList())
        // the encoder direction: pixels in, latents out (ratio_in 1, ratio_out 16)
        val px = Tensor.zeros(3, 1, 64, 96)
        val enc = VaeHost.tiled(px, 1, 16, 256, 64) { t ->
            assertEquals(listOf(3, 1, 64, 96), t.shape.toList())
            Tensor.zeros(8, 1, 4, 6)
        }
        assertEquals(listOf(8, 1, 4, 6), enc.shape.toList())
    }

    @Test
    fun decodeFollowsThePlanAndBlendsChunks() {
        val cfg = VideoVaeConfig.tiny()
        // 12 latent frames → n = 15, chunks 2, frames 2·17 + 5 = 39
        assertEquals(VaeHost.TemporalPlan(0, 2, 39), VaeHost.temporalPlan(cfg, 12))
        val z = Tensor.zeros(4, 12, 1, 1)
        val clips = mutableListOf<Int>()
        val chunks = mutableListOf<Int>()
        var call = 0
        val out = VaeHost.decode(cfg, z, { clip ->
            clips += clip.shape[1]
            // each call returns a constant frame value in normalised space, different per call
            val v = call++.toFloat()
            Tensor(FloatArray(3 * 4 * clip.shape[1]) { v }, intArrayOf(3, 4 * clip.shape[1], 1, 1))
        }) { px -> chunks += px.shape[1] }
        assertEquals(listOf(7, 7), clips)
        assertEquals(listOf(17, 17, 5), chunks)
        assertEquals(listOf(3, 39, 1, 1), out.shape.toList())
        // frames of chunk 0 are the channel means (0 normalised); the blend into chunk 1 fades towards 0·std + mean of value 1
        assertTrue(abs(out.data[0] - 0.485f) < 1e-6)
        val f17 = out.data[17] // first frame of the second chunk: w = 0 → the overlap (call 0) → 0.485
        assertTrue(abs(f17 - 0.485f) < 1e-6)
        val f18 = out.data[18] // w = 1/5 → 0.2·1·0.229 + 0.485
        assertTrue(abs(f18 - (0.485f + 0.2f * 0.229f)) < 1e-6)
    }

    @Test
    fun normalisationAndPosterior() {
        val cfg = VideoVaeConfig.tiny()
        val z = Tensor(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(4, 1, 1, 1))
        val n = VaeHost.normalizeLatents(cfg, z)
        assertTrue(abs(n.data[0] - (1f - 0.1f) / 1.1f) < 1e-6)
        val back = VaeHost.denormalizeLatents(cfg, n)
        for (i in 0 until 4) assertTrue(abs(back.data[i] - z.data[i]) < 1e-5)
        val px = Tensor(floatArrayOf(0.485f, 0.456f, 0.406f), intArrayOf(3, 1, 1, 1))
        val np = VaeHost.normalizePixels(px)
        assertTrue(np.data.all { abs(it) < 1e-6 })
        assertEquals(listOf(1f, 1f, 1f), VaeHost.denormalizePixels(Tensor(floatArrayOf(9f, 9f, 9f), intArrayOf(3, 1, 1, 1))).data.toList())
        val moments = Tensor(floatArrayOf(0.5f, -0.5f, 0f, 2f), intArrayOf(4, 1, 1, 1)) // mean 0.5, −0.5; logvar 0, 2
        val eps = Tensor(floatArrayOf(1f, 1f), intArrayOf(2, 1, 1, 1))
        val s = VaeHost.posteriorSample(moments, eps)
        assertEquals(listOf(2, 1, 1, 1), s.shape.toList())
        assertTrue(abs(s.data[0] - 1.5f) < 1e-6)
        assertTrue(abs(s.data[1] - (-0.5f + kotlin.math.exp(1.0f))) < 1e-5)
    }

    @Test
    fun encodePadsToClipsAndDropsTheTail() {
        val cfg = VideoVaeConfig.tiny()
        val calls = mutableListOf<Int>()
        val enc: (Tensor) -> Tensor = { x -> calls += x.shape[1]; Tensor.zeros(8, if (x.shape[1] == 1) 1 else 5, 1, 1) }
        val single = VaeHost.encode(cfg, Tensor.zeros(3, 1, 16, 16), enc)
        assertEquals(listOf(8, 1, 1, 1), single.shape.toList())
        val clip = VaeHost.encode(cfg, Tensor.zeros(3, 20, 16, 16), enc)
        assertEquals(listOf(1, 17, 17), calls)
        assertEquals(listOf(8, 10 - 3, 1, 1), clip.shape.toList())
    }
}
