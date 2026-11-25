package dev.h3.litert.core

import kotlinx.serialization.Serializable
import kotlin.math.sqrt

/**
 * Pixel ↔ latent ↔ packed-token geometry of one H3 request (`packing.py`; h3-core `packing.rs`;
 * FastVideo `packing.py`). Positions are computed in f64 exactly as the reference does; the rotary
 * tables are then formed in f32 ([Rope]).
 */
public object Packing {
    public const val FPS: Int = 24
    public const val SHORT_EDGE: Int = 768
    public const val MAX_PIXELS: Int = 768 * 1344
    public const val CANVAS_MULTIPLE: Int = 32
    public const val MIN_ASPECT: Double = 0.25
    public const val MAX_ASPECT: Double = 4.0
    public const val FRAMES_PER_CHUNK: Int = 17
    public const val LATENTS_PER_CHUNK: Int = 5
    public const val AUDIO_LATENTS_PER_SECOND: Int = 40
    public const val AUDIO_CHANNELS: Int = 2
    public const val MIN_ALIGNED_FRAMES: Int = 124
    public const val MAX_ALIGNED_FRAMES: Int = 362
    public val CONTINUATION_FRAMES: List<Int> = listOf(5, 22, 39)

    private const val ROPE_FRAME_RESCALE = 5.0 / 3.0
    private val ROPE_FRAMES_PER_LATENT = doubleArrayOf(1.0, 4.0, 4.0, 4.0, 4.0)
    private const val ROPE_SPATIAL_SCALE = 32.0

    /** Canvas `(height, width)` for an aspect ratio, snapped to the 32-pixel lattice. */
    public fun canvasSize(aspectW: Double, aspectH: Double, shortEdge: Int = SHORT_EDGE, maxPixels: Int = MAX_PIXELS): Pair<Int, Int> {
        if (aspectW <= 0 || aspectH <= 0) throw IllegalArgumentException("aspect ratio must be positive, got $aspectW:$aspectH")
        val ratio = aspectW / aspectH
        if (ratio < MIN_ASPECT || ratio > MAX_ASPECT) {
            throw IllegalArgumentException("aspect ratios run from 1:4 to 4:1, got $aspectW:$aspectH")
        }
        var w: Double
        var h: Double
        if (ratio >= 1.0) { w = shortEdge * ratio; h = shortEdge.toDouble() } else { w = shortEdge.toDouble(); h = shortEdge / ratio }
        val area = w * h
        if (area > maxPixels) {
            val s = sqrt(maxPixels / area)
            w *= s
            h *= s
        }
        val m = CANVAS_MULTIPLE.toDouble()
        // Python's round() is half-to-even: Math.rint.
        fun snap(v: Double): Int = maxOf(Math.rint(v / m) * m, m).toInt()
        return snap(h) to snap(w)
    }

    public fun alignNumFrames(n: Int): Int {
        var v = maxOf(n, 1)
        while (v % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK) v++
        return v
    }

    public fun resolveNumFrames(n: Int): Int {
        val a = alignNumFrames(n)
        if (a < MIN_ALIGNED_FRAMES || a > MAX_ALIGNED_FRAMES) {
            throw IllegalArgumentException("H3 generates 5-15 s at 24 fps; aligned num_frames $a is outside 124-362")
        }
        return a
    }

    /** `17n + 5` frames → `5n + 2` latent frames. */
    public fun videoLatentFrames(n: Int): Int {
        if (n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK) throw IllegalArgumentException("num_frames must be 17n + 5, got $n")
        return (n - LATENTS_PER_CHUNK) / FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2
    }

    /** `round(frames / 24 · 40)` with Python's half-even rounding. */
    public fun audioLatents(frames: Int): Int = Math.rint(frames / FPS.toDouble() * AUDIO_LATENTS_PER_SECOND).toInt()

    private fun spatialPositionGrid(dim: Int, patch: Int, sqrtArea: Double): DoubleArray {
        val ratio = dim / sqrtArea
        val left = (1.0 - ratio) / 2.0
        val n = dim / patch
        val step = ratio / n
        return DoubleArray(n) { i -> (left + i * step) * ROPE_SPATIAL_SCALE }
    }

    /** Temporal rotary positions of `n` latent frames from `origin`: the `5/3 · [1, 4, 4, 4, 4]` clock. */
    public fun temporalPositionGrid(n: Int, origin: Double): DoubleArray {
        val out = DoubleArray(n)
        var acc = 0.0
        for (i in 0 until n) {
            out[i] = origin + acc
            acc += ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % 5]
        }
        return out
    }

    public fun temporalSpan(n: Int): Double {
        var s = 0.0
        for (i in 0 until n) s += ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % 5]
        return s
    }

    /** `(h, w)` positions of one frame's patches, h-major, plus the w grid. */
    private class FrameGrid(val hw: Array<DoubleArray>, val wg: DoubleArray)

    private fun frameGrid(lh: Int, lw: Int, ph: Int, pw: Int): FrameGrid {
        val sqrtArea = sqrt((lh * lw).toDouble())
        val hg = spatialPositionGrid(lh, ph, sqrtArea)
        val wg = spatialPositionGrid(lw, pw, sqrtArea)
        val grid = ArrayList<DoubleArray>(hg.size * wg.size)
        for (h in hg) for (w in wg) grid += doubleArrayOf(h, w)
        return FrameGrid(grid.toTypedArray(), wg)
    }

    private fun fillAudio(pos: Positions, start: Int, n: Int, time0: Double, wg: DoubleArray) {
        for (ch in 0 until AUDIO_CHANNELS) {
            val wpos = if (ch == 0) wg[0] else wg[wg.size - 1]
            for (i in 0 until n) pos.set(start + ch * n + i, time0 + i, 0.0, wpos)
        }
    }

    private fun checkTextTags(tags: List<Int>) {
        if (tags.any { it != TEXT_TAG && it != VIDEO_TAG }) throw IllegalArgumentException("text token tags may be text or vision only")
    }

    /**
     * T2VA / FL2VA: `[text | condition keyframes | audio | video]`; anchors are "first" / "last".
     */
    public fun buildLayout(textTags: List<Int>, lt: Int, lh: Int, lw: Int, na: Int, patch: List<Int>, anchors: List<String>): Layout {
        checkTextTags(textTags)
        val (pt, ph, pw) = Triple(patch[0], patch[1], patch[2])
        if (lt % pt != 0 || lh % ph != 0 || lw % pw != 0) {
            throw IllegalArgumentException("latent ($lt, $lh, $lw) is not divisible by the patch $patch")
        }
        val rpf = (lh / ph) * (lw / pw)
        val nText = textTags.size
        val nCond = anchors.size * rpf
        val nAudio = na * AUDIO_CHANNELS
        val nVideo = (lt / pt) * rpf
        val total = nText + nCond + nAudio + nVideo
        val condStart = nText
        val audioStart = nText + nCond
        val videoStart = nText + nCond + nAudio
        val pos = Positions(total)
        for (i in 0 until nText) pos.set(i, i.toDouble(), 0.0, 0.0)
        val fg = frameGrid(lh, lw, ph, pw)
        for ((k, a) in anchors.withIndex()) {
            val t = when (a) {
                "first" -> nText.toDouble()
                "last" -> nText + temporalSpan(lt) - ROPE_FRAME_RESCALE
                else -> throw IllegalArgumentException("anchor `$a` is not first or last")
            }
            for ((j, g) in fg.hw.withIndex()) pos.set(condStart + k * rpf + j, t, g[0], g[1])
        }
        fillAudio(pos, audioStart, na, nText.toDouble(), fg.wg)
        val tgrid = temporalPositionGrid(lt, nText.toDouble())
        for (f in 0 until lt) for ((j, g) in fg.hw.withIndex()) pos.set(videoStart + f * rpf + j, tgrid[f], g[0], g[1])
        val tags = IntArray(total)
        for (i in 0 until nText) tags[i] = textTags[i]
        for (i in condStart until audioStart) tags[i] = VIDEO_TAG
        for (i in audioStart until videoStart) tags[i] = AUDIO_TAG
        for (i in videoStart until total) tags[i] = VIDEO_TAG
        val videoIdx = IntArray(nCond + nVideo) { if (it < nCond) condStart + it else videoStart + (it - nCond) }
        val audioIdx = IntArray(nAudio) { audioStart + it }
        return Layout(total, pos, tags, videoIdx, audioIdx, IntArray(nText) { it }, nCond, 0, lt, lh, lw, na, emptyList())
    }

    /** Ref2VA: `[text | ordered references | continuation clip | target audio | target video]`. */
    public fun buildRef2vaLayout(
        textTags: List<Int>,
        refs: List<RefGeom>,
        lt: Int,
        lh: Int,
        lw: Int,
        na: Int,
        patch: List<Int>,
        continuation: Pair<Int, Int>? = null,
    ): Layout {
        if (patch != listOf(1, 2, 2)) throw IllegalArgumentException("Ref2VA requires patch (1, 2, 2), got $patch")
        checkTextTags(textTags)
        if (refs.isEmpty()) throw IllegalArgumentException("Ref2VA requires at least one reference")
        if (lt == 0 || lh % 2 != 0 || lw % 2 != 0 || na == 0) {
            throw IllegalArgumentException("Ref2VA target geometry must be positive and patch-aligned")
        }
        val nText = textTags.size
        fun rowsOf(t: Int, h: Int, w: Int): Int {
            if (t == 0 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0) {
                throw IllegalArgumentException("visual reference geometry ($t, $h, $w) is not patch-aligned")
            }
            return t * (h / 2) * (w / 2)
        }
        val rptf = (lh / 2) * (lw / 2)
        val nTargetVideo = lt * rptf
        val nTargetAudio = na * AUDIO_CHANNELS
        var nRefVideo = 0
        var nRefAudio = 0
        for (r in refs) {
            when (r.kind) {
                "image" -> nRefVideo += rowsOf(1, r.latentH, r.latentW)
                "video" -> { nRefVideo += rowsOf(r.latentT, r.latentH, r.latentW); nRefAudio += r.audioLatents * AUDIO_CHANNELS }
                "audio" -> {
                    if (r.audioLatents == 0) throw IllegalArgumentException("an audio reference has no latents")
                    nRefAudio += r.audioLatents * AUDIO_CHANNELS
                }
                else -> throw IllegalArgumentException("reference kind `${r.kind}`")
            }
        }
        if (continuation != null) {
            val (ct, ca) = continuation
            if (ct == 0 || ct >= lt) throw IllegalArgumentException("continuation clip of $ct latent frames does not fit a target of $lt")
            nRefVideo += ct * rptf
            nRefAudio += ca * AUDIO_CHANNELS
        }
        val total = nText + nRefVideo + nRefAudio + nTargetAudio + nTargetVideo
        val pos = Positions(total)
        for (i in 0 until nText) pos.set(i, i.toDouble(), 0.0, 0.0)
        val target = frameGrid(lh, lw, 2, 2)
        val videoIdx = ArrayList<Int>()
        val audioIdx = ArrayList<Int>()
        val segments = ArrayList<Segment>()
        var cursor = nText
        var clock = nText.toDouble()
        for (r in refs) {
            when (r.kind) {
                "image" -> {
                    val n = rowsOf(1, r.latentH, r.latentW)
                    segments += Segment("image", n, Dims3(1, r.latentH, r.latentW))
                    val fg = frameGrid(r.latentH, r.latentW, 2, 2)
                    for ((j, g) in fg.hw.withIndex()) pos.set(cursor + j, clock, g[0], g[1])
                    for (i in cursor until cursor + n) videoIdx += i
                    cursor += n
                    clock += 1.0
                }
                "audio" -> {
                    val n = r.audioLatents * AUDIO_CHANNELS
                    segments += Segment("audio", n, Dims3(0, 0, 0))
                    fillAudio(pos, cursor, r.audioLatents, clock, target.wg)
                    for (i in cursor until cursor + n) audioIdx += i
                    cursor += n
                    clock += r.audioLatents.toDouble()
                }
                else -> {
                    val naR = r.audioLatents * AUDIO_CHANNELS
                    val nv = rowsOf(r.latentT, r.latentH, r.latentW)
                    if (naR > 0) segments += Segment("audio", naR, Dims3(0, 0, 0))
                    segments += Segment("video", nv, Dims3(r.latentT, r.latentH, r.latentW))
                    val fg = frameGrid(r.latentH, r.latentW, 2, 2)
                    if (naR > 0) {
                        fillAudio(pos, cursor, r.audioLatents, clock, fg.wg)
                        for (i in cursor until cursor + naR) audioIdx += i
                        cursor += naR
                    }
                    val tg = temporalPositionGrid(r.latentT, clock)
                    for (f in 0 until r.latentT) for ((j, g) in fg.hw.withIndex()) pos.set(cursor + f * fg.hw.size + j, tg[f], g[0], g[1])
                    for (i in cursor until cursor + nv) videoIdx += i
                    cursor += nv
                    clock += maxOf(r.audioLatents.toDouble(), temporalSpan(r.latentT))
                }
            }
        }
        if (continuation != null) {
            val (ct, ca) = continuation
            val naC = ca * AUDIO_CHANNELS
            val nv = ct * rptf
            if (naC > 0) segments += Segment("audio", naC, Dims3(0, 0, 0))
            segments += Segment("continuation", nv, Dims3(ct, lh, lw))
            clock += maxOf(ca.toDouble(), temporalSpan(ct))
            if (naC > 0) {
                fillAudio(pos, cursor, ca, clock, target.wg)
                for (i in cursor until cursor + naC) audioIdx += i
                cursor += naC
            }
            val tg = temporalPositionGrid(ct, clock)
            for (f in 0 until ct) for ((j, g) in target.hw.withIndex()) pos.set(cursor + f * rptf + j, tg[f], g[0], g[1])
            for (i in cursor until cursor + nv) videoIdx += i
            cursor += nv
        }
        val audioStart = cursor
        val videoStart = audioStart + nTargetAudio
        fillAudio(pos, audioStart, na, clock, target.wg)
        val tg = temporalPositionGrid(lt, clock)
        for (f in 0 until lt) for ((j, g) in target.hw.withIndex()) pos.set(videoStart + f * rptf + j, tg[f], g[0], g[1])
        for (i in audioStart until videoStart) audioIdx += i
        for (i in videoStart until total) videoIdx += i
        check(videoStart + nTargetVideo == total)
        val tags = IntArray(total) { VIDEO_TAG }
        for (i in 0 until nText) tags[i] = textTags[i]
        for (i in audioIdx) tags[i] = AUDIO_TAG
        for (i in videoIdx) tags[i] = VIDEO_TAG
        return Layout(
            total, pos, tags, videoIdx.toIntArray(), audioIdx.toIntArray(), IntArray(nText) { it },
            nRefVideo, nRefAudio, lt, lh, lw, na, segments,
        )
    }

    public class RowTimesteps(public val unique: FloatArray, public val index: IntArray)

    /**
     * Per-row clean times of one step (f32 values): the sorted unique times and each row's index
     * into them. Target video rows at `videoT`, condition video at `condVideoT`, target audio at
     * `audioT`, condition audio at `condAudioT`; text rows share the video time.
     */
    public fun rowTimesteps(layout: Layout, videoT: Float, audioT: Float, condVideoT: Float, condAudioT: Float): RowTimesteps {
        val rows = FloatArray(layout.sequenceLength) { videoT }
        for (k in 0 until layout.numConditionVideoRows) rows[layout.videoIndices[k]] = condVideoT
        for (k in layout.numConditionAudioRows until layout.audioIndices.size) rows[layout.audioIndices[k]] = audioT
        for (k in 0 until layout.numConditionAudioRows) rows[layout.audioIndices[k]] = condAudioT
        val unique = rows.toSet().sorted().toFloatArray()
        val pos = HashMap<Float, Int>()
        for ((k, u) in unique.withIndex()) pos[u] = k
        return RowTimesteps(unique, IntArray(rows.size) { pos[rows[it]]!! })
    }

    /** Contiguous runs of a modality's packed indices. */
    public fun runs(indices: IntArray): List<Run> {
        val out = ArrayList<Run>()
        for ((off, i) in indices.withIndex()) {
            val last = out.lastOrNull()
            if (last != null && last.start + last.rows == i) out[out.size - 1] = Run(last.start, last.rows + 1, last.offset)
            else out += Run(i, 1, off)
        }
        return out
    }

    /** Maximal runs of rows sharing `(timestep index, tag)`. */
    public fun modRuns(tIdx: IntArray, tags: IntArray): List<ModRun> {
        require(tIdx.size == tags.size) { "modRuns: ${tIdx.size} timestep indices vs ${tags.size} tags" }
        val out = ArrayList<ModRun>()
        for (i in tIdx.indices) {
            val last = out.lastOrNull()
            if (last != null && last.tIdx == tIdx[i] && last.tag == tags[i]) out[out.size - 1] = ModRun(last.start, last.rows + 1, last.tIdx, last.tag)
            else out += ModRun(i, 1, tIdx[i], tags[i])
        }
        return out
    }

    /** `[C, T, H, W]` latent → `[rows, C·pt·ph·pw]` patch rows (channel-major features). */
    public fun patchRows(z: Tensor, patch: List<Int>): Tensor {
        require(z.rank == 4) { "patchRows wants [C, T, H, W], got ${z.shapeString()}" }
        val c = z.shape[0]; val t = z.shape[1]; val h = z.shape[2]; val w = z.shape[3]
        val pt = patch[0]; val ph = patch[1]; val pw = patch[2]
        require(t % pt == 0 && h % ph == 0 && w % pw == 0) { "latent ${z.shapeString()} is not divisible by the patch $patch" }
        val tt = t / pt; val hh = h / ph; val ww = w / pw
        val feat = c * pt * ph * pw
        val out = FloatArray(tt * hh * ww * feat)
        val src = z.data
        for (it in 0 until tt) for (ih in 0 until hh) for (iw in 0 until ww) {
            val base = ((it * hh + ih) * ww + iw) * feat
            for (cc in 0 until c) for (kt in 0 until pt) for (kh in 0 until ph) for (kw in 0 until pw) {
                out[base + ((cc * pt + kt) * ph + kh) * pw + kw] =
                    src[((cc * t + it * pt + kt) * h + ih * ph + kh) * w + iw * pw + kw]
            }
        }
        return Tensor(out, intArrayOf(tt * hh * ww, feat))
    }

    public fun unpatchRows(rows: Tensor, c: Int, t: Int, h: Int, w: Int, patch: List<Int>): Tensor {
        val pt = patch[0]; val ph = patch[1]; val pw = patch[2]
        val tt = t / pt; val hh = h / ph; val ww = w / pw
        val feat = c * pt * ph * pw
        require(rows.rank == 2 && rows.shape[0] == tt * hh * ww && rows.shape[1] == feat) {
            "unpatchRows wants [${tt * hh * ww}, $feat], got ${rows.shapeString()}"
        }
        val out = FloatArray(c * t * h * w)
        val src = rows.data
        for (it in 0 until tt) for (ih in 0 until hh) for (iw in 0 until ww) {
            val base = ((it * hh + ih) * ww + iw) * feat
            for (cc in 0 until c) for (kt in 0 until pt) for (kh in 0 until ph) for (kw in 0 until pw) {
                out[((cc * t + it * pt + kt) * h + ih * ph + kh) * w + iw * pw + kw] =
                    src[base + ((cc * pt + kt) * ph + kh) * pw + kw]
            }
        }
        return Tensor(out, intArrayOf(c, t, h, w))
    }
}

/** `(t, h, w)` of a latent. */
public data class Dims3(val t: Int, val h: Int, val w: Int) {
    val count: Int get() = t * h * w
}

/** Rotary positions `(t, h, w)` of every packed row, in f64. */
public class Positions(public val rows: Int) {
    public val data: DoubleArray = DoubleArray(rows * 3)

    public fun set(i: Int, t: Double, h: Double, w: Double) {
        data[i * 3] = t; data[i * 3 + 1] = h; data[i * 3 + 2] = w
    }

    public fun t(i: Int): Double = data[i * 3]
    public fun h(i: Int): Double = data[i * 3 + 1]
    public fun w(i: Int): Double = data[i * 3 + 2]

    public fun get(i: Int): DoubleArray = doubleArrayOf(data[i * 3], data[i * 3 + 1], data[i * 3 + 2])
}

/** Kind, rows and latent dims of one Ref2VA segment. */
public data class Segment(val kind: String, val rows: Int, val dims: Dims3)

/** One Ref2VA reference: an image, a clip (with its audio latents) or audio alone. */
public data class RefGeom(val kind: String, val latentT: Int = 0, val latentH: Int = 0, val latentW: Int = 0, val audioLatents: Int = 0)

public class Layout(
    public val sequenceLength: Int,
    public val positions: Positions,
    public val tokenTags: IntArray,
    public val videoIndices: IntArray,
    public val audioIndices: IntArray,
    public val textIndices: IntArray,
    public val numConditionVideoRows: Int,
    public val numConditionAudioRows: Int,
    public val numVideoLatentFrames: Int,
    public val latentHeight: Int,
    public val latentWidth: Int,
    public val numAudioLatents: Int,
    public val referenceSegments: List<Segment>,
) {
    public val numTextTokens: Int get() = textIndices.size
    public val numTargetVideoRows: Int get() = videoIndices.size - numConditionVideoRows
    public val numTargetAudioRows: Int get() = audioIndices.size - numConditionAudioRows
}

/** A contiguous run of packed rows: `(packed start, rows, offset in the modality's own row buffer)`. */
public data class Run(val start: Int, val rows: Int, val offset: Int)

/** A maximal run of rows sharing `(timestep index, tag)`. */
public data class ModRun(val start: Int, val rows: Int, val tIdx: Int, val tag: Int)

/** Pixel ↔ latent shapes of an H3 clip: a `WxHxF` size of the manifest's `sizes`. */
@Serializable
public data class Geometry(val width: Int, val height: Int, val frames: Int) {
    public fun validate(spatialRatio: Int) {
        if (width <= 0 || height <= 0 || width % Packing.CANVAS_MULTIPLE != 0 || height % Packing.CANVAS_MULTIPLE != 0) {
            throw IllegalArgumentException("${width}x$height is not a positive multiple of 32")
        }
        if (width % spatialRatio != 0 || height % spatialRatio != 0) {
            throw IllegalArgumentException("${width}x$height is not divisible by the VAE ratio $spatialRatio")
        }
        Packing.videoLatentFrames(frames)
    }

    /** Latent `(T, h, w)`. */
    public fun latent(spatialRatio: Int): Dims3 = Dims3(Packing.videoLatentFrames(frames), height / spatialRatio, width / spatialRatio)

    public fun audioLatents(): Int = Packing.audioLatents(frames)

    override fun toString(): String = "${width}x${height}x$frames"

    public companion object {
        /** `WxHxF`, e.g. `1344x768x124`. */
        public fun parse(s: String): Geometry {
            val v = s.lowercase().split("x").map { it.toIntOrNull() }
            if (v.size != 3 || v.any { it == null }) throw IllegalArgumentException("geometry `$s` is not WIDTHxHEIGHTxFRAMES")
            return Geometry(v[0]!!, v[1]!!, v[2]!!)
        }

        public val H3_768P: Geometry = Geometry(1344, 768, 124)
        public val H3_480P: Geometry = Geometry(832, 480, 124)

        /** The tiny fixture's clip: latent `(7, 4, 6)`, 37 audio latents. */
        public val TINY: Geometry = Geometry(96, 64, 22)
    }
}
