package dev.h3.litert.core

/**
 * Sampling schedules (`schedule.py`; h3-core `schedule.rs`). Clean-time clock `t = 1 − σ`;
 * `x0 = x + σ·v`. Every sigma is computed in f32 as the reference does.
 *
 * A manifest carries its whole schedule precomputed (`sampler.steps`); this file exists for the
 * tests that pin the host math against h3-rs and for hosts that want to show a variant's ladder.
 */
public object Schedules {
    public val FASTH3_V1_RUNGS: List<Int> = listOf(999, 749, 500, 250)
    public val FASTH3_V2_RUNGS: List<Int> = listOf(999, 874, 749, 624, 500, 375, 250, 125)
    public val PIKA_KNOTS: List<Int> = listOf(1000, 750, 500, 250, 100, 40)
    public val PLUG4_KNOTS: List<Int> = listOf(1000, 750, 500, 250)
    public const val H3_BASE_POINTS: Int = 50
    public const val KEYFRAME_NOISE_AUG: Float = 0.999f
    public const val PDD_GRID_MAX_T: Double = 0.999

    /** `σ = s·b / (1 + (s − 1)·b)` in f32. */
    public fun shiftSigma(base: Float, shift: Float): Float = (shift * base) / (1.0f + (shift - 1.0f) * base)

    /** `torch.linspace` in f32: symmetric fill from both ends. */
    public fun linspaceF32(start: Float, end: Float, n: Int): FloatArray {
        if (n == 1) return floatArrayOf(start)
        val step = (end - start) / (n - 1).toFloat()
        return FloatArray(n) { i -> if (i < n / 2) start + step * i.toFloat() else end - step * (n - 1 - i).toFloat() }
    }

    public fun linspaceF64(start: Double, end: Double, n: Int): DoubleArray {
        if (n == 1) return doubleArrayOf(start)
        val step = (end - start) / (n - 1)
        return DoubleArray(n) { i -> if (i < n / 2) start + step * i else end - step * (n - 1 - i) }
    }

    public fun pddShift(u: Double, shift: Double, maxT: Double): Double =
        if (shift == 1.0) u else shift * u * maxT / (u * (shift - 1.0) + maxT)

    public fun pddShiftDelta(a: Double, b: Double, shift: Double, maxT: Double): Double {
        if (shift == 1.0) return b - a
        val da = maxT + (shift - 1.0) * a
        val db = maxT + (shift - 1.0) * b
        return shift * maxT * maxT * (b - a) / (da * db)
    }
}

public data class Shifts(val video: Double, val audio: Double) {
    public companion object {
        public val H3: Shifts = Shifts(12.0, 3.0)
        public val FASTH3_V2: Shifts = Shifts(10.0, 3.0)
    }
}

/** One modality's sigma grid: `steps + 1` strictly decreasing f32 values ending at 0. */
public data class Schedule(val sigmas: List<Float>, val rule: String) {
    init {
        require(sigmas.size >= 2 && sigmas.zipWithNext().all { (a, b) -> b < a } && sigmas.last() == 0.0f) {
            "sigmas must hold >= 2 strictly decreasing values ending at 0, got $sigmas"
        }
        require(rule == "euler" || rule == "renoise") { "rule `$rule` is not euler or renoise" }
    }

    public val steps: Int get() = sigmas.size - 1

    /** Clean time `1 − σ_i` in f32. */
    public fun timestep(i: Int): Float = 1.0f - sigmas[i]

    public companion object {
        public fun fromRungs(rungs: List<Int>, shift: Double, rule: String = "euler"): Schedule {
            require(rungs.isNotEmpty() && rungs.all { it in 1..1000 } && rungs.zipWithNext().all { (a, b) -> b < a }) {
                "rungs must be strictly decreasing in (0, 1000], got $rungs"
            }
            val s = shift.toFloat()
            return Schedule(rungs.map { Schedules.shiftSigma(it.toFloat() / 1000.0f, s) } + 0.0f, rule)
        }

        public fun uniform(points: Int, shift: Double): Schedule {
            require(points >= 2) { "a uniform schedule needs >= 2 grid points" }
            val s = shift.toFloat()
            val sig = Schedules.linspaceF32(1.0f, 0.0f, points).map { Schedules.shiftSigma(it, s) }
            val dedup = ArrayList<Float>()
            dedup += sig[0]
            for (i in 1 until sig.size) if (sig[i] != sig[i - 1]) dedup += sig[i]
            return Schedule(dedup, "euler")
        }
    }
}

/** FastH3 Omni's Parallel Decoding Distillation grid and fused-head weights. */
public data class PddPlan(
    val fineGrid: List<Double>,
    val indices: List<Int>,
    val videoSigmas: List<Float>,
    val audioSigmas: List<Float>,
    val videoWeights: List<Double>,
    val audioWeights: List<Double>,
) {
    public fun numSteps(): Int = indices.size - 1

    public fun block(i: Int): Pair<Int, Int> = indices[i] to indices[i + 1]

    public fun schedules(): Pair<Schedule, Schedule> = Schedule(videoSigmas, "euler") to Schedule(audioSigmas, "euler")

    public companion object {
        public fun of(indices: List<Int>, shifts: Shifts): PddPlan {
            require(indices.size >= 2 && indices[0] == 0 && indices.zipWithNext().all { (a, b) -> b > a }) {
                "pdd_step_indices must increase strictly from 0, got $indices"
            }
            val steps = indices.last()
            require(steps >= 2) { "a PDD grid needs >= 2 intervals" }
            val maxT = Schedules.PDD_GRID_MAX_T
            val grid = Schedules.linspaceF64(maxT, 0.0, steps + 1).map { minOf(it, maxT) }
            fun weights(s: Double) = grid.zipWithNext().map { (a, b) -> Schedules.pddShiftDelta(a, b, s, maxT) }
            fun nodes(s: Double) = indices.map { Schedules.pddShift(grid[it], s, maxT).toFloat() }
            val vw = weights(shifts.video)
            val aw = weights(shifts.audio)
            require((vw + aw).all { it.isFinite() && it != 0.0 }) { "PDD integration weights must be finite and non-zero" }
            return PddPlan(grid, indices, nodes(shifts.video), nodes(shifts.audio), vw, aw)
        }

        /** The weights of fine intervals `[start, end)` normalised to sum 1, in f32. */
        public fun fusedWeights(weights: List<Double>, start: Int, end: Int): List<Float> {
            val total = weights.subList(start, end).sum()
            return weights.subList(start, end).map { (it / total).toFloat() }
        }
    }
}
