package dev.h3.litert.core

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Deterministic standard-normal noise (`rng.py`; h3-core `rng.rs`, the same generator as fastwan-rs): SplitMix64 → Box–Muller,
 * bit for bit the same stream on every platform, so one seed gives one latent in Rust, Python,
 * TypeScript, Kotlin and Swift.
 */
public class Rng(seed: Long) {
    private var state: ULong = seed.toULong() xor GOLDEN

    public fun nextU64(): ULong {
        state += GOLDEN
        var z = state
        z = (z xor (z shr 30)) * M1
        z = (z xor (z shr 27)) * M2
        return z xor (z shr 31)
    }

    /** Uniform in `(0, 1]`. */
    public fun unit(): Double = ((nextU64() shr 11) + 1uL).toDouble() / TWO_53

    /** `n` standard normals as f32 (Box–Muller in f64, pairs; the odd tail draw is discarded). */
    public fun normal(n: Int): FloatArray {
        val out = FloatArray(n)
        var i = 0
        while (i < n) {
            val u1 = unit()
            val u2 = unit()
            val r = sqrt(-2.0 * ln(u1))
            val th = (2.0 * PI) * u2
            out[i] = (r * cos(th)).toFloat()
            if (i + 1 < n) out[i + 1] = (r * sin(th)).toFloat()
            i += 2
        }
        return out
    }

    /** `n` normals shaped as a tensor. */
    public fun normal(vararg shape: Int): Tensor = Tensor(normal(Tensor.numel(shape)), shape)

    /** Uniform in `[-a, a)` as f32; for synthetic weights. */
    public fun uniform(n: Int, a: Float): FloatArray = FloatArray(n) {
        ((unit() * 2.0 - 1.0).toFloat()) * a
    }

    public companion object {
        /** `Rng(seed)` from an unsigned seed (`Long` and `ULong` share one JVM signature). */
        public operator fun invoke(seed: ULong): Rng = Rng(seed.toLong())

        public const val GOLDEN: ULong = 0x9E3779B97F4A7C15uL
        private const val M1: ULong = 0xBF58476D1CE4E5B9uL
        private const val M2: ULong = 0x94D049BB133111EBuL
        private const val TWO_53: Double = 9007199254740992.0
    }
}
