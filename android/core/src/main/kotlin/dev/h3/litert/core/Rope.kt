package dev.h3.litert.core

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/** `cos` / `sin` of shape `[rows, 3·rope_freq_dim]`: `(t, h, w)` axis angles concatenated. */
public class RopeTable(public val cos: Tensor, public val sin: Tensor) {
    public val rows: Int get() = cos.shape[0]
    public val half: Int get() = cos.shape[1]

    /** Rows `[from, to)` of both tables. */
    public fun slice(from: Int, to: Int): RopeTable = RopeTable(cos.sliceRows(from, to), sin.sliceRows(from, to))
}

/**
 * Rotary tables (`rope.py`; h3-core `rope.rs`): rotate-half over the first `r = 6·rope_freq_dim`
 * channels of a head, `r/2` angles per row. The inverse frequencies are formed in f64 and cast to
 * f32, the positions cast to f32, the angles multiplied in f32 and their cos / sin taken in f32,
 * as the numpy reference.
 */
public object Rope {
    /** `inv[k] = θ^(−2k / 2r)` for `k < rope_freq_dim`, as f32. */
    public fun inverseFrequencies(ropeFreqDim: Int, theta: Double): FloatArray =
        FloatArray(ropeFreqDim) { k -> (1.0 / theta.pow(2.0 * k / (2 * ropeFreqDim))).toFloat() }

    /** DiT table for packed positions `[rows][3]` (f64): `[rows, 3·rope_freq_dim]` cos / sin. */
    public fun h3Rope(positions: Positions, ropeFreqDim: Int, theta: Double): RopeTable {
        val inv = inverseFrequencies(ropeFreqDim, theta)
        val n = positions.rows
        val width = 3 * ropeFreqDim
        val c = FloatArray(n * width)
        val s = FloatArray(n * width)
        for (i in 0 until n) {
            for (axis in 0 until 3) {
                val p = positions.data[i * 3 + axis].toFloat()
                for (k in 0 until ropeFreqDim) {
                    val a = p * inv[k] // f32 product, as numpy
                    val o = i * width + axis * ropeFreqDim + k
                    c[o] = cos(a.toDouble()).toFloat()
                    s[o] = sin(a.toDouble()).toFloat()
                }
            }
        }
        val shape = intArrayOf(n, width)
        return RopeTable(Tensor(c, shape), Tensor(s, shape.copyOf()))
    }

    /** Convenience over a flat `(t, h, w)` list. */
    public fun h3Rope(positions: List<Triple<Double, Double, Double>>, ropeFreqDim: Int, theta: Double): RopeTable {
        val p = Positions(positions.size)
        for ((i, v) in positions.withIndex()) p.set(i, v.first, v.second, v.third)
        return h3Rope(p, ropeFreqDim, theta)
    }
}
