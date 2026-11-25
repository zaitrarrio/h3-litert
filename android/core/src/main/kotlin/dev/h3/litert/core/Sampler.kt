package dev.h3.litert.core

/**
 * One denoising step per modality (`sampler.py`; h3-core `backend::FlowStep`;
 * `MiniMaxH3Scheduler.step`). Every arithmetic step is single precision in the reference's order.
 */
public object Sampler {
    /**
     * `x0 = x + σ·v`; Euler / PDD: `(σ'/σ)·x + (1 − σ'/σ)·x0`; renoise: `(1 − σ')·x0 + σ'·ε`
     * (`x0` when `σ' = 0`).
     */
    public fun flowStep(x: Tensor, v: Tensor, sigma: Float, next: Float, rule: String, noise: Tensor?): Tensor {
        require(x.size == v.size) { "flowStep: x ${x.shapeString()} vs v ${v.shapeString()}" }
        val xd = x.data
        val vd = v.data
        val out = FloatArray(x.size)
        when (rule) {
            "euler", "pdd" -> {
                val r = next / sigma
                val oneMinus = 1.0f - r
                for (i in out.indices) {
                    val x0 = xd[i] + sigma * vd[i]
                    out[i] = r * xd[i] + oneMinus * x0
                }
            }
            "renoise" -> {
                if (next == 0.0f) {
                    for (i in out.indices) out[i] = xd[i] + sigma * vd[i]
                } else {
                    requireNotNull(noise) { "renoise needs noise when σ' != 0" }
                    require(noise.size == x.size) { "flowStep: noise ${noise.shapeString()} vs x ${x.shapeString()}" }
                    val nd = noise.data
                    val oneMinus = 1.0f - next
                    for (i in out.indices) {
                        val x0 = xd[i] + sigma * vd[i]
                        out[i] = oneMinus * x0 + next * nd[i]
                    }
                }
            }
            else -> throw IllegalArgumentException("sampler rule `$rule` is not euler, renoise or pdd")
        }
        return Tensor(out, x.shape.copyOf())
    }

    public fun needsNoise(rule: String, next: Float): Boolean = rule == "renoise" && next != 0.0f
}
