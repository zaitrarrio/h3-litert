// Timestep conditioning (python timestep.py; h3-core `timestep.rs`): the
// sinusoid `Timesteps(256, flip_sin_to_cos=True)` of a clean time. The
// exporter feeds it through the time embedder and every block's `adaln_proj`
// once and ships the result as `adaln.safetensors`; the runtime only looks the
// rows up. Kept for the tests that pin the host math.

import Foundation

public enum Timestep {
    /// `[cos(t·f) | sin(t·f)]`, `f_i = exp(−ln 10⁴ · i / half)`, in f32.
    public static func sinusoid(_ t: Float, dim: Int) -> [Float] {
        let half = dim / 2
        let lnTheta = Float(log(10000.0))
        var out = [Float](repeating: 0, count: 2 * half)
        for i in 0..<half {
            let f = exp(-lnTheta * Float(i) / Float(half))
            let a = t * f
            out[i] = cos(a)
            out[half + i] = sin(a)
        }
        return out
    }

    public static func silu(_ x: Float) -> Float {
        x / (1.0 + exp(-x))
    }
}
