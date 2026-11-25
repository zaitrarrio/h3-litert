// One denoising step per modality (python sampler.py; h3-core
// `backend::FlowStep`; `MiniMaxH3Scheduler.step`). Float32 arithmetic, as numpy.

import Foundation

public enum Sampler {
    /// `x0 = x + σ·v`; Euler (and PDD) `(σ'/σ)·x + (1 − σ'/σ)·x0`; renoise `(1 − σ')·x0 + σ'·ε` (`x0` when σ' = 0).
    public static func flowStep(x: Tensor, v: Tensor, sigma: Double, next: Double, rule: SamplerRule, noise: Tensor?) throws -> Tensor {
        guard x.shape == v.shape else {
            throw H3Error.shape("flow step: x is \(x.shape), v is \(v.shape)")
        }
        let s = Float(sigma), n = Float(next)
        let xd = x.data, vd = v.data
        var out = [Float](repeating: 0, count: xd.count)
        switch rule {
        case .euler, .pdd:
            let r = n / s
            let oneMinus = Float(1.0) - r
            for i in 0..<out.count {
                let x0 = xd[i] + s * vd[i]
                out[i] = r * xd[i] + oneMinus * x0
            }
        case .renoise:
            if n == 0.0 {
                for i in 0..<out.count {
                    out[i] = xd[i] + s * vd[i]
                }
            } else {
                guard let nz = noise, nz.shape == x.shape else {
                    throw H3Error.shape("renoise step needs noise of shape \(x.shape)")
                }
                let nd = nz.data
                let oneMinus = Float(1.0) - n
                for i in 0..<out.count {
                    let x0 = xd[i] + s * vd[i]
                    out[i] = oneMinus * x0 + n * nd[i]
                }
            }
        }
        return Tensor(shape: x.shape, data: out)
    }

    /// Only the renoise rule draws fresh noise, and not on its last step.
    public static func needsNoise(rule: SamplerRule, next: Double) -> Bool {
        rule == .renoise && next != 0.0
    }
}
