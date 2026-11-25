// Deterministic standard-normal noise (python rng.py; h3-core `rng.rs`, the same
// generator as fastwan-rs): SplitMix64 → Box–Muller, bit for bit the same
// stream on every platform, so the same seed gives the same latent in Rust,
// Python, TypeScript, Kotlin and Swift.

import Foundation

public struct Rng {
    static let golden: UInt64 = 0x9E37_79B9_7F4A_7C15
    static let m1: UInt64 = 0xBF58_476D_1CE4_E5B9
    static let m2: UInt64 = 0x94D0_49BB_1331_11EB

    public private(set) var state: UInt64

    public init(seed: UInt64) {
        state = seed ^ Rng.golden
    }

    public mutating func nextU64() -> UInt64 {
        state = state &+ Rng.golden
        var z = state
        z = (z ^ (z >> 30)) &* Rng.m1
        z = (z ^ (z >> 27)) &* Rng.m2
        return z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`.
    public mutating func unit() -> Double {
        Double((nextU64() >> 11) + 1) / 9007199254740992.0  // 2^53
    }

    /// `n` standard normals as f32 (Box–Muller in f64, pairs).
    public mutating func normal(_ n: Int) -> [Float] {
        let pairs = (n + 1) / 2
        var out = [Float](repeating: 0, count: 2 * pairs)
        var i = 0
        for _ in 0..<pairs {
            let u1 = unit()
            let u2 = unit()
            let r = (-2.0 * log(u1)).squareRoot()
            let th = (2.0 * Double.pi) * u2
            out[i] = Float(r * cos(th))
            out[i + 1] = Float(r * sin(th))
            i += 2
        }
        if out.count != n { out.removeLast(out.count - n) }
        return out
    }

    /// `n` standard normals shaped as a tensor.
    public mutating func normal(shape: [Int]) -> Tensor {
        Tensor(shape: shape, data: normal(shape.reduce(1, *)))
    }

    /// Uniform in `[-a, a)`; for synthetic weights.
    public mutating func uniform(_ n: Int, _ a: Float) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let u = unit()
            out[i] = Float(u * 2.0 - 1.0) * a
        }
        return out
    }
}
