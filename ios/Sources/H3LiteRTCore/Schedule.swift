// Sampling schedules (python schedule.py; h3-core `schedule.rs`; FastVideo
// `scheduling_minimax_h3.py`, `layers/pdd.py`). Clean-time clock `t = 1 − σ`;
// `x0 = x + σ·v`. Every sigma is computed in f32 as the reference does. A
// runtime reads the precomputed `sampler.steps` of the manifest; these are the
// formulas the exporter used, kept so the tests pin them (the LongLive-Plug
// grid equals `MiniMaxH3Scheduler.set_timesteps(5)` to the last bit).

import Foundation

/// `σ = s·b / (1 + (s − 1)·b)` in f32.
public func shiftSigma(base: Float, shift: Float) -> Float {
    (shift * base) / (Float(1.0) + (shift - Float(1.0)) * base)
}

/// `torch.linspace` in f32: symmetric fill from both ends.
public func linspaceF32(start: Float, end: Float, n: Int) -> [Float] {
    if n == 1 { return [start] }
    let step = (end - start) / Float(n - 1)
    var out: [Float] = []
    out.reserveCapacity(n)
    for i in 0..<n {
        let v: Float = i < n / 2 ? start + step * Float(i) : end - step * Float(n - 1 - i)
        out.append(v)
    }
    return out
}

public func linspaceF64(start: Double, end: Double, n: Int) -> [Double] {
    if n == 1 { return [start] }
    let step = (end - start) / Double(n - 1)
    return (0..<n).map { i in i < n / 2 ? start + step * Double(i) : end - step * Double(n - 1 - i) }
}

public struct Shifts: Equatable, Codable, Sendable {
    public var video: Double
    public var audio: Double

    public init(video: Double, audio: Double) {
        self.video = video
        self.audio = audio
    }

    public static let h3 = Shifts(video: 12.0, audio: 3.0)
    public static let fasth3V2 = Shifts(video: 10.0, audio: 3.0)
}

public enum SamplerRule: String, Codable, Equatable, Sendable {
    case euler
    case renoise
    case pdd
}

/// One modality's sigma grid: `steps + 1` strictly decreasing f32 values ending at 0.
public struct Schedule: Equatable {
    public var sigmas: [Float]
    public var rule: SamplerRule

    public init(sigmas: [Float], rule: SamplerRule) throws {
        guard sigmas.count >= 2, sigmas.last == 0.0, zip(sigmas, sigmas.dropFirst()).allSatisfy({ $1 < $0 }) else {
            throw H3Error.manifest("sigmas must hold >= 2 strictly decreasing values ending at 0, got \(sigmas)")
        }
        guard rule != .pdd else {
            throw H3Error.manifest("rule pdd is not euler or renoise")
        }
        self.sigmas = sigmas
        self.rule = rule
    }

    /// Rungs in `(0, 1000]`, strictly decreasing: `σ_i = shift(r_i / 1000)`, then 0.
    public static func fromRungs(_ rungs: [Int], shift: Double, rule: SamplerRule = .euler) throws -> Schedule {
        guard !rungs.isEmpty, rungs.allSatisfy({ $0 > 0 && $0 <= 1000 }), zip(rungs, rungs.dropFirst()).allSatisfy({ $1 < $0 }) else {
            throw H3Error.manifest("rungs must be strictly decreasing in (0, 1000], got \(rungs)")
        }
        var sig = rungs.map { shiftSigma(base: Float($0) / Float(1000.0), shift: Float(shift)) }
        sig.append(0.0)
        return try Schedule(sigmas: sig, rule: rule)
    }

    /// `points` f32-linspaced base sigmas from 1 to 0, shifted, consecutive duplicates merged; Euler.
    public static func uniform(points: Int, shift: Double) throws -> Schedule {
        guard points >= 2 else { throw H3Error.manifest("a uniform schedule needs >= 2 grid points") }
        let sig = linspaceF32(start: 1.0, end: 0.0, n: points).map { shiftSigma(base: $0, shift: Float(shift)) }
        var dedup = [sig[0]]
        for i in 1..<sig.count where sig[i] != sig[i - 1] {
            dedup.append(sig[i])
        }
        return try Schedule(sigmas: dedup, rule: .euler)
    }

    public var steps: Int { sigmas.count - 1 }

    /// The clean time of step `i`: `1 − σ_i` in f32.
    public func timestep(_ i: Int) -> Float { Float(1.0) - sigmas[i] }
}

public let fasth3V1Rungs = [999, 749, 500, 250]
public let fasth3V2Rungs = [999, 874, 749, 624, 500, 375, 250, 125]
public let pikaKnots = [1000, 750, 500, 250, 100, 40]
public let plug4Knots = [1000, 750, 500, 250]
public let h3BasePoints = 50
public let keyframeNoiseAug: Float = 0.999
public let keyframeEncodeSeed: UInt64 = 42

// MARK: Parallel Decoding Distillation (FastH3 Omni)

public let pddGridMaxT = 0.999

public func pddShift(_ u: Double, shift: Double, maxT: Double) -> Double {
    shift == 1.0 ? u : shift * u * maxT / (u * (shift - 1.0) + maxT)
}

public func pddShiftDelta(_ a: Double, _ b: Double, shift: Double, maxT: Double) -> Double {
    if shift == 1.0 { return b - a }
    let da = maxT + (shift - 1.0) * a
    let db = maxT + (shift - 1.0) * b
    return shift * maxT * maxT * (b - a) / (da * db)
}

public struct PddPlan: Equatable {
    public var fineGrid: [Double]
    public var indices: [Int]
    public var videoSigmas: [Float]
    public var audioSigmas: [Float]
    public var videoWeights: [Double]
    public var audioWeights: [Double]

    public init(indices: [Int], shifts: Shifts) throws {
        guard indices.count >= 2, indices[0] == 0, zip(indices, indices.dropFirst()).allSatisfy({ $1 > $0 }) else {
            throw H3Error.manifest("pdd_step_indices must increase strictly from 0, got \(indices)")
        }
        let steps = indices[indices.count - 1]
        guard steps >= 2 else { throw H3Error.manifest("a PDD grid needs >= 2 intervals") }
        let grid = linspaceF64(start: pddGridMaxT, end: 0.0, n: steps + 1).map { min($0, pddGridMaxT) }
        func weights(_ s: Double) -> [Double] {
            zip(grid, grid.dropFirst()).map { pddShiftDelta($0, $1, shift: s, maxT: pddGridMaxT) }
        }
        func nodes(_ s: Double) -> [Float] {
            indices.map { Float(pddShift(grid[$0], shift: s, maxT: pddGridMaxT)) }
        }
        let vw = weights(shifts.video), aw = weights(shifts.audio)
        guard (vw + aw).allSatisfy({ $0.isFinite && $0 != 0.0 }) else {
            throw H3Error.manifest("PDD integration weights must be finite and non-zero")
        }
        fineGrid = grid
        self.indices = indices
        videoSigmas = nodes(shifts.video)
        audioSigmas = nodes(shifts.audio)
        videoWeights = vw
        audioWeights = aw
    }

    public var numSteps: Int { indices.count - 1 }

    public func block(_ i: Int) -> (start: Int, end: Int) { (indices[i], indices[i + 1]) }

    /// The fused heads' weights of one step, normalised to sum 1 (f32).
    public static func fusedWeights(_ weights: [Double], start: Int, end: Int) -> [Float] {
        let total = weights[start..<end].reduce(0.0, +)
        return weights[start..<end].map { Float($0 / total) }
    }

    public func schedules() throws -> (video: Schedule, audio: Schedule) {
        (try Schedule(sigmas: videoSigmas, rule: .euler), try Schedule(sigmas: audioSigmas, rule: .euler))
    }
}
