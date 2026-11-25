// Pixel ↔ latent ↔ audio-latent geometry of one H3 request (python
// packing.py; h3-core `packing.rs`): canvas rules, `17n + 5` frames,
// `5n + 2` latent frames, audio latents at 40 Hz with Python's half-even rounding.

import Foundation

public let fps = 24
public let shortEdge = 768
public let maxPixels = 768 * 1344
public let canvasMultiple = 32
public let minAspect = 0.25
public let maxAspect = 4.0
public let framesPerChunk = 17
public let latentsPerChunk = 5
public let audioLatentsPerSecond = 40
public let audioChannels = 2
public let minAlignedFrames = 124
public let maxAlignedFrames = 362
public let continuationFrames = [5, 22, 39]

/// Canvas `(height, width)` for an aspect ratio, snapped to the 32-pixel lattice.
public func canvasSize(aspectW: Double, aspectH: Double, shortEdge se: Int = shortEdge, maxPixels mp: Int = maxPixels) throws -> (height: Int, width: Int) {
    if aspectW <= 0 || aspectH <= 0 {
        throw H3Error.invalidGeometry("aspect ratio must be positive, got \(aspectW):\(aspectH)")
    }
    let ratio = aspectW / aspectH
    if !(minAspect <= ratio && ratio <= maxAspect) {
        throw H3Error.invalidGeometry("aspect ratios run from 1:4 to 4:1, got \(aspectW):\(aspectH)")
    }
    var w: Double, h: Double
    if ratio >= 1.0 {
        w = Double(se) * ratio
        h = Double(se)
    } else {
        w = Double(se)
        h = Double(se) / ratio
    }
    let area = w * h
    if area > Double(mp) {
        let s = (Double(mp) / area).squareRoot()
        w *= s
        h *= s
    }
    let m = Double(canvasMultiple)
    func snap(_ v: Double) -> Int { Int(max((v / m).rounded(.toNearestOrEven) * m, m)) }
    return (snap(h), snap(w))
}

/// The next `17n + 5` at or above `n`.
public func alignNumFrames(_ n: Int) -> Int {
    var n = max(n, 1)
    while n % framesPerChunk != latentsPerChunk {
        n += 1
    }
    return n
}

public func resolveNumFrames(_ n: Int) throws -> Int {
    let a = alignNumFrames(n)
    if !(minAlignedFrames <= a && a <= maxAlignedFrames) {
        throw H3Error.invalidGeometry("H3 generates 5-15 s at 24 fps; aligned num_frames \(a) is outside 124-362")
    }
    return a
}

/// `5n + 2` latent frames of `17n + 5` pixel frames.
public func videoLatentFrames(_ n: Int) throws -> Int {
    if n % framesPerChunk != latentsPerChunk {
        throw H3Error.invalidGeometry("num_frames must be 17n + 5, got \(n)")
    }
    return (n - latentsPerChunk) / framesPerChunk * latentsPerChunk + 2
}

/// `round(frames / 24 · 40)` with Python's half-even rounding.
public func audioLatents(frames: Int) -> Int {
    Int((Double(frames) / Double(fps) * Double(audioLatentsPerSecond)).rounded(.toNearestOrEven))
}

public struct Geometry: Codable, Equatable, Hashable, Sendable, CustomStringConvertible {
    public var width: Int
    public var height: Int
    public var frames: Int

    public init(width: Int, height: Int, frames: Int) {
        self.width = width
        self.height = height
        self.frames = frames
    }

    public func validate(spatialRatio: Int) throws {
        if width <= 0 || height <= 0 || width % canvasMultiple != 0 || height % canvasMultiple != 0 {
            throw H3Error.invalidGeometry("\(width)x\(height) is not a positive multiple of 32")
        }
        if width % spatialRatio != 0 || height % spatialRatio != 0 {
            throw H3Error.invalidGeometry("\(width)x\(height) is not divisible by the VAE ratio \(spatialRatio)")
        }
        _ = try videoLatentFrames(frames)
    }

    /// Latent `(T, H, W)` at the VAE's spatial ratio.
    public func latent(spatialRatio: Int) throws -> (t: Int, h: Int, w: Int) {
        let t = try videoLatentFrames(frames)
        return (t, height / spatialRatio, width / spatialRatio)
    }

    public var audioLatents: Int { H3LiteRTCore.audioLatents(frames: frames) }

    /// `WxHxF`, e.g. `1344x768x124`.
    public static func parse(_ s: String) throws -> Geometry {
        let parts = s.lowercased().split(separator: "x").map { Int($0) }
        guard parts.count == 3, let w = parts[0], let h = parts[1], let f = parts[2] else {
            throw H3Error.invalidGeometry("geometry `\(s)` is not WIDTHxHEIGHTxFRAMES")
        }
        return Geometry(width: w, height: h, frames: f)
    }

    public var description: String { "\(width)x\(height)x\(frames)" }

    /// H3's served 768p canvas, 5 s at 24 fps.
    public static let h3_768p = Geometry(width: 1344, height: 768, frames: 124)
    public static let h3_480p = Geometry(width: 832, height: 480, frames: 124)
    /// The tiny fixture: latent `(7, 4, 6)`, 37 audio latents.
    public static let tiny = Geometry(width: 96, height: 64, frames: 22)
}
