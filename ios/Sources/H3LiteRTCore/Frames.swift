// What the pipeline hands out: decoded frames as RGBA bytes (ready for a
// `CGImage`), the whole clip, the stereo waveform, and the 8-bit conversions
// with the reference's rounding (`(clip(v, 0, 1)·255 + 0.5) as u8`).

import Foundation

/// One block of decoded frames, RGBA row-major `[count, height, width, 4]` with alpha 255.
public struct Frames {
    public var first: Int
    public var count: Int
    public var width: Int
    public var height: Int
    public var rgba: [UInt8]

    public init(first: Int, count: Int, width: Int, height: Int, rgba: [UInt8]) {
        self.first = first
        self.count = count
        self.width = width
        self.height = height
        self.rgba = rgba
    }

    public var bytesPerFrame: Int { width * height * 4 }

    public func frame(_ i: Int) -> ArraySlice<UInt8> {
        rgba[(i * bytesPerFrame)..<((i + 1) * bytesPerFrame)]
    }

    /// `[count, height, width, 3]`, the Python layout, for comparisons against `expected.safetensors`.
    public var rgb: [UInt8] { Pixels.rgbaToRgb(rgba) }
}

/// The decoded clip, `[frames, height, width, 4]` RGBA.
public struct Video {
    public var frames: Int
    public var width: Int
    public var height: Int
    public var rgba: [UInt8]

    public init(frames: Int, width: Int, height: Int, rgba: [UInt8]) {
        self.frames = frames
        self.width = width
        self.height = height
        self.rgba = rgba
    }

    public var bytesPerFrame: Int { width * height * 4 }

    public func frame(_ i: Int) -> ArraySlice<UInt8> {
        rgba[(i * bytesPerFrame)..<((i + 1) * bytesPerFrame)]
    }

    public var rgb: [UInt8] { Pixels.rgbaToRgb(rgba) }
}

/// The decoded waveform: `channels × samples` floats in `[−1, 1]`, channel-major.
public struct Audio {
    public var sampleRate: Int
    public var channels: Int
    public var samples: Int
    public var data: [Float]

    public init(sampleRate: Int, channels: Int, samples: Int, data: [Float]) {
        precondition(data.count == channels * samples, "audio data \(data.count) for \(channels)x\(samples)")
        self.sampleRate = sampleRate
        self.channels = channels
        self.samples = samples
        self.data = data
    }

    public func channel(_ c: Int) -> ArraySlice<Float> {
        data[(c * samples)..<((c + 1) * samples)]
    }

    public var tensor: Tensor { Tensor(shape: [channels, samples], data: data) }

    /// 16-bit PCM WAV bytes (interleaved), for saving or sharing.
    public func wav() -> Data {
        var pcm = Data(capacity: channels * samples * 2)
        for i in 0..<samples {
            for c in 0..<channels {
                let v = min(max(data[c * samples + i], -1.0), 1.0)
                var s = Int16(v * 32767.0).littleEndian
                withUnsafeBytes(of: &s) { pcm.append(contentsOf: $0) }
            }
        }
        var out = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { out.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { out.append(contentsOf: $0) } }
        out.append(contentsOf: Array("RIFF".utf8))
        u32(UInt32(36 + pcm.count))
        out.append(contentsOf: Array("WAVEfmt ".utf8))
        u32(16)
        u16(1)
        u16(UInt16(channels))
        u32(UInt32(sampleRate))
        u32(UInt32(sampleRate * channels * 2))
        u16(UInt16(channels * 2))
        u16(16)
        out.append(contentsOf: Array("data".utf8))
        u32(UInt32(pcm.count))
        out.append(pcm)
        return out
    }
}

public enum Pixels {
    /// numpy: clip to [0, 1] (NaN → 0 here), ·255 + 0.5, truncate.
    @inline(__always)
    static func quantize(_ v: Float) -> UInt8 {
        if v.isNaN { return 0 }
        let c = min(max(v, 0.0), 1.0)
        return UInt8(c * 255.0 + 0.5)  // in [0.5, 255.5]; truncation never exceeds 255
    }

    /// `[3, F, H, W]` in `[0, 1]` (the VAE decoder's layout) → `[F, H, W, 4]` RGBA.
    public static func rgbaFromChannelMajor(_ px: Tensor) -> (rgba: [UInt8], frames: Int, height: Int, width: Int) {
        precondition(px.rank == 4 && px.shape[0] == 3, "rgbaFromChannelMajor wants [3, F, H, W], got \(px.shape)")
        let f = px.shape[1], h = px.shape[2], w = px.shape[3]
        let plane = h * w
        let frameStride = f * plane
        var out = [UInt8](repeating: 255, count: f * plane * 4)
        let src = px.data
        for fi in 0..<f {
            let oBase = fi * plane * 4
            for p in 0..<plane {
                let s = fi * plane + p
                out[oBase + p * 4] = quantize(src[s])
                out[oBase + p * 4 + 1] = quantize(src[frameStride + s])
                out[oBase + p * 4 + 2] = quantize(src[2 * frameStride + s])
            }
        }
        return (out, f, h, w)
    }

    /// `[F, 3, H, W]` in `[0, 1]` (TAEH3's layout) → `[F, H, W, 4]` RGBA.
    public static func rgbaFromFrameMajor(_ px: Tensor) -> (rgba: [UInt8], frames: Int, height: Int, width: Int) {
        precondition(px.rank == 4 && px.shape[1] == 3, "rgbaFromFrameMajor wants [F, 3, H, W], got \(px.shape)")
        let f = px.shape[0], h = px.shape[2], w = px.shape[3]
        let plane = h * w
        var out = [UInt8](repeating: 255, count: f * plane * 4)
        let src = px.data
        for fi in 0..<f {
            let fBase = fi * 3 * plane
            let oBase = fi * plane * 4
            for p in 0..<plane {
                for c in 0..<3 {
                    out[oBase + p * 4 + c] = quantize(src[fBase + c * plane + p])
                }
            }
        }
        return (out, f, h, w)
    }

    /// `[F, H, W, 4]` RGBA → `[F, H, W, 3]` RGB.
    public static func rgbaToRgb(_ rgba: [UInt8]) -> [UInt8] {
        let pixels = rgba.count / 4
        var out = [UInt8](repeating: 0, count: pixels * 3)
        for p in 0..<pixels {
            out[p * 3] = rgba[p * 4]
            out[p * 3 + 1] = rgba[p * 4 + 1]
            out[p * 3 + 2] = rgba[p * 4 + 2]
        }
        return out
    }

    /// One RGBA (or RGBX) image `[H, W, 4]` → `[3, 1, H, W]` floats in `[0, 1]`: a keyframe for FL2VA.
    public static func keyframePixels(rgba: [UInt8], width: Int, height: Int) -> Tensor {
        precondition(rgba.count == width * height * 4, "keyframe bytes \(rgba.count) for \(width)x\(height) RGBA")
        let plane = width * height
        var out = [Float](repeating: 0, count: 3 * plane)
        for p in 0..<plane {
            for c in 0..<3 {
                out[c * plane + p] = Float(rgba[p * 4 + c]) / 255.0
            }
        }
        return Tensor(shape: [3, 1, height, width], data: out)
    }
}
