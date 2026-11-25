// Rotary tables (python rope.py; h3-core `rope.rs`): rotate-half over the
// first `r = 6 · rope_freq_dim` channels of a head, `r/2` angles per row,
// formed in f32 as the reference.

import Foundation

public enum Rope {
    /// DiT table for packed positions `[rows][3]` (flat, `t h w t h w …`):
    /// `(cos, sin)` of shape `[rows, 3 · rope_freq_dim]`, one axis after another.
    public static func h3Rope(positions: [Float], ropeFreqDim: Int, theta: Double) -> (cos: Tensor, sin: Tensor) {
        precondition(positions.count % 3 == 0, "positions are \(positions.count) floats, not rows of 3")
        let rows = positions.count / 3
        // `1 / θ^(2k / 2r)` in f64, then f32 (numpy's `astype`).
        let inv: [Float] = (0..<ropeFreqDim).map { k in
            Float(1.0 / pow(theta, 2.0 * Double(k) / Double(2 * ropeFreqDim)))
        }
        let width = 3 * ropeFreqDim
        var cosv = [Float](repeating: 0, count: rows * width)
        var sinv = [Float](repeating: 0, count: rows * width)
        for r in 0..<rows {
            for axis in 0..<3 {
                let p = positions[r * 3 + axis]
                for k in 0..<ropeFreqDim {
                    let a = p * inv[k]
                    let j = r * width + axis * ropeFreqDim + k
                    cosv[j] = cos(a)
                    sinv[j] = sin(a)
                }
            }
        }
        return (Tensor(shape: [rows, width], data: cosv), Tensor(shape: [rows, width], data: sinv))
    }

    /// The same for `Position`s (the layout's f64 positions cast to f32 first, as `prepare` does).
    public static func h3Rope(positions: [Position], ropeFreqDim: Int, theta: Double) -> (cos: Tensor, sin: Tensor) {
        var flat = [Float](repeating: 0, count: positions.count * 3)
        for (i, p) in positions.enumerated() {
            flat[3 * i] = Float(p.t)
            flat[3 * i + 1] = Float(p.h)
            flat[3 * i + 2] = Float(p.w)
        }
        return h3Rope(positions: flat, ropeFreqDim: ropeFreqDim, theta: theta)
    }

    /// The ViT decoder's table (a constant inside `vae_dec_in`; here for the tests):
    /// `rot` rotated channels, three axes over length-normalised `[−1, 1)` grids,
    /// `(t, h, w)` token order, then `suffix` rows at position 0.
    public static func vaeDecoderRope(grid: (t: Int, h: Int, w: Int), rot: Int, theta: Double, suffix: Int) -> (cos: Tensor, sin: Tensor) {
        let perAxis = rot / 6
        let inv: [Double] = (0..<perAxis).map { k in 1.0 / pow(theta, Double(k) * 6.0 / Double(rot)) }
        func coords(_ n: Int) -> [Double] { (0..<n).map { 2.0 * ((Double($0) + 0.5) / Double(n)) - 1.0 } }
        let ct = coords(grid.t), ch = coords(grid.h), cw = coords(grid.w)
        var p: [[Double]] = []
        for t in ct {
            for h in ch {
                for w in cw {
                    p.append([t, h, w])
                }
            }
        }
        for _ in 0..<suffix {
            p.append([0.0, 0.0, 0.0])
        }
        let width = 3 * perAxis
        var cosv = [Float](repeating: 0, count: p.count * width)
        var sinv = [Float](repeating: 0, count: p.count * width)
        for (r, row) in p.enumerated() {
            for axis in 0..<3 {
                for k in 0..<perAxis {
                    let a = Float(2.0 * Double.pi * row[axis] * inv[k])
                    let j = r * width + axis * perAxis + k
                    cosv[j] = cos(a)
                    sinv[j] = sin(a)
                }
            }
        }
        return (Tensor(shape: [p.count, width], data: cosv), Tensor(shape: [p.count, width], data: sinv))
    }
}
