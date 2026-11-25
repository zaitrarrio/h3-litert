// A dense row-major float32 tensor: `[Float]` plus a shape. Every graph input
// and output in the contract is one of these (no batch dimension). The axis
// helpers (`slice`, `concat`, `repeated`) serve the VAE host side, which cuts
// and blends `[C, T, H, W]` pixel and latent blocks along any axis.

import Foundation

public struct Tensor: Equatable {
    public var shape: [Int]
    public var data: [Float]

    public init(shape: [Int], data: [Float]) {
        precondition(shape.reduce(1, *) == data.count, "tensor data \(data.count) does not match shape \(shape)")
        self.shape = shape
        self.data = data
    }

    public init(scalar: Float) {
        self.init(shape: [1], data: [scalar])
    }

    public static func zeros(_ shape: [Int]) -> Tensor {
        Tensor(shape: shape, data: [Float](repeating: 0, count: shape.reduce(1, *)))
    }

    public var count: Int { data.count }

    public var rank: Int { shape.count }

    /// Elements per row (everything but the first axis).
    public var rowStride: Int { shape.dropFirst().reduce(1, *) }

    /// The first axis.
    public var rows: Int { shape.first ?? 1 }

    public func reshaped(_ newShape: [Int]) -> Tensor {
        precondition(newShape.reduce(1, *) == data.count, "cannot reshape \(shape) to \(newShape)")
        return Tensor(shape: newShape, data: data)
    }

    /// Rows `r` of the first axis (the shape keeps its trailing dims).
    public func rows(_ r: Range<Int>) -> Tensor {
        precondition(r.lowerBound >= 0 && r.upperBound <= rows, "rows \(r) out of \(rows)")
        let s = rowStride
        let slice = Array(data[(r.lowerBound * s)..<(r.upperBound * s)])
        return Tensor(shape: [r.count] + Array(shape.dropFirst()), data: slice)
    }

    /// Row `i` of the first axis with that axis dropped: `[a, b, c][i]` is `[b, c]`.
    public func row(_ i: Int) -> Tensor {
        precondition(i >= 0 && i < rows, "row \(i) out of \(rows)")
        let s = rowStride
        return Tensor(shape: Array(shape.dropFirst()), data: Array(data[(i * s)..<((i + 1) * s)]))
    }

    /// Overwrite rows `r` of the first axis with `t` (same trailing dims, `t.rows == r.count`).
    public mutating func setRows(_ r: Range<Int>, _ t: Tensor) {
        precondition(r.lowerBound >= 0 && r.upperBound <= rows, "rows \(r) out of \(rows)")
        precondition(t.rows == r.count && Array(t.shape.dropFirst()) == Array(shape.dropFirst()), "setRows \(t.shape) into \(shape) rows \(r)")
        let s = rowStride
        data.replaceSubrange((r.lowerBound * s)..<(r.upperBound * s), with: t.data)
    }

    /// Concatenate along the first axis; the trailing dims must agree.
    public static func concatRows(_ parts: [Tensor]) -> Tensor {
        precondition(!parts.isEmpty, "concat of nothing")
        let tail = Array(parts[0].shape.dropFirst())
        var total = 0
        var data: [Float] = []
        data.reserveCapacity(parts.reduce(0) { $0 + $1.count })
        for p in parts {
            precondition(Array(p.shape.dropFirst()) == tail, "concat shapes \(parts[0].shape) and \(p.shape)")
            total += p.rows
            data.append(contentsOf: p.data)
        }
        return Tensor(shape: [total] + tail, data: data)
    }

    public subscript(flat i: Int) -> Float {
        get { data[i] }
        set { data[i] = newValue }
    }

    public func shapeEquals(_ other: [Int]) -> Bool { shape == other }

    // MARK: axis helpers

    /// A negative axis counts from the end (numpy); the result is in `0..<rank`.
    func normalized(axis: Int) -> Int {
        let a = axis < 0 ? axis + rank : axis
        precondition(a >= 0 && a < rank, "axis \(axis) out of rank \(rank)")
        return a
    }

    /// `(outer, n, inner)`: elements before the axis, the axis length, elements after it.
    func strides(axis a: Int) -> (outer: Int, n: Int, inner: Int) {
        let outer = shape[0..<a].reduce(1, *)
        let inner = shape[(a + 1)...].reduce(1, *)
        return (outer, shape[a], inner)
    }

    /// `x[..., r, ...]` along `axis` (numpy slicing).
    public func slice(axis: Int, _ r: Range<Int>) -> Tensor {
        let a = normalized(axis: axis)
        let (outer, n, inner) = strides(axis: a)
        precondition(r.lowerBound >= 0 && r.upperBound <= n, "slice \(r) out of \(n) along axis \(a) of \(shape)")
        var newShape = shape
        newShape[a] = r.count
        var out = [Float](repeating: 0, count: outer * r.count * inner)
        var dst = 0
        for o in 0..<outer {
            let base = (o * n + r.lowerBound) * inner
            let len = r.count * inner
            out.replaceSubrange(dst..<(dst + len), with: data[base..<(base + len)])
            dst += len
        }
        return Tensor(shape: newShape, data: out)
    }

    /// Concatenate along `axis`; every other dim must agree.
    public static func concat(_ parts: [Tensor], axis: Int) -> Tensor {
        precondition(!parts.isEmpty, "concat of nothing")
        let a = parts[0].normalized(axis: axis)
        var newShape = parts[0].shape
        var total = 0
        for p in parts {
            precondition(p.rank == parts[0].rank, "concat ranks \(parts[0].shape) and \(p.shape)")
            for d in 0..<p.rank where d != a {
                precondition(p.shape[d] == parts[0].shape[d], "concat shapes \(parts[0].shape) and \(p.shape) along axis \(a)")
            }
            total += p.shape[a]
        }
        newShape[a] = total
        let (outer, _, inner) = parts[0].strides(axis: a)
        var out: [Float] = []
        out.reserveCapacity(outer * total * inner)
        for o in 0..<outer {
            for p in parts {
                let n = p.shape[a]
                let base = o * n * inner
                out.append(contentsOf: p.data[base..<(base + n * inner)])
            }
        }
        return Tensor(shape: newShape, data: out)
    }

    /// `np.repeat(x.take([index], axis), times, axis)`: one slab along `axis`, `times` times.
    public func repeated(axis: Int, index: Int, times: Int) -> Tensor {
        let a = normalized(axis: axis)
        let one = slice(axis: a, index..<(index + 1))
        return Tensor.concat([Tensor](repeating: one, count: times), axis: a)
    }

    // MARK: elementwise

    public func map(_ f: (Float) -> Float) -> Tensor {
        Tensor(shape: shape, data: data.map(f))
    }

    public func scaled(_ s: Float) -> Tensor {
        map { $0 * s }
    }

    /// Per-channel `x · scale[c] + shift[c]` over the first axis (`[C, …]`).
    public func channelAffine(scale: [Float], shift: [Float]) -> Tensor {
        precondition(scale.count == rows && shift.count == rows, "channelAffine wants \(rows) scales and shifts, got \(scale.count)/\(shift.count)")
        var out = data
        let s = rowStride
        for c in 0..<rows {
            let sc = scale[c], sh = shift[c]
            for i in (c * s)..<((c + 1) * s) {
                out[i] = out[i] * sc + sh
            }
        }
        return Tensor(shape: shape, data: out)
    }
}
