// `adaln.safetensors` (docs/graphs.md): the precomputed AdaLN modulation rows
// of the variant's whole timestep ladder. `timesteps` f32 `[U]`, `blocks` f16
// `[U, layers, 3, 6, d]`, `head` f16 `[U, 2, d]`. The tables stay in half
// precision in memory (170 MB for an 8-step student); one `[6, d]` or `[2, d]`
// row is decoded to float per graph call.

import Foundation

public final class AdalnTables {
    public let timesteps: [Float]
    public let layers: Int
    public let hidden: Int
    private let blocks: [UInt16]  // [U, layers, 3, 6, d]
    private let head: [UInt16]  // [U, 2, d]

    public init(timesteps: [Float], blocksHalf: [UInt16], blocksShape: [Int], headHalf: [UInt16], headShape: [Int]) throws {
        let u = timesteps.count
        guard blocksShape.count == 5, blocksShape[0] == u, blocksShape[2] == modalities, blocksShape[3] == modSlots else {
            throw H3Error.safetensors("adaln `blocks` is \(blocksShape), expected [\(u), layers, 3, 6, d]")
        }
        guard headShape == [u, 2, blocksShape[4]] else {
            throw H3Error.safetensors("adaln `head` is \(headShape), expected [\(u), 2, \(blocksShape[4])]")
        }
        guard blocksHalf.count == blocksShape.reduce(1, *), headHalf.count == headShape.reduce(1, *) else {
            throw H3Error.safetensors("adaln tables do not match their shapes")
        }
        self.timesteps = timesteps
        self.layers = blocksShape[1]
        self.hidden = blocksShape[4]
        self.blocks = blocksHalf
        self.head = headHalf
    }

    /// From float tables (rounded to f16 once, as the exporter does): for tests and synthetic exports.
    public convenience init(timesteps: [Float], blocks: Tensor, head: Tensor) throws {
        try self.init(
            timesteps: timesteps, blocksHalf: blocks.data.map { Half.fromFloat($0) }, blocksShape: blocks.shape,
            headHalf: head.data.map { Half.fromFloat($0) }, headShape: head.shape
        )
    }

    public convenience init(safetensors st: SafeTensors) throws {
        let ts = try st.float32("timesteps")
        guard ts.rank == 1 else { throw H3Error.safetensors("adaln `timesteps` is \(ts.shape), expected [U]") }
        let (bShape, bBits) = try st.halfBits("blocks")
        let (hShape, hBits) = try st.halfBits("head")
        try self.init(timesteps: ts.data, blocksHalf: bBits, blocksShape: bShape, headHalf: hBits, headShape: hShape)
    }

    public convenience init(url: URL) throws {
        try self.init(safetensors: try SafeTensors(url: url))
    }

    public var count: Int { timesteps.count }

    /// The table row of a clean time (`np.isclose(…, atol=1e-6)` on the f32 values).
    public func indexOf(_ t: Float) throws -> Int {
        if let i = timesteps.firstIndex(where: { abs($0 - t) <= 1e-6 }) {
            return i
        }
        throw H3Error.noAdalnRow(timestep: t, have: timesteps)
    }

    public func indexOf(_ t: Double) throws -> Int { try indexOf(Float(t)) }

    /// `blocks[u, layer, tag]` as float `[6, d]`: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
    public func mods(u: Int, layer: Int, tag: Int) -> Tensor {
        precondition(u >= 0 && u < count && layer >= 0 && layer < layers && tag >= 0 && tag < modalities, "adaln mods [\(u), \(layer), \(tag)] out of [\(count), \(layers), 3]")
        let n = modSlots * hidden
        let base = ((u * layers + layer) * modalities + tag) * n
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            out[i] = Half.toFloat(blocks[base + i])
        }
        return Tensor(shape: [modSlots, hidden], data: out)
    }

    /// `head[u]` as float `[2, d]`: shift (row 0), scale (row 1).
    public func headMods(u: Int) -> Tensor {
        precondition(u >= 0 && u < count, "adaln head row \(u) out of \(count)")
        let n = 2 * hidden
        let base = u * n
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            out[i] = Half.toFloat(head[base + i])
        }
        return Tensor(shape: [2, hidden], data: out)
    }

    /// Bytes the tables occupy in memory (for the plan).
    public var bytes: Int { (blocks.count + head.count) * 2 + timesteps.count * 4 }
}
