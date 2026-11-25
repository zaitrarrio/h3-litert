// The runtime boundary: one `.tflite` graph and a store of them. The core
// drives these protocols; `H3LiteRTRuntime` implements them over LiteRT and
// the tests over a fake.

import Foundation

/// One graph of the export. Inputs are bound by signature name (`q, k, v, mask`
/// for attention), outputs are returned in signature order (`output_0`,
/// `output_1`, …; docs/graphs.md).
public protocol Graph: AnyObject {
    /// The file name inside the export, for messages.
    var name: String { get }
    func run(_ inputs: [String: Tensor]) throws -> [Tensor]
}

/// Graphs of one manifest, opened on demand. `fixed` graphs (text_refine, the
/// embeds and heads, attention, the VAE decoder and encoder graphs, the audio
/// VAE, TAEH3) stay open; `block` graphs (the DiT blocks' qkv / post) are kept
/// least-recently-used up to a residency limit, so a 50-block model streams
/// through memory a few blocks at a time.
public protocol GraphStore: AnyObject {
    var manifest: Manifest { get }
    func fixed(_ file: String) throws -> Graph
    func block(_ file: String) throws -> Graph
    func dropBlocks()
}

/// The LRU residency policy of block graphs, shared by every store
/// implementation (python runtime.GraphStore). Not thread-safe: the pipeline
/// runs on one thread.
public final class ResidentGraphCache {
    public let capacity: Int
    private var order: [String] = []  // least recently used first
    private var graphs: [String: Graph] = [:]

    public init(capacity: Int) {
        self.capacity = max(1, capacity)
    }

    public var residentFiles: [String] { order }

    public func get(_ file: String, open: () throws -> Graph) throws -> Graph {
        if let g = graphs[file] {
            if let i = order.firstIndex(of: file) {
                order.remove(at: i)
            }
            order.append(file)
            return g
        }
        while graphs.count >= capacity, let oldest = order.first {
            order.removeFirst()
            graphs[oldest] = nil
        }
        let g = try open()
        graphs[file] = g
        order.append(file)
        return g
    }

    public func clear() {
        order.removeAll()
        graphs.removeAll()
    }
}
