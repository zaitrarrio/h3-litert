// One `.tflite` graph on LiteRT for iOS, through the TensorFlow Lite Swift API
// (`import TensorFlowLite`; CocoaPods `TensorFlowLiteSwift` ~> 2.17 with the
// `Metal` subspec, or the `LiteRTSwift` pod, see ios/README.md).
//
// The whole file is behind `#if canImport(TensorFlowLite)` so the package
// resolves and the core tests build on a machine without the framework.
//
// API surface used (verified against tensorflow/lite/swift/Sources at the time
// of writing): `Interpreter(modelPath:options:delegates:)`, `Interpreter.signatureKeys`,
// `Interpreter.signatureRunner(with:)`, `SignatureRunner.inputs/outputs`,
// `SignatureRunner.invoke(with: [String: Data])`, `SignatureRunner.output(named:)`,
// `Tensor.data`, `Tensor.shape.dimensions`, `Tensor.dataType`, `MetalDelegate(options:)`.
//
// Inputs are bound by signature name (`q, k, v, mask` for attention, `mem_0 … mem_8`
// for TAEH3, `embeds, mask` for the text refiner), all float32 with exact shapes;
// outputs come back in `output_0, output_1, …` order.

import Foundation
import H3LiteRTCore

#if canImport(TensorFlowLite)
import TensorFlowLite

/// How graphs are compiled.
public struct LiteRTOptions {
    /// Use the Metal (GPU) delegate when the device has one; CPU (XNNPACK) otherwise.
    public var useMetal: Bool = true
    /// Let the GPU run f32 activations as f16 (the contract judges that path at 2e-2).
    public var allowPrecisionLoss: Bool = true
    /// CPU threads for the interpreter (nil: LiteRT decides).
    public var threadCount: Int? = nil
    /// Block graphs kept open at once.
    public var residentBlocks: Int = 2

    public init() {}
}

public final class LiteRTGraph: Graph {
    public let name: String
    public let url: URL
    private let interpreter: Interpreter
    private let runner: SignatureRunner
    private let delegate: Delegate?
    /// Signature input names.
    public let inputNames: [String]
    /// Signature output names in contract order (`output_0`, `output_1`, …).
    public let outputNames: [String]
    private let lock = NSLock()

    public init(url: URL, options: LiteRTOptions) throws {
        self.name = url.lastPathComponent
        self.url = url
        var delegates: [Delegate] = []
        var delegate: Delegate? = nil
        if options.useMetal {
            var mo = MetalDelegate.Options()
            mo.isPrecisionLossAllowed = options.allowPrecisionLoss
            mo.waitType = .passive
            let d = MetalDelegate(options: mo)
            delegates.append(d)
            delegate = d
        }
        self.delegate = delegate
        var io = Interpreter.Options()
        io.threadCount = options.threadCount
        // XNNPACK gives the CPU fallback its float kernels; harmless when the GPU takes the graph.
        io.isXNNPackEnabled = true  // TODO verify: combination with the Metal delegate on the same interpreter
        do {
            interpreter = try Interpreter(modelPath: url.path, options: io, delegates: delegates.isEmpty ? nil : delegates)
        } catch {
            throw H3Error.graph("\(url.lastPathComponent): cannot create the interpreter: \(error)")
        }
        guard let key = interpreter.signatureKeys.first else {
            throw H3Error.graph("\(url.lastPathComponent): the graph has no signature (docs/graphs.md binds inputs by signature name)")
        }
        do {
            runner = try interpreter.signatureRunner(with: key)
            try runner.allocateTensors()
        } catch {
            throw H3Error.graph("\(url.lastPathComponent): cannot create the signature runner `\(key)`: \(error)")
        }
        inputNames = runner.inputs
        outputNames = LiteRTGraph.contractOrder(runner.outputs)
    }

    /// Outputs are bound by position: the converter names them `output_0`, `output_1`, …
    /// and the signature's own listing is not guaranteed to be in that order.
    static func contractOrder(_ names: [String]) -> [String] {
        func index(_ n: String) -> Int {
            var digits = ""
            for ch in n.reversed() {
                if ch.isNumber { digits.insert(ch, at: digits.startIndex) } else { break }
            }
            return Int(digits) ?? 0
        }
        return names.sorted { index($0) < index($1) }
    }

    public func run(_ inputs: [String: H3LiteRTCore.Tensor]) throws -> [H3LiteRTCore.Tensor] {
        let want = Set(inputNames), got = Set(inputs.keys)
        guard want == got else {
            throw H3Error.graph("\(name): inputs \(inputNames.sorted()), got \(inputs.keys.sorted())")
        }
        var feed: [String: Data] = [:]
        for (k, t) in inputs {
            let expected: TensorFlowLite.Tensor
            do {
                expected = try runner.input(named: k)
            } catch {
                throw H3Error.graph("\(name): no input `\(k)`: \(error)")
            }
            guard expected.dataType == .float32 else {
                throw H3Error.graph("\(name): input `\(k)` is \(expected.dataType), the contract is float32")
            }
            guard expected.shape.dimensions == t.shape else {
                throw H3Error.shape("\(name): input `\(k)` is \(t.shape), the graph wants \(expected.shape.dimensions)")
            }
            feed[k] = t.data.withUnsafeBufferPointer { Data(buffer: $0) }
        }
        lock.lock()
        defer { lock.unlock() }
        do {
            try runner.invoke(with: feed)
        } catch {
            throw H3Error.graph("\(name): invoke failed: \(error)")
        }
        var outs: [H3LiteRTCore.Tensor] = []
        outs.reserveCapacity(outputNames.count)
        for n in outputNames {
            let o: TensorFlowLite.Tensor
            do {
                o = try runner.output(named: n)
            } catch {
                throw H3Error.graph("\(name): cannot read output `\(n)`: \(error)")
            }
            guard o.dataType == .float32 else {
                throw H3Error.graph("\(name): output `\(n)` is \(o.dataType), the contract is float32")
            }
            let count = o.data.count / 4
            var floats = [Float](repeating: 0, count: count)
            _ = floats.withUnsafeMutableBytes { dst in o.data.copyBytes(to: dst) }
            outs.append(H3LiteRTCore.Tensor(shape: o.shape.dimensions, data: floats))
        }
        return outs
    }
}

/// `GraphStore` over a directory of `.tflite` files (the export the user
/// copied into the app's Documents folder).
public final class LiteRTGraphStore: GraphStore {
    public let manifest: Manifest
    public let directory: URL
    public let options: LiteRTOptions
    private var fixedGraphs: [String: LiteRTGraph] = [:]
    private let blocks: ResidentGraphCache

    public init(directory: URL, options: LiteRTOptions = LiteRTOptions()) throws {
        self.manifest = try Manifest(directory: directory)
        self.directory = directory
        self.options = options
        self.blocks = ResidentGraphCache(capacity: options.residentBlocks)
    }

    public init(manifest: Manifest, directory: URL, options: LiteRTOptions = LiteRTOptions()) {
        self.manifest = manifest
        self.directory = directory
        self.options = options
        self.blocks = ResidentGraphCache(capacity: options.residentBlocks)
    }

    func open(_ file: String) throws -> LiteRTGraph {
        let url = directory.appendingPathComponent(file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw H3Error.io("\(file) is missing from \(directory.path); copy the whole export directory")
        }
        return try LiteRTGraph(url: url, options: options)
    }

    public func fixed(_ file: String) throws -> Graph {
        if let g = fixedGraphs[file] { return g }
        let g = try open(file)
        fixedGraphs[file] = g
        return g
    }

    public func block(_ file: String) throws -> Graph {
        try blocks.get(file) { try open(file) }
    }

    public func dropBlocks() {
        blocks.clear()
    }

    /// Also close the fixed graphs (between generations, to give memory back).
    public func dropAll() {
        blocks.clear()
        fixedGraphs.removeAll()
    }

    /// The AdaLN table that ships with this export.
    public func adalnTables() throws -> AdalnTables {
        try AdalnTables(url: directory.appendingPathComponent(manifest.adaln.file))
    }

    /// Every file the manifest names (graphs and the AdaLN table) that is not in the directory.
    public func missingFiles() -> [String] {
        manifest.allFiles.filter { !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }
}

extension Pipeline {
    /// A pipeline over an export directory: the graphs through LiteRT and its `adaln.safetensors`.
    public convenience init(store: LiteRTGraphStore) throws {
        self.init(store: store, adaln: try store.adalnTables())
    }
}

#else

/// Built without the TensorFlow Lite framework: the LiteRT backend is unavailable.
/// Link `TensorFlowLiteSwift` (CocoaPods, with the `Metal` subspec) into the
/// target that compiles these sources; see ios/README.md.
public enum LiteRTUnavailable {
    public static let reason = "H3LiteRTRuntime was compiled without the TensorFlowLite module"
}

#endif
