// A tiny safetensors reader: 8-byte little-endian header length, a JSON
// header (`{name: {dtype, shape, data_offsets}, "__metadata__": {...}}`), then
// the raw little-endian tensor bytes. Enough for prompt embeddings
// (`prompt_embeds` F32 [rows, 5120] + `text_tags`), `adaln.safetensors`
// (F32 timesteps, F16 tables) and the fixture's `expected.safetensors`.

import Foundation

public struct SafeTensors {
    public struct Entry {
        public var dtype: String
        public var shape: [Int]
        public var begin: Int
        public var end: Int
    }

    public let metadata: [String: String]
    public let entries: [String: Entry]
    private let data: Data
    private let dataStart: Int

    public init(data: Data) throws {
        guard data.count >= 8 else { throw H3Error.safetensors("file is shorter than its 8-byte header length") }
        var n: UInt64 = 0
        for i in 0..<8 {
            n |= UInt64(data[data.startIndex + i]) << (8 * UInt64(i))
        }
        guard n <= UInt64(data.count - 8) else { throw H3Error.safetensors("header length \(n) exceeds the file") }
        let headerLen = Int(n)
        let headerData = data.subdata(in: (data.startIndex + 8)..<(data.startIndex + 8 + headerLen))
        let json: Any
        do {
            json = try JSONSerialization.jsonObject(with: headerData, options: [])
        } catch {
            throw H3Error.safetensors("header is not JSON: \(error)")
        }
        guard let dict = json as? [String: Any] else { throw H3Error.safetensors("header is not a JSON object") }
        var meta: [String: String] = [:]
        var entries: [String: Entry] = [:]
        for (k, v) in dict {
            if k == "__metadata__" {
                if let m = v as? [String: Any] {
                    for (mk, mv) in m {
                        meta[mk] = "\(mv)"
                    }
                }
                continue
            }
            guard let e = v as? [String: Any],
                let dtype = e["dtype"] as? String,
                let shapeAny = e["shape"] as? [Any],
                let offsets = e["data_offsets"] as? [Any], offsets.count == 2
            else {
                throw H3Error.safetensors("tensor `\(k)` has a malformed header entry")
            }
            let shape = shapeAny.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeAny.count,
                let b = (offsets[0] as? NSNumber)?.intValue, let en = (offsets[1] as? NSNumber)?.intValue, b <= en
            else {
                throw H3Error.safetensors("tensor `\(k)` has a malformed shape or data_offsets")
            }
            entries[k] = Entry(dtype: dtype, shape: shape, begin: b, end: en)
        }
        self.metadata = meta
        self.entries = entries
        self.data = data
        self.dataStart = data.startIndex + 8 + headerLen
    }

    public init(url: URL) throws {
        let d: Data
        do {
            d = try Data(contentsOf: url)
        } catch {
            throw H3Error.io("cannot read \(url.path): \(error)")
        }
        try self.init(data: d)
    }

    public var names: [String] { entries.keys.sorted() }

    public func contains(_ name: String) -> Bool { entries[name] != nil }

    func bytes(of name: String) throws -> (Entry, Data) {
        guard let e = entries[name] else {
            throw H3Error.safetensors("no tensor `\(name)` (it has \(names))")
        }
        let lo = dataStart + e.begin, hi = dataStart + e.end
        guard hi <= data.endIndex else {
            throw H3Error.safetensors("tensor `\(name)` runs past the end of the file")
        }
        return (e, data.subdata(in: lo..<hi))
    }

    /// An F32 (or F16) tensor as float32. Assumes a little-endian host (every Apple platform).
    public func float32(_ name: String) throws -> Tensor {
        let (e, raw) = try bytes(of: name)
        let count = e.shape.reduce(1, *)
        switch e.dtype {
        case "F32":
            guard raw.count == count * 4 else {
                throw H3Error.safetensors("tensor `\(name)`: \(raw.count) bytes for shape \(e.shape) F32")
            }
            var out = [Float](repeating: 0, count: count)
            _ = out.withUnsafeMutableBytes { dst in
                raw.copyBytes(to: dst)
            }
            return Tensor(shape: e.shape, data: out)
        case "F16":
            let (_, bits) = try halfBits(name)
            return Tensor(shape: e.shape, data: bits.map { Half.toFloat($0) })
        default:
            throw H3Error.safetensors("tensor `\(name)` is \(e.dtype); only F32 and F16 are read as float")
        }
    }

    /// An F16 tensor as its raw binary16 bit patterns (for tables kept in half precision).
    public func halfBits(_ name: String) throws -> (shape: [Int], bits: [UInt16]) {
        let (e, raw) = try bytes(of: name)
        guard e.dtype == "F16" else {
            throw H3Error.safetensors("tensor `\(name)` is \(e.dtype), expected F16")
        }
        let count = e.shape.reduce(1, *)
        guard raw.count == count * 2 else {
            throw H3Error.safetensors("tensor `\(name)`: \(raw.count) bytes for shape \(e.shape) F16")
        }
        var out = [UInt16](repeating: 0, count: count)
        raw.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
            for i in 0..<count {
                out[i] = UInt16(buf[2 * i]) | (UInt16(buf[2 * i + 1]) << 8)
            }
        }
        return (e.shape, out)
    }

    /// A U8 tensor and its shape.
    public func uint8(_ name: String) throws -> (shape: [Int], data: [UInt8]) {
        let (e, raw) = try bytes(of: name)
        guard e.dtype == "U8" else {
            throw H3Error.safetensors("tensor `\(name)` is \(e.dtype), expected U8")
        }
        let count = e.shape.reduce(1, *)
        guard raw.count == count else {
            throw H3Error.safetensors("tensor `\(name)`: \(raw.count) bytes for shape \(e.shape) U8")
        }
        return (e.shape, [UInt8](raw))
    }

    /// Kept for source compatibility with the sibling runtime.
    static func halfToFloat(_ h: UInt16) -> Float { Half.toFloat(h) }
}

/// IEEE binary16 conversions in integer math (portable: `Float16` is unavailable
/// on Intel Macs, where the core tests also run).
public enum Half {
    /// binary16 → Float.
    public static func toFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32((h >> 10) & 0x1F)
        var mant = UInt32(h & 0x03FF)
        var bits: UInt32
        if exp == 0 {
            if mant == 0 {
                bits = sign
            } else {
                // subnormal: normalise
                var e: UInt32 = 127 - 15 + 1
                while (mant & 0x0400) == 0 {
                    mant <<= 1
                    e -= 1
                }
                mant &= 0x03FF
                bits = sign | (e << 23) | (mant << 13)
            }
        } else if exp == 0x1F {
            bits = sign | 0x7F80_0000 | (mant << 13)
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
        }
        return Float(bitPattern: bits)
    }

    /// Float → binary16, round to nearest even (numpy's `astype(float16)`).
    public static func fromFloat(_ f: Float) -> UInt16 {
        let x = f.bitPattern
        let sign = UInt16((x >> 16) & 0x8000)
        let exp = Int((x >> 23) & 0xFF)
        let mant = x & 0x7F_FFFF
        if exp == 0xFF {
            if mant == 0 { return sign | 0x7C00 }  // ±inf
            return sign | 0x7E00 | UInt16((mant >> 13) & 0x01FF)  // NaN, quiet
        }
        let e = exp - 127 + 15
        if e >= 0x1F { return sign | 0x7C00 }  // overflow → ±inf
        if e <= 0 {
            // Below the smallest half normal: a half subnormal (or zero).
            if e < -10 { return sign }  // < 2^-25: rounds to ±0 even at the tie
            let m = mant | 0x80_0000
            let shift = UInt32(14 - e)  // 13 bits of precision plus the (1 − e) binades below normal
            var half = m >> shift
            let rem = m & ((UInt32(1) << shift) - 1)
            let halfway = UInt32(1) << (shift - 1)
            if rem > halfway || (rem == halfway && (half & 1) == 1) { half += 1 }
            return sign | UInt16(half)
        }
        var half = (UInt32(e) << 10) | (mant >> 13)
        let rem = mant & 0x1FFF
        if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
            half += 1  // a carry into the exponent is the right next binade (or ±inf at 0x7C00)
        }
        return sign | UInt16(half)
    }

    /// Round a Float through binary16 and back (`x.astype(np.float16).astype(np.float32)`).
    public static func round(_ f: Float) -> Float { toFloat(fromFloat(f)) }
}
