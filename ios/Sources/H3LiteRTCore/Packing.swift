// The packed sequence of one H3 request (python packing.py; h3-core
// `packing.rs`; FastVideo `packing.py`): `[text | condition video rows | audio
// rows | video rows]` for T2VA / FL2VA, `[text | ordered references |
// continuation | target audio | target video]` for Ref2VA, each row tagged
// video / text / audio with a real-valued `(t, h, w)` rotary position computed
// in f64 exactly as the reference does. Also the per-step row timesteps, the
// runs the host walks, and patch / unpatch of the video latent.

import Foundation

let ropeFrameRescale = 5.0 / 3.0
let ropeFramesPerLatent: [Double] = [1.0, 4.0, 4.0, 4.0, 4.0]
let ropeSpatialScale = 32.0

/// One row's rotary position.
public struct Position: Equatable {
    public var t: Double
    public var h: Double
    public var w: Double

    public init(_ t: Double, _ h: Double, _ w: Double) {
        self.t = t
        self.h = h
        self.w = w
    }

    public static let zero = Position(0, 0, 0)
}

/// A reference of a Ref2VA request: `kind` is "image", "video" or "audio".
public struct RefGeom {
    public var kind: String
    public var latentT: Int
    public var latentH: Int
    public var latentW: Int
    public var audioLatents: Int

    public init(kind: String, latentT: Int = 0, latentH: Int = 0, latentW: Int = 0, audioLatents: Int = 0) {
        self.kind = kind
        self.latentT = latentT
        self.latentH = latentH
        self.latentW = latentW
        self.audioLatents = audioLatents
    }
}

/// One segment of a Ref2VA sequence, in order: `("image" | "audio" | "video" | "continuation", rows, (t, h, w))`.
public struct Segment: Equatable {
    public var kind: String
    public var rows: Int
    public var latentT: Int
    public var latentH: Int
    public var latentW: Int
}

public struct Layout {
    public var sequenceLength: Int
    public var positionIds: [Position]
    public var tokenTags: [Int]
    public var videoIndices: [Int]
    public var audioIndices: [Int]
    public var textIndices: [Int]
    public var numConditionVideoRows: Int
    public var numConditionAudioRows: Int
    public var numVideoLatentFrames: Int
    public var latentHeight: Int
    public var latentWidth: Int
    public var numAudioLatents: Int
    public var referenceSegments: [Segment]

    public var numTextTokens: Int { textIndices.count }
    public var numTargetVideoRows: Int { videoIndices.count - numConditionVideoRows }
    public var numTargetAudioRows: Int { audioIndices.count - numConditionAudioRows }
}

/// A contiguous run of packed rows: `(packed start, rows, offset in the modality's own row buffer)`.
public struct Run: Equatable {
    public var start: Int
    public var rows: Int
    public var offset: Int

    public init(start: Int, rows: Int, offset: Int) {
        self.start = start
        self.rows = rows
        self.offset = offset
    }

    public var range: Range<Int> { start..<(start + rows) }
}

/// A maximal run of rows sharing `(timestep index, tag)`.
public struct ModRun: Equatable {
    public var start: Int
    public var rows: Int
    public var tIdx: Int
    public var tag: Int

    public init(start: Int, rows: Int, tIdx: Int, tag: Int) {
        self.start = start
        self.rows = rows
        self.tIdx = tIdx
        self.tag = tag
    }

    public var range: Range<Int> { start..<(start + rows) }
}

func spatialPositionGrid(dim: Int, patch: Int, sqrtArea: Double) -> [Double] {
    let ratio = Double(dim) / sqrtArea
    let left = (1.0 - ratio) / 2.0
    let n = dim / patch
    let step = ratio / Double(n)
    return (0..<n).map { (left + Double($0) * step) * ropeSpatialScale }
}

/// The frame clock: `origin`, then `5/3 · [1, 4, 4, 4, 4]` per latent frame, cycling.
public func temporalPositionGrid(numLatentFrames: Int, origin: Double) -> [Double] {
    var out: [Double] = []
    var acc = 0.0
    for i in 0..<numLatentFrames {
        out.append(origin + acc)
        acc += ropeFrameRescale * ropeFramesPerLatent[i % 5]
    }
    return out
}

func temporalSpan(_ n: Int) -> Double {
    var s = 0.0
    for i in 0..<n {
        s += ropeFrameRescale * ropeFramesPerLatent[i % 5]
    }
    return s
}

func frameGrid(lh: Int, lw: Int, ph: Int, pw: Int) -> (grid: [(h: Double, w: Double)], wg: [Double]) {
    let sqrtArea = Double(lh * lw).squareRoot()
    let hg = spatialPositionGrid(dim: lh, patch: ph, sqrtArea: sqrtArea)
    let wg = spatialPositionGrid(dim: lw, patch: pw, sqrtArea: sqrtArea)
    var grid: [(h: Double, w: Double)] = []
    grid.reserveCapacity(hg.count * wg.count)
    for h in hg {
        for w in wg {
            grid.append((h, w))
        }
    }
    return (grid, wg)
}

/// Two channels of `n` audio rows at `time0 + i`, the left channel at the left-most
/// width position of the target grid and the right one at the right-most.
func fillAudio(_ pos: inout [Position], start: Int, n: Int, time0: Double, wg: [Double]) {
    for ch in 0..<audioChannels {
        let wpos = ch == 0 ? wg[0] : wg[wg.count - 1]
        for i in 0..<n {
            pos[start + ch * n + i] = Position(time0 + Double(i), 0.0, wpos)
        }
    }
}

func checkTextTags(_ tags: [Int]) throws {
    if tags.contains(where: { $0 != textTag && $0 != videoTag }) {
        throw H3Error.layout("text token tags may be text or vision only")
    }
}

/// T2VA / FL2VA: `[text | condition keyframes | audio | video]`; anchors are "first" / "last".
public func buildLayout(textTags: [Int], lt: Int, lh: Int, lw: Int, na: Int, patch: [Int], anchors: [String]) throws -> Layout {
    try checkTextTags(textTags)
    guard patch.count == 3 else { throw H3Error.layout("patch \(patch) is not (pt, ph, pw)") }
    let pt = patch[0], ph = patch[1], pw = patch[2]
    if lt % pt != 0 || lh % ph != 0 || lw % pw != 0 {
        throw H3Error.layout("latent (\(lt), \(lh), \(lw)) is not divisible by the patch \(patch)")
    }
    let rpf = (lh / ph) * (lw / pw)
    let nText = textTags.count
    let nCond = anchors.count * rpf
    let nAudio = na * audioChannels
    let nVideo = (lt / pt) * rpf
    let total = nText + nCond + nAudio + nVideo
    let condStart = nText, audioStart = nText + nCond, videoStart = nText + nCond + nAudio
    var pos = [Position](repeating: .zero, count: total)
    for i in 0..<nText {
        pos[i] = Position(Double(i), 0.0, 0.0)
    }
    let (grid, wg) = frameGrid(lh: lh, lw: lw, ph: ph, pw: pw)
    for (k, a) in anchors.enumerated() {
        let t: Double
        switch a {
        case "first": t = Double(nText)
        case "last": t = Double(nText) + temporalSpan(lt) - ropeFrameRescale
        default: throw H3Error.layout("anchor `\(a)` is not first or last")
        }
        for (j, g) in grid.enumerated() {
            pos[condStart + k * rpf + j] = Position(t, g.h, g.w)
        }
    }
    fillAudio(&pos, start: audioStart, n: na, time0: Double(nText), wg: wg)
    for (f, t) in temporalPositionGrid(numLatentFrames: lt, origin: Double(nText)).enumerated() {
        for (j, g) in grid.enumerated() {
            pos[videoStart + f * rpf + j] = Position(t, g.h, g.w)
        }
    }
    var tags = textTags
    tags.append(contentsOf: [Int](repeating: videoTag, count: nCond))
    tags.append(contentsOf: [Int](repeating: audioTag, count: nAudio))
    tags.append(contentsOf: [Int](repeating: videoTag, count: nVideo))
    return Layout(
        sequenceLength: total, positionIds: pos, tokenTags: tags,
        videoIndices: Array(condStart..<audioStart) + Array(videoStart..<total),
        audioIndices: Array(audioStart..<videoStart), textIndices: Array(0..<nText),
        numConditionVideoRows: nCond, numConditionAudioRows: 0,
        numVideoLatentFrames: lt, latentHeight: lh, latentWidth: lw, numAudioLatents: na, referenceSegments: []
    )
}

/// Ref2VA: `[text | ordered references | continuation clip | target audio | target video]`.
public func buildRef2vaLayout(
    textTags: [Int], refs: [RefGeom], lt: Int, lh: Int, lw: Int, na: Int, patch: [Int], continuation: (frames: Int, audioLatents: Int)? = nil
) throws -> Layout {
    if patch != [1, 2, 2] {
        throw H3Error.layout("Ref2VA requires patch (1, 2, 2), got \(patch)")
    }
    try checkTextTags(textTags)
    if refs.isEmpty {
        throw H3Error.layout("Ref2VA requires at least one reference")
    }
    if lt == 0 || lh % 2 != 0 || lw % 2 != 0 || na == 0 {
        throw H3Error.layout("Ref2VA target geometry must be positive and patch-aligned")
    }
    let nText = textTags.count

    func rowsOf(_ t: Int, _ h: Int, _ w: Int) throws -> Int {
        if t == 0 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0 {
            throw H3Error.layout("visual reference geometry (\(t), \(h), \(w)) is not patch-aligned")
        }
        return t * (h / 2) * (w / 2)
    }

    let rptf = (lh / 2) * (lw / 2)
    let nTargetVideo = lt * rptf, nTargetAudio = na * audioChannels
    var nRefVideo = 0, nRefAudio = 0
    for r in refs {
        switch r.kind {
        case "image":
            nRefVideo += try rowsOf(1, r.latentH, r.latentW)
        case "video":
            nRefVideo += try rowsOf(r.latentT, r.latentH, r.latentW)
            nRefAudio += r.audioLatents * audioChannels
        case "audio":
            if r.audioLatents == 0 {
                throw H3Error.layout("an audio reference has no latents")
            }
            nRefAudio += r.audioLatents * audioChannels
        default:
            throw H3Error.layout("reference kind `\(r.kind)`")
        }
    }
    if let c = continuation {
        if c.frames == 0 || c.frames >= lt {
            throw H3Error.layout("continuation clip of \(c.frames) latent frames does not fit a target of \(lt)")
        }
        nRefVideo += c.frames * rptf
        nRefAudio += c.audioLatents * audioChannels
    }
    let total = nText + nRefVideo + nRefAudio + nTargetAudio + nTargetVideo
    var pos = [Position](repeating: .zero, count: total)
    for i in 0..<nText {
        pos[i] = Position(Double(i), 0.0, 0.0)
    }
    let (targetGrid, targetWg) = frameGrid(lh: lh, lw: lw, ph: 2, pw: 2)
    var videoIndices: [Int] = [], audioIndices: [Int] = [], segments: [Segment] = []
    var cursor = nText
    var clock = Double(nText)
    for r in refs {
        switch r.kind {
        case "image":
            let n = try rowsOf(1, r.latentH, r.latentW)
            segments.append(Segment(kind: "image", rows: n, latentT: 1, latentH: r.latentH, latentW: r.latentW))
            let (grid, _) = frameGrid(lh: r.latentH, lw: r.latentW, ph: 2, pw: 2)
            for (j, g) in grid.enumerated() {
                pos[cursor + j] = Position(clock, g.h, g.w)
            }
            videoIndices.append(contentsOf: cursor..<(cursor + n))
            cursor += n
            clock += 1.0
        case "audio":
            let n = r.audioLatents * audioChannels
            segments.append(Segment(kind: "audio", rows: n, latentT: 0, latentH: 0, latentW: 0))
            fillAudio(&pos, start: cursor, n: r.audioLatents, time0: clock, wg: targetWg)
            audioIndices.append(contentsOf: cursor..<(cursor + n))
            cursor += n
            clock += Double(r.audioLatents)
        default:  // "video" (checked above)
            let naR = r.audioLatents * audioChannels
            let nv = try rowsOf(r.latentT, r.latentH, r.latentW)
            if naR > 0 {
                segments.append(Segment(kind: "audio", rows: naR, latentT: 0, latentH: 0, latentW: 0))
            }
            segments.append(Segment(kind: "video", rows: nv, latentT: r.latentT, latentH: r.latentH, latentW: r.latentW))
            let (grid, wg) = frameGrid(lh: r.latentH, lw: r.latentW, ph: 2, pw: 2)
            if naR > 0 {
                fillAudio(&pos, start: cursor, n: r.audioLatents, time0: clock, wg: wg)
                audioIndices.append(contentsOf: cursor..<(cursor + naR))
                cursor += naR
            }
            for (f, tt) in temporalPositionGrid(numLatentFrames: r.latentT, origin: clock).enumerated() {
                for (j, g) in grid.enumerated() {
                    pos[cursor + f * grid.count + j] = Position(tt, g.h, g.w)
                }
            }
            videoIndices.append(contentsOf: cursor..<(cursor + nv))
            cursor += nv
            clock += max(Double(r.audioLatents), temporalSpan(r.latentT))
        }
    }
    if let c = continuation {
        let naC = c.audioLatents * audioChannels, nv = c.frames * rptf
        if naC > 0 {
            segments.append(Segment(kind: "audio", rows: naC, latentT: 0, latentH: 0, latentW: 0))
        }
        segments.append(Segment(kind: "continuation", rows: nv, latentT: c.frames, latentH: lh, latentW: lw))
        clock += max(Double(c.audioLatents), temporalSpan(c.frames))
        if naC > 0 {
            fillAudio(&pos, start: cursor, n: c.audioLatents, time0: clock, wg: targetWg)
            audioIndices.append(contentsOf: cursor..<(cursor + naC))
            cursor += naC
        }
        for (f, tt) in temporalPositionGrid(numLatentFrames: c.frames, origin: clock).enumerated() {
            for (j, g) in targetGrid.enumerated() {
                pos[cursor + f * rptf + j] = Position(tt, g.h, g.w)
            }
        }
        videoIndices.append(contentsOf: cursor..<(cursor + nv))
        cursor += nv
    }
    let audioStart = cursor
    let videoStart = audioStart + nTargetAudio
    fillAudio(&pos, start: audioStart, n: na, time0: clock, wg: targetWg)
    for (f, tt) in temporalPositionGrid(numLatentFrames: lt, origin: clock).enumerated() {
        for (j, g) in targetGrid.enumerated() {
            pos[videoStart + f * rptf + j] = Position(tt, g.h, g.w)
        }
    }
    audioIndices.append(contentsOf: audioStart..<videoStart)
    videoIndices.append(contentsOf: videoStart..<total)
    precondition(videoStart + nTargetVideo == total, "Ref2VA layout rows do not add up")
    var tags = [Int](repeating: videoTag, count: total)
    for i in 0..<nText {
        tags[i] = textTags[i]
    }
    for i in audioIndices {
        tags[i] = audioTag
    }
    for i in videoIndices {
        tags[i] = videoTag
    }
    return Layout(
        sequenceLength: total, positionIds: pos, tokenTags: tags, videoIndices: videoIndices, audioIndices: audioIndices,
        textIndices: Array(0..<nText), numConditionVideoRows: nRefVideo, numConditionAudioRows: nRefAudio,
        numVideoLatentFrames: lt, latentHeight: lh, latentWidth: lw, numAudioLatents: na, referenceSegments: segments
    )
}

/// Per-row clean times of one step (f32 values): the sorted unique times and each row's index into them.
public func rowTimesteps(layout: Layout, videoT: Double, audioT: Double, condVideoT: Double, condAudioT: Double) -> (unique: [Float], index: [Int]) {
    var rows = [Float](repeating: Float(videoT), count: layout.sequenceLength)
    for i in layout.videoIndices[0..<layout.numConditionVideoRows] {
        rows[i] = Float(condVideoT)
    }
    for i in layout.audioIndices[layout.numConditionAudioRows...] {
        rows[i] = Float(audioT)
    }
    for i in layout.audioIndices[0..<layout.numConditionAudioRows] {
        rows[i] = Float(condAudioT)
    }
    let unique = Array(Set(rows)).sorted()
    var pos: [Float: Int] = [:]
    for (k, u) in unique.enumerated() {
        pos[u] = k
    }
    return (unique, rows.map { pos[$0]! })
}

/// Maximal contiguous runs of `indices`, with each run's offset into the modality's own row buffer.
public func runs(_ indices: [Int]) -> [Run] {
    var out: [Run] = []
    for (off, i) in indices.enumerated() {
        if let last = out.last, last.start + last.rows == i {
            out[out.count - 1].rows += 1
        } else {
            out.append(Run(start: i, rows: 1, offset: off))
        }
    }
    return out
}

/// Maximal runs of rows sharing `(timestep index, tag)`.
public func modRuns(tIdx: [Int], tags: [Int]) -> [ModRun] {
    precondition(tIdx.count == tags.count, "modRuns: \(tIdx.count) timestep indices for \(tags.count) tags")
    var out: [ModRun] = []
    for i in 0..<tIdx.count {
        if let last = out.last, last.tIdx == tIdx[i], last.tag == tags[i] {
            out[out.count - 1].rows += 1
        } else {
            out.append(ModRun(start: i, rows: 1, tIdx: tIdx[i], tag: tags[i]))
        }
    }
    return out
}

/// `[C, T, H, W]` latent → `[rows, C·pt·ph·pw]` patch rows (channel-major features),
/// rows in `(t, h, w)` order.
public func patchRows(_ z: Tensor, patch: [Int]) -> Tensor {
    precondition(z.rank == 4, "patchRows wants [C, T, H, W], got \(z.shape)")
    precondition(patch.count == 3, "patch \(patch) is not (pt, ph, pw)")
    let c = z.shape[0], t = z.shape[1], h = z.shape[2], w = z.shape[3]
    let pt = patch[0], ph = patch[1], pw = patch[2]
    precondition(t % pt == 0 && h % ph == 0 && w % pw == 0, "latent \(z.shape) is not divisible by the patch \(patch)")
    let tt = t / pt, hh = h / ph, ww = w / pw
    let rows = tt * hh * ww
    let feats = c * pt * ph * pw
    var out = [Float](repeating: 0, count: rows * feats)
    let src = z.data
    for ti in 0..<tt {
        for hi in 0..<hh {
            for wi in 0..<ww {
                let base = ((ti * hh + hi) * ww + wi) * feats
                for ci in 0..<c {
                    for kt in 0..<pt {
                        for kh in 0..<ph {
                            let srcRow = ((ci * t + ti * pt + kt) * h + hi * ph + kh) * w + wi * pw
                            let f = ((ci * pt + kt) * ph + kh) * pw
                            for kw in 0..<pw {
                                out[base + f + kw] = src[srcRow + kw]
                            }
                        }
                    }
                }
            }
        }
    }
    return Tensor(shape: [rows, feats], data: out)
}

/// The inverse of `patchRows`: `[rows, C·pt·ph·pw]` → `[C, T, H, W]`.
public func unpatchRows(_ rows: Tensor, c: Int, t: Int, h: Int, w: Int, patch: [Int]) -> Tensor {
    precondition(patch.count == 3, "patch \(patch) is not (pt, ph, pw)")
    let pt = patch[0], ph = patch[1], pw = patch[2]
    let tt = t / pt, hh = h / ph, ww = w / pw
    let feats = c * pt * ph * pw
    precondition(rows.shape == [tt * hh * ww, feats], "unpatchRows rows \(rows.shape) for c=\(c) t=\(t) h=\(h) w=\(w) patch \(patch)")
    var out = [Float](repeating: 0, count: c * t * h * w)
    let src = rows.data
    for ti in 0..<tt {
        for hi in 0..<hh {
            for wi in 0..<ww {
                let base = ((ti * hh + hi) * ww + wi) * feats
                for ci in 0..<c {
                    for kt in 0..<pt {
                        for kh in 0..<ph {
                            let dstRow = ((ci * t + ti * pt + kt) * h + hi * ph + kh) * w + wi * pw
                            let f = ((ci * pt + kt) * ph + kh) * pw
                            for kw in 0..<pw {
                                out[dstRow + kw] = src[base + f + kw]
                            }
                        }
                    }
                }
            }
        }
    }
    return Tensor(shape: [c, t, h, w], data: out)
}
