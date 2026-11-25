// The host math against the values h3-rs's own tests pin down
// (python/tests/test_host.py), plus the manifest, the AdaLN table, the
// safetensors reader and the VAE host side (h3-rs `vae.rs` tests).

import Foundation
import XCTest
@testable import H3LiteRTCore

final class HostTests: XCTestCase {
    // MARK: schedules

    func testV2Ladder() throws {
        let s = try Schedule.fromRungs(fasth3V2Rungs, shift: 10.0)
        XCTAssertEqual(s.steps, 8)
        XCTAssertEqual(Double(s.sigmas[0]), Double(shiftSigma(base: 0.999, shift: 10.0)), accuracy: 1e-7)
        XCTAssertEqual(Double(s.sigmas[4]), 10.0 * 0.5 / 5.5, accuracy: 1e-6)
        XCTAssertEqual(s.sigmas[s.sigmas.count - 1], 0.0)
        XCTAssertEqual(Double(s.timestep(4)), 1.0 - 10.0 * 0.5 / 5.5, accuracy: 1e-6)
        let a = try Schedule.fromRungs(fasth3V2Rungs, shift: 3.0)
        XCTAssert(a.sigmas[4] > 0.74 && a.sigmas[4] < 0.76, "\(a.sigmas[4])")
    }

    func testUniformAndPika() throws {
        let s = try Schedule.uniform(points: 50, shift: 12.0)
        XCTAssertEqual(s.steps, 49)
        XCTAssertEqual(s.sigmas[0], 1.0)
        let p = try Schedule.fromRungs(pikaKnots, shift: 12.0, rule: .renoise)
        XCTAssertEqual(p.steps, 6)
        XCTAssertEqual(p.sigmas[0], 1.0)
        XCTAssertEqual(Double(p.sigmas[5]), Double(shiftSigma(base: 0.04, shift: 12.0)), accuracy: 1e-7)
        XCTAssertThrowsError(try Schedule.fromRungs([500, 999], shift: 12.0))
        XCTAssertThrowsError(try Schedule.uniform(points: 1, shift: 12.0))
    }

    func testPlug4GridMatchesDiffusersBits() throws {
        let want: [Double: [UInt32]] = [
            12.0: [0x3F80_0000, 0x3F79_14C2, 0x3F6C_4EC5, 0x3F4C_CCCD, 0],
            3.0: [0x3F80_0000, 0x3F66_6666, 0x3F40_0000, 0x3F00_0000, 0],
        ]
        for (shift, bits) in want {
            let s = try Schedule.fromRungs(plug4Knots, shift: shift, rule: .renoise)
            XCTAssertEqual(s.sigmas.map { $0.bitPattern }, bits, "shift \(shift)")
            XCTAssertEqual(try Schedule.uniform(points: 5, shift: shift).sigmas, s.sigmas, "shift \(shift)")
            for i in 0..<4 {
                XCTAssertEqual(1.0 - Double(s.timestep(i)), Double(s.sigmas[i]))
            }
        }
    }

    func testPddOmniPlan() throws {
        let p = try PddPlan(indices: [0, 4, 8, 12, 16, 20, 24, 28, 32], shifts: .h3)
        XCTAssertEqual(p.fineGrid.count - 1, 32)
        XCTAssertEqual(p.numSteps, 8)
        XCTAssertEqual(p.block(0).start, 0)
        XCTAssertEqual(p.block(0).end, 4)
        XCTAssertEqual(Double(p.videoSigmas[0]), 0.999, accuracy: 1e-6)
        XCTAssertEqual(p.videoSigmas[p.videoSigmas.count - 1], 0.0)
        XCTAssert(p.videoWeights.allSatisfy { $0 < 0 })
        XCTAssertEqual(PddPlan.fusedWeights(p.videoWeights, start: 0, end: 4).reduce(0.0) { $0 + Double($1) }, 1.0, accuracy: 1e-6)
        XCTAssertEqual(p.videoWeights[4..<8].reduce(0.0, +), Double(p.videoSigmas[2]) - Double(p.videoSigmas[1]), accuracy: 1e-6)
        XCTAssertThrowsError(try PddPlan(indices: [1, 4], shifts: .h3))
        let (v, a) = try p.schedules()
        XCTAssertEqual(v.steps, 8)
        XCTAssertEqual(a.steps, 8)
    }

    // MARK: geometry

    func testFrameAndLatentCounts() throws {
        XCTAssertEqual([alignNumFrames(120), alignNumFrames(240), alignNumFrames(360)], [124, 243, 362])
        XCTAssertEqual(try videoLatentFrames(124), 37)
        XCTAssertEqual(try videoLatentFrames(362), 107)
        XCTAssertEqual(audioLatents(frames: 124), 207)
        XCTAssertEqual(audioLatents(frames: 243), 405)
        XCTAssertEqual(audioLatents(frames: 22), 37)
        XCTAssertEqual(try resolveNumFrames(5 * 24), 124)
        XCTAssertThrowsError(try videoLatentFrames(120))
        XCTAssertThrowsError(try resolveNumFrames(400))
        XCTAssertEqual(try Geometry.parse("1344x768x124"), Geometry.h3_768p)
        XCTAssertThrowsError(try Geometry.parse("1344x768"))
        XCTAssertNoThrow(try Geometry.tiny.validate(spatialRatio: 16))
        XCTAssertThrowsError(try Geometry(width: 100, height: 64, frames: 22).validate(spatialRatio: 16))
        XCTAssertThrowsError(try Geometry(width: 96, height: 64, frames: 21).validate(spatialRatio: 16))
    }

    func testCanvasRules() throws {
        XCTAssertEqual(try canvasSize(aspectW: 16, aspectH: 9).height, 768)
        XCTAssertEqual(try canvasSize(aspectW: 16, aspectH: 9).width, 1344)
        XCTAssertEqual(try canvasSize(aspectW: 9, aspectH: 16).height, 1344)
        XCTAssertEqual(try canvasSize(aspectW: 9, aspectH: 16).width, 768)
        XCTAssertEqual(try canvasSize(aspectW: 1, aspectH: 1).height, 768)
        XCTAssertEqual(try canvasSize(aspectW: 1, aspectH: 1).width, 768)
        let (h, w) = try canvasSize(aspectW: 21, aspectH: 9)
        XCTAssert(h * w <= 768 * 1344 + 32 * 32 && h % 32 == 0 && w % 32 == 0 && w > h, "\(h)x\(w)")
        XCTAssertThrowsError(try canvasSize(aspectW: 5, aspectH: 1))
        XCTAssertThrowsError(try canvasSize(aspectW: 0, aspectH: 1))
    }

    // MARK: packing

    func testT2vaLayout768p() throws {
        let (t, h, w) = try Geometry.h3_768p.latent(spatialRatio: 16)
        XCTAssertEqual([t, h, w], [37, 48, 84])
        let na = Geometry.h3_768p.audioLatents
        let l = try buildLayout(textTags: [Int](repeating: textTag, count: 10), lt: t, lh: h, lw: w, na: na, patch: [1, 2, 2], anchors: [])
        let video = 37 * 24 * 42
        XCTAssertEqual(l.sequenceLength, 10 + 2 * 207 + video)
        XCTAssertEqual(l.audioIndices.count, 414)
        XCTAssertEqual(l.positionIds[3], Position(3.0, 0.0, 0.0))
        let a0 = l.positionIds[l.audioIndices[0]]
        XCTAssertEqual(a0.t, 10.0)
        XCTAssertEqual(a0.h, 0.0)
        let aR = l.positionIds[l.audioIndices[207]]
        XCTAssertEqual(aR.t, a0.t)
        XCTAssertGreaterThan(aR.w, a0.w)
        let v0 = l.positionIds[l.videoIndices[0]]
        let v1 = l.positionIds[l.videoIndices[24 * 42]]
        let v2 = l.positionIds[l.videoIndices[2 * 24 * 42]]
        XCTAssertEqual(v0.t, 10.0)
        XCTAssertEqual(v1.t, 10 + 5.0 / 3.0, accuracy: 1e-9)
        XCTAssertEqual(v2.t, 10 + 5.0 / 3.0 + 4 * 5.0 / 3.0, accuracy: 1e-9)
        XCTAssertEqual(l.numTargetVideoRows, video)
        XCTAssertEqual(l.tokenTags[0], textTag)
        XCTAssertEqual(l.tokenTags[l.audioIndices[0]], audioTag)
        XCTAssertEqual(l.tokenTags[l.videoIndices[0]], videoTag)
        XCTAssertThrowsError(try buildLayout(textTags: [audioTag], lt: t, lh: h, lw: w, na: na, patch: [1, 2, 2], anchors: []))
        XCTAssertThrowsError(try buildLayout(textTags: [textTag], lt: t, lh: h, lw: w, na: na, patch: [1, 2, 2], anchors: ["middle"]))
    }

    func testFl2vaAnchorsAndRowTimesteps() throws {
        let l = try buildLayout(textTags: [textTag, textTag, videoTag], lt: 7, lh: 4, lw: 6, na: 20, patch: [1, 2, 2], anchors: ["first", "last"])
        XCTAssertEqual(l.numConditionVideoRows, 12)
        XCTAssertEqual(l.positionIds[l.videoIndices[0]].t, 3.0)
        XCTAssertEqual(l.positionIds[l.videoIndices[6]].t, 3.0 + 5.0 / 3.0 * 22.0 - 5.0 / 3.0, accuracy: 1e-9)
        let (u, inv) = rowTimesteps(layout: l, videoT: 0.3, audioT: 0.6, condVideoT: 0.999, condAudioT: 1.0)
        XCTAssertEqual(u.count, 3)
        XCTAssertEqual(Double(u[0]), 0.3, accuracy: 1e-6)
        XCTAssertEqual(Double(u[1]), 0.6, accuracy: 1e-6)
        XCTAssertEqual(Double(u[2]), 0.999, accuracy: 1e-6)
        XCTAssertEqual(inv[l.videoIndices[0]], 2)
        XCTAssertEqual(inv[l.audioIndices[0]], 1)
        XCTAssertEqual(inv[0], 0)
        let mr = modRuns(tIdx: inv, tags: l.tokenTags)
        // text(0,text) | vision placeholder(0,video) | cond video (2, video) | audio (1, audio) | video (0, video)
        XCTAssertEqual(mr.map { [$0.rows, $0.tIdx, $0.tag] }, [[2, 0, 1], [1, 0, 0], [12, 2, 0], [40, 1, 2], [42, 0, 0]])
        XCTAssertEqual(mr.map { $0.start }, [0, 2, 3, 15, 55])
    }

    func testRef2vaLayoutOrdersSegments() throws {
        let refs = [
            RefGeom(kind: "image", latentT: 0, latentH: 4, latentW: 4),
            RefGeom(kind: "video", latentT: 2, latentH: 4, latentW: 6, audioLatents: 3),
            RefGeom(kind: "audio", audioLatents: 5),
        ]
        let l = try buildRef2vaLayout(textTags: [Int](repeating: textTag, count: 4), refs: refs, lt: 7, lh: 4, lw: 6, na: 20, patch: [1, 2, 2])
        XCTAssertEqual(l.numConditionVideoRows, 4 + 12)
        XCTAssertEqual(l.numConditionAudioRows, 16)
        XCTAssertEqual(l.referenceSegments.map { $0.kind }, ["image", "audio", "video", "audio"])
        XCTAssertEqual(l.sequenceLength, 4 + 4 + 6 + 12 + 10 + 40 + 42)
        XCTAssertEqual(l.videoIndices.count, 16 + 42)
        XCTAssertEqual(l.audioIndices.count, 16 + 40)
        XCTAssertEqual(l.tokenTags.filter { $0 == audioTag }.count, 56)
        XCTAssertThrowsError(try buildRef2vaLayout(textTags: [textTag], refs: [], lt: 7, lh: 4, lw: 6, na: 20, patch: [1, 2, 2]))
        XCTAssertThrowsError(try buildRef2vaLayout(textTags: [textTag], refs: refs, lt: 7, lh: 4, lw: 6, na: 20, patch: [1, 2, 2], continuation: (7, 20)))
        let c = try buildRef2vaLayout(textTags: [textTag], refs: refs, lt: 7, lh: 4, lw: 6, na: 20, patch: [1, 2, 2], continuation: (2, 5))
        XCTAssertEqual(c.referenceSegments.map { $0.kind }, ["image", "audio", "video", "audio", "audio", "continuation"])
        XCTAssertEqual(c.numConditionVideoRows, 16 + 12)
        XCTAssertEqual(c.numConditionAudioRows, 16 + 10)
    }

    func testRunsAndPatchRows() {
        XCTAssertEqual(runs([3, 4, 5, 9, 10]), [Run(start: 3, rows: 3, offset: 0), Run(start: 9, rows: 2, offset: 3)])
        XCTAssertEqual(runs([]), [])
        var rng = Rng(seed: 0)
        let z = rng.normal(shape: [4, 3, 4, 6])
        let rows = patchRows(z, patch: [1, 2, 2])
        XCTAssertEqual(rows.shape, [3 * 2 * 3, 16])
        XCTAssertEqual(rows.data[0 * 16 + 1 * 4 + 1 * 2 + 0], z.data[((1 * 3 + 0) * 4 + 1) * 6 + 0])
        XCTAssertEqual(unpatchRows(rows, c: 4, t: 3, h: 4, w: 6, patch: [1, 2, 2]), z)
    }

    func testRopeTables() {
        let (cos, _) = Rope.h3Rope(positions: [0, 0, 0, 1, 2, 3], ropeFreqDim: 16, theta: 10000.0)
        XCTAssertEqual(cos.shape, [2, 48])
        XCTAssert(cos.row(0).data.allSatisfy { $0 == 1.0 })
        XCTAssertEqual(Double(cos.data[48 + 0]), Foundation.cos(1.0), accuracy: 1e-6)
        XCTAssertEqual(Double(cos.data[48 + 16]), Foundation.cos(2.0), accuracy: 1e-6)
        XCTAssertEqual(Double(cos.data[48 + 32]), Foundation.cos(3.0), accuracy: 1e-6)
        let fromPositions = Rope.h3Rope(positions: [Position(0, 0, 0), Position(1, 2, 3)], ropeFreqDim: 16, theta: 10000.0)
        XCTAssertEqual(fromPositions.cos, cos)
        let (c, _) = Rope.vaeDecoderRope(grid: (t: 1, h: 2, w: 2), rot: 48, theta: 100.0, suffix: 5)
        XCTAssertEqual(c.shape, [9, 24])
        for j in 0..<8 {
            XCTAssertEqual(Double(c.data[j]), 1.0, accuracy: 1e-6)
        }
        XCTAssertEqual(Double(c.data[8]), -1.0, accuracy: 1e-5)
        for i in 4..<9 {
            for j in 0..<24 {
                XCTAssertEqual(Double(c.data[i * 24 + j]), 1.0, accuracy: 1e-6)
            }
        }
    }

    // MARK: configs

    func testConfigsAndTaeRule() {
        XCTAssertEqual(H3Config.h3.rotaryDim, 96)
        XCTAssertEqual(H3Config.tiny.rotaryDim, 12)
        XCTAssertEqual(H3Config.fasth3Trim.layers, 42)
        XCTAssertEqual(H3Config.fasth3Omni.pddSteps, 32)
        XCTAssertEqual(H3Config.h3.dim, 7168)
        XCTAssertEqual(H3Config.h3.videoPatchDim, 96)
        XCTAssertEqual(H3Config.tiny.videoPatchDim, 16)
        let v = VideoVaeConfig.h3
        XCTAssertEqual([v.spatialRatio, v.temporalRatio, v.framePrePadding, v.tokensChunk, v.tokenOverlap, v.frameOverlap], [16, 4, 3, 5, 2, 5])
        XCTAssertEqual(v.decoderRows(latentT: 7, latentH: 16, latentW: 16), 1797)
        XCTAssertEqual(VideoVaeConfig.tiny.decoderRows(latentT: 7, latentH: 4, latentW: 6), 171)
        XCTAssertEqual(VideoVaeConfig.fasth3Trim.decoderLayers, 26)
        XCTAssertEqual(AudioVaeConfig.h3.hop, 800)
        XCTAssertEqual(AudioVaeConfig.tiny.hop, 10)
        func kept(_ n: Int) -> Int { (0..<(n * 4)).filter { TaeH3Config.keepsRawFrame($0) }.count }
        XCTAssertEqual([kept(5), kept(37), kept(2)], [17, 124, 5])
        XCTAssertEqual(TaeH3Config.taeh3.tUpscale, 4)
        XCTAssertEqual(TaeH3Config.taeh3.spatialScale, 16)
    }

    // MARK: rng, half, sinusoid

    func testRngIsSplitMix64BoxMuller() {
        // h3-core seeds the state as `seed ^ GOLDEN`, so the canonical SplitMix64 stream (state 0:
        // first output 0xE220A8397B1DCDAF) appears at seed GOLDEN.
        var r = Rng(seed: 0x9E37_79B9_7F4A_7C15)
        XCTAssertEqual(r.nextU64(), 0xE220_A839_7B1D_CDAF)
        XCTAssertEqual(r.nextU64(), 0x6E78_9E6A_A1B9_65F4)
        var g = Rng(seed: 7)
        let v = g.normal(200_000)
        XCTAssertEqual(v.count, 200_000)
        var mean = 0.0
        for x in v { mean += Double(x) }
        mean /= Double(v.count)
        var variance = 0.0
        for x in v { variance += (Double(x) - mean) * (Double(x) - mean) }
        variance /= Double(v.count)
        XCTAssertLessThan(abs(mean), 0.01)
        XCTAssertLessThan(abs(variance - 1.0), 0.02)
        // An odd count drops the second member of the last pair and consumes the pair.
        var c1 = Rng(seed: 3)
        let odd = c1.normal(7)
        var c2 = Rng(seed: 3)
        let even = c2.normal(8)
        XCTAssertEqual(odd, Array(even[0..<7]))
        XCTAssertEqual(c1.state, c2.state)
        let e = Timestep.sinusoid(0.5, dim: 8)
        XCTAssertEqual(Double(e[0]), Foundation.cos(0.5), accuracy: 1e-7)
        XCTAssertEqual(Double(e[4]), Foundation.sin(0.5), accuracy: 1e-7)
    }

    func testHalfConversions() {
        XCTAssertEqual(Half.toFloat(0x3C00), 1.0)
        XCTAssertEqual(Half.toFloat(0xC000), -2.0)
        XCTAssertEqual(Half.toFloat(0x3555), 0.333251953125)
        XCTAssertEqual(Half.toFloat(0x0001), 5.960464477539063e-08)
        XCTAssertEqual(Half.fromFloat(1.0), 0x3C00)
        XCTAssertEqual(Half.fromFloat(-2.0), 0xC000)
        XCTAssertEqual(Half.fromFloat(0.333251953125), 0x3555)
        XCTAssertEqual(Half.fromFloat(5.960464477539063e-08), 0x0001)
        XCTAssertEqual(Half.fromFloat(0.0), 0x0000)
        XCTAssertEqual(Half.fromFloat(-0.0), 0x8000)
        XCTAssertEqual(Half.fromFloat(65504.0), 0x7BFF)
        XCTAssertEqual(Half.fromFloat(70000.0), 0x7C00)  // overflow → inf
        XCTAssertEqual(Half.fromFloat(Float.infinity), 0x7C00)
        XCTAssert(Half.toFloat(Half.fromFloat(Float.nan)).isNaN)
        // Round to nearest even: 1 + 2^-11 is the tie between 1.0 and 1 + 2^-10.
        XCTAssertEqual(Half.fromFloat(1.0 + Float(1.0 / 2048.0)), 0x3C00)
        XCTAssertEqual(Half.fromFloat(1.0 + Float(3.0 / 2048.0)), 0x3C02)
        XCTAssertEqual(Half.round(0.1), 0.0999755859375)
        // Every half round-trips.
        for bits in stride(from: 0, through: 0xFFFF, by: 7) {
            let h = UInt16(bits)
            let f = Half.toFloat(h)
            if f.isNaN { continue }
            XCTAssertEqual(Half.fromFloat(f), h, "0x\(String(h, radix: 16))")
        }
    }

    // MARK: tensor

    func testTensorAxisHelpers() {
        let t = Tensor(shape: [2, 3, 2], data: (0..<12).map { Float($0) })
        XCTAssertEqual(t.row(1), Tensor(shape: [3, 2], data: [6, 7, 8, 9, 10, 11]))
        XCTAssertEqual(t.rows(1..<2).shape, [1, 3, 2])
        XCTAssertEqual(Tensor.concatRows([t.rows(0..<1), t.rows(1..<2)]), t)
        XCTAssertEqual(t.rowStride, 6)
        XCTAssertEqual(t.slice(axis: 1, 1..<3), Tensor(shape: [2, 2, 2], data: [2, 3, 4, 5, 8, 9, 10, 11]))
        XCTAssertEqual(t.slice(axis: -1, 1..<2), Tensor(shape: [2, 3, 1], data: [1, 3, 5, 7, 9, 11]))
        XCTAssertEqual(Tensor.concat([t.slice(axis: 1, 0..<1), t.slice(axis: 1, 1..<3)], axis: 1), t)
        XCTAssertEqual(Tensor.concat([t.slice(axis: 2, 0..<1), t.slice(axis: 2, 1..<2)], axis: -1), t)
        XCTAssertEqual(t.repeated(axis: 1, index: 2, times: 2), Tensor(shape: [2, 2, 2], data: [4, 5, 4, 5, 10, 11, 10, 11]))
        var u = t
        u.setRows(1..<2, Tensor(shape: [1, 3, 2], data: [Float](repeating: -1, count: 6)))
        XCTAssertEqual(u.data[6...].allSatisfy { $0 == -1 }, true)
        XCTAssertEqual(Array(u.data[0..<6]), Array(t.data[0..<6]))
        XCTAssertEqual(Tensor(shape: [2, 2], data: [1, 2, 3, 4]).channelAffine(scale: [2, 3], shift: [1, 0]), Tensor(shape: [2, 2], data: [3, 5, 9, 12]))
    }

    // MARK: sampler

    func testFlowStep() throws {
        let x = Tensor(shape: [2], data: [1.0, -2.0])
        let v = Tensor(shape: [2], data: [0.5, 0.5])
        let last = try Sampler.flowStep(x: x, v: v, sigma: 0.5, next: 0.0, rule: .euler, noise: nil)
        // r = 0: x0 = x + σ·v
        XCTAssertEqual(last.data[0], 1.0 + Float(0.5) * 0.5)
        XCTAssertEqual(last.data[1], -2.0 + Float(0.5) * 0.5)
        let mid = try Sampler.flowStep(x: x, v: v, sigma: 1.0, next: 0.75, rule: .euler, noise: nil)
        let r: Float = 0.75
        XCTAssertEqual(mid.data[0], r * 1.0 + (1 - r) * 1.5)
        XCTAssertEqual(mid.data[1], r * -2.0 + (1 - r) * -1.5)
        let noise = Tensor(shape: [2], data: [2.0, 3.0])
        let re = try Sampler.flowStep(x: x, v: v, sigma: 1.0, next: 0.75, rule: .renoise, noise: noise)
        XCTAssertEqual(re.data[0], (1 - r) * 1.5 + r * 2.0)
        XCTAssertEqual(re.data[1], (1 - r) * -1.5 + r * 3.0)
        let reLast = try Sampler.flowStep(x: x, v: v, sigma: 0.5, next: 0.0, rule: .renoise, noise: nil)
        XCTAssertEqual(reLast, last)
        XCTAssertThrowsError(try Sampler.flowStep(x: x, v: v, sigma: 1.0, next: 0.75, rule: .renoise, noise: nil))
        XCTAssertThrowsError(try Sampler.flowStep(x: x, v: Tensor(shape: [3], data: [0, 0, 0]), sigma: 1.0, next: 0.5, rule: .euler, noise: nil))
        XCTAssert(Sampler.needsNoise(rule: .renoise, next: 0.5))
        XCTAssertFalse(Sampler.needsNoise(rule: .renoise, next: 0.0))
        XCTAssertFalse(Sampler.needsNoise(rule: .euler, next: 0.5))
        XCTAssertFalse(Sampler.needsNoise(rule: .pdd, next: 0.5))
    }

    // MARK: VAE host

    func testSplitTilesAndTemporalPlan() throws {
        let (starts, lengths, overlaps) = VaeHost.splitTiles(length: 768, tile: 256, minOverlap: 64, ratio: 16)
        XCTAssertEqual(starts, [0, 160, 336, 512])
        XCTAssertEqual(lengths, [256, 256, 256, 256])
        XCTAssertEqual(overlaps, [96, 80, 80])
        let one = VaeHost.splitTiles(length: 96, tile: 256, minOverlap: 64, ratio: 16)
        XCTAssertEqual(one.starts, [0])
        XCTAssertEqual(one.lengths, [96])
        XCTAssertEqual(one.overlaps, [])
        let wide = VaeHost.splitTiles(length: 1344, tile: 256, minOverlap: 64, ratio: 16)
        XCTAssertEqual(wide.starts[wide.starts.count - 1] + 256, 1344)
        let h3 = VideoVaeConfig.h3
        XCTAssertEqual(try VaeHost.temporalPlan(h3, latentFrames: 37).pad, 0)
        XCTAssertEqual(try VaeHost.temporalPlan(h3, latentFrames: 37).chunks, 7)
        XCTAssertEqual(try VaeHost.temporalPlan(h3, latentFrames: 37).frames, 124)
        XCTAssertEqual(try VaeHost.temporalPlan(h3, latentFrames: 7).pad, 0)
        XCTAssertEqual(try VaeHost.temporalPlan(h3, latentFrames: 7).chunks, 1)
        XCTAssertEqual(try VaeHost.temporalPlan(h3, latentFrames: 7).frames, 22)
        XCTAssertEqual(try VaeHost.temporalPlan(h3, latentFrames: 2).pad, 5)
        XCTAssertEqual(try VaeHost.temporalPlan(h3, latentFrames: 2).chunks, 1)
        XCTAssertEqual(try VaeHost.temporalPlan(h3, latentFrames: 2).frames, 5)
        XCTAssertEqual(try VaeHost.temporalPlan(h3, latentFrames: 107).frames, 362)
        XCTAssertThrowsError(try VaeHost.temporalPlan(h3, latentFrames: 0))
    }

    func testBlendStitchAndTiled() throws {
        // blend: a's last 2 fade into b's first 2 along the last axis, then the rest of b.
        let a = Tensor(shape: [1, 1, 1, 4], data: [0, 0, 10, 10])
        let b = Tensor(shape: [1, 1, 1, 4], data: [20, 20, 30, 40])
        XCTAssertEqual(VaeHost.blend(a, b, extent: 2, axis: -1), Tensor(shape: [1, 1, 1, 4], data: [10, 15, 30, 40]))
        XCTAssertEqual(VaeHost.blend(a, b, extent: 0, axis: -1), b)
        // tiled over a 1-channel, 1-frame [1, 1, 4, 8] "latent" at ratio 1 with 4-px tiles and 2-px overlap:
        // identity tiles stitch back to the input exactly (cross-fading equal values is the identity).
        let x = Tensor(shape: [1, 1, 4, 8], data: (0..<32).map { Float($0) })
        let (xs, _, xo) = VaeHost.splitTiles(length: 8, tile: 4, minOverlap: 2, ratio: 1)
        XCTAssertEqual(xs, [0, 2, 4])
        XCTAssertEqual(xo, [2, 2])
        var seen: [[Int]] = []
        let y = try VaeHost.tiled(x, ratioIn: 1, ratioOut: 1, tilePx: 4, overlapPx: 2) { t in
            seen.append(t.shape)
            return t
        }
        XCTAssertEqual(seen.count, 3)
        XCTAssertEqual(seen[0], [1, 1, 4, 4])
        XCTAssertEqual(y, x)
        // A single tile is returned as is.
        let z = try VaeHost.tiled(x, ratioIn: 1, ratioOut: 1, tilePx: 16, overlapPx: 2) { $0.scaled(2) }
        XCTAssertEqual(z, x.scaled(2))
    }

    func testDecodeChunksAndBlends() throws {
        // Tiny config, 7 latent frames: one chunk of 7, frames 3..<20 kept, then the 5 overlap frames.
        let cfg = VideoVaeConfig.tiny
        var calls: [[Int]] = []
        let z = Tensor(shape: [4, 7, 1, 1], data: (0..<28).map { Float($0) })
        var chunks: [Int] = []
        let px = try VaeHost.decode(cfg, z: z, decodeClip: { clip in
            calls.append(clip.shape)
            // ImageNet-normalised constant 0 → mean after denormalisation; frames numbered on channel 0.
            var out = Tensor.zeros([3, clip.shape[1] * 4, 1, 1])
            for f in 0..<out.shape[1] { out.data[f] = Float(f) * 0.01 }
            return out
        }, onChunk: { chunks.append($0.shape[1]) })
        XCTAssertEqual(calls, [[4, 7, 1, 1]])
        XCTAssertEqual(chunks, [17, 5])
        XCTAssertEqual(px.shape, [3, 22, 1, 1])
        // frame 0 of the output is raw frame 3: 0.03·std + mean on channel 0.
        XCTAssertEqual(px.data[0], 0.03 * 0.229 + 0.485, accuracy: 1e-6)
        XCTAssertEqual(px.data[22 + 0], 0.456, accuracy: 1e-6)  // channel 1 is the mean
        // 2 latent frames: padded by 5 repeats of the last frame, 5 output frames.
        let short = Tensor(shape: [4, 2, 1, 1], data: (0..<8).map { Float($0) })
        var seenZ: [Tensor] = []
        let p2 = try VaeHost.decode(cfg, z: short, decodeClip: { clip in
            seenZ.append(clip)
            return Tensor.zeros([3, clip.shape[1] * 4, 1, 1])
        })
        XCTAssertEqual(p2.shape, [3, 5, 1, 1])
        XCTAssertEqual(seenZ.count, 1)
        XCTAssertEqual(seenZ[0].shape, [4, 7, 1, 1])
        XCTAssertEqual(Array(seenZ[0].data[0..<7]), [0, 1, 1, 1, 1, 1, 1])  // channel 0: frames 0, 1, then 1 repeated
        let empty = try VaeHost.decode(cfg, z: short, decodeClip: { Tensor.zeros([3, $0.shape[1] * 4, 1, 1]) }, collect: false)
        XCTAssertEqual(empty.shape, [3, 0, 1, 1])
    }

    func testEncodeNormaliseAndPosterior() throws {
        let cfg = VideoVaeConfig.tiny
        // A single frame goes through alone.
        var shapes: [[Int]] = []
        let one = try VaeHost.encode(cfg, pixels: Tensor.zeros([3, 1, 16, 16])) { x in
            shapes.append(x.shape)
            return Tensor.zeros([8, 1, 1, 1])
        }
        XCTAssertEqual(shapes, [[3, 1, 16, 16]])
        XCTAssertEqual(one.shape, [8, 1, 1, 1])
        // 20 frames: padded to 34 (two 17-frame clips), 5 latents each, the trailing 3 dropped.
        shapes = []
        let many = try VaeHost.encode(cfg, pixels: Tensor.zeros([3, 20, 16, 16])) { x in
            shapes.append(x.shape)
            return Tensor.zeros([8, 5, 1, 1])
        }
        XCTAssertEqual(shapes, [[3, 17, 16, 16], [3, 17, 16, 16]])
        XCTAssertEqual(many.shape, [8, 7, 1, 1])
        // Pixel normalisation round-trips (inside [0, 1]).
        let px = Tensor(shape: [3, 1, 1, 2], data: [0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        let back = VaeHost.denormalizePixels(VaeHost.normalizePixels(px))
        for i in 0..<6 {
            XCTAssertEqual(back.data[i], px.data[i], accuracy: 1e-6)
        }
        XCTAssertEqual(VaeHost.denormalizePixels(Tensor(shape: [3, 1, 1, 1], data: [10, -10, 0])).data, [1.0, 0.0, 0.406])
        // Latent normalisation round-trips.
        let z = Tensor(shape: [4, 1, 1, 1], data: [1, 2, 3, 4])
        let zz = VaeHost.normalizeLatents(cfg, VaeHost.denormalizeLatents(cfg, z))
        for i in 0..<4 {
            XCTAssertEqual(zz.data[i], z.data[i], accuracy: 1e-6)
        }
        XCTAssertEqual(VaeHost.denormalizeLatents(cfg, z).data[0], 1 * 1.1 + 0.1, accuracy: 1e-6)
        // Posterior: mean + exp(logvar/2)·ε with the logvar clamped.
        let moments = Tensor(shape: [2, 1, 1, 1], data: [0.5, 0.0])
        XCTAssertEqual(VaeHost.posteriorSample(moments: moments, noise: Tensor(shape: [1, 1, 1, 1], data: [2.0])).data, [2.5])
        let clamped = Tensor(shape: [2, 1, 1, 1], data: [0.0, 100.0])
        XCTAssertEqual(Double(VaeHost.posteriorSample(moments: clamped, noise: Tensor(shape: [1, 1, 1, 1], data: [1.0])).data[0]), exp(10.0), accuracy: 1.0)
    }

    func testPixelConversions() {
        // [3, F=1, H=1, W=2]: pixel 0 = (0.5, -1, 0), pixel 1 = (1, 0.999, 2).
        let cm = Tensor(shape: [3, 1, 1, 2], data: [0.5, 1.0, -1.0, 0.999, 0.0, 2.0])
        let a = Pixels.rgbaFromChannelMajor(cm)
        XCTAssertEqual(a.rgba, [128, 0, 0, 255, 255, 255, 255, 255])
        XCTAssertEqual([a.frames, a.height, a.width], [1, 1, 2])
        let fm = Tensor(shape: [1, 3, 1, 2], data: [0.5, 1.0, -1.0, 0.999, 0.0, 2.0])
        XCTAssertEqual(Pixels.rgbaFromFrameMajor(fm).rgba, a.rgba)
        XCTAssertEqual(Pixels.rgbaToRgb(a.rgba), [128, 0, 0, 255, 255, 255])
        let kf = Pixels.keyframePixels(rgba: [255, 0, 0, 255, 0, 255, 0, 255], width: 2, height: 1)
        XCTAssertEqual(kf.shape, [3, 1, 1, 2])
        XCTAssertEqual(kf.data, [1, 0, 0, 1, 0, 0])
        let audio = Audio(sampleRate: 320, channels: 2, samples: 2, data: [0.5, -0.5, 1.0, -1.0])
        XCTAssertEqual(Array(audio.channel(1)), [1.0, -1.0])
        let wav = audio.wav()
        XCTAssertEqual(wav.count, 44 + 8)
        XCTAssertEqual(Array(wav[0..<4]), Array("RIFF".utf8))
    }

    // MARK: manifest, adaln, safetensors, plan

    func testManifestDecodesAndRefusesMismatches() throws {
        let m = try Manifest(json: tinyManifestData())
        XCTAssertEqual(m.format, "h3-litert/1")
        XCTAssertEqual(m.model, "tiny")
        XCTAssertEqual(m.variant, "tiny")
        XCTAssertEqual(m.config, H3Config.tiny)
        XCTAssertEqual(m.vae, VideoVaeConfig.tiny)
        XCTAssertEqual(m.audioVae, AudioVaeConfig.tiny)
        XCTAssertEqual(m.taeh3, TaeH3Config.tiny)
        XCTAssertEqual(m.chunk, 16)
        XCTAssertEqual(m.headGroup, 1)
        XCTAssertEqual(m.weights, "f32")
        XCTAssertEqual(m.graphs.blocks.count, 2)
        XCTAssertEqual(m.graphs.blocks[1].qkv, "block_01_qkv.tflite")
        XCTAssertEqual(m.graphs.blocks[1].post, "block_01_post.tflite")
        XCTAssertEqual(m.graphs.textRefine.map { $0.rows }, [8, 16])
        XCTAssertEqual(m.sampler.rule, .euler)
        XCTAssertEqual(m.sampler.steps.count, 2)
        XCTAssertEqual(m.sampler.shifts, Shifts.fasth3V2)
        XCTAssertEqual(m.sampler.condVideoT, 0.999)
        XCTAssertEqual(m.sampler.condAudioT, 1.0)
        XCTAssertEqual(m.sampler.steps[1].sigmaVideoNext, 0.0)
        XCTAssertEqual(m.sampler.steps[0].sigmaAudioNext, 0.75)
        XCTAssertNil(m.sampler.steps[0].pddVideoWeights)
        XCTAssertEqual(m.adaln.file, "adaln.safetensors")
        XCTAssertEqual(m.adaln.timesteps.count, 6)
        XCTAssertEqual(try m.attentionFor(tokens: 124), Manifest.AttentionEntry(tokens: 124, qChunk: 32, heads: 1, file: "attn_s124_q32_h1.tflite"))
        XCTAssertEqual(try m.textRefineFor(rows: 6).rows, 8)
        XCTAssertEqual(try m.textRefineFor(rows: 8).rows, 8)
        XCTAssertEqual(try m.textRefineFor(rows: 9).file, "text_refine_16.tflite")
        XCTAssertEqual(try m.audioFor(latents: 37).decoder, "avae_dec_37.tflite")
        XCTAssertEqual(try m.audioFor(latents: 37).encoder, "avae_enc_37.tflite")
        XCTAssertEqual(try m.taeh3For(latentH: 4, latentW: 6).file, "taeh3_f5_4x6.tflite")
        XCTAssertEqual(m.videoVae?.decoder.rows, 171)
        XCTAssertEqual(m.videoVae?.decoder.input, "vae_dec_in.tflite")
        XCTAssertEqual(m.videoVae?.decoder.output, "vae_dec_out.tflite")
        XCTAssertEqual(m.videoVae?.decoder.blocks.count, 2)
        XCTAssertEqual(m.videoVae?.encoderImage, "vae_enc_image.tflite")
        XCTAssertEqual(m.videoVae?.tile, 256)
        XCTAssertEqual(m.sizes, [Geometry.tiny])
        XCTAssertEqual(m.files?["block_00_qkv.tflite"]?.bytes, 24904)
        XCTAssertEqual(m.graphFiles.count, 26)
        XCTAssertEqual(m.allFiles.count, 27)
        XCTAssertThrowsError(try m.attentionFor(tokens: 38396)) { e in
            XCTAssert("\(e)".contains("38396") && "\(e)".contains("124"), "\(e)")
        }
        XCTAssertThrowsError(try m.textRefineFor(rows: 17)) { e in
            XCTAssert("\(e)".contains("17") && "\(e)".contains("16"), "\(e)")
        }
        XCTAssertThrowsError(try m.audioFor(latents: 207)) { e in
            XCTAssert("\(e)".contains("207") && "\(e)".contains("37"), "\(e)")
        }
        XCTAssertThrowsError(try m.taeh3For(latentH: 48, latentW: 84)) { e in
            XCTAssert("\(e)".contains("48x84"), "\(e)")
        }
        var raw = try JSONSerialization.jsonObject(with: tinyManifestData()) as! [String: Any]
        raw["format"] = "h3-litert/2"
        let other = try JSONSerialization.data(withJSONObject: raw)
        XCTAssertThrowsError(try Manifest(json: other)) { e in
            XCTAssert("\(e)".contains("h3-litert/2") && "\(e)".contains("h3-litert/1"), "\(e)")
        }
    }

    func testAdalnTablesFromTheFixture() throws {
        let m = try Manifest(json: tinyManifestData())
        let t = try AdalnTables(url: tinyAdalnURL())
        XCTAssertEqual(t.count, 6)
        XCTAssertEqual(t.layers, 2)
        XCTAssertEqual(t.hidden, 32)
        XCTAssertEqual(t.timesteps.map { Double($0) }, m.adaln.timesteps)
        for (i, ts) in m.adaln.timesteps.enumerated() {
            XCTAssertEqual(try t.indexOf(ts), i)
        }
        XCTAssertEqual(try t.indexOf(m.sampler.steps[0].tVideo), 0)
        XCTAssertEqual(try t.indexOf(m.sampler.steps[0].tAudio), 1)
        XCTAssertEqual(try t.indexOf(m.sampler.steps[1].tVideo), 2)
        XCTAssertEqual(try t.indexOf(m.sampler.steps[1].tAudio), 3)
        XCTAssertEqual(try t.indexOf(max(m.sampler.steps[0].tVideo, m.sampler.condVideoT)), 4)
        XCTAssertEqual(try t.indexOf(m.sampler.condAudioT), 5)
        XCTAssertThrowsError(try t.indexOf(Float(0.5))) { e in
            XCTAssert("\(e)".contains("0.5"), "\(e)")
        }
        let mods = t.mods(u: 0, layer: 1, tag: audioTag)
        XCTAssertEqual(mods.shape, [6, 32])
        XCTAssert(mods.data.allSatisfy { $0.isFinite })
        XCTAssertEqual(t.headMods(u: 5).shape, [2, 32])
        // The same table built from floats rounds through f16 and reads back identically.
        let st = try SafeTensors(url: tinyAdalnURL())
        let rebuilt = try AdalnTables(timesteps: t.timesteps, blocks: try st.float32("blocks"), head: try st.float32("head"))
        XCTAssertEqual(rebuilt.mods(u: 3, layer: 0, tag: textTag), t.mods(u: 3, layer: 0, tag: textTag))
        XCTAssertEqual(rebuilt.headMods(u: 2), t.headMods(u: 2))
        XCTAssertEqual(t.bytes, (6 * 2 * 3 * 6 * 32 + 6 * 2 * 32) * 2 + 6 * 4)
    }

    func testSafeTensorsReader() throws {
        // Hand-built file: header length, JSON header, then F32 [2, 2], U8 [3] and F16 [2] data.
        let header = "{\"__metadata__\":{\"seed\":\"7\"},\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
            + "\"b\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[16,19]},\"c\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[19,23]}}"
        var file = Data()
        var n = UInt64(header.utf8.count).littleEndian
        file.append(Data(bytes: &n, count: 8))
        file.append(header.data(using: .utf8)!)
        let floats: [Float] = [1.0, -2.5, 3.25, 0.0]
        for f in floats {
            var bits = f.bitPattern.littleEndian
            file.append(Data(bytes: &bits, count: 4))
        }
        file.append(contentsOf: [7, 8, 9])
        file.append(contentsOf: [0x00, 0x3C, 0x00, 0xC0])  // 1.0, -2.0 in binary16
        let st = try SafeTensors(data: file)
        XCTAssertEqual(st.metadata["seed"], "7")
        XCTAssertEqual(st.names, ["a", "b", "c"])
        XCTAssert(st.contains("a"))
        XCTAssertFalse(st.contains("zzz"))
        XCTAssertEqual(try st.float32("a"), Tensor(shape: [2, 2], data: floats))
        let b = try st.uint8("b")
        XCTAssertEqual(b.shape, [3])
        XCTAssertEqual(b.data, [7, 8, 9])
        XCTAssertEqual(try st.float32("c"), Tensor(shape: [2], data: [1.0, -2.0]))
        XCTAssertEqual(try st.halfBits("c").bits, [0x3C00, 0xC000])
        XCTAssertThrowsError(try st.float32("b"))
        XCTAssertThrowsError(try st.halfBits("a"))
        XCTAssertThrowsError(try st.float32("zzz"))
    }

    func testPlanOnTheTinyExport() throws {
        let m = try Manifest(json: tinyManifestData())
        let p = try Plan.make(manifest: m, geometry: .tiny, textRows: 6, budgetBytes: 1 << 30)
        XCTAssert(p.fits, p.description)
        XCTAssertEqual(p.tokens, 124)
        XCTAssertEqual(p.textBucket, 8)
        XCTAssertEqual(p.qChunk, 32)
        XCTAssertEqual(p.headGroup, 1)
        XCTAssertEqual(p.blockBytes, 54368)
        XCTAssertEqual(p.weightsTotal, m.files!.values.reduce(0) { $0 + $1.bytes })
        XCTAssertEqual(p.adalnBytes, 14832)
        XCTAssert(p.notes.contains { $0.contains("padded to the 8-row text bucket") }, "\(p.notes)")
        let fl = try Plan.make(manifest: m, geometry: .tiny, textRows: 6, keyframes: 2, budgetBytes: 1 << 30)
        XCTAssertEqual(fl.tokens, 136)
        let preview = try Plan.make(manifest: m, geometry: .tiny, textRows: 6, preview: true, budgetBytes: 1 << 30)
        XCTAssert(preview.fits)
        XCTAssertFalse(try Plan.make(manifest: m, geometry: .tiny, textRows: 6, budgetBytes: 1 << 10).fits)
        XCTAssertThrowsError(try Plan.make(manifest: m, geometry: .tiny, textRows: 17, budgetBytes: 1 << 30))
        XCTAssertThrowsError(try Plan.make(manifest: m, geometry: .h3_768p, textRows: 6, budgetBytes: 1 << 30))
    }
}

/// The tiny fixture's manifest (fixtures/tiny/manifest.json), copied into the test resources.
func tinyManifestData() throws -> Data {
    guard let url = Bundle.module.url(forResource: "manifest", withExtension: "json") else {
        throw H3Error.io("manifest.json is missing from the test resources")
    }
    return try Data(contentsOf: url)
}

/// The tiny fixture's AdaLN table (fixtures/tiny/adaln.safetensors), copied into the test resources.
func tinyAdalnURL() throws -> URL {
    guard let url = Bundle.module.url(forResource: "adaln", withExtension: "safetensors") else {
        throw H3Error.io("adaln.safetensors is missing from the test resources")
    }
    return url
}
