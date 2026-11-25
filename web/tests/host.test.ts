/**
 * The host math against the values h3-rs's tests pin down: every case of
 * `python/tests/test_host.py`, plus the VAE host side (`vae_host.py`), the
 * f16 reader, the manifest validation and the memory plan.
 */

import { describe, expect, it } from "vitest";

import { AdalnTables } from "../src/adaln.js";
import { AUDIO_TAG, H3Config, TEXT_TAG, TaeH3Config, VIDEO_TAG, VideoVaeConfig } from "../src/config.js";
import { f16BitsToF32, f32ToF16Bits, roundF16 } from "../src/f16.js";
import { parseManifest } from "../src/manifest.js";
import {
  H3_768P,
  Geometry,
  alignNumFrames,
  audioLatents,
  buildLayout,
  buildRef2vaLayout,
  canvasSize,
  modRuns,
  patchRows,
  resolveNumFrames,
  roundHalfEven,
  rowTimesteps,
  runs,
  unpatchRows,
  videoLatentFrames,
} from "../src/packing.js";
import { plan } from "../src/plan.js";
import { Rng, mul64 } from "../src/rng.js";
import { h3Rope, vaeDecoderRope } from "../src/rope.js";
import { float32Tensor, parseSafetensors } from "../src/safetensors.js";
import { flowStep } from "../src/sampler.js";
import { FASTH3_V2_RUNGS, PIKA_KNOTS, PLUG4_KNOTS, PddPlan, SHIFTS_H3, Schedule, shiftSigma, stepPlan } from "../src/schedule.js";
import { sinusoid } from "../src/timestep.js";
import { blend, nd, splitTiles, stitchTiles, temporalPlan, type Nd } from "../src/vaeHost.js";

const f = Math.fround;

function bits(x: number): number {
  const b = new Float32Array([x]);
  return new Uint32Array(b.buffer)[0];
}

describe("schedule", () => {
  it("the V2 ladder", () => {
    const s = Schedule.fromRungs(FASTH3_V2_RUNGS, 10.0);
    expect(s.steps).toBe(8);
    expect(Math.abs(s.sigmas[0] - shiftSigma(0.999, 10.0))).toBeLessThan(1e-7);
    expect(Math.abs(s.sigmas[4] - (10.0 * 0.5) / 5.5)).toBeLessThan(1e-6);
    expect(s.sigmas[s.sigmas.length - 1]).toBe(0.0);
    expect(Math.abs(s.timestep(4) - (1.0 - (10.0 * 0.5) / 5.5))).toBeLessThan(1e-6);
    const a = Schedule.fromRungs(FASTH3_V2_RUNGS, 3.0);
    expect(a.sigmas[4]).toBeGreaterThan(0.74);
    expect(a.sigmas[4]).toBeLessThan(0.76);
  });

  it("the uniform grid and the Pika knots", () => {
    const s = Schedule.uniform(50, 12.0);
    expect(s.steps).toBe(49);
    expect(s.sigmas[0]).toBe(1.0);
    const p = Schedule.fromRungs(PIKA_KNOTS, 12.0, "renoise");
    expect(p.steps).toBe(6);
    expect(p.sigmas[0]).toBe(1.0);
    expect(Math.abs(p.sigmas[5] - shiftSigma(0.04, 12.0))).toBeLessThan(1e-7);
  });

  it("the LongLive-Plug grid matches Diffusers bit for bit", () => {
    const want: Record<number, number[]> = {
      12.0: [0x3f800000, 0x3f7914c2, 0x3f6c4ec5, 0x3f4ccccd, 0],
      3.0: [0x3f800000, 0x3f666666, 0x3f400000, 0x3f000000, 0],
    };
    for (const [shift, b] of Object.entries(want)) {
      const s = Schedule.fromRungs(PLUG4_KNOTS, Number(shift), "renoise");
      expect(s.sigmas.map(bits)).toEqual(b);
      expect(Schedule.uniform(5, Number(shift)).sigmas).toEqual(s.sigmas);
      for (let i = 0; i < 4; i++) expect(1.0 - s.timestep(i)).toBe(s.sigmas[i]);
    }
  });

  it("the PDD Omni plan", () => {
    const p = PddPlan.create([0, 4, 8, 12, 16, 20, 24, 28, 32], SHIFTS_H3);
    expect(p.fineGrid.length - 1).toBe(32);
    expect(p.numSteps()).toBe(8);
    expect(p.block(0)).toEqual([0, 4]);
    expect(Math.abs(p.videoSigmas[0] - 0.999)).toBeLessThan(1e-6);
    expect(p.videoSigmas[p.videoSigmas.length - 1]).toBe(0.0);
    expect(p.videoWeights.every((w) => w < 0)).toBe(true);
    expect(Math.abs(PddPlan.fusedWeights(p.videoWeights, 0, 4).reduce((a, b) => a + b, 0) - 1.0)).toBeLessThan(1e-6);
    const s48 = p.videoWeights.slice(4, 8).reduce((a, b) => a + b, 0);
    expect(Math.abs(s48 - (p.videoSigmas[2] - p.videoSigmas[1]))).toBeLessThan(1e-6);
    const [v, a] = p.schedules();
    const steps = stepPlan(v, a, p, 32);
    expect(steps).toHaveLength(8);
    expect(steps[1].pdd_video_weights?.filter((w) => w !== 0)).toHaveLength(4);
    expect(steps[1].pdd_video_weights?.slice(0, 4)).toEqual([0, 0, 0, 0]);
  });
});

describe("sampler", () => {
  it("flow_step is float32 `x0 = x + σv` blended by σ'/σ (Euler) or re-noised", () => {
    const x = Float32Array.of(1, -2, 0.5);
    const v = Float32Array.of(0.25, 0.5, -1);
    const e = Float32Array.of(0.1, 0.2, 0.3);
    const eu = flowStep(x, v, 0.909090936183929, 0.5, "euler", null);
    const s = f(0.909090936183929);
    const r = f(f(0.5) / s);
    for (let i = 0; i < 3; i++) {
      const x0 = f(x[i] + f(s * v[i]));
      expect(eu[i]).toBe(f(f(r * x[i]) + f(f(f(1) - r) * x0)));
    }
    const last = flowStep(x, v, 0.75, 0.0, "euler", null);
    for (let i = 0; i < 3; i++) expect(last[i]).toBe(f(x[i] + f(f(0.75) * v[i])));
    const rn = flowStep(x, v, 1.0, 0.5, "renoise", e);
    for (let i = 0; i < 3; i++) expect(rn[i]).toBe(f(f(f(0.5) * f(x[i] + v[i])) + f(f(0.5) * e[i])));
    expect(flowStep(x, v, 0.5, 0.0, "renoise", null)).toEqual(last.map((_, i) => f(x[i] + f(f(0.5) * v[i]))));
    expect(() => flowStep(x, v, 1.0, 0.5, "renoise", null)).toThrow();
  });
});

describe("packing", () => {
  it("frame and latent counts", () => {
    expect([alignNumFrames(120), alignNumFrames(240), alignNumFrames(360)]).toEqual([124, 243, 362]);
    expect([videoLatentFrames(124), videoLatentFrames(362)]).toEqual([37, 107]);
    expect([audioLatents(124), audioLatents(243)]).toEqual([207, 405]);
    expect(resolveNumFrames(5 * 24)).toBe(124);
    expect(() => videoLatentFrames(120)).toThrow(/17n \+ 5/);
    expect([roundHalfEven(0.5), roundHalfEven(1.5), roundHalfEven(2.5), roundHalfEven(2.6)]).toEqual([0, 2, 2, 3]);
  });

  it("canvas rules", () => {
    expect(canvasSize(16, 9)).toEqual([768, 1344]);
    expect(canvasSize(9, 16)).toEqual([1344, 768]);
    expect(canvasSize(1, 1)).toEqual([768, 768]);
    const [h, w] = canvasSize(21, 9);
    expect(h * w).toBeLessThanOrEqual(768 * 1344 + 32 * 32);
    expect(h % 32).toBe(0);
    expect(w % 32).toBe(0);
    expect(w).toBeGreaterThan(h);
    expect(() => canvasSize(5, 1)).toThrow(/1:4 to 4:1/);
  });

  it("the 768p T2VA layout", () => {
    const [t, h, w] = H3_768P.latent(16);
    expect([t, h, w]).toEqual([37, 48, 84]);
    const l = buildLayout(new Array<number>(10).fill(TEXT_TAG), t, h, w, H3_768P.audioLatents(), [1, 2, 2], []);
    const video = 37 * 24 * 42;
    expect(l.sequenceLength).toBe(10 + 2 * 207 + video);
    expect(l.audioIndices).toHaveLength(414);
    expect(l.positionIds[3]).toEqual([3.0, 0.0, 0.0]);
    const a0 = l.positionIds[l.audioIndices[0]];
    expect(a0[0]).toBe(10.0);
    expect(a0[1]).toBe(0.0);
    const ar = l.positionIds[l.audioIndices[207]];
    expect(ar[0]).toBe(a0[0]);
    expect(ar[2]).toBeGreaterThan(a0[2]);
    const [v0, v1, v2] = [0, 1, 2].map((k) => l.positionIds[l.videoIndices[k * 24 * 42]]);
    expect(v0[0]).toBe(10.0);
    expect(Math.abs(v1[0] - (10 + 5 / 3))).toBeLessThan(1e-9);
    expect(Math.abs(v2[0] - (10 + 5 / 3 + (4 * 5) / 3))).toBeLessThan(1e-9);
  });

  it("FL2VA anchors, row timesteps and modulation runs", () => {
    const l = buildLayout([TEXT_TAG, TEXT_TAG, VIDEO_TAG], 7, 4, 6, 20, [1, 2, 2], ["first", "last"]);
    expect(l.numConditionVideoRows).toBe(12);
    expect(l.positionIds[l.videoIndices[0]][0]).toBe(3.0);
    expect(Math.abs(l.positionIds[l.videoIndices[6]][0] - (3.0 + (5.0 / 3.0) * 22.0 - 5.0 / 3.0))).toBeLessThan(1e-9);
    const { unique, index } = rowTimesteps(l, 0.3, 0.6, 0.999, 1.0);
    expect(unique.map((x) => Math.round(x * 1e6) / 1e6)).toEqual([0.3, 0.6, 0.999]);
    expect(index[l.videoIndices[0]]).toBe(2);
    expect(index[l.audioIndices[0]]).toBe(1);
    expect(index[0]).toBe(0);
    const mr = modRuns(index, l.tokenTags);
    // text(0,text) | vision placeholder(0,video) | cond video (2, video) | audio (1, audio) | video (0, video)
    expect(mr.map((r) => [r.rows, r.tIdx, r.tag])).toEqual([
      [2, 0, TEXT_TAG],
      [1, 0, VIDEO_TAG],
      [12, 2, VIDEO_TAG],
      [40, 1, AUDIO_TAG],
      [42, 0, VIDEO_TAG],
    ]);
  });

  it("Ref2VA orders the segments", () => {
    const refs = [
      { kind: "image" as const, latentT: 0, latentH: 4, latentW: 4 },
      { kind: "video" as const, latentT: 2, latentH: 4, latentW: 6, audioLatents: 3 },
      { kind: "audio" as const, audioLatents: 5 },
    ];
    const l = buildRef2vaLayout(new Array<number>(4).fill(TEXT_TAG), refs, 7, 4, 6, 20, [1, 2, 2]);
    expect(l.numConditionVideoRows).toBe(4 + 12);
    expect(l.numConditionAudioRows).toBe(16);
    expect(l.referenceSegments.map((s) => s.kind)).toEqual(["image", "audio", "video", "audio"]);
    expect(l.sequenceLength).toBe(4 + 4 + 6 + 12 + 10 + 40 + 42);
    expect(() => buildRef2vaLayout([TEXT_TAG], [], 7, 4, 6, 20, [1, 2, 2])).toThrow(/at least one reference/);
  });

  it("runs and patch rows round-trip", () => {
    expect(runs([3, 4, 5, 9, 10]).map((r) => [r.start, r.rows, r.offset])).toEqual([
      [3, 3, 0],
      [9, 2, 3],
    ]);
    const [c, t, h, w] = [4, 3, 4, 6];
    const z = new Rng(0).normal(c * t * h * w);
    const rows = patchRows(z, c, t, h, w, [1, 2, 2]);
    expect(rows.length).toBe(3 * 2 * 3 * 16);
    expect(rows[1 * 4 + 1 * 2 + 0]).toBe(z[((1 * t + 0) * h + 1) * w + 0]);
    expect(unpatchRows(rows, c, t, h, w, [1, 2, 2])).toEqual(z);
  });

  it("geometry validates against the VAE ratio and the frame rule", () => {
    const g = Geometry.parse("1344x768x124");
    g.validate(16);
    expect(g.equals(H3_768P)).toBe(true);
    expect(g.toString()).toBe("1344x768x124");
    expect(() => new Geometry(1340, 768, 124).validate(16)).toThrow(/multiple of 32/);
    expect(() => new Geometry(1344, 768, 120).validate(16)).toThrow(/17n \+ 5/);
    expect(() => Geometry.parse("1344x768")).toThrow(/WIDTHxHEIGHTxFRAMES/);
    expect(new Geometry(96, 64, 22).latent(16)).toEqual([7, 4, 6]);
    expect(new Geometry(96, 64, 22).audioLatents()).toBe(37);
  });
});

describe("rope", () => {
  it("DiT and ViT decoder tables", () => {
    const { cos, rows, width } = h3Rope([0, 0, 0, 1, 2, 3], 16, 10000.0);
    expect([rows, width]).toEqual([2, 48]);
    for (let i = 0; i < 48; i++) expect(cos[i]).toBe(1.0);
    expect(Math.abs(cos[48 + 0] - Math.cos(1.0))).toBeLessThan(1e-6);
    expect(Math.abs(cos[48 + 16] - Math.cos(2.0))).toBeLessThan(1e-6);
    expect(Math.abs(cos[48 + 32] - Math.cos(3.0))).toBeLessThan(1e-6);
    const c = vaeDecoderRope([1, 2, 2], 48, 100.0, 5);
    expect([c.rows, c.width]).toEqual([9, 24]);
    for (let i = 0; i < 8; i++) expect(Math.abs(c.cos[i] - 1.0)).toBeLessThan(1e-6);
    expect(Math.abs(c.cos[8] + 1.0)).toBeLessThan(1e-5);
    for (let i = 4 * 24; i < 9 * 24; i++) expect(Math.abs(c.cos[i] - 1.0)).toBeLessThan(1e-6);
  });
});

describe("configs", () => {
  it("rotary widths, Trim / Omni, the VAE's derived numbers, the TAEH3 frame rule", () => {
    expect(H3Config.h3().rotaryDim()).toBe(96);
    expect(H3Config.tiny().rotaryDim()).toBe(12);
    expect(H3Config.fasth3Trim().layers).toBe(42);
    expect(H3Config.fasth3Omni().pddSteps).toBe(32);
    expect(H3Config.h3().dim).toBe(7168);
    expect(H3Config.h3().videoPatchDim()).toBe(96);
    const v = VideoVaeConfig.h3();
    expect([v.spatialRatio(), v.temporalRatio(), v.framePrePadding(), v.tokensChunk(), v.tokenOverlap(), v.frameOverlap()]).toEqual([16, 4, 3, 5, 2, 5]);
    expect(VideoVaeConfig.fasth3Trim().decoderLayers).toBe(26);
    const kept = (n: number): number => {
      let k = 0;
      for (let i = 0; i < n * 4; i++) if (TaeH3Config.keepsRawFrame(i)) k++;
      return k;
    };
    expect([kept(5), kept(37), kept(2)]).toEqual([17, 124, 5]);
    expect(TaeH3Config.tiny().spatialScale()).toBe(16);
    expect(H3Config.fromJson(H3Config.fasth3Trim().toJson()).keptBlockIndices).toHaveLength(42);
  });
});

describe("rng", () => {
  it("mul64 wraps like u64", () => {
    expect(mul64(0, 0xffffffff, 0, 0xffffffff)).toEqual([0xfffffffe, 0x00000001]);
    const p = BigInt.asUintN(64, 0x9e3779b97f4a7c15n * 0xbf58476d1ce4e5b9n);
    expect(mul64(0x9e3779b9, 0x7f4a7c15, 0xbf58476d, 0x1ce4e5b9)).toEqual([Number(p >> 32n), Number(p & 0xffffffffn)]);
  });

  it("is SplitMix64 → Box–Muller", () => {
    const r = new Rng(0x9e3779b97f4a7c15n);
    expect(r.nextU64()).toBe(0xe220a8397b1dcdafn);
    expect(r.nextU64()).toBe(0x6e789e6aa1b965f4n);
    const v = new Rng(7).normal(200_000);
    let sum = 0;
    let sq = 0;
    for (const x of v) {
      sum += x;
      sq += x * x;
    }
    const mean = sum / v.length;
    expect(Math.abs(mean)).toBeLessThan(0.01);
    expect(Math.abs(sq / v.length - mean * mean - 1.0)).toBeLessThan(0.02);
    expect(new Rng(3).normal(7)).toEqual(new Rng(3).normal(8).subarray(0, 7));
    const x = new Rng(7).nextU64();
    expect(new Rng(7).unit()).toBe(Number((x >> 11n) + 1n) / 2 ** 53);
  });

  it("the sinusoid", () => {
    const e = sinusoid(0.5, 8);
    expect(Math.abs(e[0] - Math.cos(f(0.5)))).toBeLessThan(1e-7);
    expect(Math.abs(e[4] - Math.sin(f(0.5)))).toBeLessThan(1e-7);
  });
});

describe("vae host", () => {
  it("split_tiles(768, 256, 64, 16) and the temporal plan", () => {
    expect(splitTiles(768, 256, 64, 16)).toEqual([
      [0, 160, 336, 512],
      [256, 256, 256, 256],
      [96, 80, 80],
    ]);
    expect(splitTiles(64, 256, 64, 16)).toEqual([[0], [64], []]);
    const h3 = VideoVaeConfig.h3();
    expect(temporalPlan(h3, 37)).toEqual([0, 7, 124]);
    expect(temporalPlan(h3, 7)).toEqual([0, 1, 22]);
    expect(temporalPlan(h3, 2)).toEqual([5, 1, 5]);
    expect(() => temporalPlan(h3, 0)).toThrow(/no latent frames/);
  });

  it("blend cross-fades linearly and stitch trims the overlaps", () => {
    const a = nd([1, 4], Float32Array.of(1, 1, 1, 1));
    const b = nd([1, 4], Float32Array.of(0, 0, 0, 0));
    const out = blend(a, b, 2, -1);
    expect(out.shape).toEqual([1, 4]);
    expect(Array.from(out.data)).toEqual([1, 0.5, 0, 0]);
    expect(blend(a, b, 0, -1)).toBe(b);
    // two 1x4 tiles along W with overlap 2 → width 6; the overlap column 0 of the right tile is blended
    const left = nd([1, 1, 4], Float32Array.of(1, 1, 1, 1));
    const right = nd([1, 1, 4], Float32Array.of(3, 3, 3, 3));
    const st = stitchTiles([[left, right]], [], [2]);
    expect(st.shape).toEqual([1, 1, 6]);
    expect(Array.from(st.data)).toEqual([1, 1, 1, 2, 3, 3]);
    const tiles: Nd[][] = [
      [nd([1, 2, 2], Float32Array.of(1, 1, 1, 1)), nd([1, 2, 2], Float32Array.of(2, 2, 2, 2))],
      [nd([1, 2, 2], Float32Array.of(3, 3, 3, 3)), nd([1, 2, 2], Float32Array.of(4, 4, 4, 4))],
    ];
    const both = stitchTiles(tiles, [1], [1]);
    expect(both.shape).toEqual([1, 3, 3]);
    // the reference blends with w = i/extent, so an overlap of 1 copies the left / upper tile's edge
    expect(Array.from(both.data)).toEqual([1, 1, 2, 1, 3, 2, 3, 3, 4]);
  });
});

describe("f16 and safetensors", () => {
  it("round-trips half precision with ties to even", () => {
    expect(f16BitsToF32(0x3c00)).toBe(1.0);
    expect(f16BitsToF32(0xc000)).toBe(-2.0);
    expect(f16BitsToF32(0x0001)).toBe(2 ** -24);
    expect(f32ToF16Bits(1.0)).toBe(0x3c00);
    expect(f32ToF16Bits(65504)).toBe(0x7bff);
    expect(f32ToF16Bits(70000)).toBe(0x7c00);
    expect(roundF16(1.0 + 2 ** -11)).toBe(1.0); // tie → even (down)
    expect(roundF16(1.0 + 3 * 2 ** -11)).toBe(1.0 + 2 ** -9); // tie → even (up)
    expect(roundF16(0.1)).toBe(f16BitsToF32(0x2e66));
    for (const x of [0.3, -1234.5, 1e-5, 3.14159, -0.0078]) {
      const r = roundF16(x);
      expect(Math.abs(r - x)).toBeLessThanOrEqual(Math.abs(x) * 2 ** -11 + 2 ** -25);
    }
  });

  it("reads F16 tensors and the AdaLN table layout", () => {
    const u = 2;
    const layers = 1;
    const d = 4;
    const blocksBytes = u * layers * 3 * 6 * d * 2;
    const headBytes = u * 2 * d * 2;
    const header = {
      timesteps: { dtype: "F32", shape: [u], data_offsets: [0, 8] },
      blocks: { dtype: "F16", shape: [u, layers, 3, 6, d], data_offsets: [8, 8 + blocksBytes] },
      head: { dtype: "F16", shape: [u, 2, d], data_offsets: [8 + blocksBytes, 8 + blocksBytes + headBytes] },
    };
    const hdr = new TextEncoder().encode(JSON.stringify(header));
    const total = 8 + hdr.length + 8 + blocksBytes + headBytes;
    const buf = new Uint8Array(total);
    new DataView(buf.buffer).setBigUint64(0, BigInt(hdr.length), true);
    buf.set(hdr, 8);
    const data = new DataView(buf.buffer, 8 + hdr.length);
    data.setFloat32(0, 0.25, true);
    data.setFloat32(4, 1.0, true);
    const blocks = new Uint16Array(u * layers * 3 * 6 * d);
    for (let i = 0; i < blocks.length; i++) blocks[i] = f32ToF16Bits(i / 8);
    new Uint8Array(buf.buffer, 8 + hdr.length + 8, blocks.byteLength).set(new Uint8Array(blocks.buffer));
    const head = new Uint16Array(u * 2 * d).fill(f32ToF16Bits(-1.5));
    new Uint8Array(buf.buffer, 8 + hdr.length + 8 + blocks.byteLength, head.byteLength).set(new Uint8Array(head.buffer));
    const file = parseSafetensors(buf);
    expect(float32Tensor(file, "timesteps").data).toEqual(Float32Array.of(0.25, 1.0));
    expect(float32Tensor(file, "blocks").shape).toEqual([u, layers, 3, 6, d]);
    const t = new AdalnTables(file);
    expect(t.indexOf(1.0)).toBe(1);
    expect(t.indexOf(0.25 + 5e-7)).toBe(0);
    expect(() => t.indexOf(0.5)).toThrow(/no AdaLN table row for timestep 0.5/);
    const m = t.mods(1, 0, 2); // key = (1·1 + 0)·3 + 2 = 5 → values 5·24 … in steps of 1/8
    expect(m.length).toBe(6 * d);
    expect(m[0]).toBe(roundF16((5 * 24) / 8));
    expect(m[23]).toBe(roundF16((5 * 24 + 23) / 8));
    expect(t.headMods(0)).toEqual(new Float32Array(2 * d).fill(-1.5));
    expect(t.bytes()).toBe(8 + blocks.byteLength + head.byteLength);
  });
});

const tinyManifest = {
  format: "h3-litert/1",
  model: "tiny",
  variant: "tiny",
  config: H3Config.tiny().toJson(),
  vae: VideoVaeConfig.tiny().toJson(),
  audio_vae: {
    encoder_dim: 4,
    encoder_rates: [2, 5],
    latent_dim: 24,
    latent_channels: 6,
    decoder_dim: 8,
    decoder_rates: [5, 2],
    decoder_kernels: [9, 4],
    heads: 2,
    resblock_kernels: [3, 7],
    resblock_dilations: [[1, 3], [1, 3]],
    sample_rate: 320,
    latents_mean: [0.0, 0.1, -0.1, 0.2, 0.05, -0.3],
    latents_std: [1.0, 1.5, 0.8, 1.2, 0.9, 1.1],
  },
  taeh3: TaeH3Config.tiny().toJson(),
  sampler: {
    rule: "euler",
    shifts: { video: 10.0, audio: 3.0 },
    steps: [
      { t_video: 0.0001, t_audio: 0.00033, sigma_video: 0.9999, sigma_video_next: 0.90909, sigma_audio: 0.99967, sigma_audio_next: 0.75, pdd_video_weights: null, pdd_audio_weights: null },
      { t_video: 0.09091, t_audio: 0.25, sigma_video: 0.90909, sigma_video_next: 0.0, sigma_audio: 0.75, sigma_audio_next: 0.0, pdd_video_weights: null, pdd_audio_weights: null },
    ],
    cond_video_t: 0.999,
    cond_audio_t: 1.0,
  },
  adaln: { file: "adaln.safetensors", timesteps: [0.0001, 0.00033, 0.09091, 0.25, 0.999, 1.0] },
  weights: "f16",
  chunk: 16,
  head_group: 1,
  graphs: {
    text_refine: [
      { rows: 16, file: "text_refine_16.tflite" },
      { rows: 8, file: "text_refine_8.tflite" },
    ],
    embed_video: "embed_video.tflite",
    embed_audio: "embed_audio.tflite",
    head_video: "head_video.tflite",
    head_audio: "head_audio.tflite",
    blocks: [
      { qkv: "block_00_qkv.tflite", post: "block_00_post.tflite", bytes: 31472 },
      { qkv: "block_01_qkv.tflite", post: "block_01_post.tflite", bytes: 31472 },
    ],
  },
  attention: [
    { tokens: 124, q_chunk: 32, heads: 1, file: "attn_s124_q32_h1.tflite" },
    { tokens: 132, q_chunk: 32, heads: 1, file: "attn_s132_q32_h1.tflite" },
  ],
  video_vae: {
    decoder: { latent_t: 7, latent_h: 4, latent_w: 6, rows: 171, in: "vae_dec_in.tflite", blocks: ["vae_dec_block_00.tflite", "vae_dec_block_01.tflite"], out: "vae_dec_out.tflite" },
    tile: 256,
    overlap: 64,
    encoder_clip: "vae_enc_clip.tflite",
    encoder_image: "vae_enc_image.tflite",
  },
  audio: [{ latents: 37, decoder: "avae_dec_37.tflite", encoder: "avae_enc_37.tflite" }],
  taeh3_graphs: [{ frames: 5, latent_h: 4, latent_w: 6, file: "taeh3_f5_4x6.tflite" }],
  sizes: [{ width: 96, height: 64, frames: 22 }],
  files: { "head_video.tflite": { bytes: 7268, sha256: "x" }, "adaln.safetensors": { bytes: 14832 } },
};

describe("manifest", () => {
  it("parses the tiny manifest and picks graphs exactly", () => {
    const m = parseManifest(tinyManifest);
    expect(m.cfg.dim).toBe(32);
    expect(m.vae.spatialRatio()).toBe(16);
    expect(m.audioVae.hop()).toBe(10);
    expect(m.attentionFor(124).file).toBe("attn_s124_q32_h1.tflite");
    expect(m.textRefineFor(6).rows).toBe(8);
    expect(m.textRefineFor(8).rows).toBe(8);
    expect(m.textRefineFor(9).rows).toBe(16);
    expect(m.audioFor(37).decoder).toBe("avae_dec_37.tflite");
    expect(m.taeh3For(4, 6).frames).toBe(5);
    expect(() => m.attentionFor(38396)).toThrow(/no attention graph for 38396 rows \(it has \[124, 132\]\)/);
    expect(() => m.textRefineFor(17)).toThrow(/largest text_refine bucket is 16/);
    expect(() => m.audioFor(207)).toThrow(/no audio VAE graphs for 207 latents \(it has \[37\]\)/);
    expect(() => m.taeh3For(48, 84)).toThrow(/no TAEH3 graph for latent 48x84/);
    expect(m.bytesOf("head_video.tflite")).toBe(7268);
    expect(m.decoderRows()).toBe(171);
    expect(m.allFiles()).toHaveLength(2 + 4 + 4 + 2 + 4 + 2 + 2 + 1 + 1);
  });

  it("refuses another format or a malformed field", () => {
    expect(() => parseManifest({ ...tinyManifest, format: "fastwan-litert/1" })).toThrow(/format "fastwan-litert\/1", expected "h3-litert\/1"/);
    expect(() => parseManifest({ ...tinyManifest, chunk: "16" })).toThrow(/chunk is not a number/);
    expect(() => parseManifest({ ...tinyManifest, weights: "int2" })).toThrow(/weights/);
    expect(() => parseManifest({ ...tinyManifest, sampler: { ...tinyManifest.sampler, rule: "ddim" } })).toThrow(/sampler.rule/);
    expect(() => parseManifest({ ...tinyManifest, graphs: { ...tinyManifest.graphs, blocks: tinyManifest.graphs.blocks.slice(0, 1) } })).toThrow(/lists 1 blocks but config.layers is 2/);
    expect(() => parseManifest({ ...tinyManifest, head_group: 3 })).toThrow(/head_group/);
    expect(() => parseManifest({ ...tinyManifest, adaln: { file: "a", timesteps: [0.5, 0.25] } })).toThrow(/sorted ascending/);
    expect(() => parseManifest({ ...tinyManifest, sampler: { ...tinyManifest.sampler, rule: "pdd" } })).toThrow(/no PDD weights/);
  });
});

describe("plan", () => {
  it("the tiny export fits a phone; H3 at 768p does not", () => {
    const m = parseManifest(tinyManifest);
    const p = plan(m, { geometry: new Geometry(96, 64, 22), textRows: 6, keyframes: 0, preview: false, resident: 2, budgetBytes: 2 * 2 ** 30 });
    expect(p.tokens).toBe(6 + 74 + 42);
    expect(p.paddedTokens).toBe(124);
    expect(p.attentionFile).toBe("attn_s124_q32_h1.tflite");
    expect(p.attentionScratch).toBe((1 * 32 * 124 + 3 * 124 * 16 + 32 * 16) * 4);
    expect(p.fits).toBe(true);
    const big = parseManifest({
      ...tinyManifest,
      config: H3Config.h3().toJson(),
      vae: VideoVaeConfig.h3().toJson(),
      graphs: { ...tinyManifest.graphs, text_refine: [{ rows: 128, file: "text_refine_128.tflite" }], blocks: new Array(50).fill({ qkv: "q", post: "p" }) },
      attention: [{ tokens: 128 + 414 + 37296, q_chunk: 128, heads: 8, file: "attn.tflite" }],
      head_group: 8,
      video_vae: { ...tinyManifest.video_vae, decoder: { ...tinyManifest.video_vae.decoder, latent_h: 16, latent_w: 16, rows: 7 * 256 + 5, blocks: new Array(36).fill("b") } },
      audio: [{ latents: 207, decoder: "d", encoder: null }],
      weights: "int4",
      files: {},
    });
    const phone = plan(big, { geometry: H3_768P, textRows: 10, keyframes: 0, preview: false, resident: 1, budgetBytes: 2 * 2 ** 30 });
    expect(phone.paddedTokens).toBe(128 + 414 + 37296);
    expect(phone.fits).toBe(false);
    expect(phone.activations).toBeGreaterThan(4 * 37838 * 7168 * 4);
    const workstation = plan(big, { geometry: H3_768P, textRows: 10, keyframes: 0, preview: false, resident: 1, budgetBytes: 64 * 2 ** 30 });
    expect(workstation.fits).toBe(true);
  });
});
