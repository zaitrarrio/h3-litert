/**
 * Prompt embeddings → packed layout → denoising loop → decoded video and
 * audio, driving the graphs exactly as `docs/graphs.md`'s host loop says
 * (port of `python/h3_litert/pipeline.py`). T2VA and FL2VA (keyframes through
 * the encoder graphs); Euler, renoise and PDD samplers; the ViT VAE decoder
 * with tiles and temporal chunks, or the TAEH3 preview.
 */

import type { AdalnTables } from "./adaln.js";
import { TEXT_TAG, TaeH3Config, type H3Config } from "./config.js";
import { roundF16 } from "./f16.js";
import type { Graph, GraphInput, GraphStore } from "./graphs.js";
import type { AttentionEntry, Manifest } from "./manifest.js";
import {
  buildLayout,
  modRuns,
  patchRows,
  rowTimesteps,
  runs,
  unpatchRows,
  type Anchor,
  type Geometry,
  type Layout,
} from "./packing.js";
import { Rng } from "./rng.js";
import { h3Rope, type RopeTable } from "./rope.js";
import { flowStep, needsNoise } from "./sampler.js";
import { KEYFRAME_NOISE_AUG, type SamplerStep } from "./schedule.js";
import * as vaeHost from "./vaeHost.js";
import { nd, prod, type Nd } from "./vaeHost.js";

export const MASK_PAD = -1e9;
export const KEYFRAME_ENCODE_SEED = 42;

const f = Math.fround;

// ---- requests and events ---------------------------------------------------------

export interface Keyframe {
  anchor: Anchor;
  /** `[3, H, W]` in `[0, 1]` at the canvas size. */
  pixels: Float32Array;
}

export interface Request {
  /** `[rows, text_dim]` prompt embeddings. */
  embeds: Float32Array;
  /** One tag per prompt row: `1` text, `0` vision. */
  textTags: readonly number[];
  geometry: Geometry;
  seed: number | bigint;
  keyframes?: readonly Keyframe[];
  /** TAEH3 instead of the VAE decoder. */
  preview?: boolean;
}

export interface PreparedEvent {
  type: "prepared";
  ms: number;
  /** Packed rows before the text padding. */
  tokens: number;
}

export interface StepEvent {
  type: "step";
  index: number;
  total: number;
  ms: number;
}

/** One RGBA frame (`width·height·4`, alpha 255), accepted by `new ImageData(frame, width, height)`. */
export type RgbaFrame = Uint8ClampedArray<ArrayBuffer>;

export interface FramesEvent {
  type: "frames";
  first: number;
  count: number;
  width: number;
  height: number;
  rgba: RgbaFrame[];
}

export interface AudioEvent {
  type: "audio";
  sampleRate: number;
  /** Left and right channel waveforms in `[−1, 1]`. */
  stereo: [Float32Array, Float32Array];
}

export type PipelineEvent = PreparedEvent | StepEvent | FramesEvent | AudioEvent;
export type OnEvent = (e: PipelineEvent) => void;

export interface Timings {
  prepareMs: number;
  stepsMs: number[];
  decodeMs: number;
  audioMs: number;
}

export function newTimings(): Timings {
  return { prepareMs: 0, stepsMs: [], decodeMs: 0, audioMs: 0 };
}

/** A latent `[channels, frames, height, width]` row-major. */
export interface Latent {
  data: Float32Array;
  channels: number;
  frames: number;
  height: number;
  width: number;
}

export interface Video {
  frames: number;
  width: number;
  height: number;
  rgba: RgbaFrame[];
}

export interface Audio {
  sampleRate: number;
  stereo: [Float32Array, Float32Array];
}

export interface Denoised {
  latent: Latent;
  /** Normalised target audio rows `[2·A, A_c]`. */
  audioRows: Float32Array;
  /** `A`, the audio latents per channel. */
  audioLatents: number;
  layout: Layout;
  /** Packed rows before and after the text padding (the attention graph's `S`). */
  tokens: number;
  paddedTokens: number;
}

/** Everything `forward` needs that does not change between steps. */
export interface Prepared {
  layout: Layout;
  /** `[T, d]` refined text rows (the bucket's rows, padding included). */
  text: Float32Array;
  nText: number;
  pad: number;
  /** `[len(videoIdx), P]`: condition rows, then the target rows. */
  videoRows: Float32Array;
  /** `[2·A, A_c]`. */
  audioRows: Float32Array;
  rope: RopeTable;
  /** `[S]`: 0 for a real row, −1e9 for a padded text row. */
  mask: Float32Array;
  /** Packed indices after the text padding shift. */
  videoIdx: number[];
  audioIdx: number[];
  /** Packed tags after the shift. */
  tags: number[];
  /** Padded sequence length `S`. */
  s: number;
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/** `[3, n, H, W]` in `[0, 1]` → `n` RGBA frames with the reference's rounding (`(clip(v)·255 + 0.5)` truncated, f32). */
export function toRgba8(px: Nd): RgbaFrame[] {
  const [c, n, h, w] = px.shape;
  if (c !== 3 || px.shape.length !== 4) throw new Error(`to_rgb8: shape [${px.shape.join(", ")}] is not [3, n, H, W]`);
  const plane = h * w;
  const out: RgbaFrame[] = [];
  for (let i = 0; i < n; i++) {
    const frame: RgbaFrame = new Uint8ClampedArray(new ArrayBuffer(plane * 4));
    for (let j = 0; j < plane; j++) {
      for (let ch = 0; ch < 3; ch++) {
        const v = px.data[(ch * n + i) * plane + j];
        const cl = v < 0 ? 0 : v > 1 ? 1 : v;
        frame[j * 4 + ch] = Math.floor(f(f(cl * 255.0) + 0.5));
      }
      frame[j * 4 + 3] = 255;
    }
    out.push(frame);
  }
  return out;
}

interface RowInput {
  /** At least `nReal · rowSize` values from offset 0. */
  data: Float32Array;
  rowSize: number;
}

export class Pipeline {
  readonly m: Manifest;
  readonly cfg: H3Config;

  constructor(
    readonly store: GraphStore,
    readonly adaln: AdalnTables,
  ) {
    this.m = store.m;
    this.cfg = this.m.cfg;
    const want = this.m.adaln.timesteps;
    if (adaln.count !== want.length) {
      throw new Error(`adaln.safetensors has ${adaln.count} timesteps, the manifest lists ${want.length}`);
    }
    for (const t of want) adaln.indexOf(t);
    if (adaln.layers !== this.cfg.layers || adaln.dim !== this.cfg.hidden) {
      throw new Error(`adaln.safetensors is [${adaln.count}, ${adaln.layers}, 3, 6, ${adaln.dim}], the config wants ${this.cfg.layers} layers of ${this.cfg.hidden}`);
    }
  }

  // ---- helpers ------------------------------------------------------------------------

  /** Run a per-row graph over `nReal` rows in chunks of `chunk` (zero-padded); outputs trimmed to `nReal` rows. */
  private async rows(graph: Graph, chunk: number, nReal: number, rowInputs: Record<string, RowInput>, fixed: Record<string, GraphInput>): Promise<Float32Array[]> {
    const outs: Float32Array[][] = [];
    const bufs = new Map<string, Float32Array>();
    for (const [k, v] of Object.entries(rowInputs)) bufs.set(k, new Float32Array(chunk * v.rowSize));
    for (let r = 0; r < nReal; r += chunk) {
      const n = Math.min(chunk, nReal - r);
      const inputs: Record<string, GraphInput> = { ...fixed };
      for (const [k, v] of Object.entries(rowInputs)) {
        const buf = bufs.get(k) as Float32Array;
        buf.fill(0);
        buf.set(v.data.subarray(r * v.rowSize, (r + n) * v.rowSize));
        inputs[k] = buf;
      }
      outs.push(await graph.run(inputs));
    }
    const result: Float32Array[] = [];
    for (let i = 0; i < outs[0].length; i++) {
      const rowSize = outs[0][i].length / chunk;
      if (!Number.isInteger(rowSize)) throw new Error(`${graph.file}: output ${i} has ${outs[0][i].length} values, not a multiple of ${chunk} rows`);
      const all = new Float32Array(nReal * rowSize);
      let off = 0;
      for (const o of outs) {
        const take = Math.min(o[i].length, all.length - off);
        if (take <= 0) break;
        all.set(o[i].subarray(0, take), off);
        off += take;
      }
      result.push(all);
    }
    return result;
  }

  // ---- prompt + layout --------------------------------------------------------------------

  async prepare(req: Request): Promise<Prepared> {
    const c = this.cfg;
    const m = this.m;
    const ratio = m.vae.spatialRatio();
    const g = req.geometry;
    g.validate(ratio);
    const [lt, lh, lw] = g.latent(ratio);
    const na = g.audioLatents();
    const nText = req.textTags.length;
    if (req.embeds.length !== nText * c.textDim) {
      throw new Error(`prompt embeds have ${req.embeds.length} values, expected [${nText}, ${c.textDim}]`);
    }
    const bucket = m.textRefineFor(nText);
    const T = bucket.rows;
    const pad = T - nText;
    const emb = new Float32Array(T * c.textDim);
    emb.set(req.embeds);
    const tmask = new Float32Array(T);
    tmask.fill(MASK_PAD, nText);
    const refine = await this.store.fixed(bucket.file);
    const [text] = await refine.run({ embeds: { data: emb, shape: [T, c.textDim] }, mask: { data: tmask, shape: [T] } });
    if (text.length !== T * c.hidden) throw new Error(`${bucket.file}: text has ${text.length} values, expected [${T}, ${c.hidden}]`);
    const keyframes = req.keyframes ?? [];
    const layout = buildLayout(req.textTags, lt, lh, lw, na, c.patch, keyframes.map((k) => k.anchor));
    const rng = new Rng(req.seed);
    const P = c.videoPatchDim();
    let cond = new Float32Array(0);
    if (keyframes.length) {
      const parts: Float32Array[] = [];
      for (const k of keyframes) parts.push(await this.encodeKeyframe(k.pixels, g));
      const total = parts.reduce((a, p) => a + p.length, 0);
      cond = new Float32Array(total);
      let off = 0;
      for (const p of parts) {
        cond.set(p, off);
        off += p.length;
      }
      const noise = rng.normal(cond.length);
      const aug = f(KEYFRAME_NOISE_AUG);
      const rest = f(1.0 - KEYFRAME_NOISE_AUG);
      for (let i = 0; i < cond.length; i++) cond[i] = f(f(aug * cond[i]) + f(rest * noise[i]));
    }
    if (cond.length !== layout.numConditionVideoRows * P) {
      throw new Error(`keyframes gave ${cond.length / P} condition rows, the layout has ${layout.numConditionVideoRows}`);
    }
    const noise = rng.normal(c.inChannels * lt * lh * lw);
    const target = patchRows(noise, c.inChannels, lt, lh, lw, c.patch);
    const videoRows = new Float32Array(cond.length + target.length);
    videoRows.set(cond);
    videoRows.set(target, cond.length);
    const audioRows = rng.normal(na * 2 * c.audioInChannels);
    const shift = (i: number): number => (i < nText ? i : i + pad);
    const s = layout.sequenceLength + pad;
    const pos = new Float32Array(s * 3);
    const tags = new Array<number>(s).fill(TEXT_TAG);
    layout.positionIds.forEach((p, i) => {
      const j = shift(i);
      pos[j * 3] = p[0];
      pos[j * 3 + 1] = p[1];
      pos[j * 3 + 2] = p[2];
      tags[j] = layout.tokenTags[i];
    });
    const rope = h3Rope(pos, c.ropeFreqDim, c.ropeTheta);
    const mask = new Float32Array(s);
    mask.fill(MASK_PAD, nText, T);
    return {
      layout,
      text,
      nText,
      pad,
      videoRows,
      audioRows,
      rope,
      mask,
      videoIdx: layout.videoIndices.map(shift),
      audioIdx: layout.audioIndices.map(shift),
      tags,
      s,
    };
  }

  /** One keyframe `[3, H, W]` → its `rpf` normalised patch rows, through the encoder graphs (seed 42, f16 round). */
  async encodeKeyframe(pixels: Float32Array, g: Geometry): Promise<Float32Array> {
    const vc = this.m.vae;
    const ratio = vc.spatialRatio();
    if (pixels.length !== 3 * g.height * g.width) {
      throw new Error(`keyframe has ${pixels.length} values, the canvas ${g.width}x${g.height} wants [3, ${g.height}, ${g.width}]`);
    }
    const px = vaeHost.normalizePixels(nd([3, 1, g.height, g.width], pixels));
    const moments = await vaeHost.encode(vc, px, (x) => this.encodeClip(x));
    const lz = moments.shape[0] / 2;
    const rng = new Rng(KEYFRAME_ENCODE_SEED);
    const z = vaeHost.posteriorSample(moments, rng.normal(lz * prod(moments.shape.slice(1))));
    for (let i = 0; i < z.data.length; i++) z.data[i] = roundF16(z.data[i]);
    const zn = vaeHost.normalizeLatents(vc, z);
    const [, t, h, w] = zn.shape;
    return patchRows(zn.data, lz, t, h, w, this.cfg.patch);
  }

  /** `encode_clip` over the encoder graphs with the manifest's tiling: pixels `[3, F, tile, tile]` → moments. */
  private async encodeClip(x: Nd): Promise<Nd> {
    const vv = this.m.videoVae;
    if (!vv) throw new Error("this export has no video VAE");
    const name = x.shape[1] === 1 ? vv.encoder_image : vv.encoder_clip;
    if (!name) throw new Error("this export has no VAE encoder graphs (exported with --no-encoders)");
    const ratio = this.m.vae.spatialRatio();
    const lz2 = 2 * this.m.vae.latentChannels;
    return vaeHost.tiled(x, 1, ratio, vv.tile, vv.overlap, async (t) => {
      const g = await this.store.fixed(name);
      const [mo] = await g.run({ x: { data: t.data, shape: t.shape } });
      const h = t.shape[2] / ratio;
      const w = t.shape[3] / ratio;
      const tt = mo.length / (lz2 * h * w);
      if (!Number.isInteger(tt)) throw new Error(`${name}: ${mo.length} moment values do not form [${lz2}, T, ${h}, ${w}]`);
      return nd([lz2, tt, h, w], mo);
    });
  }

  // ---- one forward ---------------------------------------------------------------------------

  /** One DiT forward: the velocities of every video row `[len(videoIdx), P]` and audio row `[2A, A_c]`. */
  async forward(p: Prepared, step: SamplerStep, attn: AttentionEntry): Promise<[Float32Array, Float32Array]> {
    const c = this.cfg;
    const m = this.m;
    const ad = this.adaln;
    const C = m.chunk;
    const G = m.headGroup;
    const d = c.hidden;
    const inner = c.dim;
    const hd = c.headDim;
    const H = c.heads;
    const s = p.s;
    const condVideoT = Math.max(step.t_video, m.sampler.cond_video_t);
    const { unique, index: inv } = rowTimesteps(p.layout, step.t_video, step.t_audio, condVideoT, m.sampler.cond_audio_t);
    const uOf = unique.map((t) => ad.indexOf(t));
    const shift = (i: number): number => (i < p.nText ? i : i + p.pad);
    const tIdx = new Array<number>(s).fill(0);
    inv.forEach((ti, i) => {
      tIdx[shift(i)] = ti;
    });
    for (let i = p.nText; i < p.nText + p.pad; i++) tIdx[i] = inv[0];

    const h = new Float32Array(s * d);
    h.set(p.text);
    const embV = await this.store.fixed(m.graphs.embed_video);
    const embA = await this.store.fixed(m.graphs.embed_audio);
    const P = c.videoPatchDim();
    const Ac = c.audioInChannels;
    for (const r of runs(p.videoIdx)) {
      const [e] = await this.rows(embV, C, r.rows, { rows: { data: p.videoRows.subarray(r.offset * P), rowSize: P } }, {});
      h.set(e, r.start * d);
    }
    for (const r of runs(p.audioIdx)) {
      const [e] = await this.rows(embA, C, r.rows, { rows: { data: p.audioRows.subarray(r.offset * Ac), rowSize: Ac } }, {});
      h.set(e, r.start * d);
    }
    const mruns = modRuns(tIdx, p.tags);
    const attnGraph = await this.store.fixed(attn.file);
    const cq = attn.q_chunk;
    const q = new Float32Array(s * inner);
    const k = new Float32Array(s * inner);
    const v = new Float32Array(s * inner);
    const o = new Float32Array(s * inner);
    const ropeW = p.rope.width;
    const qg = new Float32Array(s * G * hd);
    const kg = new Float32Array(s * G * hd);
    const vg = new Float32Array(s * G * hd);
    for (let l = 0; l < m.graphs.blocks.length; l++) {
      const b = m.graphs.blocks[l];
      const gq = await this.store.block(b.qkv);
      for (const r of mruns) {
        const mods: GraphInput = { data: ad.mods(uOf[r.tIdx], l, r.tag), shape: [6, d] };
        const [qq, kk, vv] = await this.rows(
          gq,
          C,
          r.rows,
          {
            h: { data: h.subarray(r.start * d), rowSize: d },
            cos: { data: p.rope.cos.subarray(r.start * ropeW), rowSize: ropeW },
            sin: { data: p.rope.sin.subarray(r.start * ropeW), rowSize: ropeW },
          },
          { mods },
        );
        q.set(qq, r.start * inner);
        k.set(kk, r.start * inner);
        v.set(vv, r.start * inner);
      }
      for (let g = 0; g < H / G; g++) {
        const h0 = g * G * hd;
        for (let i = 0; i < s; i++) {
          qg.set(q.subarray(i * inner + h0, i * inner + h0 + G * hd), i * G * hd);
          kg.set(k.subarray(i * inner + h0, i * inner + h0 + G * hd), i * G * hd);
          vg.set(v.subarray(i * inner + h0, i * inner + h0 + G * hd), i * G * hd);
        }
        const [og] = await this.rows(
          attnGraph,
          cq,
          s,
          { q: { data: qg, rowSize: G * hd } },
          { k: { data: kg, shape: [s, G, hd] }, v: { data: vg, shape: [s, G, hd] }, mask: { data: p.mask, shape: [s] } },
        );
        for (let i = 0; i < s; i++) o.set(og.subarray(i * G * hd, (i + 1) * G * hd), i * inner + h0);
      }
      const gp = await this.store.block(b.post);
      for (const r of mruns) {
        const mods: GraphInput = { data: ad.mods(uOf[r.tIdx], l, r.tag), shape: [6, d] };
        const [hh] = await this.rows(
          gp,
          C,
          r.rows,
          { h: { data: h.subarray(r.start * d), rowSize: d }, o: { data: o.subarray(r.start * inner), rowSize: inner } },
          { mods },
        );
        h.set(hh, r.start * d);
      }
    }
    const hv = await this.store.fixed(m.graphs.head_video);
    const ha = await this.store.fixed(m.graphs.head_audio);
    const pdd = m.sampler.rule === "pdd";
    const vOut = new Float32Array(p.videoIdx.length * P);
    const aOut = new Float32Array(p.audioIdx.length * Ac);
    const heads: [Graph, number[], Float32Array, number, number[] | null][] = [
      [hv, p.videoIdx, vOut, P, pdd ? step.pdd_video_weights : null],
      [ha, p.audioIdx, aOut, Ac, pdd ? step.pdd_audio_weights : null],
    ];
    for (const [graph, idx, out, width, weights] of heads) {
      for (const r of runs(idx)) {
        // a run of one modality may still span two timesteps (condition rows vs target rows)
        let start = r.start;
        while (start < r.start + r.rows) {
          const ti = tIdx[start];
          let end = start;
          while (end < r.start + r.rows && tIdx[end] === ti) end++;
          const n = end - start;
          const off = r.offset + (start - r.start);
          const fixed: Record<string, GraphInput> = { mods: { data: ad.headMods(uOf[ti]), shape: [2, d] } };
          if (weights) fixed.weights = { data: Float32Array.from(weights), shape: [weights.length] };
          const [vals] = await this.rows(graph, C, n, { h: { data: h.subarray(start * d), rowSize: d } }, fixed);
          out.set(vals, off * width);
          start = end;
        }
      }
    }
    return [vOut, aOut];
  }

  // ---- denoise ------------------------------------------------------------------------------

  async denoise(req: Request, on?: OnEvent, timings?: Timings): Promise<Denoised> {
    const c = this.cfg;
    const m = this.m;
    const tm = timings ?? newTimings();
    const t0 = now();
    const p = await this.prepare(req);
    const attn = m.attentionFor(p.s);
    tm.prepareMs = now() - t0;
    on?.({ type: "prepared", ms: tm.prepareMs, tokens: p.layout.sequenceLength });
    const rule = m.sampler.rule;
    const steps = m.sampler.steps;
    const P = c.videoPatchDim();
    const offV = p.layout.numConditionVideoRows * P;
    if (rule === "pdd") {
      const sv = f(steps[0].sigma_video);
      const sa = f(steps[0].sigma_audio);
      for (let i = offV; i < p.videoRows.length; i++) p.videoRows[i] = f(p.videoRows[i] * sv);
      for (let i = 0; i < p.audioRows.length; i++) p.audioRows[i] = f(p.audioRows[i] * sa);
    }
    const renoise = new Rng(BigInt(req.seed) + 1n);
    for (let i = 0; i < steps.length; i++) {
      const st = steps[i];
      const ts = now();
      const [vel, aud] = await this.forward(p, st, attn);
      const tgt = p.videoRows.subarray(offV);
      let nz = needsNoise(rule, st.sigma_video_next) ? renoise.normal(tgt.length) : null;
      tgt.set(flowStep(tgt, vel.subarray(offV), st.sigma_video, st.sigma_video_next, rule, nz));
      nz = needsNoise(rule, st.sigma_audio_next) ? renoise.normal(p.audioRows.length) : null;
      p.audioRows.set(flowStep(p.audioRows, aud, st.sigma_audio, st.sigma_audio_next, rule, nz));
      tm.stepsMs.push(now() - ts);
      on?.({ type: "step", index: i, total: steps.length, ms: tm.stepsMs[tm.stepsMs.length - 1] });
    }
    const [lt, lh, lw] = req.geometry.latent(m.vae.spatialRatio());
    const latent: Latent = {
      data: unpatchRows(p.videoRows.subarray(offV), c.inChannels, lt, lh, lw, c.patch),
      channels: c.inChannels,
      frames: lt,
      height: lh,
      width: lw,
    };
    return { latent, audioRows: p.audioRows, audioLatents: req.geometry.audioLatents(), layout: p.layout, tokens: p.layout.sequenceLength, paddedTokens: p.s };
  }

  // ---- decode -------------------------------------------------------------------------------

  /** Normalised latent → RGBA frames, through the VAE decoder (tiles, temporal chunks) or TAEH3 (`preview`). */
  async decodeVideo(latent: Latent, preview = false, on?: OnEvent, timings?: Timings): Promise<Video> {
    const tm = timings ?? newTimings();
    const t0 = now();
    const video = preview ? await this.decodeTaeh3(latent, on) : await this.decodeVae(latent, on);
    tm.decodeMs = now() - t0;
    return video;
  }

  private async decodeVae(latent: Latent, on?: OnEvent): Promise<Video> {
    const m = this.m;
    const vv = m.videoVae;
    if (!vv) throw new Error("this export has no video VAE decoder");
    const dec = vv.decoder;
    const vc = m.vae;
    const ratio = vc.spatialRatio();
    const tr = vc.temporalRatio();
    if (latent.channels !== vc.latentChannels) throw new Error(`latent has ${latent.channels} channels, the VAE wants ${vc.latentChannels}`);
    const gIn = await this.store.fixed(dec.in);
    const blocks: Graph[] = [];
    for (const b of dec.blocks) blocks.push(await this.store.fixed(b));
    const gOut = await this.store.fixed(dec.out);
    const R = m.decoderRows();
    const ddim = vc.decoderDim();
    const ropeW = vc.decoderRotaryDim() / 2;
    const clipTile = async (z: Nd): Promise<Nd> => {
      const [, t, h, w] = z.shape;
      if (t !== dec.latent_t || h !== dec.latent_h || w !== dec.latent_w) {
        throw new Error(`decoder tile (${t}, ${h}, ${w}) but the export has graphs for (${dec.latent_t}, ${dec.latent_h}, ${dec.latent_w})`);
      }
      let [hid, cos, sin] = await gIn.run({ z: { data: z.data, shape: z.shape } });
      for (const b of blocks) {
        [hid] = await b.run({ h: { data: hid, shape: [R, ddim] }, cos: { data: cos, shape: [R, ropeW] }, sin: { data: sin, shape: [R, ropeW] } });
      }
      const [pix] = await gOut.run({ h: { data: hid, shape: [R, ddim] } });
      const shape = [3, tr * t, ratio * h, ratio * w];
      if (pix.length !== prod(shape)) throw new Error(`${dec.out}: ${pix.length} pixel values, expected [${shape.join(", ")}]`);
      return nd(shape, pix);
    };
    const z = vaeHost.denormalizeLatents(vc, nd([latent.channels, latent.frames, latent.height, latent.width], latent.data));
    const frames: RgbaFrame[] = [];
    let width = 0;
    let height = 0;
    await vaeHost.decode(
      vc,
      z,
      (chunk) => vaeHost.tiled(chunk, ratio, 1, vv.tile, vv.overlap, clipTile),
      (px) => {
        const rgba = toRgba8(px);
        height = px.shape[2];
        width = px.shape[3];
        on?.({ type: "frames", first: frames.length, count: rgba.length, width, height, rgba });
        frames.push(...rgba);
      },
    );
    return { frames: frames.length, width, height, rgba: frames };
  }

  private async decodeTaeh3(latent: Latent, on?: OnEvent): Promise<Video> {
    const m = this.m;
    const tc = m.taeh3;
    if (!tc) throw new Error("this export has no TAEH3 graph");
    const { channels: c, frames: t, height: h, width: w } = latent;
    if (c !== tc.latentChannels) throw new Error(`latent has ${c} channels, TAEH3 wants ${tc.latentChannels}`);
    const entry = m.taeh3For(h, w);
    const graph = await this.store.fixed(entry.file);
    const fr = entry.frames;
    const nmem = 3 * tc.memBlocks;
    const memShape = (i: number): number[] => {
      const st = Math.floor(i / tc.memBlocks);
      return [1, tc.widths[st], h << st, w << st];
    };
    let mems: Float32Array[] = [];
    for (let i = 0; i < nmem; i++) mems.push(new Float32Array(prod(memShape(i))));
    const up = tc.tUpscale();
    const scale = tc.spatialScale();
    const pxH = scale * h;
    const pxW = scale * w;
    const plane = h * w;
    const frames: RgbaFrame[] = [];
    let rawIndex = 0;
    for (let f0 = 0; f0 < t; f0 += fr) {
      const n = Math.min(fr, t - f0);
      const z = new Float32Array(fr * c * plane);
      for (let fi = 0; fi < n; fi++) {
        for (let ci = 0; ci < c; ci++) {
          const src = (ci * t + f0 + fi) * plane;
          z.set(latent.data.subarray(src, src + plane), (fi * c + ci) * plane);
        }
      }
      const inputs: Record<string, GraphInput> = { z: { data: z, shape: [fr, c, h, w] } };
      for (let i = 0; i < nmem; i++) inputs[`mem_${i}`] = { data: mems[i], shape: memShape(i) };
      const out = await graph.run(inputs);
      if (out.length !== 1 + nmem) throw new Error(`${entry.file}: ${out.length} outputs, expected rgb + ${nmem} memories`);
      const frameSize = 3 * pxH * pxW;
      if (out[0].length !== fr * up * frameSize) throw new Error(`${entry.file}: rgb has ${out[0].length} values, expected [${fr * up}, 3, ${pxH}, ${pxW}]`);
      mems = out.slice(1);
      const real = n * up;
      const keep: number[] = [];
      for (let i = 0; i < real; i++) if (TaeH3Config.keepsRawFrame(rawIndex + i)) keep.push(i);
      rawIndex += real;
      if (keep.length) {
        // gather the kept frames as [3, n, H, W] for the shared quantiser
        const px = new Float32Array(3 * keep.length * pxH * pxW);
        keep.forEach((fi, j) => {
          for (let ch = 0; ch < 3; ch++) {
            const src = fi * frameSize + ch * pxH * pxW;
            px.set(out[0].subarray(src, src + pxH * pxW), (ch * keep.length + j) * pxH * pxW);
          }
        });
        const rgba = toRgba8(nd([3, keep.length, pxH, pxW], px));
        on?.({ type: "frames", first: frames.length, count: rgba.length, width: pxW, height: pxH, rgba });
        frames.push(...rgba);
      }
    }
    return { frames: frames.length, width: pxW, height: pxH, rgba: frames };
  }

  /** Normalised target audio rows `[2A, A_c]` → stereo waveforms in `[−1, 1]`. */
  async decodeAudio(audioRows: Float32Array, na: number, timings?: Timings): Promise<Audio> {
    const tm = timings ?? newTimings();
    const t0 = now();
    const ac = this.m.audioVae;
    const Ac = ac.latentChannels;
    if (audioRows.length !== 2 * na * Ac) throw new Error(`audio rows have ${audioRows.length} values, expected [${2 * na}, ${Ac}]`);
    const entry = this.m.audioFor(na);
    const g = await this.store.fixed(entry.decoder);
    const mean = ac.latentsMean.map(f);
    const std = ac.latentsStd.map(f);
    const out: Float32Array[] = [];
    for (let ch = 0; ch < 2; ch++) {
      const z = new Float32Array(Ac * na);
      for (let i = 0; i < na; i++) {
        for (let c = 0; c < Ac; c++) z[c * na + i] = f(f(audioRows[(ch * na + i) * Ac + c] * std[c]) + mean[c]);
      }
      const [wave] = await g.run({ z: { data: z, shape: [Ac, na] } });
      out.push(wave);
    }
    tm.audioMs = now() - t0;
    return { sampleRate: ac.sampleRate, stereo: [out[0], out[1]] };
  }

  async generate(req: Request, on?: OnEvent): Promise<{ video: Video; audio: Audio; latent: Latent; timings: Timings }> {
    const timings = newTimings();
    const { latent, audioRows, audioLatents } = await this.denoise(req, on, timings);
    const video = await this.decodeVideo(latent, req.preview ?? false, on, timings);
    const audio = await this.decodeAudio(audioRows, audioLatents, timings);
    on?.({ type: "audio", sampleRate: audio.sampleRate, stereo: audio.stereo });
    return { video, audio, latent, timings };
  }
}
