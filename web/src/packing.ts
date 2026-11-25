/**
 * Pixel ↔ latent ↔ packed-token geometry of one H3 request (h3-core
 * `packing.rs`; FastVideo `packing.py`; port of `python/h3_litert/packing.py`).
 * Positions are computed in f64 exactly as the reference does; the rotary
 * tables are then formed in f32 (`rope.ts`).
 */

import { AUDIO_TAG, TEXT_TAG, VIDEO_TAG, type Patch } from "./config.js";

export const FPS = 24;
export const SHORT_EDGE = 768;
export const MAX_PIXELS = 768 * 1344;
export const CANVAS_MULTIPLE = 32;
export const MIN_ASPECT = 0.25;
export const MAX_ASPECT = 4.0;
export const FRAMES_PER_CHUNK = 17;
export const LATENTS_PER_CHUNK = 5;
export const AUDIO_LATENTS_PER_SECOND = 40;
export const AUDIO_CHANNELS = 2;
export const MIN_ALIGNED_FRAMES = 124;
export const MAX_ALIGNED_FRAMES = 362;
export const CONTINUATION_FRAMES: readonly number[] = [5, 22, 39];
const ROPE_FRAME_RESCALE = 5.0 / 3.0;
const ROPE_FRAMES_PER_LATENT: readonly number[] = [1.0, 4.0, 4.0, 4.0, 4.0];
const ROPE_SPATIAL_SCALE = 32.0;

/** Python's `round()`: half to even. */
export function roundHalfEven(x: number): number {
  const fl = Math.floor(x);
  const diff = x - fl;
  if (diff > 0.5) return fl + 1;
  if (diff < 0.5) return fl;
  return fl % 2 === 0 ? fl : fl + 1;
}

/** Canvas `[height, width]` for an aspect ratio, snapped to the 32-pixel lattice. */
export function canvasSize(aspectW: number, aspectH: number, shortEdge = SHORT_EDGE, maxPixels = MAX_PIXELS): [number, number] {
  if (aspectW <= 0 || aspectH <= 0) throw new Error(`aspect ratio must be positive, got ${aspectW}:${aspectH}`);
  const ratio = aspectW / aspectH;
  if (!(ratio >= MIN_ASPECT && ratio <= MAX_ASPECT)) throw new Error(`aspect ratios run from 1:4 to 4:1, got ${aspectW}:${aspectH}`);
  let w: number;
  let h: number;
  if (ratio >= 1.0) {
    w = shortEdge * ratio;
    h = shortEdge;
  } else {
    w = shortEdge;
    h = shortEdge / ratio;
  }
  const area = w * h;
  if (area > maxPixels) {
    const s = Math.sqrt(maxPixels / area);
    w *= s;
    h *= s;
  }
  const m = CANVAS_MULTIPLE;
  const snap = (v: number): number => Math.max(roundHalfEven(v / m) * m, m);
  return [snap(h), snap(w)];
}

export function alignNumFrames(n: number): number {
  n = Math.max(n, 1);
  while (n % FRAMES_PER_CHUNK !== LATENTS_PER_CHUNK) n += 1;
  return n;
}

export function resolveNumFrames(n: number): number {
  const a = alignNumFrames(n);
  if (!(a >= MIN_ALIGNED_FRAMES && a <= MAX_ALIGNED_FRAMES)) {
    throw new Error(`H3 generates 5-15 s at 24 fps; aligned num_frames ${a} is outside 124-362`);
  }
  return a;
}

/** `17n + 5` frames → `5n + 2` latent frames. */
export function videoLatentFrames(n: number): number {
  if (n % FRAMES_PER_CHUNK !== LATENTS_PER_CHUNK) throw new Error(`num_frames must be 17n + 5, got ${n}`);
  return ((n - LATENTS_PER_CHUNK) / FRAMES_PER_CHUNK) * LATENTS_PER_CHUNK + 2;
}

/** `round(frames / 24 · 40)` with Python's half-even rounding. */
export function audioLatents(frames: number): number {
  return roundHalfEven((frames / FPS) * AUDIO_LATENTS_PER_SECOND);
}

export interface GeometryJson {
  width: number;
  height: number;
  frames: number;
}

export class Geometry {
  constructor(
    readonly width: number,
    readonly height: number,
    readonly frames: number,
  ) {}

  validate(spatialRatio: number): void {
    if (this.width <= 0 || this.height <= 0 || this.width % CANVAS_MULTIPLE !== 0 || this.height % CANVAS_MULTIPLE !== 0) {
      throw new Error(`${this.width}x${this.height} is not a positive multiple of 32`);
    }
    if (this.width % spatialRatio !== 0 || this.height % spatialRatio !== 0) {
      throw new Error(`${this.width}x${this.height} is not divisible by the VAE ratio ${spatialRatio}`);
    }
    videoLatentFrames(this.frames);
  }

  /** Latent `[T, H, W]`. */
  latent(spatialRatio: number): [number, number, number] {
    return [videoLatentFrames(this.frames), this.height / spatialRatio, this.width / spatialRatio];
  }

  audioLatents(): number {
    return audioLatents(this.frames);
  }

  toJson(): GeometryJson {
    return { width: this.width, height: this.height, frames: this.frames };
  }

  /** `WxHxF`, e.g. `1344x768x124`. */
  static parse(s: string): Geometry {
    const parts = s.toLowerCase().split("x");
    const nums = parts.map((p) => (/^\d+$/.test(p) ? Number(p) : NaN));
    if (nums.length !== 3 || nums.some((n) => !Number.isFinite(n))) throw new Error(`geometry \`${s}\` is not WIDTHxHEIGHTxFRAMES`);
    return new Geometry(nums[0], nums[1], nums[2]);
  }

  equals(o: Geometry): boolean {
    return this.width === o.width && this.height === o.height && this.frames === o.frames;
  }

  toString(): string {
    return `${this.width}x${this.height}x${this.frames}`;
  }
}

export const H3_768P = new Geometry(1344, 768, 124);
export const H3_480P = new Geometry(832, 480, 124);

export type RefKind = "image" | "video" | "audio";

export interface RefGeom {
  kind: RefKind;
  latentT?: number;
  latentH?: number;
  latentW?: number;
  audioLatents?: number;
}

export type Position = readonly [number, number, number];

export type SegmentKind = "image" | "audio" | "video" | "continuation";

export interface ReferenceSegment {
  kind: SegmentKind;
  rows: number;
  latent: readonly [number, number, number];
}

export interface Layout {
  sequenceLength: number;
  positionIds: Position[];
  tokenTags: number[];
  videoIndices: number[];
  audioIndices: number[];
  textIndices: number[];
  numConditionVideoRows: number;
  numConditionAudioRows: number;
  numVideoLatentFrames: number;
  latentHeight: number;
  latentWidth: number;
  numAudioLatents: number;
  referenceSegments: ReferenceSegment[];
}

export function numTargetVideoRows(l: Layout): number {
  return l.videoIndices.length - l.numConditionVideoRows;
}

export function numTargetAudioRows(l: Layout): number {
  return l.audioIndices.length - l.numConditionAudioRows;
}

function spatialPositionGrid(dim: number, patch: number, sqrtArea: number): number[] {
  const ratio = dim / sqrtArea;
  const left = (1.0 - ratio) / 2.0;
  const n = Math.floor(dim / patch);
  const step = ratio / n;
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push((left + i * step) * ROPE_SPATIAL_SCALE);
  return out;
}

export function temporalPositionGrid(numLatentFrames: number, origin: number): number[] {
  const out: number[] = [];
  let acc = 0.0;
  for (let i = 0; i < numLatentFrames; i++) {
    out.push(origin + acc);
    acc += ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % 5];
  }
  return out;
}

function temporalSpan(n: number): number {
  let s = 0.0;
  for (let i = 0; i < n; i++) s += ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % 5];
  return s;
}

function frameGrid(lh: number, lw: number, ph: number, pw: number): { grid: [number, number][]; wg: number[] } {
  const sqrtArea = Math.sqrt(lh * lw);
  const hg = spatialPositionGrid(lh, ph, sqrtArea);
  const wg = spatialPositionGrid(lw, pw, sqrtArea);
  const grid: [number, number][] = [];
  for (const h of hg) for (const w of wg) grid.push([h, w]);
  return { grid, wg };
}

function fillAudio(pos: Position[], start: number, n: number, time0: number, wg: number[]): void {
  for (let ch = 0; ch < AUDIO_CHANNELS; ch++) {
    const wpos = ch === 0 ? wg[0] : wg[wg.length - 1];
    for (let i = 0; i < n; i++) pos[start + ch * n + i] = [time0 + i, 0.0, wpos];
  }
}

function checkTextTags(tags: readonly number[]): void {
  if (tags.some((t) => t !== TEXT_TAG && t !== VIDEO_TAG)) throw new Error("text token tags may be text or vision only");
}

function range(a: number, b: number): number[] {
  const out: number[] = [];
  for (let i = a; i < b; i++) out.push(i);
  return out;
}

export type Anchor = "first" | "last";

/** T2VA / FL2VA: `[text | condition keyframes | audio | video]`; anchors are "first" / "last". */
export function buildLayout(textTags: readonly number[], lt: number, lh: number, lw: number, na: number, patch: Patch, anchors: readonly Anchor[]): Layout {
  checkTextTags(textTags);
  const [pt, ph, pw] = patch;
  if (lt % pt !== 0 || lh % ph !== 0 || lw % pw !== 0) throw new Error(`latent (${lt}, ${lh}, ${lw}) is not divisible by the patch (${patch.join(", ")})`);
  const rpf = (lh / ph) * (lw / pw);
  const nText = textTags.length;
  const nCond = anchors.length * rpf;
  const nAudio = na * AUDIO_CHANNELS;
  const nVideo = (lt / pt) * rpf;
  const total = nText + nCond + nAudio + nVideo;
  const condStart = nText;
  const audioStart = nText + nCond;
  const videoStart = nText + nCond + nAudio;
  const pos: Position[] = new Array<Position>(total).fill([0.0, 0.0, 0.0]);
  for (let i = 0; i < nText; i++) pos[i] = [i, 0.0, 0.0];
  const { grid, wg } = frameGrid(lh, lw, ph, pw);
  anchors.forEach((a, k) => {
    let t: number;
    if (a === "first") t = nText;
    else if (a === "last") t = nText + temporalSpan(lt) - ROPE_FRAME_RESCALE;
    else throw new Error(`anchor ${JSON.stringify(a)} is not first or last`);
    grid.forEach((g, j) => {
      pos[condStart + k * rpf + j] = [t, g[0], g[1]];
    });
  });
  fillAudio(pos, audioStart, na, nText, wg);
  temporalPositionGrid(lt, nText).forEach((t, f) => {
    grid.forEach((g, j) => {
      pos[videoStart + f * rpf + j] = [t, g[0], g[1]];
    });
  });
  const tags = [...textTags, ...new Array<number>(nCond).fill(VIDEO_TAG), ...new Array<number>(nAudio).fill(AUDIO_TAG), ...new Array<number>(nVideo).fill(VIDEO_TAG)];
  return {
    sequenceLength: total,
    positionIds: pos,
    tokenTags: tags,
    videoIndices: [...range(condStart, audioStart), ...range(videoStart, total)],
    audioIndices: range(audioStart, videoStart),
    textIndices: range(0, nText),
    numConditionVideoRows: nCond,
    numConditionAudioRows: 0,
    numVideoLatentFrames: lt,
    latentHeight: lh,
    latentWidth: lw,
    numAudioLatents: na,
    referenceSegments: [],
  };
}

/** Ref2VA: `[text | ordered references | continuation clip | target audio | target video]`. */
export function buildRef2vaLayout(
  textTags: readonly number[],
  refs: readonly RefGeom[],
  lt: number,
  lh: number,
  lw: number,
  na: number,
  patch: Patch,
  continuation: readonly [number, number] | null = null,
): Layout {
  if (patch[0] !== 1 || patch[1] !== 2 || patch[2] !== 2) throw new Error(`Ref2VA requires patch (1, 2, 2), got (${patch.join(", ")})`);
  checkTextTags(textTags);
  if (!refs.length) throw new Error("Ref2VA requires at least one reference");
  if (lt === 0 || lh % 2 !== 0 || lw % 2 !== 0 || na === 0) throw new Error("Ref2VA target geometry must be positive and patch-aligned");
  const nText = textTags.length;
  const rowsOf = (t: number, h: number, w: number): number => {
    if (t === 0 || h === 0 || w === 0 || h % 2 !== 0 || w % 2 !== 0) throw new Error(`visual reference geometry (${t}, ${h}, ${w}) is not patch-aligned`);
    return t * (h / 2) * (w / 2);
  };
  const rptf = (lh / 2) * (lw / 2);
  const nTargetVideo = lt * rptf;
  const nTargetAudio = na * AUDIO_CHANNELS;
  let nRefVideo = 0;
  let nRefAudio = 0;
  for (const r of refs) {
    const rt = r.latentT ?? 0;
    const rh = r.latentH ?? 0;
    const rw = r.latentW ?? 0;
    const ra = r.audioLatents ?? 0;
    if (r.kind === "image") nRefVideo += rowsOf(1, rh, rw);
    else if (r.kind === "video") {
      nRefVideo += rowsOf(rt, rh, rw);
      nRefAudio += ra * AUDIO_CHANNELS;
    } else if (r.kind === "audio") {
      if (ra === 0) throw new Error("an audio reference has no latents");
      nRefAudio += ra * AUDIO_CHANNELS;
    } else throw new Error(`reference kind ${JSON.stringify(r.kind)}`);
  }
  if (continuation) {
    const [ct, ca] = continuation;
    if (ct === 0 || ct >= lt) throw new Error(`continuation clip of ${ct} latent frames does not fit a target of ${lt}`);
    nRefVideo += ct * rptf;
    nRefAudio += ca * AUDIO_CHANNELS;
  }
  const total = nText + nRefVideo + nRefAudio + nTargetAudio + nTargetVideo;
  const pos: Position[] = new Array<Position>(total).fill([0.0, 0.0, 0.0]);
  for (let i = 0; i < nText; i++) pos[i] = [i, 0.0, 0.0];
  const { grid: targetGrid, wg: targetWg } = frameGrid(lh, lw, 2, 2);
  const videoIndices: number[] = [];
  const audioIndices: number[] = [];
  const segments: ReferenceSegment[] = [];
  let cursor = nText;
  let clock = nText;
  for (const r of refs) {
    const rt = r.latentT ?? 0;
    const rh = r.latentH ?? 0;
    const rw = r.latentW ?? 0;
    const ra = r.audioLatents ?? 0;
    if (r.kind === "image") {
      const n = rowsOf(1, rh, rw);
      segments.push({ kind: "image", rows: n, latent: [1, rh, rw] });
      const { grid } = frameGrid(rh, rw, 2, 2);
      grid.forEach((g, j) => {
        pos[cursor + j] = [clock, g[0], g[1]];
      });
      videoIndices.push(...range(cursor, cursor + n));
      cursor += n;
      clock += 1.0;
    } else if (r.kind === "audio") {
      const n = ra * AUDIO_CHANNELS;
      segments.push({ kind: "audio", rows: n, latent: [0, 0, 0] });
      fillAudio(pos, cursor, ra, clock, targetWg);
      audioIndices.push(...range(cursor, cursor + n));
      cursor += n;
      clock += ra;
    } else {
      const naR = ra * AUDIO_CHANNELS;
      const nv = rowsOf(rt, rh, rw);
      if (naR) segments.push({ kind: "audio", rows: naR, latent: [0, 0, 0] });
      segments.push({ kind: "video", rows: nv, latent: [rt, rh, rw] });
      const { grid, wg } = frameGrid(rh, rw, 2, 2);
      if (naR) {
        fillAudio(pos, cursor, ra, clock, wg);
        audioIndices.push(...range(cursor, cursor + naR));
        cursor += naR;
      }
      temporalPositionGrid(rt, clock).forEach((tt, f) => {
        grid.forEach((g, j) => {
          pos[cursor + f * grid.length + j] = [tt, g[0], g[1]];
        });
      });
      videoIndices.push(...range(cursor, cursor + nv));
      cursor += nv;
      clock += Math.max(ra, temporalSpan(rt));
    }
  }
  if (continuation) {
    const [ct, ca] = continuation;
    const naC = ca * AUDIO_CHANNELS;
    const nv = ct * rptf;
    if (naC) segments.push({ kind: "audio", rows: naC, latent: [0, 0, 0] });
    segments.push({ kind: "continuation", rows: nv, latent: [ct, lh, lw] });
    clock += Math.max(ca, temporalSpan(ct));
    if (naC) {
      fillAudio(pos, cursor, ca, clock, targetWg);
      audioIndices.push(...range(cursor, cursor + naC));
      cursor += naC;
    }
    temporalPositionGrid(ct, clock).forEach((tt, f) => {
      targetGrid.forEach((g, j) => {
        pos[cursor + f * rptf + j] = [tt, g[0], g[1]];
      });
    });
    videoIndices.push(...range(cursor, cursor + nv));
    cursor += nv;
  }
  const audioStart = cursor;
  const videoStart = audioStart + nTargetAudio;
  fillAudio(pos, audioStart, na, clock, targetWg);
  temporalPositionGrid(lt, clock).forEach((tt, f) => {
    targetGrid.forEach((g, j) => {
      pos[videoStart + f * rptf + j] = [tt, g[0], g[1]];
    });
  });
  audioIndices.push(...range(audioStart, videoStart));
  videoIndices.push(...range(videoStart, total));
  if (videoStart + nTargetVideo !== total) throw new Error("Ref2VA layout does not add up");
  const tags = new Array<number>(total).fill(VIDEO_TAG);
  for (let i = 0; i < nText; i++) tags[i] = textTags[i];
  for (const i of audioIndices) tags[i] = AUDIO_TAG;
  for (const i of videoIndices) tags[i] = VIDEO_TAG;
  return {
    sequenceLength: total,
    positionIds: pos,
    tokenTags: tags,
    videoIndices,
    audioIndices,
    textIndices: range(0, nText),
    numConditionVideoRows: nRefVideo,
    numConditionAudioRows: nRefAudio,
    numVideoLatentFrames: lt,
    latentHeight: lh,
    latentWidth: lw,
    numAudioLatents: na,
    referenceSegments: segments,
  };
}

export interface RowTimesteps {
  /** The sorted unique f32 clean times of this step. */
  unique: number[];
  /** Each row's index into `unique`. */
  index: number[];
}

/** Per-row clean times of one step (f32 values); the sorted unique times and each row's index into them. */
export function rowTimesteps(layout: Layout, videoT: number, audioT: number, condVideoT: number, condAudioT: number): RowTimesteps {
  const f = Math.fround;
  const rows = new Array<number>(layout.sequenceLength).fill(f(videoT));
  for (const i of layout.videoIndices.slice(0, layout.numConditionVideoRows)) rows[i] = f(condVideoT);
  for (const i of layout.audioIndices.slice(layout.numConditionAudioRows)) rows[i] = f(audioT);
  for (const i of layout.audioIndices.slice(0, layout.numConditionAudioRows)) rows[i] = f(condAudioT);
  const unique = [...new Set(rows)].sort((a, b) => a - b);
  const pos = new Map<number, number>(unique.map((u, k) => [u, k]));
  return { unique, index: rows.map((r) => pos.get(r) as number) };
}

/** A contiguous run of packed rows: packed `start`, `rows`, and the `offset` in the modality's own row buffer. */
export interface Run {
  start: number;
  rows: number;
  offset: number;
}

export function runs(indices: readonly number[]): Run[] {
  const out: Run[] = [];
  indices.forEach((i, off) => {
    const last = out[out.length - 1];
    if (last && last.start + last.rows === i) last.rows += 1;
    else out.push({ start: i, rows: 1, offset: off });
  });
  return out;
}

/** A maximal run of rows sharing `(timestep index, tag)`. */
export interface ModRun {
  start: number;
  rows: number;
  tIdx: number;
  tag: number;
}

export function modRuns(tIdx: readonly number[], tags: readonly number[]): ModRun[] {
  if (tIdx.length !== tags.length) throw new Error(`mod_runs: ${tIdx.length} timestep indices for ${tags.length} tags`);
  const out: ModRun[] = [];
  for (let i = 0; i < tIdx.length; i++) {
    const last = out[out.length - 1];
    if (last && last.tIdx === tIdx[i] && last.tag === tags[i]) last.rows += 1;
    else out.push({ start: i, rows: 1, tIdx: tIdx[i], tag: tags[i] });
  }
  return out;
}

/** `[C, T, H, W]` latent → `[rows, C·pt·ph·pw]` patch rows (rows in `(t, h, w)` order, channel-major features). */
export function patchRows(z: Float32Array, c: number, t: number, h: number, w: number, patch: Patch): Float32Array {
  const [pt, ph, pw] = patch;
  if (z.length !== c * t * h * w) throw new Error(`patch_rows: ${z.length} values for [${c}, ${t}, ${h}, ${w}]`);
  if (t % pt !== 0 || h % ph !== 0 || w % pw !== 0) throw new Error(`patch_rows: (${t}, ${h}, ${w}) is not divisible by the patch (${patch.join(", ")})`);
  const tt = t / pt;
  const hh = h / ph;
  const ww = w / pw;
  const feat = c * pt * ph * pw;
  const out = new Float32Array(tt * hh * ww * feat);
  let row = 0;
  for (let a = 0; a < tt; a++) {
    for (let b = 0; b < hh; b++) {
      for (let d = 0; d < ww; d++, row++) {
        let k = row * feat;
        for (let cc = 0; cc < c; cc++) {
          for (let kt = 0; kt < pt; kt++) {
            for (let kh = 0; kh < ph; kh++) {
              const src = ((cc * t + a * pt + kt) * h + b * ph + kh) * w + d * pw;
              for (let kw = 0; kw < pw; kw++) out[k++] = z[src + kw];
            }
          }
        }
      }
    }
  }
  return out;
}

/** The inverse of `patchRows`: `[rows, C·pt·ph·pw]` → `[C, T, H, W]`. */
export function unpatchRows(rows: Float32Array, c: number, t: number, h: number, w: number, patch: Patch): Float32Array {
  const [pt, ph, pw] = patch;
  const tt = t / pt;
  const hh = h / ph;
  const ww = w / pw;
  const feat = c * pt * ph * pw;
  if (rows.length !== tt * hh * ww * feat) throw new Error(`unpatch_rows: ${rows.length} values for ${tt * hh * ww} rows of ${feat}`);
  const out = new Float32Array(c * t * h * w);
  let row = 0;
  for (let a = 0; a < tt; a++) {
    for (let b = 0; b < hh; b++) {
      for (let d = 0; d < ww; d++, row++) {
        let k = row * feat;
        for (let cc = 0; cc < c; cc++) {
          for (let kt = 0; kt < pt; kt++) {
            for (let kh = 0; kh < ph; kh++) {
              const dst = ((cc * t + a * pt + kt) * h + b * ph + kh) * w + d * pw;
              for (let kw = 0; kw < pw; kw++) out[dst + kw] = rows[k++];
            }
          }
        }
      }
    }
  }
  return out;
}
