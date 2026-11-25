/**
 * The video VAE's host side (h3-rs `vae.rs`, Diffusers `AutoencoderKLMiniMaxH3`;
 * port of `python/h3_litert/vae_host.py`): spatial tiles and their stitch, the
 * temporal chunk plan and frame blend, the posterior, pixel and latent
 * normalisation. Graphs are called through async callbacks, so the pipeline and
 * the tests share this code. Tensors are row-major `Float32Array`s with an
 * explicit shape (`Nd`), float32 arithmetic reproduced with `Math.fround`.
 */

import { PIXEL_MEAN, PIXEL_STD, type VideoVaeConfig } from "./config.js";

const f = Math.fround;

export const TILE = 256;
export const OVERLAP = 64;

/** An N-D float32 tensor, row-major. */
export interface Nd {
  data: Float32Array;
  shape: number[];
}

export function nd(shape: readonly number[], data?: Float32Array): Nd {
  const n = prod(shape);
  if (data && data.length !== n) throw new Error(`tensor of shape [${shape.join(", ")}] cannot hold ${data.length} values`);
  return { data: data ?? new Float32Array(n), shape: [...shape] };
}

export function prod(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function axisOf(x: Nd, axis: number): number {
  const a = axis < 0 ? x.shape.length + axis : axis;
  if (a < 0 || a >= x.shape.length) throw new Error(`axis ${axis} out of range for shape [${x.shape.join(", ")}]`);
  return a;
}

function outerInner(shape: readonly number[], a: number): [number, number] {
  return [prod(shape.slice(0, a)), prod(shape.slice(a + 1))];
}

/** `x[..., start:end, ...]` along `axis` (a copy). */
export function sliceAxis(x: Nd, axis: number, start: number, end: number): Nd {
  const a = axisOf(x, axis);
  const len = x.shape[a];
  start = Math.max(0, Math.min(start, len));
  end = Math.max(start, Math.min(end, len));
  const [outer, inner] = outerInner(x.shape, a);
  const n = end - start;
  const shape = [...x.shape];
  shape[a] = n;
  const out = new Float32Array(outer * n * inner);
  for (let o = 0; o < outer; o++) {
    out.set(x.data.subarray((o * len + start) * inner, (o * len + end) * inner), o * n * inner);
  }
  return { data: out, shape };
}

/** Concatenate along `axis`; every other dimension must agree. */
export function concatAxis(xs: readonly Nd[], axis: number): Nd {
  if (!xs.length) throw new Error("concat of nothing");
  const a = axisOf(xs[0], axis);
  const shape = [...xs[0].shape];
  let total = 0;
  for (const x of xs) {
    if (x.shape.length !== shape.length || x.shape.some((d, i) => i !== a && d !== shape[i])) {
      throw new Error(`concat: shape [${x.shape.join(", ")}] does not match [${shape.join(", ")}] off axis ${a}`);
    }
    total += x.shape[a];
  }
  shape[a] = total;
  const [outer, inner] = outerInner(shape, a);
  const out = new Float32Array(outer * total * inner);
  for (let o = 0; o < outer; o++) {
    let off = o * total * inner;
    for (const x of xs) {
      const n = x.shape[a];
      out.set(x.data.subarray(o * n * inner, (o + 1) * n * inner), off);
      off += n * inner;
    }
  }
  return { data: out, shape };
}

/** `x` with its last slice along `axis` repeated `n` more times (`np.repeat(x[..., -1:], n)` appended). */
export function padRepeatLast(x: Nd, axis: number, n: number): Nd {
  if (n <= 0) return x;
  const a = axisOf(x, axis);
  const last = sliceAxis(x, a, x.shape[a] - 1, x.shape[a]);
  const reps: Nd[] = [x];
  for (let i = 0; i < n; i++) reps.push(last);
  return concatAxis(reps, a);
}

/** `_split_tiles`: `[starts, lengths, overlaps]` along one pixel axis. */
export function splitTiles(length: number, tile: number, minOverlap: number, ratio: number): [number[], number[], number[]] {
  if (tile >= length) return [[0], [length], []];
  let n = Math.ceil(length / tile);
  while (tile * n - minOverlap * (n - 1) - length < 0) n += 1;
  const overlaps = new Array<number>(n - 1).fill(minOverlap);
  const remaining = tile * n - overlaps.reduce((s, v) => s + v, 0) - length;
  for (let i = 0; i < Math.floor(remaining / ratio); i++) overlaps[i % (n - 1)] += ratio;
  const starts = [0];
  for (let i = 0; i < n - 1; i++) starts.push(starts[starts.length - 1] + tile - overlaps[i]);
  return [starts, new Array<number>(n).fill(tile), overlaps];
}

/** Linear cross-fade of `a`'s last `extent` with `b`'s first `extent` along `axis`, then the rest of `b`. */
export function blend(a: Nd, b: Nd, extent: number, axis: number): Nd {
  const ax = axisOf(b, axis);
  extent = Math.min(a.shape[ax], b.shape[ax], extent);
  if (extent === 0) return b;
  const la = a.shape[ax];
  const lb = b.shape[ax];
  const [outer, inner] = outerInner(b.shape, ax);
  const [outerA, innerA] = outerInner(a.shape, ax);
  if (outerA !== outer || innerA !== inner) throw new Error(`blend: [${a.shape.join(", ")}] and [${b.shape.join(", ")}] disagree off axis ${ax}`);
  const w = new Float32Array(extent);
  for (let i = 0; i < extent; i++) w[i] = f(f(i) / f(extent));
  const shape = [...b.shape];
  shape[ax] = extent;
  const out = new Float32Array(outer * extent * inner);
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < extent; i++) {
      const wb = w[i];
      const wa = f(f(1.0) - wb);
      const sa = (o * la + la - extent + i) * inner;
      const sb = (o * lb + i) * inner;
      const d = (o * extent + i) * inner;
      for (let k = 0; k < inner; k++) out[d + k] = f(f(a.data[sa + k] * wa) + f(b.data[sb + k] * wb));
    }
  }
  const mixed: Nd = { data: out, shape };
  if (extent === lb) return mixed;
  return concatAxis([mixed, sliceAxis(b, ax, extent, lb)], ax);
}

/** Tiles `[row][col]` (each `[..., H, W]`) → one `[..., H_total, W_total]` with the overlaps cross-faded. */
export function stitchTiles(tiles: readonly (readonly Nd[])[], hOverlaps: readonly number[], wOverlaps: readonly number[]): Nd {
  const rows: Nd[] = [];
  tiles.forEach((row, i) => {
    const outRow: Nd[] = [];
    row.forEach((tile0, j) => {
      let tile = tile0;
      if (i > 0) tile = blend(tiles[i - 1][j], tile, hOverlaps[i - 1], -2);
      if (j > 0) tile = blend(row[j - 1], tile, wOverlaps[j - 1], -1);
      if (i < tiles.length - 1) tile = sliceAxis(tile, -2, 0, tile.shape[tile.shape.length - 2] - hOverlaps[i]);
      if (j < row.length - 1) tile = sliceAxis(tile, -1, 0, tile.shape[tile.shape.length - 1] - wOverlaps[j]);
      outRow.push(tile);
    });
    rows.push(concatAxis(outRow, -1));
  });
  return concatAxis(rows, -2);
}

/**
 * Run `run` on tiles of `x [C, T, H, W]` laid out in pixel space (`H·ratioIn`
 * pixels), stitch in the output's units (`ratioOut` pixels per output cell).
 */
export async function tiled(x: Nd, ratioIn: number, ratioOut: number, tilePx: number, overlapPx: number, run: (t: Nd) => Promise<Nd>): Promise<Nd> {
  const H = x.shape[x.shape.length - 2];
  const W = x.shape[x.shape.length - 1];
  const [ys, yl, yo] = splitTiles(H * ratioIn, tilePx, overlapPx, 16);
  const [xs, xl, xo] = splitTiles(W * ratioIn, tilePx, overlapPx, 16);
  const rows: Nd[][] = [];
  for (let i = 0; i < ys.length; i++) {
    const row: Nd[] = [];
    for (let j = 0; j < xs.length; j++) {
      const y0 = ys[i] / ratioIn;
      const y1 = (ys[i] + yl[i]) / ratioIn;
      const x0 = xs[j] / ratioIn;
      const x1 = (xs[j] + xl[j]) / ratioIn;
      const t = sliceAxis(sliceAxis(x, -2, y0, y1), -1, x0, x1);
      row.push(await run(t));
    }
    rows.push(row);
  }
  if (rows.length === 1 && rows[0].length === 1) return rows[0][0];
  return stitchTiles(
    rows,
    yo.map((o) => Math.floor(o / ratioOut)),
    xo.map((o) => Math.floor(o / ratioOut)),
  );
}

/** `[pad latent frames, chunks, output frames]` of `_decode`. */
export function temporalPlan(cfg: VideoVaeConfig, latentFrames: number): [number, number, number] {
  if (latentFrames === 0) throw new Error("no latent frames to decode");
  const drop = cfg.tokenDrop;
  const chunk = cfg.tokensChunk();
  const tr = cfg.temporalRatio();
  const n = latentFrames + drop;
  let pad = (chunk - (n % chunk)) % chunk;
  let chunks = (n + pad) / chunk - (drop > 0 ? 1 : 0);
  if (chunks < 1) {
    pad += chunk;
    chunks = 1;
  }
  let frames = chunks * (chunk * tr - cfg.framePrePadding());
  if (drop > 0) frames += cfg.frameOverlap();
  if (pad > 0) {
    const intra = cfg.clipLength % tr;
    let padFrames = 0;
    for (let off = 0; off < pad; off++) padFrames += intra && (latentFrames + off) % chunk === 0 ? intra : tr;
    frames -= padFrames;
  }
  if (frames <= 0) throw new Error(`the decode plan yields no frames for ${latentFrames} latent frames`);
  return [pad, chunks, frames];
}

/**
 * Denormalised latents `[lz, T, h, w]` → pixels in `[0, 1]`, delivered per
 * temporal chunk as `[3, n, H, W]` through `onChunk`; returns the frame count.
 * `decodeClip` decodes one `[lz, 7, h, w]` (tiled by the caller or not) into
 * ImageNet-normalised `[3, 28, 16h, 16w]`.
 */
export async function decode(cfg: VideoVaeConfig, z0: Nd, decodeClip: (z: Nd) => Promise<Nd>, onChunk: (px: Nd) => void): Promise<number> {
  const [pad, chunks, total] = temporalPlan(cfg, z0.shape[1]);
  const z = pad ? padRepeatLast(z0, 1, pad) : z0;
  const chunkTok = cfg.tokensChunk();
  const tr = cfg.temporalRatio();
  const pre = cfg.framePrePadding();
  const chunkFrames = chunkTok * tr;
  let written = 0;
  let overlap: Nd | null = null;
  for (let ci = 0; ci < chunks; ci++) {
    const start = ci * chunkTok;
    const n = Math.min(chunkTok + cfg.tokenOverlap(), z.shape[1] - start);
    const clip = await decodeClip(sliceAxis(z, 1, start, start + n)); // [3, 4n, H, W]
    let chunk = sliceAxis(clip, 1, pre, chunkFrames);
    if (overlap) chunk = blend(overlap, chunk, cfg.frameOverlap(), 1);
    overlap = cfg.tokenDrop > 0 && clip.shape[1] > chunkFrames + pre ? sliceAxis(clip, 1, chunkFrames + pre, clip.shape[1]) : null;
    const keep = Math.min(chunk.shape[1], total - written);
    if (keep > 0) {
      const px = denormalizePixels(sliceAxis(chunk, 1, 0, keep));
      written += keep;
      onChunk(px);
    }
  }
  if (overlap && written < total) {
    const keep = Math.min(total - written, overlap.shape[1]);
    const px = denormalizePixels(sliceAxis(overlap, 1, 0, keep));
    written += keep;
    onChunk(px);
  }
  if (written !== total) throw new Error(`decoded ${written} frames, the plan expected ${total}`);
  return written;
}

function perChannel(x: Nd, op: (v: number, c: number) => number): Nd {
  const c = x.shape[0];
  const plane = x.data.length / c;
  const out = new Float32Array(x.data.length);
  for (let ch = 0; ch < c; ch++) {
    const base = ch * plane;
    for (let i = 0; i < plane; i++) out[base + i] = op(x.data[base + i], ch);
  }
  return { data: out, shape: [...x.shape] };
}

/** `clip(x·std + mean, 0, 1)` over the channel axis of `[3, …]`. */
export function denormalizePixels(x: Nd): Nd {
  const m = PIXEL_MEAN.map(f);
  const s = PIXEL_STD.map(f);
  return perChannel(x, (v, c) => {
    const y = f(f(v * s[c]) + m[c]);
    return y < 0 ? 0 : y > 1 ? 1 : y;
  });
}

/** `(x − mean) / std` over the channel axis of `[3, …]`. */
export function normalizePixels(x: Nd): Nd {
  const m = PIXEL_MEAN.map(f);
  const s = PIXEL_STD.map(f);
  return perChannel(x, (v, c) => f(f(v - m[c]) / s[c]));
}

export function normalizeLatents(cfg: VideoVaeConfig, z: Nd): Nd {
  const m = cfg.latentsMean.map(f);
  const s = cfg.latentsStd.map(f);
  return perChannel(z, (v, c) => f(f(v - m[c]) / s[c]));
}

export function denormalizeLatents(cfg: VideoVaeConfig, z: Nd): Nd {
  const m = cfg.latentsMean.map(f);
  const s = cfg.latentsStd.map(f);
  return perChannel(z, (v, c) => f(f(v * s[c]) + m[c]));
}

/** `mean + exp(logvar/2)·ε` from `[2·lz, …]` moments (logvar clamped to [−30, 20]). */
export function posteriorSample(moments: Nd, noise: Float32Array): Nd {
  const lz = moments.shape[0] / 2;
  const plane = moments.data.length / 2;
  if (noise.length !== plane) throw new Error(`posterior_sample: ${noise.length} noise values for ${plane} latents`);
  const out = new Float32Array(plane);
  for (let i = 0; i < plane; i++) {
    const mean = moments.data[i];
    let lv = moments.data[plane + i];
    lv = lv < -30 ? -30 : lv > 20 ? 20 : lv;
    out[i] = f(mean + f(f(Math.exp(f(0.5 * lv))) * noise[i]));
  }
  return { data: out, shape: [lz, ...moments.shape.slice(1)] };
}

/**
 * ImageNet-normalised pixels `[3, F, H, W]` → moments `[2·lz, T, h, w]`
 * (`_encode`: 17-frame clips, the trailing `token_drop` latent frames dropped;
 * a single frame goes through alone).
 */
export async function encode(cfg: VideoVaeConfig, pixels: Nd, encodeClip: (x: Nd) => Promise<Nd>): Promise<Nd> {
  const fr = pixels.shape[1];
  if (fr === 1) return encodeClip(pixels);
  const cl = cfg.clipLength;
  const px = fr % cl ? padRepeatLast(pixels, 1, cl - (fr % cl)) : pixels;
  const parts: Nd[] = [];
  for (let i = 0; i < px.shape[1] / cl; i++) parts.push(await encodeClip(sliceAxis(px, 1, i * cl, (i + 1) * cl)));
  let moments = concatAxis(parts, 1);
  if (cfg.tokenDrop > 0) moments = sliceAxis(moments, 1, 0, moments.shape[1] - cfg.tokenDrop);
  return moments;
}
