/**
 * Memory plan (`docs/design.md`, "Tiers"): what a generation holds at once,
 * before anything is allocated. Hosts call it before `generate` to refuse what
 * will not fit: resident blocks × block bytes + the fixed graphs + the AdaLN
 * table + `q/k/v/o/h` at `S × 7168 / 5376` f32 + the attention score chunk
 * `G·Cq·S` + the decoder's peak.
 */

import type { Manifest, WeightFormat } from "./manifest.js";
import type { Geometry } from "./packing.js";

export const BYTES_PER_WEIGHT: Readonly<Record<WeightFormat, number>> = { f32: 4.0, f16: 2.0, int8: 1.0, int4: 0.5 };

export interface PlanRequest {
  geometry: Geometry;
  /** Prompt rows (before bucketing). */
  textRows: number;
  /** Keyframes (FL2VA): 0, 1 or 2. */
  keyframes: number;
  /** TAEH3 instead of the VAE decoder. */
  preview: boolean;
  /** Block graphs kept resident. */
  resident: number;
  budgetBytes: number;
}

export interface Plan {
  /** Packed rows before the text padding, and with it (the attention graph's `S`). */
  tokens: number;
  paddedTokens: number;
  textBucket: number;
  chunk: number;
  qChunk: number;
  headGroup: number;
  /** The attention graph the run would use, or null when the export has none for `S`. */
  attentionFile: string | null;
  blockBytes: number;
  weightsResident: number;
  weightsFixed: number;
  adalnTable: number;
  activations: number;
  attentionScratch: number;
  decodePeak: number;
  peak: number;
  budget: number;
  fits: boolean;
  notes: string[];
}

function sum(xs: readonly number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}

export function plan(m: Manifest, r: PlanRequest): Plan {
  const c = m.cfg;
  const vc = m.vae;
  const notes: string[] = [];
  const a = 4; // f32 activations
  const ratio = vc.spatialRatio();
  const [lt, lh, lw] = r.geometry.latent(ratio);
  const na = r.geometry.audioLatents();
  const [pt, ph, pw] = c.patch;
  const rpf = (lh / ph) * (lw / pw);
  const bucket = m.textRefineFor(r.textRows);
  const T = bucket.rows;
  const nVideo = r.keyframes * rpf + (lt / pt) * rpf;
  const nAudio = 2 * na;
  const tokens = r.textRows + nVideo + nAudio;
  const S = T + nVideo + nAudio;
  const d = c.hidden;
  const I = c.dim;
  const hd = c.headDim;
  const G = m.headGroup;
  let attention = m.attention.find((e) => e.tokens === S) ?? null;
  if (!attention) {
    notes.push(`no attention graph for ${S} rows (the export has [${[...new Set(m.attention.map((e) => e.tokens))].sort((x, y) => x - y).join(", ")}])`);
    attention = m.attention[0] ?? null;
  }
  const cq = attention?.q_chunk ?? m.chunk;

  let unknown = 0;
  const bytesOf = (file: string | null | undefined): number => {
    if (!file) return 0;
    const b = m.bytesOf(file);
    if (b === undefined) unknown++;
    return b ?? 0;
  };
  const wb = BYTES_PER_WEIGHT[m.weights];
  const blockBytes = Math.max(
    0,
    ...m.graphs.blocks.map((b) => b.bytes ?? (m.bytesOf(b.qkv) !== undefined && m.bytesOf(b.post) !== undefined ? bytesOf(b.qkv) + bytesOf(b.post) : Math.trunc(c.blockParams() * wb))),
  );
  const resident = Math.max(1, Math.min(r.resident, c.layers));
  const weightsResident = resident * blockBytes;
  const fixedFiles: (string | null)[] = [bucket.file, m.graphs.embed_video, m.graphs.embed_audio, m.graphs.head_video, m.graphs.head_audio];
  if (attention) fixedFiles.push(attention.file);
  const audio = m.audio.find((e) => e.latents === na) ?? null;
  if (audio) fixedFiles.push(audio.decoder);
  else notes.push(`no audio VAE graphs for ${na} latents`);
  if (r.keyframes > 0 && m.videoVae) fixedFiles.push(m.videoVae.encoder_image);
  let decodePeak = 0;
  const F = r.geometry.frames;
  const frameBytes = F * r.geometry.height * r.geometry.width * 4; // RGBA frames held for playback
  if (r.preview) {
    const tae = m.taeh3;
    const entry = m.taeh3Graphs.find((t) => t.latent_h === lh && t.latent_w === lw) ?? null;
    if (!tae || !entry) notes.push(`no TAEH3 graph for latent ${lh}x${lw}`);
    else {
      fixedFiles.push(entry.file);
      const f = Math.min(entry.frames, lt);
      const nf = tae.widths;
      const px = lh * lw;
      const s0 = f * nf[0] * px;
      const s1 = f * tae.tgrow[0] * nf[1] * 4 * px;
      const s2 = f * tae.tgrow[0] * tae.tgrow[1] * nf[2] * 16 * px;
      const out = f * tae.tUpscale() * nf[3] * 64 * px;
      decodePeak = Math.max(5 * s0, 5 * s1, 5 * s2, s2 + 2 * out) * a + frameBytes;
    }
  } else if (m.videoVae) {
    const dec = m.videoVae.decoder;
    fixedFiles.push(dec.in, ...dec.blocks, dec.out);
    const R = m.decoderRows();
    const tr = vc.temporalRatio();
    const tilePix = 3 * tr * dec.latent_t * ratio * dec.latent_h * ratio * dec.latent_w;
    const clipPix = 3 * tr * dec.latent_t * r.geometry.height * r.geometry.width;
    // h / h_out / cos / sin of the ViT, one decoded tile, the stitched clip and its blended chunk, the overlap.
    decodePeak = (3 * R * vc.decoderDim() + tilePix + 3 * clipPix) * a + frameBytes;
  } else notes.push("no video VAE decoder in this export");
  const weightsFixed = sum(fixedFiles.map(bytesOf));
  const adalnTable = m.bytesOf(m.adaln.file) ?? m.adaln.timesteps.length * (c.layers * 3 * 6 + 2) * d * 2;
  if (unknown) notes.push(`${unknown} graph file sizes are not in the manifest's files; the weights total is a lower bound`);
  const P = c.videoPatchDim();
  const Ac = c.audioInChannels;
  // h, text, q, k, v, o; the row buffers and their velocities; the rope tables and the mask.
  const activations = (S * d + T * d + 4 * S * I + 2 * (nVideo * P + nAudio * Ac) + S * (c.rotaryDim() + 1)) * a;
  // the score chunk, the gathered k / v / q of one head group, the chunk's output
  const attentionScratch = (G * cq * S + 3 * S * G * hd + cq * G * hd) * a;
  const denoisePeak = weightsResident + weightsFixed + adalnTable + activations + attentionScratch;
  const peak = Math.max(denoisePeak, weightsResident + weightsFixed + decodePeak);
  if (resident < c.layers) notes.push(`${resident} of ${c.layers} block graphs resident; the rest stream from the network`);
  if (S % m.chunk !== 0) notes.push(`${S} rows run as ${Math.ceil(S / m.chunk)} chunks of ${m.chunk} (${m.chunk - (S % m.chunk)} padded rows)`);
  return {
    tokens,
    paddedTokens: S,
    textBucket: T,
    chunk: m.chunk,
    qChunk: cq,
    headGroup: G,
    attentionFile: attention && attention.tokens === S ? attention.file : null,
    blockBytes,
    weightsResident,
    weightsFixed,
    adalnTable,
    activations,
    attentionScratch,
    decodePeak,
    peak,
    budget: r.budgetBytes,
    fits: peak <= r.budgetBytes,
    notes,
  };
}

export function formatPlan(p: Plan): string {
  const mib = (b: number): string => (b / (1 << 20)).toFixed(1).padStart(9) + " MiB";
  const lines = [
    `rows              ${p.tokens} (+${p.paddedTokens - p.tokens} text pad = ${p.paddedTokens}, bucket ${p.textBucket})`,
    `block graph       ${mib(p.blockBytes)}`,
    `blocks resident   ${mib(p.weightsResident)}`,
    `fixed graphs      ${mib(p.weightsFixed)}`,
    `AdaLN table       ${mib(p.adalnTable)}`,
    `DiT activations   ${mib(p.activations)} (row chunk ${p.chunk})`,
    `attention scratch ${mib(p.attentionScratch)} (query chunk ${p.qChunk}, ${p.headGroup} heads)`,
    `decode peak       ${mib(p.decodePeak)}`,
    `peak              ${mib(p.peak)} of ${(p.budget / (1 << 30)).toFixed(1)} GiB  fits: ${p.fits}`,
  ];
  for (const n of p.notes) lines.push(`note: ${n}`);
  return lines.join("\n");
}
