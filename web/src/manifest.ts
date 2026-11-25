/**
 * `manifest.json`: parsing and validation against `docs/graphs.md`. A manifest
 * of another `format`, or a size, prompt length or audio length this export
 * has no graphs for, is an error with the mismatch in its message; nothing
 * falls back silently.
 */

import {
  AudioVaeConfig,
  H3Config,
  TaeH3Config,
  VideoVaeConfig,
  type AudioVaeConfigJson,
  type H3ConfigJson,
  type TaeH3ConfigJson,
  type VideoVaeConfigJson,
} from "./config.js";
import type { GeometryJson } from "./packing.js";
import type { Rule, SamplerStep } from "./schedule.js";

export const FORMAT = "h3-litert/1";

export type WeightFormat = "f32" | "f16" | "int8" | "int4";

export interface SamplerJson {
  rule: Rule;
  shifts: { video: number; audio: number };
  steps: SamplerStep[];
  cond_video_t: number;
  cond_audio_t: number;
}

export interface AdalnJson {
  file: string;
  timesteps: number[];
}

export interface TextRefineEntry {
  rows: number;
  file: string;
}

export interface BlockGraphs {
  qkv: string;
  post: string;
  bytes?: number;
}

export interface GraphFiles {
  text_refine: TextRefineEntry[];
  embed_video: string;
  embed_audio: string;
  head_video: string;
  head_audio: string;
  blocks: BlockGraphs[];
}

export interface AttentionEntry {
  tokens: number;
  q_chunk: number;
  heads: number;
  file: string;
}

export interface VideoDecoderGraphs {
  latent_t: number;
  latent_h: number;
  latent_w: number;
  rows: number;
  in: string;
  blocks: string[];
  out: string;
}

export interface VideoVaeGraphs {
  decoder: VideoDecoderGraphs;
  tile: number;
  overlap: number;
  encoder_clip: string | null;
  encoder_image: string | null;
}

export interface AudioEntry {
  latents: number;
  decoder: string;
  encoder: string | null;
}

export interface TaeH3Entry {
  frames: number;
  latent_h: number;
  latent_w: number;
  file: string;
}

export interface FileInfo {
  bytes: number;
  sha256?: string;
}

export interface ManifestJson {
  format: string;
  model: string;
  variant: string;
  config: H3ConfigJson;
  vae: VideoVaeConfigJson;
  audio_vae: AudioVaeConfigJson;
  taeh3: TaeH3ConfigJson | null;
  sampler: SamplerJson;
  adaln: AdalnJson;
  weights: WeightFormat;
  chunk: number;
  head_group: number;
  graphs: GraphFiles;
  attention: AttentionEntry[];
  video_vae: VideoVaeGraphs | null;
  audio: AudioEntry[];
  taeh3_graphs: TaeH3Entry[];
  sizes: GeometryJson[];
  files: Record<string, FileInfo>;
}

function fail(path: string, msg: string): never {
  throw new Error(`manifest.json: ${path} ${msg}`);
}

function obj(v: unknown, path: string): Record<string, unknown> {
  if (typeof v !== "object" || v === null || Array.isArray(v)) fail(path, "is not an object");
  return v as Record<string, unknown>;
}

function num(v: unknown, path: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) fail(path, "is not a number");
  return v;
}

function int(v: unknown, path: string): number {
  const n = num(v, path);
  if (!Number.isInteger(n) || n < 0) fail(path, `is ${n}, not a non-negative integer`);
  return n;
}

function str(v: unknown, path: string): string {
  if (typeof v !== "string" || v.length === 0) fail(path, "is not a non-empty string");
  return v;
}

function strOrNull(v: unknown, path: string): string | null {
  return v === null || v === undefined ? null : str(v, path);
}

function arr(v: unknown, path: string): unknown[] {
  if (!Array.isArray(v)) fail(path, "is not an array");
  return v;
}

function numArray(v: unknown, path: string): number[] {
  return arr(v, path).map((x, i) => num(x, `${path}[${i}]`));
}

function intArray(v: unknown, path: string): number[] {
  return arr(v, path).map((x, i) => int(x, `${path}[${i}]`));
}

function strArray(v: unknown, path: string): string[] {
  return arr(v, path).map((x, i) => str(x, `${path}[${i}]`));
}

function optIntOrNull(v: unknown, path: string): number | null {
  return v === null || v === undefined ? null : int(v, path);
}

const WEIGHT_FORMATS: readonly WeightFormat[] = ["f32", "f16", "int8", "int4"];
const RULES: readonly Rule[] = ["euler", "renoise", "pdd"];

export class Manifest {
  readonly cfg: H3Config;
  readonly vae: VideoVaeConfig;
  readonly audioVae: AudioVaeConfig;
  readonly taeh3: TaeH3Config | null;
  readonly sampler: SamplerJson;
  readonly adaln: AdalnJson;
  readonly weights: WeightFormat;
  readonly chunk: number;
  readonly headGroup: number;
  readonly model: string;
  readonly variant: string;
  readonly graphs: GraphFiles;
  readonly attention: readonly AttentionEntry[];
  readonly videoVae: VideoVaeGraphs | null;
  readonly audio: readonly AudioEntry[];
  readonly taeh3Graphs: readonly TaeH3Entry[];
  readonly sizes: readonly GeometryJson[];
  readonly files: Readonly<Record<string, FileInfo>>;

  constructor(readonly raw: ManifestJson) {
    this.cfg = H3Config.fromJson(raw.config);
    this.vae = VideoVaeConfig.fromJson(raw.vae);
    this.audioVae = AudioVaeConfig.fromJson(raw.audio_vae);
    this.taeh3 = raw.taeh3 ? TaeH3Config.fromJson(raw.taeh3) : null;
    this.sampler = raw.sampler;
    this.adaln = raw.adaln;
    this.weights = raw.weights;
    this.chunk = raw.chunk;
    this.headGroup = raw.head_group;
    this.model = raw.model;
    this.variant = raw.variant;
    this.graphs = raw.graphs;
    this.attention = raw.attention;
    this.videoVae = raw.video_vae;
    this.audio = raw.audio;
    this.taeh3Graphs = raw.taeh3_graphs;
    this.sizes = raw.sizes;
    this.files = raw.files;
    if (this.graphs.blocks.length !== this.cfg.layers) {
      fail("graphs.blocks", `lists ${this.graphs.blocks.length} blocks but config.layers is ${this.cfg.layers}`);
    }
    if (this.cfg.heads % this.headGroup !== 0) fail("head_group", `${this.headGroup} does not divide ${this.cfg.heads} heads`);
    for (const a of this.attention) {
      if (a.heads !== this.headGroup) fail("attention", `${a.file} has ${a.heads} heads per call but head_group is ${this.headGroup}`);
    }
    if (this.sampler.rule === "pdd") {
      for (const [i, s] of this.sampler.steps.entries()) {
        if (!s.pdd_video_weights || !s.pdd_audio_weights) fail(`sampler.steps[${i}]`, "has no PDD weights for a pdd sampler");
      }
    }
    if (this.videoVae && this.videoVae.decoder.rows !== this.decoderRows()) {
      fail("video_vae.decoder.rows", `is ${this.videoVae.decoder.rows}, the tile and register tokens give ${this.decoderRows()}`);
    }
  }

  /** `R = latent_t·latent_h·latent_w + register_tokens + 1` of the decoder tile. */
  decoderRows(): number {
    const d = this.videoVae?.decoder;
    if (!d) return 0;
    return d.latent_t * d.latent_h * d.latent_w + this.vae.decoderRegisterTokens + 1;
  }

  /** The attention graph of a packed sequence of exactly `tokens` rows (text padded to its bucket). */
  attentionFor(tokens: number): AttentionEntry {
    for (const a of this.attention) if (a.tokens === tokens) return a;
    const have = [...new Set(this.attention.map((a) => a.tokens))].sort((x, y) => x - y);
    throw new Error(`this export has no attention graph for ${tokens} rows (it has [${have.join(", ")}]); re-export with --size`);
  }

  /** The smallest text refiner bucket holding `rows` prompt rows. */
  textRefineFor(rows: number): TextRefineEntry {
    const sorted = [...this.graphs.text_refine].sort((a, b) => a.rows - b.rows);
    for (const t of sorted) if (t.rows >= rows) return t;
    const largest = sorted.length ? sorted[sorted.length - 1].rows : 0;
    throw new Error(`the prompt has ${rows} rows; this export's largest text_refine bucket is ${largest}`);
  }

  /** The audio VAE graphs of exactly `latents` latents. */
  audioFor(latents: number): AudioEntry {
    for (const a of this.audio) if (a.latents === latents) return a;
    const have = [...new Set(this.audio.map((a) => a.latents))].sort((x, y) => x - y);
    throw new Error(`this export has no audio VAE graphs for ${latents} latents (it has [${have.join(", ")}])`);
  }

  /** The TAEH3 chunk graph of latent size `latent_h × latent_w`. */
  taeh3For(latentH: number, latentW: number): TaeH3Entry {
    for (const t of this.taeh3Graphs) if (t.latent_h === latentH && t.latent_w === latentW) return t;
    const have = [...new Set(this.taeh3Graphs.map((t) => `${t.latent_h}x${t.latent_w}`))].sort();
    throw new Error(`this export has no TAEH3 graph for latent ${latentH}x${latentW}${have.length ? ` (it has [${have.join(", ")}])` : ""}`);
  }

  /** Byte size of a file when the manifest records it. */
  bytesOf(file: string): number | undefined {
    return this.files[file]?.bytes;
  }

  /** Every file this manifest names, in download order (fixed graphs, blocks, attention, decoders, encoders, TAEH3, the AdaLN table). */
  allFiles(): string[] {
    const g = this.graphs;
    const out = [...g.text_refine.map((t) => t.file), g.embed_video, g.embed_audio, g.head_video, g.head_audio];
    for (const b of g.blocks) out.push(b.qkv, b.post);
    for (const a of this.attention) out.push(a.file);
    if (this.videoVae) {
      const d = this.videoVae.decoder;
      out.push(d.in, ...d.blocks, d.out);
      if (this.videoVae.encoder_clip) out.push(this.videoVae.encoder_clip);
      if (this.videoVae.encoder_image) out.push(this.videoVae.encoder_image);
    }
    for (const a of this.audio) {
      out.push(a.decoder);
      if (a.encoder) out.push(a.encoder);
    }
    for (const t of this.taeh3Graphs) out.push(t.file);
    out.push(this.adaln.file);
    return out;
  }
}

function parseStep(v: unknown, path: string): SamplerStep {
  const o = obj(v, path);
  const weights = (k: string): number[] | null => (o[k] === null || o[k] === undefined ? null : numArray(o[k], `${path}.${k}`));
  return {
    t_video: num(o.t_video, `${path}.t_video`),
    t_audio: num(o.t_audio, `${path}.t_audio`),
    sigma_video: num(o.sigma_video, `${path}.sigma_video`),
    sigma_video_next: num(o.sigma_video_next, `${path}.sigma_video_next`),
    sigma_audio: num(o.sigma_audio, `${path}.sigma_audio`),
    sigma_audio_next: num(o.sigma_audio_next, `${path}.sigma_audio_next`),
    pdd_video_weights: weights("pdd_video_weights"),
    pdd_audio_weights: weights("pdd_audio_weights"),
  };
}

/** Validate a parsed `manifest.json` (any JSON value) into a `Manifest`. */
export function parseManifest(json: unknown): Manifest {
  const m = obj(json, "root");
  if (m.format !== FORMAT) {
    throw new Error(`manifest.json: format ${JSON.stringify(m.format)}, expected ${JSON.stringify(FORMAT)}`);
  }
  const c = obj(m.config, "config");
  const config: H3ConfigJson = {
    hidden: int(c.hidden, "config.hidden"),
    heads: int(c.heads, "config.heads"),
    head_dim: int(c.head_dim, "config.head_dim"),
    layers: int(c.layers, "config.layers"),
    refiner_layers: int(c.refiner_layers, "config.refiner_layers"),
    ffn_dim: int(c.ffn_dim, "config.ffn_dim"),
    in_channels: int(c.in_channels, "config.in_channels"),
    audio_in_channels: int(c.audio_in_channels, "config.audio_in_channels"),
    patch: intArray(c.patch, "config.patch"),
    text_dim: int(c.text_dim, "config.text_dim"),
    freq_dim: int(c.freq_dim, "config.freq_dim"),
    time_embed_hidden: int(c.time_embed_hidden, "config.time_embed_hidden"),
    time_embed_dim: int(c.time_embed_dim, "config.time_embed_dim"),
    rope_freq_dim: int(c.rope_freq_dim, "config.rope_freq_dim"),
    rope_theta: num(c.rope_theta, "config.rope_theta"),
    norm_eps: num(c.norm_eps, "config.norm_eps"),
    qk_norm_eps: num(c.qk_norm_eps, "config.qk_norm_eps"),
    final_norm_eps: num(c.final_norm_eps, "config.final_norm_eps"),
    adaln_rank: optIntOrNull(c.adaln_rank, "config.adaln_rank"),
    pdd_steps: optIntOrNull(c.pdd_steps, "config.pdd_steps"),
    kept_block_indices: c.kept_block_indices === null || c.kept_block_indices === undefined ? null : intArray(c.kept_block_indices, "config.kept_block_indices"),
  };
  if (config.patch.length !== 3) fail("config.patch", `has ${config.patch.length} entries, expected 3`);
  const v = obj(m.vae, "vae");
  const vae: VideoVaeConfigJson = {
    in_channels: int(v.in_channels, "vae.in_channels"),
    latent_channels: int(v.latent_channels, "vae.latent_channels"),
    block_out_channels: intArray(v.block_out_channels, "vae.block_out_channels"),
    layers_per_block: int(v.layers_per_block, "vae.layers_per_block"),
    spatial_down: intArray(v.spatial_down, "vae.spatial_down"),
    temporal_down: intArray(v.temporal_down, "vae.temporal_down"),
    norm_groups: int(v.norm_groups, "vae.norm_groups"),
    norm_eps: num(v.norm_eps, "vae.norm_eps"),
    decoder_layers: int(v.decoder_layers, "vae.decoder_layers"),
    decoder_heads: int(v.decoder_heads, "vae.decoder_heads"),
    decoder_head_dim: int(v.decoder_head_dim, "vae.decoder_head_dim"),
    decoder_register_tokens: int(v.decoder_register_tokens, "vae.decoder_register_tokens"),
    decoder_ffn_mult: int(v.decoder_ffn_mult, "vae.decoder_ffn_mult"),
    decoder_rope_theta: num(v.decoder_rope_theta, "vae.decoder_rope_theta"),
    decoder_rope_dim_ratio: num(v.decoder_rope_dim_ratio, "vae.decoder_rope_dim_ratio"),
    decoder_norm_eps: num(v.decoder_norm_eps, "vae.decoder_norm_eps"),
    clip_length: int(v.clip_length, "vae.clip_length"),
    token_drop: int(v.token_drop, "vae.token_drop"),
    latents_mean: numArray(v.latents_mean, "vae.latents_mean"),
    latents_std: numArray(v.latents_std, "vae.latents_std"),
  };
  const a = obj(m.audio_vae, "audio_vae");
  const audioVae: AudioVaeConfigJson = {
    encoder_dim: int(a.encoder_dim, "audio_vae.encoder_dim"),
    encoder_rates: intArray(a.encoder_rates, "audio_vae.encoder_rates"),
    latent_dim: int(a.latent_dim, "audio_vae.latent_dim"),
    latent_channels: int(a.latent_channels, "audio_vae.latent_channels"),
    decoder_dim: int(a.decoder_dim, "audio_vae.decoder_dim"),
    decoder_rates: intArray(a.decoder_rates, "audio_vae.decoder_rates"),
    decoder_kernels: intArray(a.decoder_kernels, "audio_vae.decoder_kernels"),
    heads: int(a.heads, "audio_vae.heads"),
    resblock_kernels: intArray(a.resblock_kernels, "audio_vae.resblock_kernels"),
    resblock_dilations: arr(a.resblock_dilations, "audio_vae.resblock_dilations").map((d, i) => intArray(d, `audio_vae.resblock_dilations[${i}]`)),
    sample_rate: int(a.sample_rate, "audio_vae.sample_rate"),
    latents_mean: numArray(a.latents_mean, "audio_vae.latents_mean"),
    latents_std: numArray(a.latents_std, "audio_vae.latents_std"),
  };
  let taeh3: TaeH3ConfigJson | null = null;
  if (m.taeh3 !== null && m.taeh3 !== undefined) {
    const t = obj(m.taeh3, "taeh3");
    taeh3 = {
      latent_channels: int(t.latent_channels, "taeh3.latent_channels"),
      widths: intArray(t.widths, "taeh3.widths"),
      tgrow: intArray(t.tgrow, "taeh3.tgrow"),
      mem_blocks: int(t.mem_blocks, "taeh3.mem_blocks"),
      patch: int(t.patch, "taeh3.patch"),
    };
  }
  const s = obj(m.sampler, "sampler");
  const rule = str(s.rule, "sampler.rule");
  if (!(RULES as readonly string[]).includes(rule)) fail("sampler.rule", `is ${JSON.stringify(rule)}, not one of ${RULES.join("/")}`);
  const sh = obj(s.shifts, "sampler.shifts");
  const steps = arr(s.steps, "sampler.steps").map((st, i) => parseStep(st, `sampler.steps[${i}]`));
  if (!steps.length) fail("sampler.steps", "is empty");
  const sampler: SamplerJson = {
    rule: rule as Rule,
    shifts: { video: num(sh.video, "sampler.shifts.video"), audio: num(sh.audio, "sampler.shifts.audio") },
    steps,
    cond_video_t: num(s.cond_video_t, "sampler.cond_video_t"),
    cond_audio_t: num(s.cond_audio_t, "sampler.cond_audio_t"),
  };
  const ad = obj(m.adaln, "adaln");
  const adaln: AdalnJson = { file: str(ad.file, "adaln.file"), timesteps: numArray(ad.timesteps, "adaln.timesteps") };
  for (let i = 1; i < adaln.timesteps.length; i++) {
    if (!(adaln.timesteps[i] > adaln.timesteps[i - 1])) fail("adaln.timesteps", "is not sorted ascending");
  }
  const weights = str(m.weights, "weights");
  if (!(WEIGHT_FORMATS as readonly string[]).includes(weights)) fail("weights", `is ${JSON.stringify(weights)}, not one of ${WEIGHT_FORMATS.join("/")}`);
  const chunk = int(m.chunk, "chunk");
  if (chunk === 0) fail("chunk", "is 0");
  const headGroup = int(m.head_group, "head_group");
  if (headGroup === 0) fail("head_group", "is 0");
  const g = obj(m.graphs, "graphs");
  const graphs: GraphFiles = {
    text_refine: arr(g.text_refine, "graphs.text_refine").map((t, i) => {
      const o = obj(t, `graphs.text_refine[${i}]`);
      return { rows: int(o.rows, `graphs.text_refine[${i}].rows`), file: str(o.file, `graphs.text_refine[${i}].file`) };
    }),
    embed_video: str(g.embed_video, "graphs.embed_video"),
    embed_audio: str(g.embed_audio, "graphs.embed_audio"),
    head_video: str(g.head_video, "graphs.head_video"),
    head_audio: str(g.head_audio, "graphs.head_audio"),
    blocks: arr(g.blocks, "graphs.blocks").map((b, i): BlockGraphs => {
      const o = obj(b, `graphs.blocks[${i}]`);
      const out: BlockGraphs = { qkv: str(o.qkv, `graphs.blocks[${i}].qkv`), post: str(o.post, `graphs.blocks[${i}].post`) };
      if (o.bytes !== undefined && o.bytes !== null) out.bytes = int(o.bytes, `graphs.blocks[${i}].bytes`);
      return out;
    }),
  };
  if (!graphs.text_refine.length) fail("graphs.text_refine", "lists no buckets");
  const attention = arr(m.attention, "attention").map((e, i): AttentionEntry => {
    const o = obj(e, `attention[${i}]`);
    return {
      tokens: int(o.tokens, `attention[${i}].tokens`),
      q_chunk: int(o.q_chunk, `attention[${i}].q_chunk`),
      heads: int(o.heads, `attention[${i}].heads`),
      file: str(o.file, `attention[${i}].file`),
    };
  });
  let videoVae: VideoVaeGraphs | null = null;
  if (m.video_vae !== null && m.video_vae !== undefined) {
    const vv = obj(m.video_vae, "video_vae");
    const d = obj(vv.decoder, "video_vae.decoder");
    videoVae = {
      decoder: {
        latent_t: int(d.latent_t, "video_vae.decoder.latent_t"),
        latent_h: int(d.latent_h, "video_vae.decoder.latent_h"),
        latent_w: int(d.latent_w, "video_vae.decoder.latent_w"),
        rows: int(d.rows, "video_vae.decoder.rows"),
        in: str(d.in, "video_vae.decoder.in"),
        blocks: strArray(d.blocks, "video_vae.decoder.blocks"),
        out: str(d.out, "video_vae.decoder.out"),
      },
      tile: int(vv.tile, "video_vae.tile"),
      overlap: int(vv.overlap, "video_vae.overlap"),
      encoder_clip: strOrNull(vv.encoder_clip, "video_vae.encoder_clip"),
      encoder_image: strOrNull(vv.encoder_image, "video_vae.encoder_image"),
    };
    if (videoVae.decoder.blocks.length !== vae.decoder_layers) {
      fail("video_vae.decoder.blocks", `lists ${videoVae.decoder.blocks.length} blocks but vae.decoder_layers is ${vae.decoder_layers}`);
    }
  }
  const audio = (m.audio === null || m.audio === undefined ? [] : arr(m.audio, "audio")).map((e, i): AudioEntry => {
    const o = obj(e, `audio[${i}]`);
    return { latents: int(o.latents, `audio[${i}].latents`), decoder: str(o.decoder, `audio[${i}].decoder`), encoder: strOrNull(o.encoder, `audio[${i}].encoder`) };
  });
  const taeh3Graphs = (m.taeh3_graphs === null || m.taeh3_graphs === undefined ? [] : arr(m.taeh3_graphs, "taeh3_graphs")).map((e, i): TaeH3Entry => {
    const o = obj(e, `taeh3_graphs[${i}]`);
    return {
      frames: int(o.frames, `taeh3_graphs[${i}].frames`),
      latent_h: int(o.latent_h, `taeh3_graphs[${i}].latent_h`),
      latent_w: int(o.latent_w, `taeh3_graphs[${i}].latent_w`),
      file: str(o.file, `taeh3_graphs[${i}].file`),
    };
  });
  const sizes = arr(m.sizes, "sizes").map((e, i): GeometryJson => {
    const o = obj(e, `sizes[${i}]`);
    return { width: int(o.width, `sizes[${i}].width`), height: int(o.height, `sizes[${i}].height`), frames: int(o.frames, `sizes[${i}].frames`) };
  });
  const files: Record<string, FileInfo> = {};
  if (m.files !== undefined && m.files !== null) {
    for (const [name, fv] of Object.entries(obj(m.files, "files"))) {
      const o = obj(fv, `files[${name}]`);
      const info: FileInfo = { bytes: int(o.bytes, `files[${name}].bytes`) };
      if (o.sha256 !== undefined) info.sha256 = str(o.sha256, `files[${name}].sha256`);
      files[name] = info;
    }
  }
  return new Manifest({
    format: FORMAT,
    model: str(m.model, "model"),
    variant: typeof m.variant === "string" && m.variant.length ? m.variant : str(m.model, "model"),
    config,
    vae,
    audio_vae: audioVae,
    taeh3,
    sampler,
    adaln,
    weights: weights as WeightFormat,
    chunk,
    head_group: headGroup,
    graphs,
    attention,
    video_vae: videoVae,
    audio,
    taeh3_graphs: taeh3Graphs,
    sizes,
    files,
  });
}

/** Fetch and parse `<baseUrl>manifest.json`. */
export async function loadManifest(baseUrl: string): Promise<Manifest> {
  const url = joinUrl(baseUrl, "manifest.json");
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return parseManifest(await res.json());
}

export function joinUrl(base: string, file: string): string {
  return base.endsWith("/") ? base + file : `${base}/${file}`;
}
