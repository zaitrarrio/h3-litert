/**
 * The browser side of the fixture check: runs the tiny export's T2VA request
 * through the pipeline on the chosen accelerator and compares with
 * `expected.safetensors` the way `h3-litert check` does: `rel = max|Δ| /
 * max|ref|` on the video latent, the audio rows and the waveform, `max |Δ|` on
 * the frames. Driven by `run-fixture.mjs` through Playwright, or by hand from
 * `fixture.html`.
 */

import { loadLiteRt } from "@litertjs/core";

import {
  AdalnTables,
  Geometry,
  GraphStore,
  Pipeline,
  Rng,
  UrlSource,
  float32Tensor,
  litertCompiler,
  loadManifest,
  parseSafetensors,
  uint8Tensor,
  type Accelerator,
  type PipelineEvent,
  type Video,
} from "../src/index.js";

export interface FixtureOptions {
  /** Base URL of the export (`manifest.json`, graphs, `adaln.safetensors`, `expected.safetensors`). */
  base: string;
  /** Directory of the LiteRT.js Wasm files. */
  wasmDir: string;
  accelerator: Accelerator;
  jspi?: boolean;
  threads?: boolean;
  /** Block graphs kept resident; 1 exercises the LRU eviction on the tiny model. */
  resident?: number;
  /** Also run the TAEH3 preview decoder (reported, not judged). */
  preview?: boolean;
  /** Also run an FL2VA denoise with a synthetic first keyframe through the encoder graphs (reported, not judged). */
  fl2va?: boolean;
}

export interface RelResult {
  rel: number;
  maxAbs: number;
  refMax: number;
  tol: number;
}

export interface FixtureResult {
  ok: boolean;
  accelerator: Accelerator;
  weights: string;
  geometry: string;
  seed: number;
  tokens: number;
  paddedTokens: number;
  latent: RelResult;
  audioRows: RelResult;
  frames: { maxDelta: number; meanDelta: number; count: number; width: number; height: number; expected: number; tol: number };
  audio: RelResult & { samples: number; sampleRate: number };
  preview: { count: number; width: number; height: number; maxDeltaVsVae: number } | null;
  fl2va: { tokens: number; paddedTokens: number; conditionRows: number; latentDeltaVsT2va: number } | null;
  timings: { prepareMs: number; stepsMs: number[]; decodeMs: number; audioMs: number };
  downloads: number;
  events: string[];
}

function rel(got: Float32Array, ref: Float32Array, tol: number, what: string): RelResult {
  if (got.length !== ref.length) throw new Error(`${what} has ${got.length} values, expected ${ref.length}`);
  let maxAbs = 0;
  let refMax = 0;
  for (let i = 0; i < got.length; i++) {
    maxAbs = Math.max(maxAbs, Math.abs(got[i] - ref[i]));
    refMax = Math.max(refMax, Math.abs(ref[i]));
  }
  return { rel: maxAbs / Math.max(refMax, 1e-6), maxAbs, refMax, tol };
}

function compareFrames(video: Video, ref: { data: Uint8Array; shape: number[] }): { maxDelta: number; meanDelta: number } {
  const [nf, fh, fw, fc] = ref.shape;
  if (fc !== 3) throw new Error(`expected.frames has ${fc} channels`);
  if (video.frames !== nf || video.height !== fh || video.width !== fw) return { maxDelta: 255, meanDelta: NaN };
  let maxDelta = 0;
  let sum = 0;
  let n = 0;
  for (let f = 0; f < nf; f++) {
    const px = video.rgba[f];
    for (let p = 0; p < fh * fw; p++) {
      for (let c = 0; c < 3; c++) {
        const d = Math.abs(px[p * 4 + c] - ref.data[(f * fh * fw + p) * 3 + c]);
        maxDelta = Math.max(maxDelta, d);
        sum += d;
        n++;
      }
      if (px[p * 4 + 3] !== 255) throw new Error("alpha is not 255");
    }
  }
  return { maxDelta, meanDelta: sum / n };
}

export async function runFixture(o: FixtureOptions): Promise<FixtureResult> {
  const events: string[] = [];
  await loadLiteRt(o.wasmDir, { jspi: o.jspi ?? false, threads: o.threads ?? false });
  const m = await loadManifest(o.base);
  let downloads = 0;
  const source = new UrlSource(o.base, m.files);
  const store = new GraphStore(m, source, litertCompiler({ accelerator: o.accelerator }), {
    resident: o.resident ?? 1,
    onProgress: (p) => {
      if (p.total && p.loaded === p.total) downloads++;
    },
  });
  const adaln = AdalnTables.parse(await source.fetch(m.adaln.file, (p) => {
    if (p.total && p.loaded === p.total) downloads++;
  }));
  const pipe = new Pipeline(store, adaln);

  const res = await fetch(`${o.base}expected.safetensors`);
  if (!res.ok) throw new Error(`expected.safetensors: HTTP ${res.status}`);
  const exp = parseSafetensors(await res.arrayBuffer());
  const g = Geometry.parse(exp.metadata.geometry ?? m.sizes.map((s) => `${s.width}x${s.height}x${s.frames}`)[0]);
  const seed = Number(exp.metadata.seed ?? "7");
  const embeds = float32Tensor(exp, "prompt_embeds");
  const tags = Array.from(float32Tensor(exp, "text_tags").data, (t) => Math.round(t));
  const refLatent = float32Tensor(exp, "latent");
  const refAudioRows = float32Tensor(exp, "audio_rows");
  const refFrames = uint8Tensor(exp, "frames");
  const refAudio = float32Tensor(exp, "audio");

  const on = (e: PipelineEvent): void => {
    if (e.type === "prepared") events.push(`prepared ${e.tokens} rows in ${e.ms.toFixed(1)} ms`);
    else if (e.type === "step") events.push(`step ${e.index + 1}/${e.total}: ${e.ms.toFixed(1)} ms`);
    else if (e.type === "frames") events.push(`frames ${e.first}..${e.first + e.count - 1} ${e.width}x${e.height}`);
    else events.push(`audio ${e.stereo[0].length} samples/channel at ${e.sampleRate} Hz`);
  };
  const timings = { prepareMs: 0, stepsMs: [] as number[], decodeMs: 0, audioMs: 0 };
  // The contract's tolerances (rel = max|Δ| / max|expected|): 1e-3 and 2/255 for an f32 export on the
  // CPU backend; f16 / int8 / int4 exports and GPU delegates are judged at 2e-2 and 16/255, as `h3-litert check`.
  const exact = o.accelerator === "wasm" && m.weights === "f32";
  const tol = exact ? 1e-3 : 2e-2;
  const frameTol = exact ? 2 : 16;
  const { latent, audioRows, audioLatents, tokens, paddedTokens } = await pipe.denoise({ embeds: embeds.data, textTags: tags, geometry: g, seed }, on, timings);
  const latentRel = rel(latent.data, refLatent.data, tol, "latent");
  const rowsRel = rel(audioRows, refAudioRows.data, tol, "audio_rows");
  const video = await pipe.decodeVideo(latent, false, on, timings);
  const fr = compareFrames(video, refFrames);
  const audio = await pipe.decodeAudio(audioRows, audioLatents, timings);
  on({ type: "audio", sampleRate: audio.sampleRate, stereo: audio.stereo });
  const wave = new Float32Array(audio.stereo[0].length * 2);
  wave.set(audio.stereo[0]);
  wave.set(audio.stereo[1], audio.stereo[0].length);
  const audioRel = rel(wave, refAudio.data, 10 * tol, "audio");
  let preview: FixtureResult["preview"] = null;
  if (o.preview) {
    const pv = await pipe.decodeVideo(latent, true, on);
    let maxDelta = 0;
    if (pv.frames === video.frames && pv.width === video.width && pv.height === video.height) {
      for (let f = 0; f < pv.frames; f++) for (let i = 0; i < pv.rgba[f].length; i++) maxDelta = Math.max(maxDelta, Math.abs(pv.rgba[f][i] - video.rgba[f][i]));
    } else maxDelta = 255;
    preview = { count: pv.frames, width: pv.width, height: pv.height, maxDeltaVsVae: maxDelta };
  }
  let fl2va: FixtureResult["fl2va"] = null;
  if (o.fl2va) {
    // test_graphs.py's keyframe: Rng(99) noise around grey, [3, H, W] in [0, 1]
    const px = new Rng(99).normal(3 * g.height * g.width);
    for (let i = 0; i < px.length; i++) px[i] = Math.min(1, Math.max(0, px[i] * 0.2 + 0.5));
    const r = await pipe.denoise({ embeds: embeds.data, textTags: tags, geometry: g, seed, keyframes: [{ anchor: "first", pixels: px }] }, on);
    let d = 0;
    for (let i = 0; i < r.latent.data.length; i++) d = Math.max(d, Math.abs(r.latent.data[i] - latent.data[i]));
    fl2va = { tokens: r.tokens, paddedTokens: r.paddedTokens, conditionRows: r.layout.numConditionVideoRows, latentDeltaVsT2va: d };
  }
  const ok =
    latentRel.rel <= tol && rowsRel.rel <= tol && video.frames === refFrames.shape[0] && fr.maxDelta <= frameTol && audioRel.rel <= audioRel.tol;
  await store.deleteAll();
  return {
    ok,
    accelerator: o.accelerator,
    weights: m.weights,
    geometry: g.toString(),
    seed,
    tokens,
    paddedTokens,
    latent: latentRel,
    audioRows: rowsRel,
    frames: { ...fr, count: video.frames, width: video.width, height: video.height, expected: refFrames.shape[0], tol: frameTol },
    audio: { ...audioRel, samples: audio.stereo[0].length, sampleRate: audio.sampleRate },
    preview,
    fl2va,
    timings,
    downloads,
    events,
  };
}

declare global {
  interface Window {
    runFixture: typeof runFixture;
  }
}

window.runFixture = runFixture;
