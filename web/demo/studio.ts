/**
 * The browser studio: pick an export (a base URL with `manifest.json`), a
 * prompt from a gallery of precomputed Qwen3-VL embeddings, a size, a seed,
 * optionally a first / last keyframe (FL2VA) and the TAEH3 preview; watch the
 * block graphs download, the steps tick, the frames play on a canvas while the
 * clip is still decoding, and hear the audio through WebAudio when it is done.
 * Dependency-free beyond `@litertjs/core`.
 *
 * The memory plan is shown and enforced: H3 is a 33 B model and a phone cannot
 * hold it; only the tiny self-test export fits there, and the page says so.
 * WebGPU when the browser has a GPU adapter, else Wasm.
 */

import { getWebGpuDevice, loadLiteRt, supportsFeature } from "@litertjs/core";

import {
  AdalnTables,
  Geometry,
  GraphStore,
  Pipeline,
  UrlSource,
  float32Tensor,
  formatPlan,
  joinUrl,
  litertCompiler,
  loadManifest,
  parseSafetensors,
  plan,
  type Accelerator,
  type Anchor,
  type Audio,
  type DownloadProgress,
  type Keyframe,
  type Manifest,
  type PipelineEvent,
  type Plan,
  type RgbaFrame,
} from "../src/index.js";

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel);
  if (!el) throw new Error(`missing element ${sel}`);
  return el;
};

const ui = {
  info: $<HTMLElement>("#info"),
  base: $<HTMLInputElement>("#base"),
  galleryUrl: $<HTMLInputElement>("#galleryUrl"),
  load: $<HTMLButtonElement>("#load"),
  prompt: $<HTMLSelectElement>("#prompt"),
  size: $<HTMLSelectElement>("#size"),
  seed: $<HTMLInputElement>("#seed"),
  resident: $<HTMLInputElement>("#resident"),
  preview: $<HTMLInputElement>("#preview"),
  first: $<HTMLInputElement>("#first"),
  last: $<HTMLInputElement>("#last"),
  go: $<HTMLButtonElement>("#go"),
  stop: $<HTMLButtonElement>("#stop"),
  play: $<HTMLButtonElement>("#play"),
  view: $<HTMLCanvasElement>("#view"),
  log: $<HTMLElement>("#log"),
  blocks: $<HTMLElement>("#blocks"),
  dl: $<HTMLElement>("#dl"),
  steps: $<HTMLElement>("#steps"),
  plan: $<HTMLElement>("#plan"),
};

const FPS = 24;
const GIB = 1024 ** 3;

export const isPhone = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
/** `webgpu` only when the browser has a GPU adapter (headless Chromium exposes `navigator.gpu` without one). */
let accelerator: Accelerator = "wasm";

async function detectAccelerator(): Promise<Accelerator> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown | null> } }).gpu;
  if (!gpu) return "wasm";
  try {
    return (await gpu.requestAdapter()) ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}

function log(msg: string, cls = ""): void {
  const line = document.createElement("div");
  line.textContent = msg;
  if (cls) line.className = cls;
  ui.log.append(line);
  ui.log.scrollTop = ui.log.scrollHeight;
}

// ---- gallery ------------------------------------------------------------------

interface GalleryEntry {
  name: string;
  file: string;
}

/** `index.json`: `["fox.safetensors", …]` or `[{"name": "fox", "file": "fox.safetensors"}, …]` or `{"prompts": [...]}`. */
async function loadGallery(url: string): Promise<GalleryEntry[]> {
  const res = await fetch(joinUrl(url, "index.json"));
  if (!res.ok) throw new Error(`${joinUrl(url, "index.json")}: HTTP ${res.status}`);
  const json: unknown = await res.json();
  const list = Array.isArray(json) ? json : (json as { prompts?: unknown }).prompts;
  if (!Array.isArray(list)) throw new Error("gallery index.json is not a list");
  return list.map((e): GalleryEntry => {
    if (typeof e === "string") return { name: e.replace(/\.safetensors$/, ""), file: e };
    const o = e as { name?: unknown; file?: unknown };
    const file = typeof o.file === "string" ? o.file : typeof o.name === "string" ? `${o.name}.safetensors` : "";
    if (!file) throw new Error("gallery entry without a file");
    return { name: typeof o.name === "string" ? o.name : file.replace(/\.safetensors$/, ""), file };
  });
}

interface PromptFile {
  embeds: Float32Array;
  tags: number[];
}

/** `prompt_embeds [rows, text_dim]` (+ `text_tags [rows]`, all text when absent). */
async function loadPrompt(url: string, m: Manifest): Promise<PromptFile> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const st = parseSafetensors(await res.arrayBuffer());
  const name = st.tensors.has("prompt_embeds") ? "prompt_embeds" : st.tensors.has("embeds") ? "embeds" : "";
  if (!name) throw new Error(`${url}: no prompt_embeds`);
  const t = float32Tensor(st, name);
  if (t.shape.length !== 2 || t.shape[1] !== m.cfg.textDim) {
    throw new Error(`${url}: ${name} is [${t.shape.join(", ")}], this export wants [rows, ${m.cfg.textDim}]`);
  }
  const rows = t.shape[0];
  let tags = new Array<number>(rows).fill(1);
  if (st.tensors.has("text_tags")) {
    const tt = float32Tensor(st, "text_tags");
    if (tt.data.length !== rows) throw new Error(`${url}: text_tags has ${tt.data.length} entries for ${rows} rows`);
    tags = Array.from(tt.data, (v) => Math.round(v));
  }
  return { embeds: t.data, tags };
}

/** Rows of the selected prompt without fetching it twice: cached per URL. */
const promptCache = new Map<string, Promise<PromptFile>>();
function promptFor(url: string, m: Manifest): Promise<PromptFile> {
  let p = promptCache.get(url);
  if (!p) {
    p = loadPrompt(url, m);
    promptCache.set(url, p);
    p.catch(() => promptCache.delete(url));
  }
  return p;
}

// ---- keyframes --------------------------------------------------------------------

/** A PNG (or any image) → `[3, H, W]` in `[0, 1]` at the canvas size (scaled when the image is another size). */
async function keyframeFrom(file: File, anchor: Anchor, g: Geometry): Promise<Keyframe> {
  const bitmap = await createImageBitmap(file);
  if (bitmap.width !== g.width || bitmap.height !== g.height) {
    log(`${anchor} keyframe is ${bitmap.width}x${bitmap.height}; scaling to the canvas ${g.width}x${g.height}`);
  }
  const canvas = document.createElement("canvas");
  canvas.width = g.width;
  canvas.height = g.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2d context");
  ctx.drawImage(bitmap, 0, 0, g.width, g.height);
  bitmap.close();
  const img = ctx.getImageData(0, 0, g.width, g.height).data;
  const plane = g.width * g.height;
  const px = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    px[i] = img[i * 4] / 255;
    px[plane + i] = img[i * 4 + 1] / 255;
    px[2 * plane + i] = img[i * 4 + 2] / 255;
  }
  return { anchor, pixels: px };
}

function keyframeCount(): number {
  return (ui.first.files?.length ? 1 : 0) + (ui.last.files?.length ? 1 : 0);
}

// ---- download panel ----------------------------------------------------------------

class DownloadPanel {
  private readonly cells = new Map<string, { el: HTMLElement; bar: HTMLElement; total: number; loaded: Map<string, number> }>();
  private readonly fileToCell = new Map<string, string>();
  private bytes = 0;

  constructor(m: Manifest) {
    ui.blocks.replaceChildren();
    const sizeOf = (f: string | null): number => (f ? m.bytesOf(f) ?? 0 : 0);
    const group = (label: string, files: (string | null)[]): void => {
      const real = files.filter((f): f is string => !!f);
      if (real.length) this.add(label, real, real.reduce((a, f) => a + sizeOf(f), 0));
    };
    group("text", m.graphs.text_refine.map((t) => t.file));
    group("embed/head", [m.graphs.embed_video, m.graphs.embed_audio, m.graphs.head_video, m.graphs.head_audio]);
    group("AdaLN", [m.adaln.file]);
    m.graphs.blocks.forEach((b, i) => {
      const files = [b.qkv, b.post];
      const total = files.reduce((a, f) => a + sizeOf(f), 0) || b.bytes || 0;
      this.add(`block ${String(i).padStart(2, "0")}`, files, total);
    });
    group("attention", m.attention.map((a) => a.file));
    if (m.videoVae) {
      const d = m.videoVae.decoder;
      group("VAE dec", [d.in, ...d.blocks, d.out]);
      group("VAE enc", [m.videoVae.encoder_clip, m.videoVae.encoder_image]);
    }
    group("audio VAE", m.audio.flatMap((a) => [a.decoder, a.encoder]));
    group("TAEH3", m.taeh3Graphs.map((t) => t.file));
    this.render();
  }

  private add(label: string, files: string[], total: number): void {
    const el = document.createElement("div");
    el.className = "blk";
    const bar = document.createElement("div");
    bar.className = "bar";
    const span = document.createElement("span");
    span.textContent = label;
    el.append(bar, span);
    el.title = files.join("\n");
    ui.blocks.append(el);
    this.cells.set(label, { el, bar, total, loaded: new Map() });
    for (const f of files) this.fileToCell.set(f, label);
  }

  progress(p: DownloadProgress): void {
    const label = this.fileToCell.get(p.file);
    if (!label) return;
    const c = this.cells.get(label);
    if (!c) return;
    const before = c.loaded.get(p.file) ?? 0;
    c.loaded.set(p.file, p.loaded);
    this.bytes += p.loaded - before;
    const sum = [...c.loaded.values()].reduce((a, b) => a + b, 0);
    const frac = c.total ? Math.min(1, sum / c.total) : p.total ? Math.min(1, p.loaded / p.total) : 0;
    c.bar.style.width = `${(frac * 100).toFixed(0)}%`;
    c.el.classList.toggle("done", frac >= 1);
    for (const [l, o] of this.cells) o.el.classList.toggle("live", l === label && frac < 1);
    this.render();
  }

  private render(): void {
    ui.dl.textContent = `${(this.bytes / 2 ** 20).toFixed(1)} MiB fetched this session`;
  }
}

// ---- player ---------------------------------------------------------------------------

class Player {
  private frames: ImageData[] = [];
  private timer: number | null = null;
  private i = 0;
  private readonly ctx: CanvasRenderingContext2D;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2d context");
    this.ctx = ctx;
  }

  reset(width: number, height: number): void {
    this.stop();
    this.frames = [];
    this.i = 0;
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, width, height);
  }

  add(rgba: RgbaFrame[], width: number, height: number): void {
    if (this.canvas.width !== width || this.canvas.height !== height) this.reset(width, height);
    for (const px of rgba) this.frames.push(new ImageData(px, width, height));
    if (this.timer === null) {
      this.timer = window.setInterval(() => {
        if (!this.frames.length) return;
        this.ctx.putImageData(this.frames[this.i % this.frames.length], 0, 0);
        this.i++;
      }, 1000 / FPS);
    }
  }

  /** Restart from frame 0 (to line the loop up with the audio). */
  rewind(): void {
    this.i = 0;
  }

  stop(): void {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
  }
}

// ---- audio ----------------------------------------------------------------------------------

let lastAudio: Audio | null = null;
let audioCtx: AudioContext | null = null;

/** Browsers accept sample rates of a few kHz and up; the tiny fixture's 320 Hz is sample-held up to 32 kHz. */
function playable(a: Audio): Audio {
  if (a.sampleRate >= 8000) return a;
  const rate = 32000;
  const k = Math.ceil(rate / a.sampleRate);
  const up = (x: Float32Array): Float32Array => {
    const out = new Float32Array(x.length * k);
    for (let i = 0; i < x.length; i++) out.fill(x[i], i * k, (i + 1) * k);
    return out;
  };
  return { sampleRate: a.sampleRate * k, stereo: [up(a.stereo[0]), up(a.stereo[1])] };
}

function playAudio(audio: Audio): void {
  try {
    const a = playable(audio);
    if (!audioCtx || audioCtx.sampleRate !== a.sampleRate) {
      audioCtx?.close().catch(() => undefined);
      audioCtx = new AudioContext({ sampleRate: a.sampleRate });
    }
    const n = a.stereo[0].length;
    const buf = audioCtx.createBuffer(2, n, a.sampleRate);
    buf.getChannelData(0).set(a.stereo[0]);
    buf.getChannelData(1).set(a.stereo[1]);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    player.rewind();
    src.start();
    log(`playing ${(n / a.sampleRate).toFixed(2)} s of stereo audio at ${audio.sampleRate} Hz${a !== audio ? ` (held up to ${a.sampleRate} Hz for the browser)` : ""}`);
  } catch (e) {
    log(`audio: ${e instanceof Error ? e.message : String(e)}`, "err");
  }
}

// ---- the studio -----------------------------------------------------------------------------

interface Loaded {
  manifest: Manifest;
  base: string;
  gallery: GalleryEntry[];
  galleryUrl: string;
}

let loaded: Loaded | null = null;
let litertReady: Promise<void> | null = null;
let running = false;
let stopRequested = false;
const player = new Player(ui.view);

function wasmDir(): string {
  const q = new URLSearchParams(location.search).get("wasm");
  const attr = document.querySelector<HTMLScriptElement>("script[data-wasm-dir]")?.dataset.wasmDir;
  return q ?? attr ?? "../node_modules/@litertjs/core/wasm/";
}

async function ensureLiteRt(): Promise<void> {
  if (!litertReady) {
    litertReady = (async () => {
      // JSPI lets WebGPU graphs fall back to Wasm per op instead of per graph.
      const jspi = accelerator === "webgpu" && (await supportsFeature("jspi").catch(() => false));
      await loadLiteRt(wasmDir(), { jspi });
      if (accelerator === "webgpu" && !getWebGpuDevice()) {
        log("LiteRT.js found no WebGPU device; running on Wasm", "err");
        accelerator = "wasm";
      }
      log(`LiteRT.js loaded (${accelerator}${jspi ? ", jspi" : ""})`);
    })().catch((e: unknown) => {
      litertReady = null;
      throw e;
    });
  }
  return litertReady;
}

function budgetBytes(): number {
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return (mem ?? (isPhone ? 2 : 8)) * GIB;
}

function currentGeometry(): Geometry {
  return Geometry.parse(ui.size.value);
}

async function load(): Promise<void> {
  ui.go.disabled = true;
  ui.log.replaceChildren();
  ui.prompt.replaceChildren();
  ui.size.replaceChildren();
  ui.plan.textContent = "";
  const base = ui.base.value.trim();
  const galleryUrl = ui.galleryUrl.value.trim();
  try {
    const manifest = await loadManifest(base);
    ui.info.textContent = `${manifest.model} (${manifest.variant}) · ${manifest.weights} · ${manifest.cfg.layers} blocks · ${manifest.sampler.rule}, ${manifest.sampler.steps.length} steps · ${accelerator}${isPhone ? " · phone" : ""}`;
    if (!manifest.sizes.length) throw new Error("this export lists no sizes");
    for (const s of manifest.sizes) {
      const o = document.createElement("option");
      o.value = o.textContent = new Geometry(s.width, s.height, s.frames).toString();
      ui.size.append(o);
    }
    ui.preview.disabled = !manifest.taeh3Graphs.length;
    if (!manifest.taeh3Graphs.length) ui.preview.checked = false;
    const encoders = manifest.videoVae?.encoder_image != null;
    ui.first.disabled = ui.last.disabled = !encoders;
    if (!encoders) log("this export has no VAE encoder graphs; keyframes (FL2VA) are off");
    let gallery: GalleryEntry[] = [];
    try {
      gallery = await loadGallery(galleryUrl);
    } catch (e) {
      log(`no gallery: ${String(e instanceof Error ? e.message : e)}`, "err");
    }
    for (const g of gallery) {
      const o = document.createElement("option");
      o.value = g.file;
      o.textContent = g.name;
      ui.prompt.append(o);
    }
    if (!gallery.length) {
      log(`The gallery needs index.json listing prompt .safetensors (prompt_embeds [rows, ${manifest.cfg.textDim}] + text_tags [rows]) — see web/README.md.`);
    }
    new DownloadPanel(manifest);
    loaded = { manifest, base, gallery, galleryUrl };
    await showPlan();
    ui.go.disabled = !gallery.length;
    log(`loaded ${manifest.model} from ${base}: ${manifest.graphs.blocks.length} blocks, chunk ${manifest.chunk}, head group ${manifest.headGroup}, sizes ${manifest.sizes.map((s) => `${s.width}x${s.height}x${s.frames}`).join(" ")}`, "ok");
  } catch (e) {
    log(String(e instanceof Error ? e.message : e), "err");
  }
}

async function currentPlan(): Promise<Plan | null> {
  if (!loaded) return null;
  const m = loaded.manifest;
  try {
    const g = currentGeometry();
    let textRows = 0;
    if (ui.prompt.value) textRows = (await promptFor(joinUrl(loaded.galleryUrl, ui.prompt.value), m)).tags.length;
    return plan(m, {
      geometry: g,
      textRows: Math.max(1, textRows),
      keyframes: keyframeCount(),
      preview: ui.preview.checked,
      resident: Math.max(1, Number(ui.resident.value) || 2),
      budgetBytes: budgetBytes(),
    });
  } catch (e) {
    log(String(e instanceof Error ? e.message : e), "err");
    return null;
  }
}

async function showPlan(): Promise<void> {
  const p = await currentPlan();
  ui.plan.textContent = p ? formatPlan(p) + (p.fits ? "" : `\n${refusal(p)}`) : "";
  ui.plan.className = p && !p.fits ? "err" : "";
}

function refusal(p: Plan): string {
  const base = `the plan's peak ${(p.peak / GIB).toFixed(1)} GiB exceeds this device's ${(p.budget / GIB).toFixed(1)} GiB`;
  return isPhone ? `${base}: a phone cannot hold H3 (33 B parameters); only the tiny self-test export runs here` : `${base}: H3 needs a workstation browser (FastH3-Trim at int4 is ~11 GB of graphs)`;
}

async function generate(): Promise<void> {
  if (!loaded || running) return;
  const { manifest: m, base, galleryUrl } = loaded;
  const g = currentGeometry();
  const seed = Number(ui.seed.value) || 0;
  const p = await currentPlan();
  if (!p) return;
  if (!p.fits) {
    log(`refusing: ${refusal(p)}`, "err");
    return;
  }
  if (!p.attentionFile) {
    log(`refusing: ${p.notes.find((n) => n.startsWith("no attention graph")) ?? "no attention graph for this layout"}; re-export with --size`, "err");
    return;
  }
  running = true;
  stopRequested = false;
  ui.go.disabled = true;
  ui.stop.disabled = false;
  ui.play.disabled = true;
  lastAudio = null;
  ui.steps.replaceChildren();
  player.reset(g.width, g.height);
  const panel = new DownloadPanel(m);
  const source = new UrlSource(base, m.files);
  const resident = Math.max(1, Number(ui.resident.value) || 2);
  const store = new GraphStore(m, source, litertCompiler({ accelerator }), { resident, onProgress: (pr) => panel.progress(pr) });
  const t0 = performance.now();
  const timing = (label: string, ms: number): void => {
    const row = document.createElement("div");
    const a = document.createElement("span");
    a.textContent = label;
    const b = document.createElement("span");
    b.textContent = ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(0)} ms`;
    row.append(a, b);
    ui.steps.append(row);
  };
  const on = (e: PipelineEvent): void => {
    if (stopRequested) throw new Error("stopped");
    if (e.type === "prepared") timing(`prompt, layout + noise (${e.tokens} rows)`, e.ms);
    else if (e.type === "step") timing(`step ${e.index + 1}/${e.total}`, e.ms);
    else if (e.type === "frames") {
      player.add(e.rgba, e.width, e.height);
      log(`frames ${e.first}–${e.first + e.count - 1} (${e.width}x${e.height})`);
    } else {
      lastAudio = { sampleRate: e.sampleRate, stereo: e.stereo };
      ui.play.disabled = false;
    }
  };
  try {
    await ensureLiteRt();
    const prompt = await promptFor(joinUrl(galleryUrl, ui.prompt.value), m);
    const keyframes: Keyframe[] = [];
    if (ui.first.files?.[0]) keyframes.push(await keyframeFrom(ui.first.files[0], "first", g));
    if (ui.last.files?.[0]) keyframes.push(await keyframeFrom(ui.last.files[0], "last", g));
    log(`generating ${g} with seed ${seed} on ${accelerator}${keyframes.length ? ` (FL2VA, ${keyframes.map((k) => k.anchor).join(" + ")})` : ""}${ui.preview.checked ? " (TAEH3 preview)" : ""}…`);
    const adaln = AdalnTables.parse(await source.fetch(m.adaln.file, (pr) => panel.progress(pr)));
    const pipe = new Pipeline(store, adaln);
    const { video, audio, timings } = await pipe.generate(
      { embeds: prompt.embeds, textTags: prompt.tags, geometry: g, seed, keyframes, preview: ui.preview.checked },
      on,
    );
    timing(ui.preview.checked ? "TAEH3 decode" : "VAE decode", timings.decodeMs);
    timing("audio decode", timings.audioMs);
    timing("total", performance.now() - t0);
    log(`done: ${video.frames} frames of ${video.width}x${video.height} and ${(audio.stereo[0].length / audio.sampleRate).toFixed(2)} s of audio in ${((performance.now() - t0) / 1000).toFixed(1)} s`, "ok");
    playAudio(audio);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    log(msg === "stopped" ? "stopped" : `error: ${msg}`, msg === "stopped" ? "" : "err");
  } finally {
    await store.deleteAll().catch(() => undefined);
    running = false;
    ui.go.disabled = false;
    ui.stop.disabled = true;
  }
}

ui.load.addEventListener("click", () => void load());
ui.go.addEventListener("click", () => void generate());
ui.stop.addEventListener("click", () => {
  stopRequested = true;
});
ui.play.addEventListener("click", () => {
  if (lastAudio) playAudio(lastAudio);
});
for (const el of [ui.size, ui.resident, ui.preview, ui.first, ui.last, ui.prompt]) el.addEventListener("change", () => void showPlan());

// `?base=` and `?gallery=` prefill the two URLs (e.g. `?base=/fixtures/tiny/`).
const query = new URLSearchParams(location.search);
if (query.get("base")) ui.base.value = query.get("base") as string;
if (query.get("gallery")) ui.galleryUrl.value = query.get("gallery") as string;

ui.info.textContent = "…";
void (async () => {
  accelerator = await detectAccelerator();
  ui.info.textContent = `${accelerator}${isPhone ? " · phone" : ""}`;
  await load();
})();
