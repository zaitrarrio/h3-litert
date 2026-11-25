/**
 * Running `.tflite` graphs in the browser through LiteRT.js.
 *
 * `Graph` is the one interface the pipeline drives: inputs bound by NAME,
 * float32 outputs returned by POSITION (`output_0`, `output_1`, … of
 * `docs/graphs.md`). `LiteRtGraph` implements it over a `CompiledModel`;
 * `GraphStore` fetches graphs of a manifest on demand (from a base URL or an
 * in-memory file map), keeps at most `resident` block graphs alive (LRU, the
 * rest are deleted and re-fetched) and reports download progress.
 */

import { loadAndCompile, Tensor, type CompiledModel, type CompileOptions } from "@litertjs/core";

import type { Manifest } from "./manifest.js";
import { joinUrl } from "./manifest.js";

/** A float32 input: a bare array takes the shape the graph's signature declares. */
export type GraphInput = Float32Array | { data: Float32Array; shape: readonly number[] };

export interface Graph {
  readonly file: string;
  /** Run once; every output as a fresh `Float32Array`, in signature order. */
  run(inputs: Record<string, GraphInput>): Promise<Float32Array[]>;
  inputShape(name: string): readonly number[];
  inputNames(): readonly string[];
  /** Release the graph (and its weights). */
  delete(): void;
}

export type Accelerator = "wasm" | "webgpu";

export interface LiteRtGraphOptions {
  accelerator: Accelerator;
  /** WebGPU delegate precision; `fp16` halves the activation traffic (judged at 2e-2, `docs/graphs.md`). */
  precision?: "fp16" | "fp32";
  /** CPU threads when the threaded Wasm build is loaded (`loadLiteRt(dir, {threads: true})`). */
  numThreads?: number;
}

/** Signature output order: by the trailing number of the name (`output_0`, `output_1`, …), then by index. */
export function outputOrder(names: readonly { name: string; index: number }[]): number[] {
  const key = (n: string): number => {
    const m = /(\d+)$/.exec(n);
    return m ? Number(m[1]) : 0;
  };
  return names
    .map((d, i) => i)
    .sort((a, b) => key(names[a].name) - key(names[b].name) || names[a].index - names[b].index);
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

/** One compiled LiteRT.js graph. */
export class LiteRtGraph implements Graph {
  private readonly inputs: { name: string; shape: number[] }[];
  private readonly outputs: { name: string; dtype: string }[];
  private readonly order: number[];
  private deleted = false;

  constructor(
    readonly file: string,
    private readonly model: CompiledModel,
    readonly accelerator: Accelerator,
  ) {
    this.inputs = model.getInputDetails().map((d) => {
      if (d.dtype !== "float32") throw new Error(`${file}: input ${d.name} is ${d.dtype}, the contract is float32`);
      return { name: d.name, shape: Array.from(d.shape) };
    });
    const outs = model.getOutputDetails();
    this.outputs = outs.map((d) => ({ name: d.name, dtype: d.dtype }));
    this.order = outputOrder(outs.map((d) => ({ name: d.name, index: d.index })));
  }

  static async compile(file: string, bytes: Uint8Array, opts: LiteRtGraphOptions): Promise<LiteRtGraph> {
    const compile: CompileOptions = { accelerator: opts.accelerator };
    if (opts.precision) compile.gpuOptions = { precision: opts.precision };
    if (opts.numThreads) compile.cpuOptions = { numThreads: opts.numThreads };
    let model: CompiledModel;
    try {
      model = await loadAndCompile(bytes, compile);
    } catch (e) {
      throw new Error(`${file}: LiteRT.js could not compile the graph for ${opts.accelerator}: ${String(e)}`, { cause: e });
    }
    return new LiteRtGraph(file, model, opts.accelerator);
  }

  inputNames(): readonly string[] {
    return this.inputs.map((i) => i.name);
  }

  inputShape(name: string): readonly number[] {
    const i = this.inputs.find((d) => d.name === name);
    if (!i) throw new Error(`${this.file}: no input \`${name}\` (inputs: ${this.inputNames().join(", ")})`);
    return i.shape;
  }

  async run(inputs: Record<string, GraphInput>): Promise<Float32Array[]> {
    if (this.deleted) throw new Error(`${this.file}: graph was deleted`);
    const given = Object.keys(inputs).sort();
    const want = [...this.inputNames()].sort();
    if (given.length !== want.length || given.some((n, i) => n !== want[i])) {
      throw new Error(`${this.file}: inputs [${want.join(", ")}], got [${given.join(", ")}]`);
    }
    const tensors: Tensor[] = [];
    try {
      for (const d of this.inputs) {
        const v = inputs[d.name];
        const data = v instanceof Float32Array ? v : v.data;
        const shape = v instanceof Float32Array ? d.shape : v.shape;
        if (!sameShape(shape, d.shape)) {
          throw new Error(`${this.file}: input ${d.name} is [${shape.join(", ")}], the graph wants [${d.shape.join(", ")}]`);
        }
        const n = shape.reduce((a, b) => a * b, 1);
        if (data.length !== n) {
          throw new Error(`${this.file}: input ${d.name} has ${data.length} values for shape [${shape.join(", ")}]`);
        }
        tensors.push(new Tensor(data, [...shape]));
      }
      const outs = await this.model.run(tensors);
      const result: Float32Array[] = new Array<Float32Array>(outs.length);
      for (let i = 0; i < outs.length; i++) {
        let t = outs[i];
        try {
          if (t.accelerator !== "wasm") {
            const moved = await t.moveTo("wasm");
            t = moved;
          }
          const arr = t.toTypedArray();
          if (!(arr instanceof Float32Array)) {
            throw new Error(`${this.file}: output ${this.outputs[i].name} is ${t.dtype}, the contract is float32`);
          }
          result[i] = arr.slice();
        } finally {
          t.delete();
        }
      }
      return this.order.map((i) => result[i]);
    } finally {
      for (const t of tensors) t.delete();
    }
  }

  delete(): void {
    if (this.deleted) return;
    this.deleted = true;
    this.model.delete();
  }
}

// ---- sources ------------------------------------------------------------------

export interface DownloadProgress {
  file: string;
  loaded: number;
  /** Known from the manifest's `files` or `Content-Length`; 0 when unknown. */
  total: number;
}

export type OnProgress = (p: DownloadProgress) => void;

/** Where graph bytes come from. */
export interface GraphSource {
  fetch(file: string, onProgress?: OnProgress): Promise<Uint8Array>;
}

/** Graphs fetched from `<baseUrl>/<file>`, with streaming progress. */
export class UrlSource implements GraphSource {
  constructor(
    readonly baseUrl: string,
    private readonly sizes: Readonly<Record<string, { bytes: number }>> = {},
  ) {}

  async fetch(file: string, onProgress?: OnProgress): Promise<Uint8Array> {
    const url = joinUrl(this.baseUrl, file);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const header = Number(res.headers.get("content-length") ?? 0);
    const total = this.sizes[file]?.bytes ?? (Number.isFinite(header) ? header : 0);
    if (!res.body) {
      const buf = new Uint8Array(await res.arrayBuffer());
      onProgress?.({ file, loaded: buf.length, total: total || buf.length });
      return buf;
    }
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    onProgress?.({ file, loaded: 0, total });
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress?.({ file, loaded, total });
    }
    const out = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) {
      out.set(c, off);
      off += c.length;
    }
    if (total && loaded !== total) throw new Error(`${url}: downloaded ${loaded} bytes, the manifest says ${total}`);
    return out;
  }
}

/** Graphs from memory (tests, or a page that unpacked an archive). */
export class MapSource implements GraphSource {
  constructor(private readonly files: ReadonlyMap<string, Uint8Array>) {}

  fetch(file: string, onProgress?: OnProgress): Promise<Uint8Array> {
    const v = this.files.get(file);
    if (!v) return Promise.reject(new Error(`no graph file \`${file}\` in the source`));
    onProgress?.({ file, loaded: v.length, total: v.length });
    return Promise.resolve(v);
  }
}

// ---- store --------------------------------------------------------------------

/** Turns graph bytes into a `Graph`; `litertCompiler` is the LiteRT.js one, tests may fake it. */
export type GraphCompiler = (file: string, bytes: Uint8Array) => Promise<Graph>;

export function litertCompiler(opts: LiteRtGraphOptions): GraphCompiler {
  return (file, bytes) => LiteRtGraph.compile(file, bytes, opts);
}

export interface GraphStoreOptions {
  /** Block graphs kept alive at once (LRU); the fixed graphs always are. */
  resident?: number;
  onProgress?: OnProgress;
}

/**
 * Graphs of one manifest, fetched and compiled on demand. Block graphs beyond
 * `resident` are deleted least-recently-used first, so a 50-block model
 * streams through memory a few blocks at a time.
 */
export class GraphStore {
  readonly resident: number;
  private readonly onProgress: OnProgress | undefined;
  private readonly fixedGraphs = new Map<string, Promise<Graph>>();
  private readonly blocks = new Map<string, Promise<Graph>>();

  constructor(
    readonly m: Manifest,
    readonly source: GraphSource,
    readonly compiler: GraphCompiler,
    opts: GraphStoreOptions = {},
  ) {
    this.resident = Math.max(1, opts.resident ?? 4);
    this.onProgress = opts.onProgress;
  }

  private load(file: string): Promise<Graph> {
    return this.source.fetch(file, this.onProgress).then((bytes) => {
      const want = this.m.bytesOf(file);
      if (want !== undefined && bytes.length !== want) {
        throw new Error(`${file}: ${bytes.length} bytes, the manifest says ${want}`);
      }
      return this.compiler(file, bytes);
    });
  }

  /** A graph that stays resident (text_refine, embed_*, head_*, attention, the VAE decoders / encoders, TAEH3). */
  fixed(file: string): Promise<Graph> {
    let g = this.fixedGraphs.get(file);
    if (!g) {
      g = this.load(file);
      this.fixedGraphs.set(file, g);
      g.catch(() => this.fixedGraphs.delete(file));
    }
    return g;
  }

  /** A block graph (qkv / post), evicting the least recently used beyond `resident`. */
  block(file: string): Promise<Graph> {
    const have = this.blocks.get(file);
    if (have) {
      this.blocks.delete(file);
      this.blocks.set(file, have);
      return have;
    }
    while (this.blocks.size >= this.resident) {
      const oldest = this.blocks.keys().next().value;
      if (oldest === undefined) break;
      const p = this.blocks.get(oldest);
      this.blocks.delete(oldest);
      p?.then((g) => g.delete()).catch(() => undefined);
    }
    const g = this.load(file);
    this.blocks.set(file, g);
    g.catch(() => this.blocks.delete(file));
    return g;
  }

  /** Files currently resident (for a page's status line). */
  residentFiles(): string[] {
    return [...this.fixedGraphs.keys(), ...this.blocks.keys()];
  }

  async dropBlocks(): Promise<void> {
    const ps = [...this.blocks.values()];
    this.blocks.clear();
    for (const p of ps) {
      try {
        (await p).delete();
      } catch {
        // a graph that failed to load has nothing to delete
      }
    }
  }

  async deleteAll(): Promise<void> {
    await this.dropBlocks();
    const ps = [...this.fixedGraphs.values()];
    this.fixedGraphs.clear();
    for (const p of ps) {
      try {
        (await p).delete();
      } catch {
        // as above
      }
    }
  }
}
