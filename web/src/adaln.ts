/**
 * The precomputed AdaLN table (`adaln.safetensors`, `docs/graphs.md`):
 * `timesteps [U]` f32, `blocks [U, layers, 3, 6, d]` f16 and `head [U, 2, d]`
 * f16. The f16 rows stay packed; a `[6, d]` (or `[2, d]`) f32 row is widened
 * when a step asks for it and cached.
 */

import { f16ToF32Array } from "./f16.js";
import { float16Tensor, float32Tensor, parseSafetensors, type SafeTensorsFile } from "./safetensors.js";

export const ADALN_ATOL = 1e-6;

export class AdalnTables {
  readonly timesteps: Float32Array;
  readonly count: number;
  readonly layers: number;
  readonly dim: number;
  private readonly blocks: Uint16Array;
  private readonly head: Uint16Array;
  private readonly cache = new Map<number, Float32Array>();

  constructor(file: SafeTensorsFile) {
    const ts = float32Tensor(file, "timesteps");
    const b = float16Tensor(file, "blocks");
    const h = float16Tensor(file, "head");
    if (ts.shape.length !== 1) throw new Error(`adaln: timesteps is [${ts.shape.join(", ")}], expected [U]`);
    const u = ts.shape[0];
    if (b.shape.length !== 5 || b.shape[0] !== u || b.shape[2] !== 3 || b.shape[3] !== 6) {
      throw new Error(`adaln: blocks is [${b.shape.join(", ")}], expected [${u}, layers, 3, 6, d]`);
    }
    if (h.shape.length !== 3 || h.shape[0] !== u || h.shape[1] !== 2 || h.shape[2] !== b.shape[4]) {
      throw new Error(`adaln: head is [${h.shape.join(", ")}], expected [${u}, 2, ${b.shape[4]}]`);
    }
    for (let i = 1; i < u; i++) {
      if (!(ts.data[i] > ts.data[i - 1])) throw new Error("adaln: timesteps are not sorted ascending");
    }
    this.timesteps = ts.data;
    this.count = u;
    this.layers = b.shape[1];
    this.dim = b.shape[4];
    this.blocks = b.data;
    this.head = h.data;
  }

  static parse(bytes: ArrayBuffer | Uint8Array): AdalnTables {
    return new AdalnTables(parseSafetensors(bytes));
  }

  /** The row of clean time `t` (within 1e-6 of a table entry), or an error naming what the table covers. */
  indexOf(t: number): number {
    const tf = Math.fround(t);
    for (let i = 0; i < this.count; i++) if (Math.abs(this.timesteps[i] - tf) <= ADALN_ATOL) return i;
    throw new Error(`no AdaLN table row for timestep ${t}; this export covers [${Array.from(this.timesteps).join(", ")}]`);
  }

  /** `blocks[u, layer, tag]` as f32 `[6, d]`. */
  mods(u: number, layer: number, tag: number): Float32Array {
    if (u < 0 || u >= this.count || layer < 0 || layer >= this.layers || tag < 0 || tag > 2) {
      throw new Error(`adaln: no entry (${u}, ${layer}, ${tag})`);
    }
    const key = (u * this.layers + layer) * 3 + tag;
    let row = this.cache.get(key);
    if (!row) {
      const start = key * 6 * this.dim;
      row = f16ToF32Array(this.blocks, start, start + 6 * this.dim);
      this.cache.set(key, row);
    }
    return row;
  }

  /** `head[u]` as f32 `[2, d]`. */
  headMods(u: number): Float32Array {
    if (u < 0 || u >= this.count) throw new Error(`adaln: no head row ${u}`);
    const key = -1 - u;
    let row = this.cache.get(key);
    if (!row) {
      row = f16ToF32Array(this.head, u * 2 * this.dim, (u + 1) * 2 * this.dim);
      this.cache.set(key, row);
    }
    return row;
  }

  /** Bytes of the table as shipped (f16). */
  bytes(): number {
    return 4 * this.count + 2 * (this.blocks.length + this.head.length);
  }
}
