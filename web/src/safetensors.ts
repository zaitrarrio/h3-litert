/**
 * A minimal safetensors reader: 8-byte little-endian u64 header length, a JSON
 * header of `{name: {dtype, shape, data_offsets}}` plus `__metadata__`, then
 * the raw data. Reads F32, F16 (the AdaLN table) and U8 (the fixture's frames).
 */

import { f16ToF32Array } from "./f16.js";

export type SafeDtype = "F32" | "F16" | "BF16" | "I64" | "I32" | "U8" | "BOOL" | "F64";

export interface SafeTensor {
  dtype: SafeDtype;
  shape: number[];
  /** The raw bytes of the tensor (a copy aligned at offset 0). */
  bytes: Uint8Array;
}

export interface SafeTensorsFile {
  metadata: Record<string, string>;
  tensors: Map<string, SafeTensor>;
}

interface HeaderEntry {
  dtype: SafeDtype;
  shape: number[];
  data_offsets: [number, number];
}

function isHeaderEntry(v: unknown): v is HeaderEntry {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return typeof o.dtype === "string" && Array.isArray(o.shape) && Array.isArray(o.data_offsets) && o.data_offsets.length === 2;
}

export function parseSafetensors(buf: ArrayBuffer | Uint8Array): SafeTensorsFile {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (u8.length < 8) throw new Error("safetensors: file shorter than its header length");
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = Number(view.getBigUint64(0, true));
  if (8 + n > u8.length) throw new Error(`safetensors: header of ${n} bytes exceeds the file`);
  const header: unknown = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n)));
  if (typeof header !== "object" || header === null) throw new Error("safetensors: header is not an object");
  const data = u8.subarray(8 + n);
  const tensors = new Map<string, SafeTensor>();
  let metadata: Record<string, string> = {};
  for (const [name, entry] of Object.entries(header as Record<string, unknown>)) {
    if (name === "__metadata__") {
      if (typeof entry === "object" && entry !== null) {
        metadata = Object.fromEntries(Object.entries(entry as Record<string, unknown>).map(([k, v]) => [k, String(v)]));
      }
      continue;
    }
    if (!isHeaderEntry(entry)) throw new Error(`safetensors: malformed entry for \`${name}\``);
    const [a, b] = entry.data_offsets;
    if (a < 0 || b < a || b > data.length) throw new Error(`safetensors: \`${name}\` offsets [${a}, ${b}) exceed the data`);
    tensors.set(name, { dtype: entry.dtype, shape: entry.shape.map(Number), bytes: data.slice(a, b) });
  }
  return { metadata, tensors };
}

function get(file: SafeTensorsFile, name: string): SafeTensor {
  const t = file.tensors.get(name);
  if (!t) throw new Error(`safetensors: no tensor \`${name}\` (has ${[...file.tensors.keys()].join(", ")})`);
  return t;
}

/** The named tensor as `Float32Array`: `F32` as is, `F16` widened. */
export function float32Tensor(file: SafeTensorsFile, name: string): { data: Float32Array; shape: number[] } {
  const t = get(file, name);
  if (t.dtype === "F32") return { data: new Float32Array(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength / 4), shape: t.shape };
  if (t.dtype === "F16") return { data: f16ToF32Array(new Uint16Array(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength / 2)), shape: t.shape };
  throw new Error(`safetensors: \`${name}\` is ${t.dtype}, expected F32 or F16`);
}

/** The named `F16` tensor's raw bit patterns (for tables widened on demand). */
export function float16Tensor(file: SafeTensorsFile, name: string): { data: Uint16Array; shape: number[] } {
  const t = get(file, name);
  if (t.dtype !== "F16") throw new Error(`safetensors: \`${name}\` is ${t.dtype}, expected F16`);
  return { data: new Uint16Array(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength / 2), shape: t.shape };
}

/** The named tensor as `Uint8Array` (must be `U8`). */
export function uint8Tensor(file: SafeTensorsFile, name: string): { data: Uint8Array; shape: number[] } {
  const t = get(file, name);
  if (t.dtype !== "U8") throw new Error(`safetensors: \`${name}\` is ${t.dtype}, expected U8`);
  return { data: t.bytes, shape: t.shape };
}
