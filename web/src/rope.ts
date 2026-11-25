/**
 * Rotary tables (h3-core `rope.rs`; port of `python/h3_litert/rope.py`):
 * rotate-half over the first `rot` channels of a head, `rot/2` angles per row,
 * formed in f32 as the reference (positions and inverse frequencies cast to
 * f32, the product rounded to f32, cos / sin rounded to f32).
 */

const f = Math.fround;

export interface RopeTable {
  /** `[rows, width]` row-major. */
  cos: Float32Array;
  sin: Float32Array;
  rows: number;
  /** `3 · rope_freq_dim` (48 for H3). */
  width: number;
}

/** DiT table for packed positions `[rows][3]` (flat, `(t, h, w)` per row): `[rows, 3·rope_freq_dim]` cos / sin. */
export function h3Rope(positions: ArrayLike<number>, ropeFreqDim: number, theta: number): RopeTable {
  if (positions.length % 3 !== 0) throw new Error(`h3_rope: ${positions.length} values are not rows of 3 positions`);
  const rows = positions.length / 3;
  const inv = new Float32Array(ropeFreqDim);
  for (let i = 0; i < ropeFreqDim; i++) inv[i] = f(1.0 / Math.pow(theta, (2.0 * i) / (2 * ropeFreqDim)));
  const width = 3 * ropeFreqDim;
  const cos = new Float32Array(rows * width);
  const sin = new Float32Array(rows * width);
  for (let r = 0; r < rows; r++) {
    for (let a = 0; a < 3; a++) {
      const p = f(positions[r * 3 + a]);
      const base = r * width + a * ropeFreqDim;
      for (let i = 0; i < ropeFreqDim; i++) {
        const ang = f(p * inv[i]);
        cos[base + i] = f(Math.cos(ang));
        sin[base + i] = f(Math.sin(ang));
      }
    }
  }
  return { cos, sin, rows, width };
}

/**
 * ViT decoder table (for reference; the exporter bakes it into `vae_dec_in`):
 * `rot` rotated channels, three axes over length-normalised `[−1, 1)` grids,
 * `(t, h, w)` token order, then `suffix` rows at position 0.
 */
export function vaeDecoderRope(grid: readonly [number, number, number], rot: number, theta: number, suffix: number): RopeTable {
  const perAxis = Math.floor(rot / 6);
  const inv: number[] = [];
  for (let k = 0; k < perAxis; k++) inv.push(1.0 / Math.pow(theta, (k * 6.0) / rot));
  const coords = (n: number): number[] => Array.from({ length: n }, (_, i) => 2.0 * ((i + 0.5) / n) - 1.0);
  const [gt, gh, gw] = grid;
  const ct = coords(gt);
  const ch = coords(gh);
  const cw = coords(gw);
  const n = gt * gh * gw;
  const rows = n + suffix;
  const width = 3 * perAxis;
  const cos = new Float32Array(rows * width);
  const sin = new Float32Array(rows * width);
  const fill = (row: number, p: readonly [number, number, number]): void => {
    for (let a = 0; a < 3; a++) {
      for (let k = 0; k < perAxis; k++) {
        const ang = f(2.0 * Math.PI * p[a] * inv[k]);
        cos[row * width + a * perAxis + k] = f(Math.cos(ang));
        sin[row * width + a * perAxis + k] = f(Math.sin(ang));
      }
    }
  };
  let row = 0;
  for (const t of ct) for (const h of ch) for (const w of cw) fill(row++, [t, h, w]);
  for (let s = 0; s < suffix; s++) fill(row++, [0, 0, 0]);
  return { cos, sin, rows, width };
}
