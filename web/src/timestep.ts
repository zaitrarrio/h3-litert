/**
 * The timestep sinusoid (h3-core `timestep.rs`; `Timesteps(256, flip_sin_to_cos=True)`).
 * The exporter runs the time embedder and `adaln_proj` offline into
 * `adaln.safetensors`; the runtime only needs this for the tests that pin it.
 */

const f = Math.fround;

/** `[cos(t·f_i) | sin(t·f_i)]`, `f_i = exp(−ln(10000)·i / half)` in f32. */
export function sinusoid(t: number, dim: number): Float32Array {
  const half = Math.floor(dim / 2);
  const out = new Float32Array(2 * half);
  const tf = f(t);
  for (let i = 0; i < half; i++) {
    const freq = f(Math.exp(f(f(f(-Math.log(10000.0)) * f(i)) / f(half))));
    const a = f(tf * freq);
    out[i] = f(Math.cos(a));
    out[half + i] = f(Math.sin(a));
  }
  return out;
}
