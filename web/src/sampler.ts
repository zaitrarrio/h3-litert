/**
 * One denoising step per modality (h3-core `backend::FlowStep`;
 * `MiniMaxH3Scheduler.step`; port of `python/h3_litert/sampler.py`). Float32
 * arithmetic throughout, reproduced with `Math.fround`.
 */

import type { Rule } from "./schedule.js";

const f = Math.fround;

/**
 * `x0 = x + σ·v`; Euler `(σ'/σ)·x + (1 − σ'/σ)·x0`; renoise `(1 − σ')·x0 + σ'·ε`
 * (`x0` when σ' = 0); `pdd` steps like Euler on its own node grid.
 */
export function flowStep(x: Float32Array, v: Float32Array, sigma: number, nxt: number, rule: Rule, noise: Float32Array | null): Float32Array {
  if (v.length !== x.length) throw new Error(`flow_step: v has ${v.length} values, x has ${x.length}`);
  const s = f(sigma);
  const n = f(nxt);
  const out = new Float32Array(x.length);
  if (rule === "euler" || rule === "pdd") {
    const r = f(n / s);
    const one = f(f(1.0) - r);
    for (let i = 0; i < x.length; i++) {
      const x0 = f(x[i] + f(s * v[i]));
      out[i] = f(f(r * x[i]) + f(one * x0));
    }
    return out;
  }
  if (rule === "renoise") {
    if (n === 0.0) {
      for (let i = 0; i < x.length; i++) out[i] = f(x[i] + f(s * v[i]));
      return out;
    }
    if (noise === null || noise.length !== x.length) throw new Error("flow_step: renoise needs noise of the target's size");
    const one = f(f(1.0) - n);
    for (let i = 0; i < x.length; i++) {
      const x0 = f(x[i] + f(s * v[i]));
      out[i] = f(f(one * x0) + f(n * noise[i]));
    }
    return out;
  }
  throw new Error(`rule ${JSON.stringify(rule)}`);
}

export function needsNoise(rule: Rule, nxt: number): boolean {
  return rule === "renoise" && nxt !== 0.0;
}
