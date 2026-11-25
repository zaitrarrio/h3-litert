/**
 * Sampling schedules (h3-core `schedule.rs`; FastVideo `scheduling_minimax_h3.py`,
 * `layers/pdd.py`; port of `python/h3_litert/schedule.py`). Clean-time clock
 * `t = 1 − σ`; `x0 = x + σ·v`.
 *
 *     Euler    x ← (σ'/σ)·x + (1 − σ'/σ)·x0
 *     Renoise  x ← (1 − σ')·x0 + σ'·ε, x0 when σ' = 0
 *
 * A runtime reads its sigmas precomputed from the manifest; these functions
 * reproduce the exporter's values (f32 throughout, as the reference) for the
 * tests and for anyone building a manifest in the browser.
 */

const f = Math.fround;

export type Rule = "euler" | "renoise" | "pdd";

/** `σ = s·b / (1 + (s − 1)·b)` in f32. */
export function shiftSigma(base: number, shift: number): number {
  const b = f(base);
  const s = f(shift);
  return f(f(s * b) / f(f(1.0) + f(f(s - f(1.0)) * b)));
}

/** `torch.linspace` in f32: symmetric fill from both ends. */
export function linspaceF32(start: number, end: number, n: number): number[] {
  if (n === 1) return [f(start)];
  const a = f(start);
  const b = f(end);
  const step = f(f(b - a) / f(n - 1));
  const out: number[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    out.push(i < half ? f(a + f(step * f(i))) : f(b - f(step * f(n - 1 - i))));
  }
  return out;
}

export function linspaceF64(start: number, end: number, n: number): number[] {
  if (n === 1) return [start];
  const step = (end - start) / (n - 1);
  const out: number[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) out.push(i < half ? start + step * i : end - step * (n - 1 - i));
  return out;
}

export interface Shifts {
  video: number;
  audio: number;
}

export const SHIFTS_H3: Shifts = { video: 12.0, audio: 3.0 };
export const SHIFTS_FASTH3_V2: Shifts = { video: 10.0, audio: 3.0 };

/** One modality's sigma grid: `steps + 1` strictly decreasing f32 values ending at 0. */
export class Schedule {
  constructor(
    readonly sigmas: readonly number[],
    readonly rule: "euler" | "renoise",
  ) {
    const s = sigmas;
    if (s.length < 2 || s.some((b, i) => i > 0 && b >= s[i - 1]) || s[s.length - 1] !== 0.0) {
      throw new Error(`sigmas must hold >= 2 strictly decreasing values ending at 0, got [${s.join(", ")}]`);
    }
    if (rule !== "euler" && rule !== "renoise") throw new Error(`rule ${JSON.stringify(rule)} is not euler or renoise`);
  }

  static fromRungs(rungs: readonly number[], shift: number, rule: "euler" | "renoise" = "euler"): Schedule {
    if (!rungs.length || rungs.some((r) => r <= 0 || r > 1000) || rungs.some((b, i) => i > 0 && b >= rungs[i - 1])) {
      throw new Error(`rungs must be strictly decreasing in (0, 1000], got [${rungs.join(", ")}]`);
    }
    return new Schedule([...rungs.map((r) => shiftSigma(f(f(r) / f(1000.0)), shift)), 0.0], rule);
  }

  static uniform(points: number, shift: number): Schedule {
    if (points < 2) throw new Error("a uniform schedule needs >= 2 grid points");
    const sig = linspaceF32(1.0, 0.0, points).map((b) => shiftSigma(b, shift));
    const dedup = [sig[0]];
    for (let i = 1; i < sig.length; i++) if (sig[i] !== sig[i - 1]) dedup.push(sig[i]);
    return new Schedule(dedup, "euler");
  }

  get steps(): number {
    return this.sigmas.length - 1;
  }

  /** The clean time `1 − σ_i` in f32. */
  timestep(i: number): number {
    return f(f(1.0) - f(this.sigmas[i]));
  }
}

export const FASTH3_V1_RUNGS: readonly number[] = [999, 749, 500, 250];
export const FASTH3_V2_RUNGS: readonly number[] = [999, 874, 749, 624, 500, 375, 250, 125];
export const PIKA_KNOTS: readonly number[] = [1000, 750, 500, 250, 100, 40];
export const PLUG4_KNOTS: readonly number[] = [1000, 750, 500, 250];
export const H3_BASE_POINTS = 50;
export const KEYFRAME_NOISE_AUG = 0.999;

// ---- Parallel Decoding Distillation (FastH3 Omni) ---------------------------

export const PDD_GRID_MAX_T = 0.999;

export function pddShift(u: number, shift: number, maxT: number): number {
  return shift === 1.0 ? u : (shift * u * maxT) / (u * (shift - 1.0) + maxT);
}

export function pddShiftDelta(a: number, b: number, shift: number, maxT: number): number {
  if (shift === 1.0) return b - a;
  const da = maxT + (shift - 1.0) * a;
  const db = maxT + (shift - 1.0) * b;
  return (shift * maxT * maxT * (b - a)) / (da * db);
}

export class PddPlan {
  constructor(
    readonly fineGrid: readonly number[],
    readonly indices: readonly number[],
    readonly videoSigmas: readonly number[],
    readonly audioSigmas: readonly number[],
    readonly videoWeights: readonly number[],
    readonly audioWeights: readonly number[],
  ) {}

  static create(indices: readonly number[], shifts: Shifts): PddPlan {
    if (indices.length < 2 || indices[0] !== 0 || indices.some((b, i) => i > 0 && b <= indices[i - 1])) {
      throw new Error(`pdd_step_indices must increase strictly from 0, got [${indices.join(", ")}]`);
    }
    const steps = indices[indices.length - 1];
    if (steps < 2) throw new Error("a PDD grid needs >= 2 intervals");
    const grid = linspaceF64(PDD_GRID_MAX_T, 0.0, steps + 1).map((u) => Math.min(u, PDD_GRID_MAX_T));
    const weights = (s: number): number[] => grid.slice(1).map((b, i) => pddShiftDelta(grid[i], b, s, PDD_GRID_MAX_T));
    const nodes = (s: number): number[] => indices.map((i) => f(pddShift(grid[i], s, PDD_GRID_MAX_T)));
    const vw = weights(shifts.video);
    const aw = weights(shifts.audio);
    if ([...vw, ...aw].some((w) => !Number.isFinite(w) || w === 0.0)) {
      throw new Error("PDD integration weights must be finite and non-zero");
    }
    return new PddPlan(grid, [...indices], nodes(shifts.video), nodes(shifts.audio), vw, aw);
  }

  numSteps(): number {
    return this.indices.length - 1;
  }

  block(i: number): [number, number] {
    return [this.indices[i], this.indices[i + 1]];
  }

  /** The fused heads' weights of one step, normalised to sum to 1 (f32). */
  static fusedWeights(weights: readonly number[], start: number, end: number): number[] {
    const part = weights.slice(start, end);
    const total = part.reduce((a, b) => a + b, 0);
    return part.map((w) => f(w / total));
  }

  schedules(): [Schedule, Schedule] {
    return [new Schedule(this.videoSigmas, "euler"), new Schedule(this.audioSigmas, "euler")];
  }
}

// ---- the per-step plan a manifest carries -----------------------------------

export interface SamplerStep {
  t_video: number;
  t_audio: number;
  sigma_video: number;
  sigma_video_next: number;
  sigma_audio: number;
  sigma_audio_next: number;
  pdd_video_weights: number[] | null;
  pdd_audio_weights: number[] | null;
}

/** `sampler.steps` of the manifest: everything a runtime needs per step. */
export function stepPlan(video: Schedule, audio: Schedule, pdd: PddPlan | null = null, pddSteps: number | null = null): SamplerStep[] {
  if (video.steps !== audio.steps) throw new Error("video and audio schedules must have the same number of steps");
  const out: SamplerStep[] = [];
  for (let i = 0; i < video.steps; i++) {
    const e: SamplerStep = {
      t_video: video.timestep(i),
      t_audio: audio.timestep(i),
      sigma_video: video.sigmas[i],
      sigma_video_next: video.sigmas[i + 1],
      sigma_audio: audio.sigmas[i],
      sigma_audio_next: audio.sigmas[i + 1],
      pdd_video_weights: null,
      pdd_audio_weights: null,
    };
    if (pdd) {
      const [s, t] = pdd.block(i);
      const vw = PddPlan.fusedWeights(pdd.videoWeights, s, t);
      const aw = PddPlan.fusedWeights(pdd.audioWeights, s, t);
      const n = pddSteps ?? pdd.fineGrid.length - 1;
      const fullV = new Array<number>(n).fill(0.0);
      const fullA = new Array<number>(n).fill(0.0);
      for (let k = 0, j = s; j < t; j++, k++) {
        fullV[j] = vw[k];
        fullA[j] = aw[k];
      }
      e.pdd_video_weights = fullV;
      e.pdd_audio_weights = fullA;
    }
    out.push(e);
  }
  return out;
}
