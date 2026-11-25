"""Sampling schedules (h3-core `schedule.rs`; FastVideo `scheduling_minimax_h3.py`,
`layers/pdd.py`). Clean-time clock `t = 1 − σ`; `x0 = x + σ·v`.

    Euler    x ← (σ'/σ)·x + (1 − σ'/σ)·x0
    Renoise  x ← (1 − σ')·x0 + σ'·ε, x0 when σ' = 0

Every sigma is computed in f32 as the reference does."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

f32 = np.float32


def shift_sigma(base: float, shift: float) -> float:
    b, s = f32(base), f32(shift)
    return float((s * b) / (f32(1.0) + (s - f32(1.0)) * b))


def linspace_f32(start: float, end: float, n: int) -> list[float]:
    """`torch.linspace` in f32: symmetric fill from both ends."""
    if n == 1:
        return [float(f32(start))]
    start, end = f32(start), f32(end)
    step = (end - start) / f32(n - 1)
    out = []
    for i in range(n):
        v = start + step * f32(i) if i < n // 2 else end - step * f32(n - 1 - i)
        out.append(float(f32(v)))
    return out


def linspace_f64(start: float, end: float, n: int) -> list[float]:
    if n == 1:
        return [start]
    step = (end - start) / (n - 1)
    return [start + step * i if i < n // 2 else end - step * (n - 1 - i) for i in range(n)]


@dataclass(frozen=True)
class Shifts:
    video: float
    audio: float


SHIFTS_H3 = Shifts(12.0, 3.0)
SHIFTS_FASTH3_V2 = Shifts(10.0, 3.0)


@dataclass(frozen=True)
class Schedule:
    """One modality's sigma grid: `steps + 1` strictly decreasing f32 values ending at 0."""

    sigmas: tuple[float, ...]
    rule: str  # "euler" | "renoise"

    def __post_init__(self):
        s = self.sigmas
        if len(s) < 2 or any(b >= a for a, b in zip(s, s[1:])) or s[-1] != 0.0:
            raise ValueError(f"sigmas must hold >= 2 strictly decreasing values ending at 0, got {s}")
        if self.rule not in ("euler", "renoise"):
            raise ValueError(f"rule {self.rule!r} is not euler or renoise")

    @classmethod
    def from_rungs(cls, rungs: list[int], shift: float, rule: str = "euler") -> "Schedule":
        if not rungs or any(r <= 0 or r > 1000 for r in rungs) or any(b >= a for a, b in zip(rungs, rungs[1:])):
            raise ValueError(f"rungs must be strictly decreasing in (0, 1000], got {rungs}")
        return cls(tuple(shift_sigma(float(f32(r) / f32(1000.0)), shift) for r in rungs) + (0.0,), rule)

    @classmethod
    def uniform(cls, points: int, shift: float) -> "Schedule":
        if points < 2:
            raise ValueError("a uniform schedule needs >= 2 grid points")
        sig = [shift_sigma(b, shift) for b in linspace_f32(1.0, 0.0, points)]
        dedup = [sig[0]] + [b for a, b in zip(sig, sig[1:]) if b != a]
        return cls(tuple(dedup), "euler")

    @property
    def steps(self) -> int:
        return len(self.sigmas) - 1

    def timestep(self, i: int) -> float:
        return float(f32(1.0) - f32(self.sigmas[i]))


FASTH3_V1_RUNGS = [999, 749, 500, 250]
FASTH3_V2_RUNGS = [999, 874, 749, 624, 500, 375, 250, 125]
PIKA_KNOTS = [1000, 750, 500, 250, 100, 40]
PLUG4_KNOTS = [1000, 750, 500, 250]
H3_BASE_POINTS = 50
KEYFRAME_NOISE_AUG = 0.999

# ---- Parallel Decoding Distillation (FastH3 Omni) ---------------------------
PDD_GRID_MAX_T = 0.999


def pdd_shift(u: float, shift: float, max_t: float) -> float:
    return u if shift == 1.0 else shift * u * max_t / (u * (shift - 1.0) + max_t)


def pdd_shift_delta(a: float, b: float, shift: float, max_t: float) -> float:
    if shift == 1.0:
        return b - a
    da = max_t + (shift - 1.0) * a
    db = max_t + (shift - 1.0) * b
    return shift * max_t * max_t * (b - a) / (da * db)


@dataclass(frozen=True)
class PddPlan:
    fine_grid: tuple[float, ...]
    indices: tuple[int, ...]
    video_sigmas: tuple[float, ...]
    audio_sigmas: tuple[float, ...]
    video_weights: tuple[float, ...]
    audio_weights: tuple[float, ...]

    @classmethod
    def new(cls, indices: list[int], shifts: Shifts) -> "PddPlan":
        if len(indices) < 2 or indices[0] != 0 or any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError(f"pdd_step_indices must increase strictly from 0, got {indices}")
        steps = indices[-1]
        if steps < 2:
            raise ValueError("a PDD grid needs >= 2 intervals")
        grid = [min(u, PDD_GRID_MAX_T) for u in linspace_f64(PDD_GRID_MAX_T, 0.0, steps + 1)]
        weights = lambda s: tuple(pdd_shift_delta(a, b, s, PDD_GRID_MAX_T) for a, b in zip(grid, grid[1:]))  # noqa: E731
        nodes = lambda s: tuple(float(f32(pdd_shift(grid[i], s, PDD_GRID_MAX_T))) for i in indices)  # noqa: E731
        vw, aw = weights(float(shifts.video)), weights(float(shifts.audio))
        if any(not np.isfinite(w) or w == 0.0 for w in vw + aw):
            raise ValueError("PDD integration weights must be finite and non-zero")
        return cls(tuple(grid), tuple(indices), nodes(float(shifts.video)), nodes(float(shifts.audio)), vw, aw)

    def num_steps(self) -> int:
        return len(self.indices) - 1

    def block(self, i: int) -> tuple[int, int]:
        return self.indices[i], self.indices[i + 1]

    @staticmethod
    def fused_weights(weights, start: int, end: int) -> list[float]:
        total = sum(weights[start:end])
        return [float(f32(w / total)) for w in weights[start:end]]

    def schedules(self) -> tuple[Schedule, Schedule]:
        return Schedule(self.video_sigmas, "euler"), Schedule(self.audio_sigmas, "euler")


# ---- the per-step plan a manifest carries -----------------------------------


def step_plan(video: Schedule, audio: Schedule, pdd: PddPlan | None = None, pdd_steps: int | None = None) -> list[dict]:
    """`sampler.steps` of the manifest: everything a runtime needs per step."""
    if video.steps != audio.steps:
        raise ValueError("video and audio schedules must have the same number of steps")
    out = []
    for i in range(video.steps):
        e = {
            "t_video": video.timestep(i), "t_audio": audio.timestep(i),
            "sigma_video": video.sigmas[i], "sigma_video_next": video.sigmas[i + 1],
            "sigma_audio": audio.sigmas[i], "sigma_audio_next": audio.sigmas[i + 1],
            "pdd_video_weights": None, "pdd_audio_weights": None,
        }
        if pdd is not None:
            s, t = pdd.block(i)
            vw = PddPlan.fused_weights(pdd.video_weights, s, t)
            aw = PddPlan.fused_weights(pdd.audio_weights, s, t)
            n = pdd_steps or len(pdd.fine_grid) - 1
            # Heads are indexed by fine-grid interval; the fused graph takes a weight per head.
            full_v = [0.0] * n
            full_a = [0.0] * n
            for k, j in enumerate(range(s, t)):
                full_v[j], full_a[j] = vw[k], aw[k]
            e["pdd_video_weights"], e["pdd_audio_weights"] = full_v, full_a
        out.append(e)
    return out
