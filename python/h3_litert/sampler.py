"""One denoising step per modality (h3-core `backend::FlowStep`; `MiniMaxH3Scheduler.step`)."""

from __future__ import annotations

import numpy as np

f32 = np.float32


def flow_step(x: np.ndarray, v: np.ndarray, sigma: float, nxt: float, rule: str, noise: np.ndarray | None) -> np.ndarray:
    """`x0 = x + σ·v`; Euler `(σ'/σ)·x + (1 − σ'/σ)·x0`; renoise `(1 − σ')·x0 + σ'·ε` (`x0` when σ' = 0)."""
    s, n = f32(sigma), f32(nxt)
    x0 = x + s * v
    if rule == "euler":
        r = n / s
        return (r * x + (f32(1.0) - r) * x0).astype(np.float32)
    if rule in ("renoise", "pdd"):
        if rule == "pdd":
            r = n / s
            return (r * x + (f32(1.0) - r) * x0).astype(np.float32)
        if n == 0.0:
            return x0.astype(np.float32)
        assert noise is not None
        return ((f32(1.0) - n) * x0 + n * noise).astype(np.float32)
    raise ValueError(f"rule {rule!r}")


def needs_noise(rule: str, nxt: float) -> bool:
    return rule == "renoise" and nxt != 0.0
