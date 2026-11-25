"""Timestep conditioning on the host (h3-core `timestep.rs`): clean time
`t ∈ [0, 1]` → `Timesteps(256, flip_sin_to_cos=True)` → two-layer SiLU MLP;
the exporter feeds the result to every block's `adaln_proj` once."""

from __future__ import annotations

import math

import numpy as np


def sinusoid(t: float, dim: int) -> np.ndarray:
    half = dim // 2
    f = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / np.float32(half)).astype(np.float32)
    a = np.float32(t) * f
    return np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def matvec(w: np.ndarray, b: np.ndarray | None, x: np.ndarray) -> np.ndarray:
    y = w.astype(np.float64) @ x.astype(np.float64)
    if b is not None:
        y = y + b.astype(np.float64)
    return y.astype(np.float32)


class TimeEmbedder:
    def __init__(self, freq_dim: int, l1w, l1b, l2w, l2b, basis=None):
        self.freq_dim, self.l1w, self.l1b, self.l2w, self.l2b, self.basis = freq_dim, l1w, l1b, l2w, l2b, basis

    def temb(self, t: float) -> np.ndarray:
        return matvec(self.l2w, self.l2b, silu(matvec(self.l1w, self.l1b, sinusoid(t, self.freq_dim))))

    def adaln_input(self, t: float) -> np.ndarray:
        """`silu(temb)`, or `basis · silu(temb)` for a rank-reduced checkpoint."""
        s = silu(self.temb(t))
        return matvec(self.basis, None, s) if self.basis is not None else s
