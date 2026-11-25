"""Rotary tables (h3-core `rope.rs`): rotate-half over the first `rot` channels
of a head, `rot/2` angles per row, formed in f32 as the reference."""

from __future__ import annotations

import math

import numpy as np


def h3_rope(positions, rope_freq_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """DiT table for packed positions `[rows][3]`: `[rows, 3·rope_freq_dim]` cos / sin."""
    inv = (1.0 / theta ** (2.0 * np.arange(rope_freq_dim) / (2 * rope_freq_dim))).astype(np.float32)
    p = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    angles = (p[:, :, None] * inv[None, None, :]).reshape(p.shape[0], 3 * rope_freq_dim)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def vae_decoder_rope(grid: tuple[int, int, int], rot: int, theta: float, suffix: int) -> tuple[np.ndarray, np.ndarray]:
    """ViT decoder table: `rot` rotated channels, three axes over length-normalised
    `[−1, 1)` grids, `(t, h, w)` token order, then `suffix` rows at position 0."""
    per_axis = rot // 6
    inv = np.array([1.0 / theta ** (k * 6.0 / rot) for k in range(per_axis)])
    coords = lambda n: np.array([2.0 * ((i + 0.5) / n) - 1.0 for i in range(n)])  # noqa: E731
    ct, ch, cw = coords(grid[0]), coords(grid[1]), coords(grid[2])
    tt, hh, ww = np.meshgrid(ct, ch, cw, indexing="ij")
    p = np.stack([tt.ravel(), hh.ravel(), ww.ravel()], axis=1)  # [n, 3]
    p = np.concatenate([p, np.zeros((suffix, 3))], axis=0)
    angles = (2.0 * math.pi * p[:, :, None] * inv[None, None, :]).reshape(p.shape[0], 3 * per_axis).astype(np.float32)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
