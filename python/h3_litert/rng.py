"""Deterministic standard-normal noise (h3-core `rng.rs` (the same generator as fastwan-rs)): SplitMix64 →
Box–Muller, bit for bit the same stream on every platform, so the same seed
gives the same latent in Rust, Python, TypeScript, Kotlin and Swift."""

from __future__ import annotations

import math

import numpy as np

_GOLDEN = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_MASK = (1 << 64) - 1


class Rng:
    def __init__(self, seed: int):
        self.state = (seed ^ _GOLDEN) & _MASK

    def next_u64(self) -> int:
        self.state = (self.state + _GOLDEN) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * _M1) & _MASK
        z = ((z ^ (z >> 27)) * _M2) & _MASK
        return z ^ (z >> 31)

    def _unit(self) -> float:
        """Uniform in `(0, 1]`."""
        return ((self.next_u64() >> 11) + 1) / float(1 << 53)

    def _raw(self, n: int) -> np.ndarray:
        """`n` SplitMix64 outputs, vectorised: the state sequence is an
        arithmetic progression, so the mixes run on a numpy uint64 array
        (wrapping arithmetic, as the Rust)."""
        starts = (self.state + _GOLDEN * np.arange(1, n + 1, dtype=np.uint64)) & np.uint64(_MASK)
        self.state = (self.state + _GOLDEN * n) & _MASK
        with np.errstate(over="ignore"):
            z = starts
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            return z ^ (z >> np.uint64(31))

    def normal(self, n: int) -> np.ndarray:
        """`n` standard normals as f32 (Box–Muller in f64, pairs)."""
        pairs = (n + 1) // 2
        raw = self._raw(2 * pairs)
        u = ((raw >> np.uint64(11)).astype(np.float64) + 1.0) / float(1 << 53)
        u1, u2 = u[0::2], u[1::2]
        r = np.sqrt(-2.0 * np.log(u1))
        th = (2.0 * math.pi) * u2
        out = np.empty(2 * pairs, dtype=np.float64)
        out[0::2] = r * np.cos(th)
        out[1::2] = r * np.sin(th)
        return out[:n].astype(np.float32)

    def uniform(self, n: int, a: float) -> np.ndarray:
        """Uniform in `[-a, a)`; for synthetic weights."""
        raw = self._raw(n)
        u = ((raw >> np.uint64(11)).astype(np.float64) + 1.0) / float(1 << 53)
        return ((u * 2.0 - 1.0).astype(np.float32)) * np.float32(a)
