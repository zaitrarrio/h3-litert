"""What a generation asks for, shared by the reference and the graph pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .packing import Geometry


@dataclass
class Keyframe:
    anchor: str  # "first" | "last"
    pixels: np.ndarray  # [3, 1, H, W] in [0, 1] at the canvas size


@dataclass
class Request:
    embeds: np.ndarray  # [rows, text_dim]
    text_tags: list[int]
    geometry: Geometry
    seed: int = 7
    keyframes: list[Keyframe] = field(default_factory=list)
    preview: bool = False  # TAEH3 instead of the VAE decoder


KEYFRAME_ENCODE_SEED = 42
