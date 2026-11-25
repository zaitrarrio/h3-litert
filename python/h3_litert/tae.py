"""TAEH3 (madebyollin/taehv, taeh3.safetensors): fastwan-litert's TAEHV graph with
H3's 24 latent channels and the x2 pixel-shuffle head. The host keeps raw frame
i when i mod 20 >= 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TaeH3Config as TaeConfig

# ---- TAEHV -----------------------------------------------------------------


def conv3(cin: int, cout: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, padding=1, bias=bias)


class MemBlock(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.conv0 = conv3(2 * n, n)
        self.conv2 = conv3(n, n)
        self.conv4 = conv3(n, n)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv0(torch.cat([x, past], dim=1)))
        y = F.relu(self.conv2(y))
        return F.relu(self.conv4(y) + x)


class TaeStage(nn.Module):
    def __init__(self, cfg: TaeConfig, s: int):
        super().__init__()
        n, n_next, stride = cfg.widths[s], cfg.widths[s + 1], cfg.tgrow[s]
        self.n, self.stride, self.last = n, stride, s == 2
        self.mems = nn.ModuleList(MemBlock(n) for _ in range(cfg.mem_blocks))
        self.grow = nn.Conv2d(n, n * stride, 1, bias=False)
        self.conv = conv3(n, n_next, bias=False)

    def forward(self, x: torch.Tensor, mems: list[torch.Tensor]):
        """`x [f, n, h, w]`, `mems` one `[1, n, h, w]` per MemBlock →
        `(x' [f·stride, n_next, 2h, 2w], new mems)`."""
        new = []
        for m, mem in zip(self.mems, mems):
            past = torch.cat([mem, x[:-1]], dim=0)
            new.append(x[-1:])
            x = m(x, past)
        f, n, h, w = x.shape
        g = self.grow(x).reshape(f * self.stride, n, h, w)  # TGrow before the upsample (exact)
        g = F.interpolate(g, scale_factor=2.0, mode="nearest")
        y = self.conv(g)
        if self.last:
            y = F.relu(y)
        return y, new


class TaeH3(nn.Module):
    """TAEHV decoder as one chunk graph: `z[F, C, h, w]` and 9 memories →
    `rgb[4F, 3, 8ph, 8pw]` in `[0, 1]` and the 9 new memories."""

    def __init__(self, cfg: TaeConfig):
        super().__init__()
        self.cfg = cfg
        self.first = conv3(cfg.latent_channels, cfg.widths[0])
        self.stages = nn.ModuleList(TaeStage(cfg, s) for s in range(3))
        self.last = conv3(cfg.widths[3], cfg.out_channels())

    def memory_shapes(self, h: int, w: int) -> list[tuple[int, ...]]:
        return [(1, self.cfg.widths[s], h << s, w << s) for s in range(3) for _ in range(self.cfg.mem_blocks)]

    def zero_memory(self, h: int, w: int) -> list[torch.Tensor]:
        return [torch.zeros(s) for s in self.memory_shapes(h, w)]

    def forward(self, z: torch.Tensor, *mems: torch.Tensor):
        cfg = self.cfg
        x = torch.tanh(z / 3.0) * 3.0
        x = F.relu(self.first(x))
        out_mems: list[torch.Tensor] = []
        for s, stage in enumerate(self.stages):
            k = cfg.mem_blocks
            x, new = stage(x, list(mems[s * k : (s + 1) * k]))
            out_mems.extend(new)
        rgb = self.last(x)
        if cfg.patch > 1:
            rgb = F.pixel_shuffle(rgb, cfg.patch)
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return (rgb, *out_mems)

