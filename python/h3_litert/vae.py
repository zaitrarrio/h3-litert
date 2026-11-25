"""`AutoencoderKLMiniMaxH3`: the causal 3-D CNN encoder and the ViT decoder as
graph modules (`vae_dec_in`, `vae_dec_block_N`, `vae_dec_out`, `vae_enc_*`)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VideoVaeConfig
from .dit import attention, rms_norm, rotate_half


class CausalConv3d(nn.Conv3d):
    def __init__(self, cin, cout, k, stride=1, spatial_padding=0, temporal_padding=0):
        super().__init__(cin, cout, k, stride=stride, padding=0)
        self.sp, self.tp = spatial_padding, temporal_padding

    def forward(self, x):
        if self.sp:
            p = self.sp
            x = F.pad(x, (p, p, p, p, 0, 0), mode="reflect")
        if self.tp:
            x = F.pad(x, (0, 0, 0, 0, self.tp, 0))
        return F.conv3d(x, self.weight, self.bias, stride=self.stride)


class FrameGroupNorm(nn.GroupNorm):
    """GroupNorm per latent frame (`use_t_isolated_gn`)."""

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, 1, h, w)
        y = super().forward(y)
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class Resnet3d(nn.Module):
    def __init__(self, cin, cout, groups, eps):
        super().__init__()
        self.norm1 = FrameGroupNorm(groups, cin, eps=eps)
        self.conv1 = CausalConv3d(cin, cout, 3, spatial_padding=1, temporal_padding=2)
        self.norm2 = FrameGroupNorm(groups, cout, eps=eps)
        self.conv2 = CausalConv3d(cout, cout, 3, spatial_padding=1, temporal_padding=2)
        self.conv_shortcut = CausalConv3d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        r = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        if self.conv_shortcut is not None:
            r = self.conv_shortcut(r)
        return r + x


class Downsample3d(nn.Module):
    def __init__(self, c, ts, ss):
        super().__init__()
        self.ss = ss
        self.conv = CausalConv3d(c, c, 3, stride=(ts, ss, ss), temporal_padding=2)

    def forward(self, x):
        if self.ss == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class Encoder(nn.Module):
    """`vae_enc_*`: pixels `[3, T, H, W]` (ImageNet-normalised) → moments `[2·lz, T', H', W']`, `quant_conv` included."""

    def __init__(self, cfg: VideoVaeConfig):
        super().__init__()
        self.cfg = cfg
        chans = cfg.block_out_channels
        self.conv_in = CausalConv3d(cfg.in_channels, chans[0], 3, spatial_padding=1, temporal_padding=2)
        self.blocks = nn.ModuleList()
        cin = chans[0]
        for i, cout in enumerate(chans):
            res = nn.ModuleList(Resnet3d(cin if j == 0 else cout, cout, cfg.norm_groups, cfg.norm_eps) for j in range(cfg.layers_per_block))
            down = Downsample3d(cout, cfg.temporal_down[i], cfg.spatial_down[i]) if cfg.temporal_down[i] * cfg.spatial_down[i] > 1 else None
            self.blocks.append(nn.ModuleDict({"resnets": res, **({"down": down} if down else {})}))
            cin = cout
        self.norm_out = FrameGroupNorm(cfg.norm_groups, cin, eps=cfg.norm_eps)
        self.conv_out = CausalConv3d(cin, 2 * cfg.latent_channels, 3, spatial_padding=1, temporal_padding=2)
        self.quant_conv = nn.Conv3d(2 * cfg.latent_channels, 2 * cfg.latent_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[None]
        x = self.conv_in(x)
        for b in self.blocks:
            for r in b["resnets"]:
                x = r(x)
            if "down" in b:
                x = b["down"](x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return self.quant_conv(x)[0]


class DecBlock(nn.Module):
    def __init__(self, cfg: VideoVaeConfig):
        super().__init__()
        dim = cfg.decoder_dim()
        self.cfg = cfg
        self.norm1 = nn.Parameter(torch.ones(dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.Parameter(torch.ones(dim))
        self.ff_in = nn.Linear(dim, 2 * cfg.decoder_ffn_mult * dim)
        self.ff_out = nn.Linear(cfg.decoder_ffn_mult * dim, dim)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        rows, heads, hd, eps = h.shape[0], c.decoder_heads, c.decoder_head_dim, c.decoder_norm_eps
        n = rms_norm(h, self.norm1, eps)
        q = rotate_half(rms_norm(self.to_q(n).reshape(rows, heads, hd), None, eps), cos, sin, c.decoder_rotary_dim())
        k = rotate_half(rms_norm(self.to_k(n).reshape(rows, heads, hd), None, eps), cos, sin, c.decoder_rotary_dim())
        v = self.to_v(n).reshape(rows, heads, hd)
        h = h + self.to_out(attention(q, k, v)) * self.scale1
        n = rms_norm(h, self.norm2, eps)
        value, gate = self.ff_in(n).chunk(2, dim=-1)
        return h + self.ff_out(value * F.silu(gate)) * self.scale2


class Decoder(nn.Module):
    """The ViT decoder; `dec_in` / `blocks[N]` / `dec_out` are the graphs."""

    def __init__(self, cfg: VideoVaeConfig):
        super().__init__()
        self.cfg = cfg
        dim, lz = cfg.decoder_dim(), cfg.latent_channels
        self.post_quant_conv = nn.Linear(lz, lz)  # the 1×1×1 Conv3d as a linear over channels
        self.proj_in = nn.Linear(lz, dim)
        self.register_tokens = nn.Parameter(torch.zeros(cfg.decoder_register_tokens, dim))
        self.blocks = nn.ModuleList(DecBlock(cfg) for _ in range(cfg.decoder_layers))
        self.norm_out = nn.LayerNorm(dim, eps=cfg.decoder_norm_eps)
        pt, ps = cfg.temporal_ratio(), cfg.spatial_ratio()
        self.proj_out = nn.Linear(dim, cfg.in_channels * pt * ps * ps)

    def rows(self, t: int, h: int, w: int) -> int:
        return t * h * w + self.cfg.decoder_register_tokens + 1

    def rope(self, t: int, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        from .rope import vae_decoder_rope

        cos, sin = vae_decoder_rope((t, h, w), self.cfg.decoder_rotary_dim(), self.cfg.decoder_rope_theta, self.cfg.decoder_register_tokens + 1)
        return torch.from_numpy(cos), torch.from_numpy(sin)

    def dec_in(self, z: torch.Tensor) -> torch.Tensor:
        """`z [lz, T, H, W]` → `h [R, dim]` (tokens, register tokens, a zero cls row)."""
        lz, t, h, w = z.shape
        tok = self.proj_in(self.post_quant_conv(z.permute(1, 2, 3, 0).reshape(t * h * w, lz)))
        cls = torch.zeros(1, tok.shape[1])
        return torch.cat([tok, self.register_tokens, cls], dim=0)

    def dec_out(self, h: torch.Tensor, t: int, hh: int, ww: int) -> torch.Tensor:
        c = self.cfg
        n = t * hh * ww
        y = self.proj_out(self.norm_out(h[:n]))
        pt, ps = c.temporal_ratio(), c.spatial_ratio()
        # Diffusers' 7-D permute in two steps of <= 6 dims (TFLite's TRANSPOSE limit).
        y = y.reshape(t * hh * ww, c.in_channels, pt * ps * ps).permute(1, 0, 2)  # [c, n, pt·ps·ps]
        y = y.reshape(c.in_channels * t, hh, ww, pt, ps, ps).permute(0, 3, 1, 4, 2, 5)  # [c·t, pt, hh, ps, ww, ps]
        return y.reshape(c.in_channels, t * pt, hh * ps, ww * ps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Uncut decode of one clip tile `[lz, T, H, W]` → `[3, 4T, 16H, 16W]` (ImageNet-normalised)."""
        _, t, h, w = z.shape
        x = self.dec_in(z)
        cos, sin = self.rope(t, h, w)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.dec_out(x, t, h, w)


class DecIn(nn.Module):
    """`vae_dec_in`: the RoPE tables of the fixed tile shape ride along as constants."""

    def __init__(self, dec: Decoder, t: int, h: int, w: int):
        super().__init__()
        self.dec = dec
        cos, sin = dec.rope(t, h, w)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, z):
        return self.dec.dec_in(z), self.cos, self.sin


class DecOut(nn.Module):
    def __init__(self, dec: Decoder, t: int, h: int, w: int):
        super().__init__()
        self.dec, self.t, self.h, self.w = dec, t, h, w

    def forward(self, h):
        return self.dec.dec_out(h, self.t, self.h, self.w)


class EncoderGraph(nn.Module):
    def __init__(self, enc: Encoder):
        super().__init__()
        self.enc = enc

    def forward(self, x):
        return self.enc(x)
