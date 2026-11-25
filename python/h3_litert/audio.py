"""`AutoencoderKLMiniMaxH3Audio`: the DAC encoder with its causal-attention
projection (`avae_enc_A`) and the BigVGAN decoder (`avae_dec_A`), one mono
channel per call. Weight-normed convolutions are folded at load."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AudioVaeConfig

AA_KERNEL = 12
AA_RATIO = 2


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    half = kernel_size // 2
    att = 2.285 * (half - 1) * math.pi * (4 * half_width) + 7.95
    if att > 50.0:
        beta = 0.1102 * (att - 8.7)
    elif att >= 21.0:
        beta = 0.5842 * (att - 21) ** 0.4 + 0.07886 * (att - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = torch.arange(-half, half) + 0.5 if kernel_size % 2 == 0 else torch.arange(kernel_size) - half
    f = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    f = f / f.sum()
    return f.reshape(1, 1, kernel_size)


def replicate_pad(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """`F.pad(mode="replicate")` written as slices + concat (a converter-friendly form)."""
    parts = []
    if left:
        parts.append(x[..., :1].expand(*x.shape[:-1], left))
    parts.append(x)
    if right:
        parts.append(x[..., -1:].expand(*x.shape[:-1], right))
    return torch.cat(parts, dim=-1)


class Snake1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, c, 1))

    def forward(self, x):
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class SnakeBeta(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(c))
        self.beta = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        a = torch.exp(self.alpha)[None, :, None]
        b = torch.exp(self.beta)[None, :, None]
        return x + (b + 1e-9).reciprocal() * torch.sin(a * x).pow(2)


class LowPass(nn.Module):
    def __init__(self, cutoff, half_width, stride, k):
        super().__init__()
        even = k % 2 == 0
        self.pl, self.pr, self.stride = k // 2 - int(even), k // 2, stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, k))

    def forward(self, x):
        # Depthwise with one shared tap: channels fold into the batch axis (an ungrouped conv, converter-friendly).
        b, c, n = x.shape
        x = replicate_pad(x, self.pl, self.pr).reshape(b * c, 1, -1)
        return F.conv1d(x, self.filter, stride=self.stride).reshape(b, c, -1)


class UpSample(nn.Module):
    def __init__(self, ratio, k):
        super().__init__()
        self.ratio, self.stride = ratio, ratio
        self.pad = k // ratio - 1
        self.pl = self.pad * self.stride + (k - self.stride) // 2
        self.pr = self.pad * self.stride + (k - self.stride + 1) // 2
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, k))

    def forward(self, x):
        b, c, n = x.shape
        x = replicate_pad(x, self.pad, self.pad).reshape(b * c, 1, -1)
        y = self.ratio * F.conv_transpose1d(x, self.filter, stride=self.stride)
        y = y.reshape(b, c, -1)
        return y[..., self.pl : y.shape[-1] - self.pr]


class AliasFree(nn.Module):
    def __init__(self, act, ratio=AA_RATIO, k=AA_KERNEL):
        super().__init__()
        self.act = act
        self.upsample = UpSample(ratio, k)
        self.downsample = LowPass(0.5 / ratio, 0.6 / ratio, ratio, k)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.block = nn.Sequential(Snake1d(dim), nn.Conv1d(dim, dim, 7, dilation=dilation, padding=3 * dilation), Snake1d(dim), nn.Conv1d(dim, dim, 1))

    def forward(self, x):
        r = self.block(x)
        pad = (x.shape[-1] - r.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + r


class EncoderBlock(nn.Module):
    def __init__(self, dim, stride):
        super().__init__()
        self.block = nn.Sequential(ResidualUnit(dim // 2, 1), ResidualUnit(dim // 2, 3), ResidualUnit(dim // 2, 9), Snake1d(dim // 2),
                                   nn.Conv1d(dim // 2, dim, 2 * stride, stride=stride, padding=math.ceil(stride / 2)))

    def forward(self, x):
        return self.block(x)


class AudioEncoder(nn.Module):
    """`avae_enc_A`: `wave [800·A]` → posterior mean `[lc, A]` (encoder, `pre_block`, `mean_proj`)."""

    def __init__(self, cfg: AudioVaeConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.encoder_dim
        layers: list[nn.Module] = [nn.Conv1d(1, d, 7, padding=3)]
        for s in cfg.encoder_rates:
            d *= 2
            layers.append(EncoderBlock(d, s))
        layers += [Snake1d(d), nn.Conv1d(d, cfg.latent_dim, 3, padding=1)]
        self.block = nn.Sequential(*layers)
        ld, lc, heads = cfg.latent_dim, cfg.latent_channels, cfg.heads
        self.norm1 = nn.LayerNorm(ld)
        self.qkv = nn.Linear(ld, 3 * ld, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(ld))
        self.v_bias = nn.Parameter(torch.zeros(ld))
        self.attn_proj = nn.Linear(lc, lc)
        self.proj = nn.Linear(ld, lc)
        self.norm3 = nn.LayerNorm(ld)
        self.norm2 = nn.LayerNorm(lc)
        self.mlp_norm = nn.LayerNorm(lc)
        self.w0 = nn.Linear(lc, 2 * lc)
        self.w1 = nn.Linear(lc, 2 * lc)
        self.w2 = nn.Linear(2 * lc, lc)
        self.mean_proj = nn.Conv1d(lc, lc, 1)
        self.heads = heads

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        """Causal attention over `[T, ld]`, heads mean-pooled, head dim average-pooled to `lc`."""
        t, ld = x.shape
        hd = ld // self.heads
        bias = torch.cat([self.q_bias, torch.zeros_like(self.q_bias), self.v_bias])
        qkv = F.linear(x, self.qkv.weight, bias).reshape(t, 3, self.heads, hd)
        q, k, v = qkv[:, 0].permute(1, 0, 2), qkv[:, 1].permute(1, 0, 2), qkv[:, 2].permute(1, 0, 2)
        s = torch.matmul(q, k.transpose(1, 2)) * (1.0 / math.sqrt(hd))
        causal = torch.triu(torch.full((t, t), -1e9), diagonal=1)
        p = torch.softmax(s + causal[None], dim=-1)
        o = torch.matmul(p, v).mean(0)  # [t, hd]
        lc = self.cfg.latent_channels
        o = o.reshape(t, lc, hd // lc).mean(-1)  # adaptive average pool hd → lc
        return self.attn_proj(o)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        h = self.block(wave.reshape(1, 1, -1))[0].T  # [A, ld]
        h = self.proj(self.norm3(h)) + self.attn(self.norm1(h))
        n = self.mlp_norm(self.norm2(h))
        h = h + self.w2(F.gelu(self.w0(n), approximate="tanh") * self.w1(n))
        return self.mean_proj(h.T[None])[0]  # [lc, A]


class AmpBlock(nn.Module):
    def __init__(self, c, k, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList(nn.Conv1d(c, c, k, dilation=d, padding=(k * d - d) // 2) for d in dilations)
        self.convs2 = nn.ModuleList(nn.Conv1d(c, c, k, dilation=1, padding=(k - 1) // 2) for _ in dilations)
        self.activations = nn.ModuleList(AliasFree(SnakeBeta(c)) for _ in range(2 * len(dilations)))

    def forward(self, x):
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            r = c1(self.activations[2 * i](x))
            r = c2(self.activations[2 * i + 1](r))
            x = x + r
        return x


class AudioDecoder(nn.Module):
    """`avae_dec_A`: denormalised latents `[lc, A]` → `wave [800·A]` in `[−1, 1]` (`dec_in_proj` + BigVGAN)."""

    def __init__(self, cfg: AudioVaeConfig):
        super().__init__()
        self.cfg = cfg
        self.dec_in_proj = nn.Conv1d(cfg.latent_channels, cfg.latent_dim, 1)
        dd = cfg.decoder_dim
        self.conv_pre = nn.Conv1d(cfg.latent_dim, dd, 7, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        c = dd
        for rate, k in zip(cfg.decoder_rates, cfg.decoder_kernels):
            self.ups.append(nn.ConvTranspose1d(c, c // 2, k, rate, padding=(k - rate) // 2))
            c //= 2
            for rk, dil in zip(cfg.resblock_kernels, cfg.resblock_dilations):
                self.resblocks.append(AmpBlock(c, rk, dil))
        self.activation_post = AliasFree(SnakeBeta(c))
        self.conv_post = nn.Conv1d(c, 1, 7, padding=3, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        nk = len(self.cfg.resblock_kernels)
        x = self.conv_pre(self.dec_in_proj(z[None]))
        for i, up in enumerate(self.ups):
            x = up(x)
            acc = None
            for j in range(nk):
                y = self.resblocks[i * nk + j](x)
                acc = y if acc is None else acc + y
            x = acc / nk
        x = self.conv_post(self.activation_post(x))
        return torch.clamp(x, -1.0, 1.0).reshape(-1)
