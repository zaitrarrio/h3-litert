"""Checkpoint access and the mapping of Diffusers-named tensors onto the graph
modules. A missing key or a shape mismatch is an error, never a default."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .config import AudioVaeConfig, H3Config, TaeH3Config, VideoVaeConfig
from .rng import Rng


class WeightSource(Protocol):
    def get(self, name: str, shape: tuple[int, ...]) -> np.ndarray: ...
    def contains(self, name: str) -> bool: ...


def _squeeze(s):
    return tuple(d for d in s if d != 1)



class SafetensorsDir:
    """All `*.safetensors` under a directory (or one file), read lazily; also
    folds `weight_norm` pairs (`X.weight_g` / `X.weight_v`) when `X.weight` is asked for."""

    def __init__(self, path: Path | str):
        from safetensors import safe_open

        path = Path(path)
        files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"no .safetensors in {path}")
        # torch framework: bf16 checkpoints (MiniMax-H3, FastH3) read where numpy's reader refuses them
        self._files = [safe_open(str(f), framework="pt") for f in files]
        self._index = {k: i for i, f in enumerate(self._files) for k in f.keys()}

    def contains(self, name: str) -> bool:
        return name in self._index or (name.endswith(".weight") and name + "_v" in self._index)

    def names(self):
        return self._index.keys()

    def _raw(self, name: str) -> np.ndarray:
        if name not in self._index:
            raise KeyError(f"missing tensor `{name}`")
        import torch

        t = self._files[self._index[name]].get_tensor(name)
        return t.to(torch.float32).numpy()

    def get(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name not in self._index and name.endswith(".weight") and name + "_v" in self._index:
            g, v = self._raw(name + "_g"), self._raw(name + "_v")
            norm = np.sqrt((v.astype(np.float64) ** 2).reshape(v.shape[0], -1).sum(1)).reshape((-1,) + (1,) * (v.ndim - 1))
            t = (g.reshape(norm.shape) * v / norm).astype(np.float32)
        else:
            t = self._raw(name)
        if _squeeze(t.shape) != _squeeze(shape):
            raise ValueError(f"`{name}` is {list(t.shape)}, expected {list(shape)}")
        return t.reshape(shape)


class DictSource:
    """A PyTorch `state_dict` (the parity tests)."""

    def __init__(self, sd: dict):
        self.sd = {k: (v.detach().float().numpy() if hasattr(v, "detach") else np.asarray(v, np.float32)) for k, v in sd.items()}

    def contains(self, name: str) -> bool:
        return name in self.sd or (name.endswith(".weight") and name + "_v" in self.sd)

    def get(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name not in self.sd and name.endswith(".weight") and name + "_v" in self.sd:
            g, v = self.sd[name + "_g"], self.sd[name + "_v"]
            norm = np.sqrt((v.astype(np.float64) ** 2).reshape(v.shape[0], -1).sum(1)).reshape((-1,) + (1,) * (v.ndim - 1))
            t = (g.reshape(norm.shape) * v / norm).astype(np.float32)
        else:
            t = self.sd[name]
        if _squeeze(t.shape) != _squeeze(shape):
            raise ValueError(f"`{name}` is {list(t.shape)}, expected {list(shape)}")
        return t.reshape(shape)


class Synthetic:
    """Deterministic weights by name hash (h3-core `weights::Synthetic`'s role):
    norms near 1, biases and alphas near 0, linears scaled by fan-in."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def contains(self, name: str) -> bool:
        return True

    def get(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        n = int(np.prod(shape))
        h = 1469598103934665603
        for ch in name.encode():
            h = ((h ^ ch) * 1099511628211) & ((1 << 64) - 1)
        rng = Rng(self.seed ^ h)
        if name.endswith(("norm.weight", "norm1.weight", "norm2.weight", "norm3.weight", "norm_q.weight", "norm_k.weight",
                          "final_norm.weight", "norm_out.weight", "norm_out.norm.weight", "scale1", "scale2", "mlp.norm.weight")):
            return (1.0 + rng.uniform(n, 0.1)).reshape(shape)
        if name.endswith((".bias", "alpha", "beta", "q_bias", "v_bias", "zero_k_bias")) or "register_tokens" in name:
            return rng.uniform(n, 0.1).reshape(shape)
        if "adaln" in name or "norm_out.linear" in name:
            return rng.uniform(n, 0.02).reshape(shape)
        fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        return rng.uniform(n, float(np.sqrt(3.0 / fan_in))).reshape(shape)


def _t(src: WeightSource, name: str, shape) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(src.get(name, tuple(shape))))


def _lin(src, mod: torch.nn.Linear, key: str, shape, bias: bool):
    mod.weight.copy_(_t(src, f"{key}.weight", shape))
    if bias:
        mod.bias.copy_(_t(src, f"{key}.bias", (shape[0],)))


def _conv(src, mod, key: str, shape, bias: bool = True):
    mod.weight.copy_(_t(src, f"{key}.weight", shape))
    if bias:
        # ConvTranspose weights are `[cin, cout, k]`: the bias is `cout`.
        out = shape[1] if isinstance(mod, torch.nn.ConvTranspose1d) else shape[0]
        mod.bias.copy_(_t(src, f"{key}.bias", (out,)))


# ---- DiT ---------------------------------------------------------------------------


@torch.no_grad()
def load_attn_ff(u, src: WeightSource, p: str) -> None:
    """One attention + SwiGLU unit (`transformer_blocks.N` or `token_refiner.refiner_blocks.N`)."""
    cfg: H3Config = u.cfg
    d, inner, ffn = cfg.hidden, cfg.dim, cfg.ffn_dim
    for m in ("to_q", "to_k", "to_v"):
        _lin(src, getattr(u, m), f"{p}.attn.{m}", (inner, d), False)
    _lin(src, u.to_out, f"{p}.attn.to_out.0", (d, inner), False)
    u.norm_q.copy_(_t(src, f"{p}.attn.norm_q.weight", (cfg.head_dim,)))
    u.norm_k.copy_(_t(src, f"{p}.attn.norm_k.weight", (cfg.head_dim,)))
    u.norm1.copy_(_t(src, f"{p}.norm1.weight", (d,)))
    u.norm2.copy_(_t(src, f"{p}.norm2.weight", (d,)))
    _lin(src, u.ff_in, f"{p}.ff.net.0.proj", (2 * ffn, d), False)
    _lin(src, u.ff_out, f"{p}.ff.net.2", (d, ffn), False)


@torch.no_grad()
def load_adaln(lin: torch.nn.Linear, src: WeightSource, cfg: H3Config, l: int) -> None:
    _lin(src, lin, f"transformer_blocks.{l}.adaln_proj.linear", (18 * cfg.hidden, cfg.adaln_in()), True)


@torch.no_grad()
def load_dit(dit, src: WeightSource) -> None:
    """Fill a `dit.H3Dit` (everything it allocated: with `with_blocks=False`, the blocks are skipped)."""
    cfg: H3Config = dit.cfg
    d = cfg.hidden
    _lin(src, dit.embed_video.proj, "proj_in", (d, cfg.video_patch_dim()), True)
    _lin(src, dit.embed_audio.proj, "audio_proj_in", (d, cfg.audio_in_channels), True)
    _lin(src, dit.text.context_embedder, "context_embedder", (d, cfg.text_dim), True)
    _lin(src, dit.time.linear_1, "time_embedder.linear_1", (cfg.time_embed_hidden, cfg.freq_dim), True)
    _lin(src, dit.time.linear_2, "time_embedder.linear_2", (cfg.time_embed_dim, cfg.time_embed_hidden), True)
    if cfg.adaln_rank:
        dit.time.basis.copy_(_t(src, "adaln_basis.weight", (cfg.adaln_rank, cfg.time_embed_dim)))
    _lin(src, dit.norm_out_linear, "norm_out.linear", (2 * d, cfg.adaln_in()), True)
    dit.norm_out.copy_(_t(src, "norm_out.norm.weight", (d,)))
    heads = cfg.pdd_steps or 1
    _lin(src, dit.head_video.proj, "proj_out", (heads * cfg.video_patch_dim(), d), True)
    _lin(src, dit.head_audio.proj, "audio_proj_out", (heads * cfg.audio_in_channels, d), True)
    dit.text.final_norm.copy_(_t(src, "token_refiner.final_norm.weight", (d,)))
    for l, u in enumerate(dit.text.blocks):
        load_attn_ff(u, src, f"token_refiner.refiner_blocks.{l}")
    for l, u in enumerate(dit.blocks):
        load_attn_ff(u, src, f"transformer_blocks.{l}")
        load_adaln(dit.adaln[l], src, cfg, l)


def open_source(spec):
    """`("safetensors", path)` or `("synthetic", seed)` → a WeightSource (picklable for worker processes)."""
    kind, arg = spec
    return SafetensorsDir(arg) if kind == "safetensors" else Synthetic(int(arg))


# ---- video VAE -----------------------------------------------------------------------


@torch.no_grad()
def load_vae_encoder(enc, src: WeightSource) -> None:
    cfg: VideoVaeConfig = enc.cfg
    chans = cfg.block_out_channels
    _conv(src, enc.conv_in, "encoder.conv_in", (chans[0], cfg.in_channels, 3, 3, 3))
    cin = chans[0]
    for i, b in enumerate(enc.blocks):
        cout = chans[i]
        for j, r in enumerate(b["resnets"]):
            p = f"encoder.down_blocks.{i}.resnets.{j}"
            ci = cin if j == 0 else cout
            r.norm1.weight.copy_(_t(src, f"{p}.norm1.weight", (ci,)))
            r.norm1.bias.copy_(_t(src, f"{p}.norm1.bias", (ci,)))
            _conv(src, r.conv1, f"{p}.conv1", (cout, ci, 3, 3, 3))
            r.norm2.weight.copy_(_t(src, f"{p}.norm2.weight", (cout,)))
            r.norm2.bias.copy_(_t(src, f"{p}.norm2.bias", (cout,)))
            _conv(src, r.conv2, f"{p}.conv2", (cout, cout, 3, 3, 3))
            if r.conv_shortcut is not None:
                _conv(src, r.conv_shortcut, f"{p}.conv_shortcut", (cout, ci, 1, 1, 1))
        if "down" in b:
            _conv(src, b["down"].conv, f"encoder.down_blocks.{i}.downsamplers.0.conv", (cout, cout, 3, 3, 3))
        cin = cout
    enc.norm_out.weight.copy_(_t(src, "encoder.norm_out.weight", (cin,)))
    enc.norm_out.bias.copy_(_t(src, "encoder.norm_out.bias", (cin,)))
    _conv(src, enc.conv_out, "encoder.conv_out", (2 * cfg.latent_channels, cin, 3, 3, 3))
    _conv(src, enc.quant_conv, "quant_conv", (2 * cfg.latent_channels, 2 * cfg.latent_channels, 1, 1, 1))


@torch.no_grad()
def load_vae_decoder(dec, src: WeightSource) -> None:
    cfg: VideoVaeConfig = dec.cfg
    lz, dim, ff = cfg.latent_channels, cfg.decoder_dim(), cfg.decoder_ffn_mult * cfg.decoder_dim()
    dec.post_quant_conv.weight.copy_(_t(src, "post_quant_conv.weight", (lz, lz, 1, 1, 1)).reshape(lz, lz))
    dec.post_quant_conv.bias.copy_(_t(src, "post_quant_conv.bias", (lz,)))
    _lin(src, dec.proj_in, "decoder.proj_in", (dim, lz), True)
    dec.register_tokens.copy_(_t(src, "decoder.register_tokens", (1, cfg.decoder_register_tokens, dim)).reshape(-1, dim))
    for l, b in enumerate(dec.blocks):
        p = f"decoder.transformer_blocks.{l}"
        b.norm1.copy_(_t(src, f"{p}.norm1.weight", (dim,)))
        b.norm2.copy_(_t(src, f"{p}.norm2.weight", (dim,)))
        b.scale1.copy_(_t(src, f"{p}.scale1", (dim,)))
        b.scale2.copy_(_t(src, f"{p}.scale2", (dim,)))
        for m in ("to_q", "to_k", "to_v"):
            _lin(src, getattr(b, m), f"{p}.attn.{m}", (dim, dim), True)
        _lin(src, b.to_out, f"{p}.attn.to_out.0", (dim, dim), True)
        _lin(src, b.ff_in, f"{p}.ff.net.0.proj", (2 * ff, dim), True)
        _lin(src, b.ff_out, f"{p}.ff.net.2", (dim, ff), True)
    dec.norm_out.weight.copy_(_t(src, "decoder.norm_out.weight", (dim,)))
    dec.norm_out.bias.copy_(_t(src, "decoder.norm_out.bias", (dim,)))
    out = cfg.in_channels * cfg.temporal_ratio() * cfg.spatial_ratio() ** 2
    _lin(src, dec.proj_out, "decoder.proj_out", (out, dim), True)


# ---- audio VAE -----------------------------------------------------------------------


@torch.no_grad()
def load_audio_encoder(enc, src: WeightSource) -> None:
    cfg: AudioVaeConfig = enc.cfg
    d0 = cfg.encoder_dim
    _conv(src, enc.block[0], "encoder.block.0", (d0, 1, 7))
    c = d0
    for i, stride in enumerate(cfg.encoder_rates):
        p = f"encoder.block.{i + 1}"
        eb = enc.block[i + 1]
        for j in range(3):
            q = f"{p}.block.{j}"
            ru = eb.block[j]
            ru.block[0].alpha.copy_(_t(src, f"{q}.block.0.alpha", (1, c, 1)))
            _conv(src, ru.block[1], f"{q}.block.1", (c, c, 7))
            ru.block[2].alpha.copy_(_t(src, f"{q}.block.2.alpha", (1, c, 1)))
            _conv(src, ru.block[3], f"{q}.block.3", (c, c, 1))
        eb.block[3].alpha.copy_(_t(src, f"{p}.block.3.alpha", (1, c, 1)))
        _conv(src, eb.block[4], f"{p}.block.4", (2 * c, c, 2 * stride))
        c *= 2
    n = len(cfg.encoder_rates) + 1
    enc.block[n].alpha.copy_(_t(src, f"encoder.block.{n}.alpha", (1, c, 1)))
    _conv(src, enc.block[n + 1], f"encoder.block.{n + 1}", (cfg.latent_dim, c, 3))
    ld, lc = cfg.latent_dim, cfg.latent_channels
    for mod, key, n_ in ((enc.norm1, "pre_block.norm1", ld), (enc.norm3, "pre_block.norm3", ld), (enc.norm2, "pre_block.norm2", lc), (enc.mlp_norm, "pre_block.mlp.norm", lc)):
        mod.weight.copy_(_t(src, f"{key}.weight", (n_,)))
        mod.bias.copy_(_t(src, f"{key}.bias", (n_,)))
    enc.qkv.weight.copy_(_t(src, "pre_block.attn.qkv.weight", (3 * ld, ld)))
    enc.q_bias.copy_(_t(src, "pre_block.attn.q_bias", (ld,)))
    enc.v_bias.copy_(_t(src, "pre_block.attn.v_bias", (ld,)))
    _lin(src, enc.attn_proj, "pre_block.attn.proj", (lc, lc), True)
    _lin(src, enc.proj, "pre_block.proj", (lc, ld), True)
    _lin(src, enc.w0, "pre_block.mlp.w0", (2 * lc, lc), True)
    _lin(src, enc.w1, "pre_block.mlp.w1", (2 * lc, lc), True)
    _lin(src, enc.w2, "pre_block.mlp.w2", (lc, 2 * lc), True)
    _conv(src, enc.mean_proj, "mean_proj", (lc, lc, 1))


@torch.no_grad()
def load_audio_decoder(dec, src: WeightSource) -> None:
    cfg: AudioVaeConfig = dec.cfg
    ld, lc, dd = cfg.latent_dim, cfg.latent_channels, cfg.decoder_dim
    _conv(src, dec.dec_in_proj, "dec_in_proj", (ld, lc, 1))
    _conv(src, dec.conv_pre, "decoder.conv_pre", (dd, ld, 7))
    c = dd
    nk = len(cfg.resblock_kernels)
    for i, (rate, k) in enumerate(zip(cfg.decoder_rates, cfg.decoder_kernels)):
        _conv(src, dec.ups[i], f"decoder.ups.{i}.0", (c, c // 2, k))
        c //= 2
        for j, rk in enumerate(cfg.resblock_kernels):
            p = f"decoder.resblocks.{i * nk + j}"
            rb = dec.resblocks[i * nk + j]
            for di in range(len(cfg.resblock_dilations[j])):
                _conv(src, rb.convs1[di], f"{p}.convs1.{di}", (c, c, rk))
                _conv(src, rb.convs2[di], f"{p}.convs2.{di}", (c, c, rk))
            for a in range(2 * len(cfg.resblock_dilations[j])):
                rb.activations[a].act.alpha.copy_(_t(src, f"{p}.activations.{a}.act.alpha", (c,)))
                rb.activations[a].act.beta.copy_(_t(src, f"{p}.activations.{a}.act.beta", (c,)))
    dec.activation_post.act.alpha.copy_(_t(src, "decoder.activation_post.act.alpha", (c,)))
    dec.activation_post.act.beta.copy_(_t(src, "decoder.activation_post.act.beta", (c,)))
    _conv(src, dec.conv_post, "decoder.conv_post", (1, c, 7), bias=False)


# ---- TAEH3 -------------------------------------------------------------------------------


@torch.no_grad()
def load_tae(tae, src: WeightSource) -> None:
    cfg: TaeH3Config = tae.cfg
    nf = cfg.widths
    _conv(src, tae.first, "decoder.1", (nf[0], cfg.latent_channels, 3, 3))
    for s, stage in enumerate(tae.stages):
        base = 3 + 6 * s
        for m, mem in enumerate(stage.mems):
            p = f"decoder.{base + m}"
            _conv(src, mem.conv0, f"{p}.conv.0", (nf[s], 2 * nf[s], 3, 3))
            _conv(src, mem.conv2, f"{p}.conv.2", (nf[s], nf[s], 3, 3))
            _conv(src, mem.conv4, f"{p}.conv.4", (nf[s], nf[s], 3, 3))
        _conv(src, stage.grow, f"decoder.{base + 4}.conv", (nf[s] * cfg.tgrow[s], nf[s], 1, 1), bias=False)
        _conv(src, stage.conv, f"decoder.{base + 5}", (nf[s + 1], nf[s], 3, 3), bias=False)
    _conv(src, tae.last, "decoder.22", (cfg.out_channels(), nf[3], 3, 3))


def read_h3_config(transformer_dir: Path) -> H3Config:
    return H3Config.from_diffusers_json(json.loads((Path(transformer_dir) / "config.json").read_text()))
