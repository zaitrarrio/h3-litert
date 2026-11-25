"""The uncut PyTorch pipeline: the host loop of `docs/graphs.md` with the whole
modules in place of the graphs. It writes the tiny fixture's expectations and is
what the graph pipeline is tested against."""

from __future__ import annotations

import numpy as np
import torch

from . import vae_host
from .audio import AudioDecoder
from .config import AudioVaeConfig, VIDEO_TAG, VideoVaeConfig
from .dit import H3Dit
from .packing import Geometry, build_layout, patch_rows, row_timesteps, unpatch_rows
from .request import KEYFRAME_ENCODE_SEED, Keyframe
from .rng import Rng
from .sampler import flow_step, needs_noise
from .schedule import KEYFRAME_NOISE_AUG
from .vae import Decoder, Encoder
from .variants import Variant
from .weights import WeightSource, load_audio_decoder, load_dit, load_vae_decoder, load_vae_encoder


def f16_round(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).astype(np.float32)


def encode_keyframes(cfg: VideoVaeConfig, keyframes: list[Keyframe], encode_clip, patch) -> np.ndarray:
    """Condition rows of FL2VA keyframes: encode, sample the posterior (seed 42), f16 round, normalise, patchify."""
    rows = []
    for kf in keyframes:
        moments = vae_host.encode(cfg, vae_host.normalize_pixels(kf.pixels), encode_clip)
        rng = Rng(KEYFRAME_ENCODE_SEED)
        lz = moments.shape[0] // 2
        z = vae_host.posterior_sample(moments, rng.normal(lz * moments.shape[1] * moments.shape[2] * moments.shape[3]).reshape(lz, *moments.shape[1:]))
        z = vae_host.normalize_latents(cfg, f16_round(z))
        rows.append(patch_rows(z, patch))
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), np.float32)


def augment(rows: np.ndarray, rng: Rng) -> np.ndarray:
    noise = rng.normal(rows.size).reshape(rows.shape)
    return (np.float32(KEYFRAME_NOISE_AUG) * rows + np.float32(1.0 - KEYFRAME_NOISE_AUG) * noise).astype(np.float32)


@torch.no_grad()
def denoise(dit: H3Dit, sampler: dict, embeds, text_tags, g: Geometry, seed: int, vae_cfg: VideoVaeConfig, cond_rows: np.ndarray | None, anchors: list[str]):
    """Returns the target video latent `[C, T, H, W]` (normalised) and the target audio rows `[2A, A_c]`."""
    c = dit.cfg
    lt, lh, lw = g.latent(vae_cfg.spatial_ratio())
    na = g.audio_latents()
    layout = build_layout(text_tags, lt, lh, lw, na, c.patch, anchors)
    rng = Rng(seed)
    cond = augment(cond_rows, rng) if cond_rows is not None and cond_rows.size else np.zeros((0, c.video_patch_dim()), np.float32)
    noise = rng.normal(c.in_channels * lt * lh * lw).reshape(c.in_channels, lt, lh, lw)
    video_rows = np.concatenate([cond, patch_rows(noise, c.patch)], axis=0)
    audio_rows = rng.normal(na * 2 * c.audio_in_channels).reshape(na * 2, c.audio_in_channels)
    renoise = Rng(seed + 1)
    off_v = layout.num_condition_video_rows
    rule = sampler["rule"]
    L = lambda x: torch.tensor(x, dtype=torch.long)  # noqa: E731
    pos = torch.tensor(layout.position_ids, dtype=torch.float32)
    if rule == "pdd":
        video_rows[off_v:] *= np.float32(sampler["steps"][0]["sigma_video"])
        audio_rows *= np.float32(sampler["steps"][0]["sigma_audio"])
    for st in sampler["steps"]:
        unique, inv = row_timesteps(layout, st["t_video"], st["t_audio"], max(st["t_video"], sampler["cond_video_t"]), sampler["cond_audio_t"])
        pdd = None
        if rule == "pdd":
            pdd = (torch.tensor(st["pdd_video_weights"]), torch.tensor(st["pdd_audio_weights"]))
        v, a = dit(torch.from_numpy(video_rows), torch.from_numpy(audio_rows), torch.from_numpy(np.ascontiguousarray(embeds, dtype=np.float32)),
                   torch.tensor(unique), L(inv), L(layout.token_tags), pos, L(layout.video_indices), L(layout.audio_indices), L(layout.text_indices), pdd)
        v, a = v.numpy(), a.numpy()
        nv = needs_noise(rule, st["sigma_video_next"])
        noise_v = renoise.normal(video_rows[off_v:].size).reshape(video_rows[off_v:].shape) if nv else None
        video_rows[off_v:] = flow_step(video_rows[off_v:], v[off_v:], st["sigma_video"], st["sigma_video_next"], rule, noise_v)
        noise_a = renoise.normal(audio_rows.size).reshape(audio_rows.shape) if needs_noise(rule, st["sigma_audio_next"]) else None
        audio_rows[:] = flow_step(audio_rows, a, st["sigma_audio"], st["sigma_audio_next"], rule, noise_a)
    latent = unpatch_rows(video_rows[off_v:], c.in_channels, lt, lh, lw, c.patch)
    return latent, audio_rows


@torch.no_grad()
def generate(variant: Variant, src: WeightSource, vae_cfg: VideoVaeConfig, audio_cfg: AudioVaeConfig, embeds, text_tags, g: Geometry, seed: int,
             keyframes: list[Keyframe] | None = None) -> dict[str, np.ndarray]:
    dit = H3Dit(variant.config).eval()
    load_dit(dit, src)
    dec = Decoder(vae_cfg).eval()
    load_vae_decoder(dec, src)
    adec = AudioDecoder(audio_cfg).eval()
    load_audio_decoder(adec, src)
    anchors = [k.anchor for k in (keyframes or [])]
    cond = None
    if keyframes:
        enc = Encoder(vae_cfg).eval()
        load_vae_encoder(enc, src)
        cond = encode_keyframes(vae_cfg, keyframes, lambda x: enc(torch.from_numpy(x)).numpy(), variant.config.patch)
    latent, audio_rows = denoise(dit, variant.sampler_json(), embeds, text_tags, g, seed, vae_cfg, cond, anchors)
    ratio = vae_cfg.spatial_ratio()
    tile_run = lambda z: dec(torch.from_numpy(z)).numpy()  # noqa: E731
    frames = vae_host.decode(vae_cfg, vae_host.denormalize_latents(vae_cfg, latent),
                             lambda z: vae_host.tiled(z, ratio, 1, vae_host.TILE, vae_host.OVERLAP, tile_run))
    rgb8 = (np.clip(frames, 0, 1) * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 3, 0)  # [F, H, W, 3]
    na = g.audio_latents()
    m = np.array(audio_cfg.latents_mean, np.float32).reshape(-1, 1)
    s = np.array(audio_cfg.latents_std, np.float32).reshape(-1, 1)
    chans = []
    for ch in range(2):
        z = audio_rows[ch * na : (ch + 1) * na].T * s + m  # [lc, A]
        chans.append(adec(torch.from_numpy(np.ascontiguousarray(z))).numpy())
    return {"latent": latent.astype(np.float32), "audio_rows": audio_rows.astype(np.float32), "frames": np.ascontiguousarray(rgb8),
            "audio": np.stack(chans).astype(np.float32)}
