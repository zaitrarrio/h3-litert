"""The PyTorch modules against Diffusers' MiniMax H3 classes, tiny random weights."""

import numpy as np
import pytest
import torch

from h3_litert.audio import AudioDecoder, AudioEncoder
from h3_litert.config import AudioVaeConfig, H3Config, TEXT_TAG, VIDEO_TAG, VideoVaeConfig
from h3_litert.dit import H3Dit
from h3_litert.packing import build_layout, row_timesteps
from h3_litert.vae import Decoder, Encoder
from h3_litert.weights import DictSource, load_audio_decoder, load_audio_encoder, load_dit, load_vae_decoder, load_vae_encoder

diffusers = pytest.importorskip("diffusers")


def jitter(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ("norm" in name and name.endswith("weight")) or name.endswith(("scale1", "scale2")):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif name.endswith("bias") or "alpha" in name or "beta" in name:
                p.copy_(0.1 * torch.randn_like(p))
            elif "adaln" in name or "norm_out.linear" in name:
                p.copy_(0.02 * torch.randn_like(p))
            else:
                fan_in = p.shape[1:].numel() if p.ndim > 1 else 1
                p.copy_(torch.randn_like(p) / fan_in**0.5)


def rel(got, want):
    got, want = torch.as_tensor(got), torch.as_tensor(want)
    return ((got - want).abs().max() / want.abs().max()).item()


def test_dit_matches_diffusers():
    from diffusers import MiniMaxH3Transformer3DModel

    torch.manual_seed(0)
    c = H3Config.tiny()
    ref = MiniMaxH3Transformer3DModel(
        num_attention_heads=c.heads, attention_head_dim=c.head_dim, hidden_size=c.hidden, num_layers=c.layers,
        num_refiner_layers=c.refiner_layers, ffn_dim=c.ffn_dim, in_channels=c.in_channels, audio_in_channels=c.audio_in_channels,
        patch_size=c.patch, text_dim=c.text_dim, freq_dim=c.freq_dim, time_embed_hidden_dim=c.time_embed_hidden,
        time_embed_dim=c.time_embed_dim, rope_freq_dim=c.rope_freq_dim, rope_theta=c.rope_theta, norm_eps=c.norm_eps,
        qk_norm_eps=c.qk_norm_eps, final_norm_eps=c.final_norm_eps,
    ).eval()
    jitter(ref)
    tags = [TEXT_TAG, TEXT_TAG, VIDEO_TAG, TEXT_TAG, TEXT_TAG, TEXT_TAG]
    layout = build_layout(tags, 2, 4, 6, 5, (1, 2, 2), ["first"])
    unique, inv = row_timesteps(layout, 0.25, 0.7, 0.999, 1.0)
    video = torch.randn(len(layout.video_indices), c.video_patch_dim())
    audio = torch.randn(len(layout.audio_indices), c.audio_in_channels)
    text = torch.randn(6, c.text_dim)
    pos = torch.tensor(layout.position_ids, dtype=torch.float32)
    L = lambda x: torch.tensor(x, dtype=torch.long)  # noqa: E731
    with torch.no_grad():
        wv, wa = ref(hidden_states=video[None], audio_hidden_states=audio[None], encoder_hidden_states=text[None],
                     timestep=torch.tensor(unique), timestep_indices=L(inv), token_tags=L(layout.token_tags),
                     position_ids=pos, video_indices=L(layout.video_indices), audio_indices=L(layout.audio_indices),
                     text_indices=L(layout.text_indices), return_dict=False)
    dit = H3Dit(c).eval()
    load_dit(dit, DictSource(ref.state_dict()))
    with torch.no_grad():
        gv, ga = dit(video, audio, text, torch.tensor(unique), L(inv), L(layout.token_tags), pos, L(layout.video_indices),
                     L(layout.audio_indices), L(layout.text_indices))
    ev, ea = rel(gv, wv[0]), rel(ga, wa[0])
    print(f"H3 DiT vs diffusers: video rel {ev:e}, audio rel {ea:e}")
    assert ev < 1e-4 and ea < 1e-4


def test_video_vae_matches_diffusers():
    from diffusers import AutoencoderKLMiniMaxH3

    torch.manual_seed(0)
    c = VideoVaeConfig.tiny()
    ref = AutoencoderKLMiniMaxH3(
        in_channels=3, out_channels=3, latent_channels=c.latent_channels, block_out_channels=c.block_out_channels,
        layers_per_block=c.layers_per_block, spatial_downsample_factors=c.spatial_down, temporal_downsample_factors=c.temporal_down,
        norm_num_groups=c.norm_groups, norm_eps=c.norm_eps, decoder_num_layers=c.decoder_layers,
        decoder_num_attention_heads=c.decoder_heads, decoder_attention_head_dim=c.decoder_head_dim,
        decoder_num_register_tokens=c.decoder_register_tokens, decoder_ffn_mult=c.decoder_ffn_mult,
        decoder_rope_theta=c.decoder_rope_theta, decoder_rope_dim_ratio=c.decoder_rope_dim_ratio, decoder_norm_eps=c.decoder_norm_eps,
        clip_length=c.clip_length, token_drop=c.token_drop, latents_mean=c.latents_mean, latents_std=c.latents_std,
    ).eval()
    jitter(ref)
    src = DictSource(ref.state_dict())
    enc, dec = Encoder(c).eval(), Decoder(c).eval()
    load_vae_encoder(enc, src)
    load_vae_decoder(dec, src)
    x = torch.randn(3, 17, 64, 96)
    with torch.no_grad():
        want_m = ref.quant_conv(ref.encoder(x[None]))[0]
        got_m = enc(x)
    e1 = rel(got_m, want_m)
    z = torch.randn(c.latent_channels, 7, 4, 6)
    with torch.no_grad():
        want_px = ref.decoder(ref.post_quant_conv(z[None]))[0]
        got_px = dec(z)
    e2 = rel(got_px, want_px)
    print(f"H3 VAE vs diffusers: encode rel {e1:e}, decode rel {e2:e}")
    assert e1 < 1e-4 and e2 < 1e-4


def test_audio_vae_matches_diffusers():
    from diffusers import AutoencoderKLMiniMaxH3Audio

    torch.manual_seed(0)
    c = AudioVaeConfig.tiny()
    ref = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=c.encoder_dim, encoder_rates=c.encoder_rates, latent_dim=c.latent_dim, latent_channels=c.latent_channels,
        num_attention_heads=c.heads, decoder_dim=c.decoder_dim, decoder_rates=c.decoder_rates, decoder_kernel_sizes=c.decoder_kernels,
        resblock_kernel_sizes=c.resblock_kernels, resblock_dilation_sizes=c.resblock_dilations, sampling_rate=c.sample_rate,
    ).eval()
    jitter(ref)
    src = DictSource(ref.state_dict())
    enc, dec = AudioEncoder(c).eval(), AudioDecoder(c).eval()
    load_audio_encoder(enc, src)
    load_audio_decoder(dec, src)
    wave = torch.randn(370) * 0.3
    with torch.no_grad():
        want_mean = ref.encode(wave[None, None]).latent_dist.mode()[0]
        got_mean = enc(wave)
    e1 = rel(got_mean, want_mean)
    z = torch.randn(c.latent_channels, 37)
    with torch.no_grad():
        want_w = ref.decode(z[None]).sample[0, 0]
        got_w = dec(z)
    e2 = rel(got_w, want_w)
    print(f"H3 audio VAE vs diffusers: encode rel {e1:e}, decode rel {e2:e}")
    assert e1 < 1e-4 and e2 < 1e-4
