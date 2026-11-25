# The graph contract

An export is a directory of `.tflite` graphs, `adaln.safetensors` and
`manifest.json`. This file is the contract between the exporter
(`python/h3_litert/export.py`) and every runtime. It extends
fastwan-litert's contract (same conventions: f32 tensors, exact shapes, no
batch dimension, inputs bound by name, outputs by position as `output_0, …`).

Symbols: `d = hidden` (5376), `H = heads` (56), `hd = head_dim` (128),
`I = H·hd` (7168), `r = rotary_dim` (96, so RoPE tables have `r/2 = 48`
angles), `C` = the row chunk, `S` = rows of the packed sequence, `Cq` = the
attention query chunk, `G` = heads per attention call, `P` = video patch
features (`in_channels · 1·2·2 = 96`), `A_c = audio_in_channels` (32).
Modality tags: `0` video, `1` text, `2` audio.

## `manifest.json`

```json
{
  "format": "h3-litert/1",
  "model": "h3",                              // or "tiny"
  "variant": "fasth3-v2",
  "config": { "hidden": 5376, "heads": 56, "head_dim": 128, "layers": 50, "refiner_layers": 2,
              "ffn_dim": 14336, "in_channels": 24, "audio_in_channels": 32, "patch": [1, 2, 2],
              "text_dim": 5120, "freq_dim": 256, "time_embed_hidden": 5376, "time_embed_dim": 2688,
              "rope_freq_dim": 16, "rope_theta": 10000.0, "norm_eps": 1e-5, "qk_norm_eps": 1e-5,
              "final_norm_eps": 1e-5, "adaln_rank": null, "pdd_steps": null },
  "vae": { ...VideoVaeConfig... }, "audio_vae": { ...AudioVaeConfig... }, "taeh3": { ...TaeH3Config... },
  "sampler": {
    "rule": "euler",                          // euler | renoise | pdd
    "shifts": { "video": 10.0, "audio": 3.0 },
    "steps": [ { "t_video": 0.000999, "t_audio": 0.000333,
                 "sigma_video": 0.999001, "sigma_video_next": 0.985,
                 "sigma_audio": 0.999667, "sigma_audio_next": 0.954,
                 "pdd_video_weights": null, "pdd_audio_weights": null }, ... ],
    "cond_video_t": 0.999, "cond_audio_t": 1.0
  },
  "adaln": { "file": "adaln.safetensors", "timesteps": [0.000999, 0.015, ..., 0.999, 1.0] },
  "weights": "int8", "chunk": 1024, "head_group": 8,
  "graphs": {
    "text_refine": [ { "rows": 128, "file": "text_refine_128.tflite" }, ... ],
    "embed_video": "embed_video.tflite", "embed_audio": "embed_audio.tflite",
    "head_video": "head_video.tflite", "head_audio": "head_audio.tflite",
    "blocks": [ { "qkv": "block_00_qkv.tflite", "post": "block_00_post.tflite", "bytes": 1234 }, ... ]
  },
  "attention": [ { "tokens": 38396, "q_chunk": 128, "heads": 8, "file": "attn_s38396_q128_h8.tflite" } ],
  "video_vae": { "decoder": { "latent_t": 7, "latent_h": 16, "latent_w": 16, "rows": 1797,
                              "in": "vae_dec_in.tflite", "blocks": ["vae_dec_block_00.tflite", ...],
                              "out": "vae_dec_out.tflite" },
                 "encoder_clip": "vae_enc_clip.tflite", "encoder_image": "vae_enc_image.tflite",
                 "tile": 256, "overlap": 64 },
  "audio": [ { "latents": 207, "decoder": "avae_dec_207.tflite", "encoder": "avae_enc_207.tflite" } ],
  "taeh3_graphs": [ { "frames": 5, "latent_h": 48, "latent_w": 84, "file": "taeh3_f5_48x84.tflite" } ],
  "sizes": [ { "width": 1344, "height": 768, "frames": 124 } ],
  "files": { "block_00_qkv.tflite": { "bytes": 1234, "sha256": "..." }, ... }
}
```

The `sampler.steps` list is the whole schedule, precomputed by the exporter
from the variant preset (h3-core `schedule.rs`): a runtime never shifts a
sigma itself. `adaln.timesteps` are the sorted unique clean times the table
covers; a step whose times are not all in it is an error.

## `adaln.safetensors`

| tensor | shape | dtype |
|---|---|---|
| `timesteps` | `[U]` | f32, sorted ascending, the manifest's `adaln.timesteps` |
| `blocks` | `[U, layers, 3, 6, d]` | f16: for timestep `u`, block `N`, modality tag `m`, the six slots shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp |
| `head` | `[U, 2, d]` | f16: `norm_out.linear(silu(temb))` as shift (row 0), scale (row 1) |

Computed by the exporter as `adaln_proj.linear(adaln_input(t))` with
`adaln_input = silu(time_embedder(sinusoid(t)))`, or `adaln_basis ·` that for
a rank-reduced (Trim) checkpoint, in f32, rounded to f16 once.

## Graphs

### `text_refine_T`

`context_embedder` → `refiner_layers` pre-norm blocks (RMSNorm, attention
with per-head RMSNorm, no RoPE, SwiGLU FFN) → `final_norm`.

| | name | shape |
|---|---|---|
| in | `embeds` | `[T, text_dim]` prompt rows, zero past the real rows |
| in | `mask` | `[T]`: `0` for a real row, `-1e9` for padding; added to every query's scores over the keys |
| out | `text` | `[T, d]` (rows past the real ones are discarded) |

`T` is the smallest bucket ≥ the prompt's rows.

### `embed_video`, `embed_audio`

`proj_in` / `audio_proj_in` (linear with bias).

| | name | shape |
|---|---|---|
| in | `rows` | `[C, P]` / `[C, A_c]` |
| out | `h` | `[C, d]` |

### `block_N_qkv`

`n = rms(h)·norm1·(1 + scale_msa) + shift_msa`; `q, k, v = n·W_{q,k,v}ᵀ` (no
bias); `q, k` RMS-normalised per head (`norm_q`, `norm_k` of width `hd`,
`qk_norm_eps`); rotate-half RoPE over the first `r` channels of every head:
`y[j] = x[j]·cos[j] − x[j + r/2]·sin[j]`, `y[j + r/2] = x[j + r/2]·cos[j] +
x[j]·sin[j]`, `j < r/2`, channels `≥ r` untouched.

| | name | shape |
|---|---|---|
| in | `h` | `[C, d]` |
| in | `mods` | `[6, d]` (the run's `blocks[u, N, tag]`) |
| in | `cos` / `sin` | `[C, r/2]` |
| out | `q`, `k`, `v` | `[C, H, hd]` each |

### `attn_sS_qCq_hG`

Dense softmax attention over `G` heads, scale `1/√hd`, with an additive key
mask: the prompt's rows are padded to the text bucket `T` so that `S` is a
function of the clip size and the bucket alone, and the padded rows are
masked out as keys (`-1e9`) and dropped as queries.

| | name | shape |
|---|---|---|
| in | `q` | `[Cq, G, hd]` |
| in | `k`, `v` | `[S, G, hd]` |
| in | `mask` | `[S]`: `0` for a real row, `-1e9` for a padded text row |
| out | `o` | `[Cq, G·hd]` |

The host gathers head group `g` (heads `gG … gG + G − 1`) from the
`[S, H, hd]` tensors and writes `o` into columns `gG·hd …` of the `[S, I]`
attention output. The exporter writes one attention graph per `(size, text
bucket, number of keyframes ∈ {0, 1, 2})`; Ref2VA sequences, whose length
depends on the references, need `--rows S` at export time.

**Padded text rows.** The layout is built with the prompt's real rows (so the
video and audio clocks start at the real `n_text`); the packed buffers then
hold `[real text | T − n_text pad rows | the rest]`, every non-text row index
shifted by the pad count. Pad rows carry zeros, the text modulation entry,
any position, and are never read back.

### `block_N_post`

`h += (o·W_outᵀ)·gate_msa`; `n = rms(h)·norm2·(1 + scale_mlp) + shift_mlp`;
`ff = (value · silu(gate))·W_ff2ᵀ` with `[value | gate] = n·W_ff1ᵀ`
(Diffusers' `ff.net.0.proj` row order); `h += ff·gate_mlp`.

| | name | shape |
|---|---|---|
| in | `h` | `[C, d]` |
| in | `o` | `[C, I]` |
| in | `mods` | `[6, d]` |
| out | `h_out` | `[C, d]` |

### `head_video`, `head_audio`

`rms(h)·norm_out.norm·(1 + scale) + shift` with the run's head row, then
`proj_out` / `audio_proj_out`.

| | name | shape |
|---|---|---|
| in | `h` | `[C, d]` |
| in | `mods` | `[2, d]` (`head[u]`) |
| in | `weights` | `[pdd_steps]` — PDD variants only: the fused heads' normalised weights for this step; the graph computes all `pdd_steps` heads and sums them weighted |
| out | `v` | `[C, P]` / `[C, A_c]` |

### Video VAE decoder: `vae_dec_in`, `vae_dec_block_N`, `vae_dec_out`

One temporal chunk of `latent_t` (7) frames of one `latent_h × latent_w`
(16×16) tile, `R = latent_t·latent_h·latent_w + register_tokens + 1` rows.

| graph | in | out |
|---|---|---|
| `vae_dec_in` | `z[24, latent_t, latent_h, latent_w]` denormalised latents | `h[R, 2048]` (`post_quant_conv`, `proj_in`, the register tokens and a zero cls row appended), `cos[R, 24]`, `sin[R, 24]` (constants: the decoder's RoPE at this tile shape, `θ = 100`, `[−1, 1)` grids, suffix rows at 0) |
| `vae_dec_block_N` | `h[R, 2048]`, `cos`, `sin` | `h_out[R, 2048]` (RMSNorm, attention with per-head RMSNorm and rotate-half RoPE on 48 of 64, `× scale1` residual, RMSNorm, SwiGLU, `× scale2` residual) |
| `vae_dec_out` | `h[R, 2048]` | `pix[3, 4·latent_t, 16·latent_h, 16·latent_w]` in ImageNet-normalised space (LayerNorm, `proj_out`, unpatchify, suffix rows dropped) |

The host runs `_split_tiles` (256 px tiles, 64 px minimum overlap, slack
distributed in multiples of 16), the temporal plan (`tokens_chunk = 5`,
`token_overlap = 2`, `frame_pre_padding = 3`, `frame_overlap = 5`, the
repeated last latent frame, the trailing-frame cut), the tile stitch and the
chunk blend, then `pix·std + mean` and clamp to `[0, 1]`, exactly as h3-rs's
`vae.rs` (ported with its tests).

### Video VAE encoder: `vae_enc_clip`, `vae_enc_image`

The causal 3-D CNN (`reflect` spatial padding, causal temporal padding,
frame-isolated GroupNorm) plus `quant_conv`.

| graph | in | out |
|---|---|---|
| `vae_enc_clip` | `x[3, 17, tile, tile]` ImageNet-normalised pixels | `moments[48, 5, tile/16, tile/16]` |
| `vae_enc_image` | `x[3, 1, tile, tile]` | `moments[48, 1, tile/16, tile/16]` |

The host tiles, stitches the moments, samples the posterior with the
reference's fixed seed (42), rounds through f16, normalises and patchifies.

### Audio VAE: `avae_dec_A`, `avae_enc_A`

| graph | in | out |
|---|---|---|
| `avae_dec_A` | `z[32, A]` denormalised latents of one channel | `wave[800·A]` in `[−1, 1]` |
| `avae_enc_A` | `wave[800·A]` | `mean[32, A]` (the posterior mode) |

`A` is the audio latent count of a clip size in `sizes`; stereo is two calls.

### `taeh3_fF_HxW`

fastwan-litert's TAEHV graph (`tae_fF_HxW` there) with `latent_channels` 24
and the ×2 pixel-shuffle head: `z[F, 24, H, W]` and nine memories →
`rgb[4F, 3, 16H, 16W]` in `[0, 1]` and nine memories. The host keeps raw
frame `i` (counted from the clip start) when `i mod 20 ≥ 3`.

## The host loop

```
layout = build_layout(text_tags, lt, lh, lw, na, patch, anchors)        # or build_ref2va_layout
cos, sin = h3_rope(layout.position_ids)                                   # [S, 48]
text = text_refine(pad(embeds, T), mask)[:rows]                           # once per prompt
video_rows = cond rows (encoder graphs, seed 42, f16 round, normalise, patchify, 0.999 aug) + patchify(noise)
audio_rows = cond rows + noise
for each step i:
    unique, t_idx = row_timesteps(layout, t_video, t_audio, max(t_video, 0.999), 1.0)
    u[k] = index of unique[k] in adaln.timesteps
    h = zeros[S, d]; h[text rows] = text
    for run in runs(video_indices): h[run] = embed_video(video_rows[run])     # chunks of C, padded
    for run in runs(audio_indices): h[run] = embed_audio(audio_rows[run])
    for N in blocks:
        for run in mod_runs(t_idx, tags):                     # maximal rows sharing (t_idx, tag)
            q,k,v[run] = block_N_qkv(h[run], blocks[u, N, tag], cos[run], sin[run])
        for g in head groups, cq in query chunks: o[cq, g] = attn(q[cq, g], k[:, g], v[:, g])
        for run in mod_runs: h[run] = block_N_post(h[run], o[run], blocks[u, N, tag])
    for run in runs(video_indices): v_video[run] = head_video(h[run], head[u], [weights])
    for run in runs(audio_indices): v_audio[run] = head_audio(h[run], head[u], [weights])
    target video rows: x ← step(x, v, σ_v, σ_v')  (euler: x·σ'/σ + x0·(1 − σ'/σ), x0 = x + σ·v;
                                                  renoise: (1 − σ')·x0 + σ'·ε, x0 when σ' = 0)
    target audio rows: the same with σ_a
decode: video latent = unpatchify(target video rows); VAE decoder tiles / chunks (or TAEH3);
        audio = avae_dec per channel of the target audio rows (denormalised)
```

Rules, in addition to fastwan-litert's (padding, no implicit fallback,
events):

- **Runs never straddle a modulation entry.** A per-row graph call covers
  rows of one `(timestep index, tag)`; the last chunk of a run is
  zero-padded to `C` and its padded outputs dropped.
- **Noise order.** Keyframe / reference augmentation noise from the request
  generator, then the video latent noise `[C, T, H, W]`, then the audio noise
  `[2·A, A_c]`; renoise draws from a second generator seeded `seed + 1`,
  video before audio at each step (h3-rs's order).
- **Conditioning rows are never written** by the step; only target rows.
- **PDD.** Step `i` uses fused-head weights `sampler.steps[i].pdd_*_weights`
  and the initial target noise is scaled by `sigma_video[0]` / `sigma_audio[0]`.

## Weight formats

As fastwan-litert: `f32`, `f16`, `int8`, `int4` on the block graphs
(`qkv`, `post`) and the ViT decoder blocks; everything else f16 (f32 in an
all-f32 export). LoRA adapters are merged before quantisation.

## The tiny fixture

`h3-litert export --model tiny --out fixtures/tiny`: `H3Config.tiny()`
(hidden 32, 2 heads of 16, 2 blocks, 1 refiner block, rotary 12 of 16),
`VideoVaeConfig.tiny()`, `AudioVaeConfig.tiny()`, `TaeH3Config.tiny()`,
synthetic weights; `C = 16`, `Cq = 32`, `G = 1`; a two-step Euler ladder
(rungs 999 / 500, shifts 10 / 3); geometry `96×64×22` (latent `(7, 4, 6)`,
37 audio latents, a 6-row prompt tagged `[1, 1, 0, 1, 1, 1]`: 122 packed
rows, 124 with the two pad rows of the 8-row text bucket); the decoder at `latent_t 7, 4×6` (`R = 171`), audio `A = 37`,
TAEH3 `F = 5`. `expected.safetensors` holds the prompt, the denoised video
latent `[4, 7, 4, 6]`, the target audio rows `[74, 6]`, the decoded frames
`[22, 64, 96, 3]` uint8 and the stereo waveform `[2, 370]`, for seed 7, from
the uncut PyTorch modules.
