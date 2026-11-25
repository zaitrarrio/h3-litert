# h3-litert design

MiniMax H3 (33 B parameters, video + 32 kHz stereo audio, one packed
sequence through 50 Omni-Transformer blocks) and the FastVideo cookbook's
few-step students on [LiteRT](https://ai.google.dev/edge/litert): the
desktop first (the Python runtime on `ai-edge-litert`, which is also the
reference), then the web (LiteRT.js), Android (LiteRT `CompiledModel`) and
iOS (LiteRT, Metal) for the sizes those can hold.

It is a port of [h3-rs](https://github.com/zaitrarrio/h3-rs), made the way
[fastwan-litert](https://github.com/zaitrarrio/fastwan-litert) ports
fastwan-rs: the host-side work is kept and ported with its tests, and the
fused Rust kernels are replaced by `.tflite` graphs written once as PyTorch
modules, converted with `litert-torch`, and driven by thin hosts through one
manifest. Read fastwan-litert's `docs/design.md` first; this file says what
H3 adds.

## What stays from h3-rs

- **The packed sequence.** `[text | condition video rows | audio rows |
  video rows]` for T2VA / FL2VA, `[text | ordered references | continuation
  | target audio | target video]` for Ref2VA, each row tagged video / text /
  audio and carrying a real-valued `(t, h, w)` rotary position, built in f64
  exactly as FastVideo's `packing.py` (h3-core `packing.rs`, ported with its
  tests: canvas rules, `17n + 5` frames, `5n + 2` latent frames, audio latents
  at 40 Hz with Python's half-even rounding, the `5/3 · [1, 4, 4, 4, 4]` frame
  clock).
- **The schedules.** Clean-time clock `t = 1 − σ`, the rational shift
  `σ = s·b / (1 + (s − 1)·b)` per modality (12 / 3 for the release, 10 / 3 for
  V2 and Trim), Euler as the `x_t`/`x0` blend, Pika's renoise rule, FastH3
  Omni's Parallel Decoding Distillation grid and fused-head weights, all in f32
  as the reference and pinned bit for bit by h3-rs's tests (the LongLive-Plug
  grid equals `MiniMaxH3Scheduler.set_timesteps(5)` to the last bit).
- **Row timesteps.** Target video rows at `t_v`, condition video rows at
  `max(t_v, 0.999)`, target audio at `t_a`, condition audio at `1.0`; the
  sorted unique times index the modulation tables.
- **Precomputed AdaLN.** Every block's `adaln_proj` is a `[96768, 2688]`
  linear: 13 B of the 33 B parameters exist only to turn the timestep
  embedding into modulation rows. h3-rs streams them through the GPU once per
  generation or loads a saved table; here the **exporter** computes the table
  for the variant's whole timestep ladder and ships it as `adaln.safetensors`
  (`[U, layers, 3, 6, hidden]` f16: 170 MB for an 8-step student, 1 GB for the
  49-step base). The 13 B parameters never reach a device.
- **The decoders' host side.** Spatial tiles of 256 px with 64 px overlap and
  the linear cross-fade stitch, temporal chunks of 7 latent frames with the
  `token_drop` overlap and frame blend, ImageNet de-normalisation, the TAEH3
  `i mod 20 ≥ 3` frame rule, stereo audio as two mono decodes: all host code
  ported from `vae.rs` / `audio_vae.rs` / `taeh3.rs` with their tests.
- **Deterministic noise** (SplitMix64 → Box–Muller), the keyframe encode seed
  42, the renoise generator at `seed + 1`.
- **Prompts from outside.** The Qwen3-VL-32B conditioner (64 GB bf16) is not
  on-device work. Prompts come from `h3 encode` galleries or
  `h3-litert encode` (transformers, hidden state 50, on a big machine), as
  `[rows, 5120]` embeddings with per-row text / vision tags.

## The graph cut

The H3 block has no cross-attention and one modality-agnostic weight set;
what varies per row is only the `(timestep, modality)` modulation entry. The
packed sequence is a handful of **runs** of rows sharing one entry (text; the
condition frames; the two audio channels; the target video), so every
per-row graph takes one `[6, hidden]` modulation table and the host walks the
runs, exactly as h3-rs writes each modality's projection into its rows.

| graph | per | inputs → outputs |
|---|---|---|
| `text_refine_T` | prompt | `embeds[T,5120]`, `mask[T]` → `text[T,hidden]` (context embedder, 2 refiner blocks, final norm; `T` a bucket, padded rows masked out of the refiner's attention) |
| `embed_video` / `embed_audio` | row chunk | `rows[C,96]` / `rows[C,32]` → `h[C,hidden]` |
| `block_N_qkv` | row chunk | `h[C,hidden]`, `mods[6,hidden]`, `cos/sin[C,48]` → `q,k,v[C,56,128]` (modulated RMSNorm, projections, per-head RMSNorm, rotate-half RoPE on 96 of 128) |
| `attn_sS_qCq_hG` | query chunk × head group | `q[Cq,G,128]`, `k,v[S,G,128]`, `mask[S]` → `o[Cq,G·128]` (text rows padded to the bucket and masked) |
| `block_N_post` | row chunk | `h`, `o[C,7168]`, `mods` → `h'` (out projection, gated residual, modulated RMSNorm, SwiGLU FFN, gated residual) |
| `head_video` / `head_audio` | row chunk | `h[C,hidden]`, `mods[2,hidden]` → `[C,96]` / `[C,32]`; the Omni variant's PDD heads take `weights[32]` and fuse inside |
| `vae_dec_in`, `vae_dec_block_N`, `vae_dec_out` | 7-frame latent chunk × 256 px tile | the ViT decoder: `[24,7,16,16]` → 1797 tokens → 36 blocks → `[3,28,256,256]` |
| `vae_enc_clip` / `vae_enc_image` | 17-frame clip × tile | the causal 3-D CNN encoder for keyframes and references, `[3,17,256,256]` → `[48,5,16,16]` |
| `avae_dec_A` / `avae_enc_A` | audio channel | BigVGAN `[32,A]` → `[1,800·A]`; DAC + causal projection back |
| `taeh3_fF_HxW` | latent chunk | TAEH3 as fastwan-litert's TAEHV graph with the ×2 pixel-shuffle head |

Why this cut:

- **Attention dominates** (56 heads over up to ~38 k rows at 1344×768×124) and
  is the only coupling between rows. Head groups of `G` heads bound the score
  chunk (`G · Cq · S · 4 B`: 8 heads × 128 queries × 38 k = 156 MB) without a
  flash kernel; a block is then `G/56 · S/Cq` attention calls plus the per-row
  chunks. There is no VSA on LiteRT, so the VSA-distilled students run dense:
  the reference says FastWan's short clips are *better* dense, FastH3's are
  untested dense, and the gap is a measurement this repo has yet to make.
- **One block per file** (3 graphs, 0.8 GB at bf16-equivalent f16, 0.2 GB at
  int4) streams: the desktop keeps a few blocks resident and reads the rest
  from disk, as h3-rs streams blocks over PCIe on a 24 GB card.
- **The AdaLN table replaces 40 % of the checkpoint** with a lookup; the
  `[U, 2, hidden]` head table comes with it.
- **The ViT decoder is a plain transformer**: 36 block graphs at one fixed
  token count, because the released tiling makes every tile 256 px and every
  temporal chunk 7 latent frames, so one shape covers every canvas.
- **The text refiner is bucketed** (`T` ∈ {128, 256, 512, 1024}) with an
  additive mask, so a prompt of any length up to 1024 rows is exact.

## Tiers

| tier | what runs | sizes |
|---|---|---|
| desktop | every variant below, dense attention, blocks streamed from disk; the VAE, audio VAE and TAEH3 | 1344×768×124 (≈38 k rows; `q/k/v` are 1.1 GB each in f32) down to 832×480 |
| web / phone | the tiny self-test model, and FastH3-Trim at int4 (11 GB) on a workstation browser; nothing of this size fits a phone | a phone tier does not exist for a 33 B model: the runtimes exist to run the same contract, and `plan` says no |

`plan` refuses what will not fit: resident blocks × block bytes + the AdaLN
rows of this run + `q/k/v/o/h` at `S × 7168 / 5376` f32 + the attention
score chunk + the decoder's peak.

## Variants

| `--variant` | checkpoint | sampler (manifest `steps`) | attention here |
|---|---|---|---|
| `h3` | `MiniMaxAI/MiniMax-H3` `transformer` / `transformer_ref` | 50 uniform points, shifts 12 / 3, Euler | dense |
| `fasth3-v1` | FastH3-4-step-Preview-v1 | rungs 999/749/500/250 | dense (VSA 0.9 in the reference) |
| `fasth3-v2` | FastH3-8-Step-V2 | 8 rungs, shifts 10 / 3 | dense (VSA 0.8) |
| `fasth3-trim` | FastH3-Trim-8-Step (bf16 tree) | as V2; 42 blocks, rank-16 AdaLN, 26-block decoder | dense |
| `fasth3-omni` | FastH3-Omni-8-Step-V1 | PDD: 32 fine intervals, nodes 0,4,…,32, fused heads per step | dense |
| `pika-flex` | Pika-H3-Flex adapters (merged at export, one export per task) | renoise knots 1000/750/500/250/100/40 | dense |
| `h3-plug4` | LongLive-Plug few-step LoRA (merged at export) | knots 1000/750/500/250, renoise | dense |

Not in this port: Open-H3-Regenerate-2K (latent upscaler, Sol fast VAE),
the NVFP4 single-file Trim checkpoint (export from the bf16 tree instead),
VSA, INT8 / FP8 attention. LoRA adapters are merged into the weights at
export (exact in f32; h3-rs keeps them as branches on bf16 backends because a
merge rounds the delta away there; an f32 export does not have that problem,
and int8 / int4 quantisation happens after the merge).

## Verified, and how

The Python package carries the tests; CI runs them on CPU with the tiny
models (`H3Config.tiny()`, `VideoVaeConfig.tiny()`, `AudioVaeConfig.tiny()`,
`TaeH3Config.tiny()`: the numbers of h3-core's).

| claim | evidence |
|---|---|
| host math equals h3-rs | `tests/test_host.py`: the V2 ladder, the uniform grid, the Pika knots, the LongLive-Plug grid's f32 bits, the PDD plan, canvas sizes, `17n+5` → `5n+2`, audio latents, the rotary positions of a small layout, row timesteps, the decode plan, the TAEH3 frame rule (17 of 20, 124 of 148), SplitMix64 |
| the PyTorch DiT equals Diffusers' `MiniMaxH3Transformer3DModel` | `tests/test_parity.py`: one forward on a tiny packed layout, tiny random weights |
| the PyTorch VAEs equal Diffusers' | `AutoencoderKLMiniMaxH3` encode / decode, `AutoencoderKLMiniMaxH3Audio` encode / decode, tiny weights |
| the graph cut equals the whole module | `tests/test_graphs.py`: the chunked graph pipeline through `ai-edge-litert` against the uncut forward, with runs, head groups, query chunks and padding |
| the pipeline runs end to end | a tiny T2VA request: layout, 8 steps, VAE decode, audio decode, deterministic per seed |

Not verified: anything on a GPU delegate, the real checkpoints (33 B; the
export is an offline job of hours), and clip quality of the VSA students run
dense. `docs/runs/` is where those records go.

## Layout

```
python/     h3_litert: config, schedule, packing, rope, rng, timestep (host math, from h3-core);
            modules (PyTorch, the graph cut: DiT, refiner, heads, ViT VAE, audio VAE, TAEH3);
            weights (safetensors / synthetic, LoRA merge); adaln (table precompute);
            export; runtime; pipeline (T2VA / FL2VA / Ref2VA, Euler / renoise / PDD); cli
fixtures/   tiny: the self-test models and expected outputs
docs/       this file; graphs.md is the manifest and tensor contract
web/ android/ ios/   the thin hosts, as in fastwan-litert
```
