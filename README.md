# h3-litert

MiniMax H3 (video + 32 kHz stereo audio, 33 B parameters) and the FastVideo
cookbook's few-step students on [LiteRT](https://ai.google.dev/edge/litert).
A port of [h3-rs](https://github.com/zaitrarrio/h3-rs) made the way
[fastwan-litert](https://github.com/zaitrarrio/fastwan-litert) ports
fastwan-rs: the host-side work (packing, schedules, tiling, the AdaLN tables)
stays and is ported with its tests; the fused Rust kernels become `.tflite`
graphs that one exporter writes and thin hosts drive.

- **Model:** `MiniMaxAI/MiniMax-H3` and FastH3 V1 / V2 / Trim / Omni, Pika H3
  Flex and LongLive-Plug (LoRA merged at export), T2VA and FL2VA (Ref2VA
  layouts are built, the pipeline for references is not wired yet). Prompts
  from the Qwen3-VL conditioner are precomputed elsewhere (as in h3-rs).
- **The cut:** per transformer block a `qkv` graph and a `post` graph over
  one modulation run; a weight-free attention graph over query chunks and
  head groups with a key mask; the AdaLN projections (13 B parameters)
  replaced by a table the exporter precomputes for the variant's timestep
  ladder; the ViT VAE decoder as 36 block graphs at one tile shape; the
  causal 3-D encoder, the BigVGAN audio VAE and TAEH3 as graphs; tiling,
  blending and the samplers on the host. `docs/design.md` says why,
  `docs/graphs.md` is the contract.
- **Where it runs:** the desktop (Python on `ai-edge-litert`; the 33 B
  model streams its blocks from disk). The web / Android / iOS hosts run the
  same contract and the tiny model; a phone cannot hold H3.

| | status |
|---|---|
| PyTorch modules vs Diffusers | DiT rel 1.4e-7, VAE encode exact / decode 7e-7, audio VAE 1.2e-6 (`tests/test_parity.py`) |
| graphs vs the PyTorch pipeline | T2VA latent and audio rows within 1e-4, frames within 2/255, waveform within 1e-3; FL2VA through the encoder graphs within 1e-3; TAEH3 preview = module (`tests/test_graphs.py`) |
| host math vs h3-rs | schedules bit for bit (incl. the LongLive-Plug grid), packing, canvas rules, decode plans, TAEH3 frame rule (`tests/test_host.py`) |
| `web/` (LiteRT.js) | 24 vitest cases; the tiny fixture in headless Chromium on the Wasm backend: video latent rel 1.7e-5, audio rows rel 2e-5, frames within 1/255, waveform rel 2e-4 |
| `android/` (Kotlin `:core` + LiteRT `:app`) | 44 JUnit tests pass (`gradle :core:test -Ph3.app=false`); `:app` needs the Android SDK and a device |
| `ios/` (Swift package + TensorFlowLiteSwift / Metal) | written and self-reviewed, uncompiled (no Swift toolchain here) |
| real checkpoints, GPU delegates, dense vs VSA quality | not run: no GPU and no 33 B download here |

## Use

```bash
cd python && pip install -e ".[export,test]"
pytest                                               # 21 tests on CPU, tiny models
h3-litert export --variant tiny --out fixtures/tiny  # the self-test fixture (f32)
h3-litert check --graphs fixtures/tiny               # graph pipeline vs the PyTorch reference
h3-litert variants
```

The real model (an offline job of hours; ~40 GB of f16 graphs for V2, 10 GB
at int4):

```bash
huggingface-cli download MiniMaxAI/MiniMax-H3 --include "vae/*" "audio_vae/*" --local-dir h3
huggingface-cli download FastVideo/FastVideo-FastH3-8-Step-V2 --include "transformer/*" --local-dir fasth3-v2
h3-litert export --variant fasth3-v2 --model-dir fasth3-v2 --base-dir h3 --weights int8 \
    --out graphs/v2-int8 --size 1344x768x124 --size 832x480x124
h3-litert generate --graphs graphs/v2-int8 --prompt gallery/fox.safetensors --size 832x480x124 --out fox.y4m
ffmpeg -i fox.y4m -i fox.wav -c:v libx264 -c:a aac fox.mp4
```

`--first-frame a.png [--last-frame b.png]` is FL2VA; `--preview` decodes with
TAEH3 (`--tae taeh3.safetensors` at export). Every clip runs dense attention:
there is no VSA on LiteRT (`docs/design.md`).

## Layout

```
docs/       design.md, graphs.md (the contract)
python/     h3_litert: config, schedule, packing, rope, rng, timestep (host math); dit, vae, audio, tae
            (PyTorch graph modules); weights; export (+ AdaLN tables); runtime; pipeline; vae_host; cli
fixtures/   tiny: 26 graphs, adaln.safetensors, expected.safetensors (2.2 MB)
web/        @h3/litert on @litertjs/core + the studio page
android/    Kotlin :core (host math, pipeline, tests) and :app (LiteRT GPU, Compose)
ios/        H3LiteRTCore / H3LiteRTRuntime Swift package and a SwiftUI demo
```
