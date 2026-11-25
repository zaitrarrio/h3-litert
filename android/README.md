# h3-litert on Android

The Android runtime of h3-litert: MiniMax H3 (video + stereo audio, one packed sequence through
the Omni-Transformer blocks) and the FastH3 few-step students over LiteRT `CompiledModel` graphs,
following `docs/graphs.md` exactly. A Kotlin port of the Python reference in `python/h3_litert/`,
built the way fastwan-litert's Android runtime is built (same Gradle layout, `Tensor`, `Rng`,
`SafeTensors`, `Graph`/`GraphStore` and the LiteRT backend are shared code).

## Modules

```
android/
  core/   pure Kotlin/JVM library, package dev.h3.litert.core   (builds and tests on any JDK >= 17)
  app/    Android application: LiteRT GPU backend + Compose UI   (needs the Android SDK)
```

**`:core`** has no Android imports. It holds everything that is not a graph call:

| file | what | Python source |
|---|---|---|
| `H3Config.kt` | `H3Config` (h3 / Trim / Omni / tiny, `rotaryDim`, `videoPatchDim`), `VideoVaeConfig` (ratios, `framePrePadding`, `tokensChunk`, `tokenOverlap`, `frameOverlap`, `decoderRows`), `AudioVaeConfig` (`hop`), `TaeH3Config` (`keepsRawFrame`), the latent / pixel statistics | `config.py` |
| `Packing.kt` | `Geometry`, `canvasSize`, `alignNumFrames`, `videoLatentFrames`, `audioLatents`, `buildLayout`, `buildRef2vaLayout`, `rowTimesteps`, `runs`, `modRuns`, `patchRows`, `unpatchRows`; positions in f64 | `packing.py` |
| `Rope.kt` | `h3Rope` → `cos`/`sin` `[S, 3·rope_freq_dim]` in f32 | `rope.py` |
| `Schedule.kt` | `shiftSigma`, torch-style f32 `linspace`, `Schedule.fromRungs` / `uniform`, `PddPlan` (for the tests; a manifest ships its schedule precomputed) | `schedule.py` |
| `Sampler.kt` | `flowStep` (Euler / renoise / PDD), `needsNoise` | `sampler.py` |
| `Rng.kt` | SplitMix64 → Box–Muller, the same stream as h3-rs / fastwan-rs | `rng.py` |
| `VaeHost.kt` | `splitTiles`, `blend`, `stitchTiles`, `tiled`, `temporalPlan`, `decode`, `encode`, pixel / latent (de)normalisation, `posteriorSample` | `vae_host.py` |
| `Manifest.kt` | `manifest.json` (kotlinx.serialization), the `format` check, the AdaLN coverage check, `attentionFor(rows)`, `textRefineFor(rows)`, `audioFor(latents)`, `taeh3For(h, w)`, `plan(...)` | `runtime.py` |
| `AdalnTables.kt` | `adaln.safetensors`: `timesteps` f32, `blocks` / `head` f16 decoded per row on demand, `indexOf(t)` | `pipeline.py` `AdalnTables` |
| `Plan.kt` | the memory plan (`docs/design.md` "Tiers") | — |
| `Pipeline.kt` | `Request` / `Keyframe`, `prepare`, `KeyframeEncoder`, `forward`, `denoise`, `decodeVideo` (VAE or TAEH3), `decodeAudio`, `generate`; `Event.Prepared / Step / Frames / Audio` | `pipeline.py` |
| `Host.kt` | `padRows`, RGBA quantisation, the f16 round trip | — |
| `Graph.kt`, `Tensor.kt`, `SafeTensors.kt` | the graph interface and LRU store, the float32 tensor, a safetensors reader (F32 / F16 / BF16 / raw bytes) | fastwan-litert |

**`:app`** (`dev.h3.litert.app`):

- `LiteRtGraphStore` — `GraphStore` over `com.google.ai.edge.litert.CompiledModel` (the sibling's,
  unchanged): graphs read from the manifest directory, `Accelerator.GPU` with fp16 precision and a
  program cache, optional CPU fallback per graph. Every input is checked against the graph's own
  tensor layout, so an export/manifest mismatch fails with both shapes in the message.
- `GenerateViewModel` — runs `Pipeline.generate` on `Dispatchers.Default`, turns `Event.Frames`
  into `Bitmap`s, keeps the waveform of `Event.Audio`, the timings, the `Plan` and a log in a
  `StateFlow<UiState>`. `AudioPlayer` plays the stereo float waveform through a static
  `AudioTrack` when the clip is done (and on **Play**); `Keyframes` turns a picked picture into the
  `[3, 1, H, W]` keyframe tensor.
- `MainActivity` / `ui/GenerateScreen.kt` — Compose: the graphs directory, a prompt from
  `gallery/` (rows and vision-row count shown), a size from the manifest's `sizes`, a seed,
  optional first / last keyframe pictures (FL2VA; the photo picker, no permission), the GPU and
  TAEH3-preview switches, the memory plan, prepare / step / decode / audio timings, the frames as
  each decoder chunk arrives (24 fps loop), audio playback.

## Running the `:core` tests (no Android SDK needed)

```bash
cd android
./gradlew :core:test -Ph3.app=false      # what CI runs; or: gradle :core:test (Gradle 8.14.3, any JDK >= 17)
```

`settings.gradle.kts` includes `:app` when `-Ph3.app=true`, or when the property is unset and an
Android SDK is found (`ANDROID_HOME`, `ANDROID_SDK_ROOT`, or `sdk.dir` in `local.properties`).
`-Ph3.app=false` builds `:core` only, whatever the machine has, so the Android Gradle Plugin is
never resolved on CI.

The tests mirror `python/tests/test_host.py` and add the pipeline orchestration (44 tests):

- `HostMathTest` — the V2 ladder, the uniform grid and Pika knots, the LongLive-Plug grid's f32 bit
  patterns (`0x3F800000, 0x3F7914C2, 0x3F6C4EC5, 0x3F4CCCCD, 0` at shift 12 and
  `…, 0x3F666666, 0x3F400000, 0x3F000000, 0` at 3), the PDD plan, `17n + 5` → `5n + 2`, audio
  latents with half-even rounding, the canvas rules, the 768p layout's positions, the FL2VA anchors,
  row timesteps and modulation runs `[(2,0,1), (1,0,0), (12,2,0), (40,1,2), (42,0,0)]`, the Ref2VA
  segment order, runs / patch rows, RoPE, the VAE derived numbers `(16, 4, 3, 5, 2, 5)`, the TAEH3
  rule (17 / 124 / 5 kept), SplitMix64 (`0xE220A8397B1DCDAF`), `flowStep`, the f16 round trip.
- `VaeHostTest` — `splitTiles(768, 256, 64, 16) = ([0, 160, 336, 512], [256]×4, [96, 80, 80])`,
  `temporalPlan(h3, 37) = (0, 7, 124)`, `(…, 7) = (0, 1, 22)`, `(…, 2) = (5, 1, 5)`, the blend, the
  stitch, the tile walk, the chunked decode with its cross-fade, the posterior, `encode`.
- `ManifestTest` — the tiny manifest field by field (its schedule equals `Schedule.fromRungs([999, 500], 10 / 3)`),
  the refusals (format, an uncovered step time, a head group that does not divide the heads), the
  four lookups with the mismatch in the message, the AdaLN rows against the raw table,
  `expected.safetensors`, the plan, LRU residency.
- `PipelineTest` — a `FakeGraphStore` whose graphs check every input name and shape against the
  contract and record the calls, on the tiny manifest: the 6-row prompt goes to the 8-row bucket
  with its mask, S = 124, every non-text index shifted by 2, pad rows at the rotary origin; per
  block 11 `qkv` and 11 `post` calls (the modulation runs text 2 | vision 1 | text + pad 5 | audio
  74 | video 42, chunked by 16, never straddling an entry — the modulation table bound to each call
  is checked), 2 head groups × 4 query chunks of attention with the 124-row mask (−1e9 at rows
  6..7), the heads per modality run; two Euler steps reproduce the closed form; the VAE decoder
  runs one 7-frame chunk of a single 4×6 tile (171 rows, 2 block calls) and emits 17 + 5 frames of
  96×64; the audio decoder runs once per channel on `[6, 37]` → 370 samples; TAEH3 runs 5 + 2
  padded latents and keeps 22 of 28 raw frames; FL2VA encodes the keyframe through
  `vae_enc_image`, adds the 6-row condition run at 0.999 (S = 130, 136 with two), draws the
  augmentation noise before the latent noise; wrong prompts / tags / sizes are refused with the
  mismatch in the message.

## Building `:app`

Needs the Android SDK with platform 36 (AGP 8.13.2, Kotlin 2.3.0, Compose BOM 2025.11.00,
LiteRT 2.1.6; minSdk 26):

```bash
cd android
echo "sdk.dir=$ANDROID_HOME" > local.properties      # or export ANDROID_HOME
./gradlew :app:installDebug
```

Kotlin must be ≥ 2.2: `litert-api` 2.1.6 is compiled with Kotlin metadata 2.3.0.

### Pushing graphs and prompts

The app reads from its external files directory (no storage permission needed):

```
/sdcard/Android/data/dev.h3.litert/files/graphs/    manifest.json + adaln.safetensors + *.tflite
/sdcard/Android/data/dev.h3.litert/files/gallery/   *.safetensors with prompt_embeds [rows, 5120] and text_tags [rows]
```

The tiny fixture is the export that runs on a phone:

```bash
adb shell mkdir -p /sdcard/Android/data/dev.h3.litert/files/graphs /sdcard/Android/data/dev.h3.litert/files/gallery
adb push fixtures/tiny/. /sdcard/Android/data/dev.h3.litert/files/graphs/
adb push fixtures/tiny/expected.safetensors /sdcard/Android/data/dev.h3.litert/files/gallery/tiny.safetensors
```

Then press **Rescan**: `expected.safetensors` carries `prompt_embeds [6, 24]` and `text_tags`
(`1, 1, 0, 1, 1, 1`) for the 96×64×22 clip; seed 7 is the fixture's. A real export pushes the same
way (`h3-litert export …`, `h3-litert encode …`).

## The plan, and why a phone runs only the tiny model

Before generating, the app computes `Manifest.plan(size, prompt rows, keyframes, availMem)` and
shows it; **Generate is disabled when it does not fit**. H3 is a 33 B-parameter model: 50 block
graphs of 0.2 GB each at int4 (0.8 GB at f16), a 38 k-row packed sequence at 1344×768×124 whose
`q`/`k`/`v`/`o` are 1.1 GB each in f32, and a 36-block ViT decoder. `docs/design.md` says it
plainly: a phone tier does not exist for this model. The runtime exists to run the same contract
as the desktop and the web, and the plan says no:

```
rows              38396
weights (all)      11000.0 MiB   ...
peak               ...           of 2048 MiB  fits: false
note: the int4 weights alone (11000 MiB) exceed the 2048 MiB budget: a phone cannot hold this model, ...
```

The tiny export (`fixtures/tiny`: hidden 32, 2 blocks, 96×64×22, 124 rows) fits in a few MB and
runs the whole loop — text refiner, embedders, 2 blocks with head groups and query chunks, both
heads, the ViT decoder on one tile, the audio decoder per channel, TAEH3 on request — on a device.

Memory behaviour of the Android backend: block graphs are opened by file under an LRU of 4
(`resident`), the fixed graphs (text refiner, embedders, heads, attention, the VAEs, TAEH3) stay
open, every `TensorBuffer` is created once per graph input and reused across calls, the AdaLN
tables stay f16 and are decoded one `[6, d]` row at a time, and the GPU program cache lives in
`cacheDir/litert-gpu`.

## What is not verified here

- **No device, no SDK in this container.** `:app` has not been compiled; `:core` has
  (`gradle :core:test -Ph3.app=false`, 44 tests). Version pins are the sibling's, which were chosen
  from the Google Maven index and the LiteRT 2.1.6 AAR manifests, not from a build.
- **LiteRT API names** in `LiteRtGraphStore` are the sibling's, checked there against the
  published `litert-api` 2.1.6 classes; their runtime behaviour (GPU compile of the graphs, fp16
  accuracy, whether `output_i` is the signature's output order, the program-cache options) is
  untested and those lines carry `TODO verify` comments.
- **`AudioTrack`** in `ENCODING_PCM_FLOAT` static mode is standard API 21+ behaviour but has not
  been run; the tiny model's 320 Hz audio is upsampled by sample repetition to 8 kHz because the
  platform refuses rates below about 4 kHz.
- **Fixture parity.** The Python `check` compares the graph pipeline against
  `expected.safetensors`; on the JVM there is no LiteRT runtime, so the Kotlin pipeline is verified
  structurally (fake graphs, call sequence, shapes, masks, padding, noise order) and numerically
  only in the host math. Running the tiny fixture on a device through `:app` is the first real
  parity check.
- **Ref2VA** has its layout (`buildRef2vaLayout`, tested) but no pipeline path or UI: the
  reference's `pipeline.py` covers T2VA and FL2VA only.
