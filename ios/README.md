# h3-litert for iOS

The Swift side of h3-litert: the host math and the pipeline of
`docs/graphs.md` as a Swift package, a LiteRT (TensorFlow Lite Swift, Metal)
backend, and a SwiftUI demo. The package mirrors `python/h3_litert` one
file at a time; the Python runtime is the reference it is checked against.
It is built the way fastwan-litert's `ios/` is built, with the same
`Tensor`, `Rng`, safetensors reader, graph store and LiteRT glue.

**Status: written without a Swift toolchain.** Nothing in this directory has
been compiled, the tests have not run, and no graph has run on a device. The
code was reviewed by hand for syntax and types; expect a first `swift build`
to turn up small things. See "Not verified" at the end.

## Layout

```
ios/
  Package.swift                       swift-tools-version 5.9; iOS 16 / macOS 13
  Sources/H3LiteRTCore/               Foundation only
    Config.swift                      H3Config, VideoVaeConfig, AudioVaeConfig, TaeH3Config (config.py)
    Geometry.swift                    canvas rules, 17n+5 frames, 5n+2 latents, audio latents (packing.py)
    Packing.swift                     buildLayout, buildRef2vaLayout, rowTimesteps, runs, modRuns, patchRows (packing.py)
    Rope.swift                        h3Rope [S, 3·rope_freq_dim] cos/sin; the decoder's table for the tests (rope.py)
    Schedule.swift                    shiftSigma, f32 linspace, Schedule, PddPlan (schedule.py)
    Sampler.swift                     flowStep: Euler / renoise / PDD, f32 (sampler.py)
    Timestep.swift                    the sinusoid (timestep.py), for the tests
    Rng.swift                         SplitMix64 → Box–Muller, UInt64 wrapping arithmetic (rng.py)
    Tensor.swift                      [Float] + shape; rows / setRows / slice / concat along any axis
    VaeHost.swift                     splitTiles, blend, stitchTiles, tiled, temporalPlan, decode, encode,
                                      normalisation, posteriorSample (vae_host.py)
    Frames.swift                      Frames / Video / Audio; RGBA conversions with the reference's rounding
    Graph.swift                       Graph and GraphStore protocols, ResidentGraphCache (LRU of block graphs)
    Manifest.swift                    Codable manifest; format check; attentionFor / textRefineFor / audioFor / taeh3For throw
    Adaln.swift                       adaln.safetensors: f16 tables in memory, [6, d] / [2, d] rows decoded per call
    Plan.swift                        the memory plan (design.md "Tiers")
    Pipeline.swift                    prepare, forward, denoise, decodeVideo (VAE / TAEH3), decodeAudio, generate (pipeline.py)
    SafeTensors.swift                 F32 / F16 / U8 reader; binary16 ↔ Float in integer math
    Check.swift                       FixtureCheck: the `check` self-test against expected.safetensors
  Sources/H3LiteRTRuntime/
    LiteRTGraph.swift                 LiteRTGraph (Interpreter + SignatureRunner, MetalDelegate), LiteRTGraphStore
  Tests/H3LiteRTCoreTests/
    HostTests.swift                   every case of python/tests/test_host.py; manifest, AdaLN, safetensors, VAE host, plan
    PipelineTests.swift               the orchestration over a fake GraphStore on the tiny manifest
    Resources/manifest.json           a copy of fixtures/tiny/manifest.json
    Resources/adaln.safetensors       a copy of fixtures/tiny/adaln.safetensors
  Demo/                               SwiftUI app sources and how to make the Xcode project
```

### Core

`Pipeline` drives `Graph`s through a `GraphStore` exactly as
`docs/graphs.md`'s host loop and `pipeline.py` do:

- `prepare(request, encodeClip:)`: the text bucket (`textRefineFor(rows:)`,
  the prompt zero-padded to `T`, `mask` −1e9 past the real rows) through
  `text_refine_T` once per prompt; `buildLayout` with the real rows; FL2VA
  keyframes through the encoder graphs (`VaeHost.tiled`, posterior sampled
  with seed 42, rounded through f16, normalised, patchified, then
  `0.999·rows + 0.001·noise` from the request generator); the video latent
  noise `[C, T, h, w]`, then the audio noise `[2A, A_c]`; every non-text row
  index shifted by the pad; `cos`/`sin` `[S, r/2]` over the padded sequence
  (pad rows at position 0); the `[S]` key mask.
- `forward(prepared, step:, attention:)`: `rowTimesteps` with the condition
  times `max(t_video, 0.999)` / `1.0` and their AdaLN rows; `h[text] = text`;
  `embed_video` / `embed_audio` over the runs of each modality; per block
  `qkv` over the modulation runs (rows sharing `(timestep, tag)`, each run
  chunked by `C`, the last chunk zero-padded), attention per head group over
  query chunks of `Cq` with the exact `S` keys, values and mask, `post` over
  the same runs; `head_video` / `head_audio` over each modality's runs split
  where the timestep changes (condition rows vs target rows), with the PDD
  `weights` input when the sampler is `pdd`.
- `denoise`: the steps of `sampler.steps`; the renoise generator
  `Rng(seed + 1)`, video before audio; PDD's initial scaling by the first
  sigmas; condition rows never written; `unpatchRows` of the target rows.
- `decodeVideo`: `VaeHost.decode` (temporal plan, the repeated last latent
  frame, the `frame_pre_padding` cut, the chunk blend, the trailing overlap)
  over `VaeHost.tiled` (256 px tiles, 64 px overlap, the cross-fade stitch)
  over `vae_dec_in` → blocks → `vae_dec_out`; or TAEH3 chunks with nine
  memories and the `i mod 20 ≥ 3` rule when `preview`.
- `decodeAudio`: two mono `avae_dec_A` calls on the denormalised,
  transposed rows.
- Events: `.prepared(seconds:tokens:)`, `.step(index:total:seconds:)`,
  `.frames(Frames)` (RGBA `[count, height, width, 4]`, ready for a
  `CGImage`; `Frames.rgb` is the Python layout), `.audio(Audio)`.

All graph inputs are float32 tensors with exact shapes and no batch
dimension; inputs are bound by signature name (`q, k, v, mask` for
attention), outputs returned by position.

### Runtime

`LiteRTGraph` opens one `.tflite` file with the TensorFlow Lite Swift API:
`Interpreter(modelPath:options:delegates:)` with a `MetalDelegate` when
asked (`isPrecisionLossAllowed` on, so the GPU may run f16), the graph's
single signature through `interpreter.signatureRunner(with:)`, inputs by
name via `SignatureRunner.invoke(with: [String: Data])`, outputs by
`output(named:)` sorted into `output_0, output_1, …` order. Input shapes and
dtypes are checked against the graph before every call and a mismatch is an
error naming the graph, the input and both shapes. `LiteRTGraphStore` reads
graphs from a directory URL, keeps the fixed graphs open and the block graphs
in an LRU of `residentBlocks` (default 2), and loads `adaln.safetensors`
(`adalnTables()`); `Pipeline(store: LiteRTGraphStore)` wires both.

The Swift API is distributed as the CocoaPod `TensorFlowLiteSwift`
(2.17.0; GPU delegate in the `Metal` subspec), with `LiteRTSwift` (nightly
pods) as its successor. There is no official Swift package of either, so
`Package.swift` declares no dependency and `LiteRTGraph.swift` is wrapped in
`#if canImport(TensorFlowLite)`: on a Mac without the framework the target
compiles to a stub and `swift test` runs the core tests. In an app, link the
pod and make sure the file compiles where the pod is visible (see
`Demo/README.md` step 4). Marked `// TODO verify` in the source.

## Tests

```bash
cd ios && swift test
```

`HostTests` reproduces every pinned value of `python/tests/test_host.py`:
the V2 ladder, the uniform grid and the Pika knots, the LongLive-Plug grid's
f32 bit patterns (`0x3F800000, 0x3F7914C2, 0x3F6C4EC5, 0x3F4CCCCD, 0` at
shift 12 and `0x3F800000, 0x3F666666, 0x3F400000, 0x3F000000, 0` at shift 3),
the PDD plan, `124 → 37` / `362 → 107` latent frames and `124 → 207` /
`243 → 405` audio latents, the canvas rules (16:9 → 768×1344), the 768p
layout, the FL2VA anchors (first at 3.0, last at `3 + 5/3·22 − 5/3`), the row
timesteps and modulation runs `[(2,0,1), (1,0,0), (12,2,0), (40,1,2),
(42,0,0)]`, the Ref2VA segment order, runs and patch rows, the RoPE tables,
the VAE's derived numbers `(16, 4, 3, 5, 2, 5)`, the TAEH3 rule `(17, 124,
5)`, SplitMix64's `0xE220A8397B1DCDAF`; plus `splitTiles(768, 256, 64, 16) =
([0, 160, 336, 512], [256 × 4], [96, 80, 80])`, `temporalPlan(h3, 37) =
(0, 7, 124)`, `(…, 7) = (0, 1, 22)`, `(…, 2) = (5, 1, 5)`, blend / stitch /
tiled, decode and encode on a fake clip, the pixel conversions, the manifest,
the AdaLN table of the fixture, the safetensors reader and binary16, the
plan.

`PipelineTests` runs the pipeline on a fake store that knows every graph of
the tiny manifest and records each call: on the tiny T2VA request (6 prompt
rows → bucket 8, 2 pad rows, S = 124; chunk 16; q_chunk 32; head groups of 1
of 2 heads) it checks per block and step 11 `qkv` calls over the modulation
runs (text 2 → 1, vision placeholder 1 → 1, text 3 + 2 pad → 1, audio 74 → 5,
video 42 → 3), 2 head groups × 4 query chunks of attention with the 124-long
mask carrying −1e9 at rows 6 and 7, `post` as `qkv`, the heads per modality
run, 2 steps, the AdaLN rows handed to each call; the noise order; FL2VA with
a keyframe (the image encoder once, 6 condition rows, S = 130, the condition
run at the 0.999 table row, the rows never written); the VAE decode (one
7-frame chunk at 171 rows, 2 block calls, 22 frames in two events of 17 and
5); the TAEH3 preview (2 calls, 22 kept frames, memories carried, z
frame-major and zero-padded); the audio decode (2 calls → 370 samples); the
error paths; the LRU.

## Getting the graphs onto the device

The app reads the export from its Documents folder. With
`UIFileSharingEnabled` and `LSSupportsOpeningDocumentsInPlace` in the app's
Info.plist, that folder shows up in the Files app ("On My iPhone → H3 Demo")
and in Finder (device → Files tab), where you drop:

- `Documents/graphs/`: the whole export directory from `h3-litert export`
  (`manifest.json`, `adaln.safetensors` and every `.tflite`;
  `LiteRTGraphStore.missingFiles()` lists what is absent),
- `Documents/gallery/<name>.safetensors`: prompt embeddings from
  `h3-litert encode` (`prompt_embeds` `[rows, 5120]` F32 and `text_tags`).

The export is the same one the Python, web and Android runtimes use;
nothing is converted on the device.

## The phone tier and memory

There is no phone tier for a 33 B model (`docs/design.md`): the runtime
exists to run the same contract, and `Plan.make(manifest:geometry:textRows:
keyframes:preview:budgetBytes:residentBlocks:)` says no. It counts the
resident block graphs (× their manifest byte sizes) and the fixed graphs of
the request, the AdaLN table, the `[S, d]` / `[S, I]` activations (`h`, `q`,
`k`, `v`, `o`, the tables and row buffers), one attention call's gathered
keys and values plus its `G · Cq · S` score chunk, the decoder's peak (the
ViT tile's rows, the chunk's tiles before the stitch, the two pixel chunks of
the blend; or TAEH3's stage maps) and the RGBA frames and waveform held, and
compares the larger of the denoise and decode peaks with the budget. The
demo shows the plan for half the device's memory and refuses to generate when
it does not fit; the tiny export fits anywhere.

## Not verified

- **Nothing compiled.** No Swift toolchain or Xcode was available where this
  was written. Every file was self-reviewed for syntax and types, but the
  first build will be the first compile. The tests are written to reproduce
  the pinned values of `python/tests/test_host.py` and the orchestration
  counts above (checked by hand and, where numeric, against the Python
  reference in this repository); they have not run.
- **No device.** `LiteRTGraph` follows the TensorFlow Lite Swift API as
  published in `tensorflow/lite/swift/Sources`, but it has not been linked or
  run; the Metal delegate path, the f16 precision of its output against
  `expected.safetensors` (the contract's 2e-2), speed, and the memory of the
  LRU under a 50-block export are all untested. XNNPACK together with the
  Metal delegate on one interpreter is marked `// TODO verify`.
- **Package integration of the pod.** See "Runtime" above; the fallback of
  compiling `LiteRTGraph.swift` inside the app target is documented but not
  tried.
- The `check` path (`FixtureCheck.run`) has not been run against
  `fixtures/tiny/expected.safetensors` through LiteRT.
- `AVAudioEngine` at the zero-order-held 8 kHz of the tiny fixture, and
  the keyframe orientation out of the `CGContext` draw, are untested.
