# `@h3/litert` — MiniMax H3 on LiteRT.js

The web runtime of h3-litert: the host math of h3-rs (schedules, the packed
layout, RoPE, SplitMix64 noise, the flow step, the VAE tiles / temporal chunks,
the TAEH3 frame rule) in TypeScript, the graph pipeline of `docs/graphs.md`
over [LiteRT.js](https://www.npmjs.com/package/@litertjs/core) (`@litertjs/core`
2.6, WebGPU or Wasm), and a browser studio. ESM, strict TypeScript (no `any`),
no dependency besides `@litertjs/core`. It is fastwan-litert's web runtime
grown to H3: the same `Graph` / `GraphStore` / safetensors / test harness,
with the H3 host loop (modulation runs, head groups, the masked attention,
per-timestep head runs, two modalities, keyframes).

```
src/config.ts      H3Config, VideoVaeConfig, AudioVaeConfig, TaeH3Config (keepsRawFrame)   (python: config.py)
src/schedule.ts    shiftSigma, Schedule (rungs / uniform), PddPlan, stepPlan; all f32 as the reference
src/packing.ts     Geometry, canvasSize, 17n+5 → 5n+2, audioLatents (half-even), buildLayout,
                   buildRef2vaLayout, rowTimesteps, runs, modRuns, patchRows / unpatchRows
src/rope.ts        h3Rope (the DiT table: f32 positions · f32 inverse frequencies), vaeDecoderRope
src/rng.ts         Rng: SplitMix64 → Box–Muller, bit-exact with Rust / Python / Kotlin / Swift
src/sampler.ts     flowStep (Euler / renoise / PDD), needsNoise
src/vaeHost.ts     splitTiles, blend, stitchTiles, tiled, temporalPlan, decode, encode,
                   normalise / denormalise pixels and latents, posteriorSample     (python: vae_host.py)
src/f16.ts         binary16 ↔ binary32 (ties to even): the AdaLN table, the keyframe latent rounding
src/safetensors.ts a small safetensors reader (F32, F16, U8)
src/adaln.ts       AdalnTables: adaln.safetensors, f16 rows widened to f32 on demand
src/manifest.ts    parseManifest / loadManifest: the contract, validated; attentionFor / textRefineFor /
                   audioFor / taeh3For throw with the mismatch, nothing falls back
src/graphs.ts      Graph (inputs by name, outputs by position), LiteRtGraph over @litertjs/core,
                   GraphStore (on-demand fetch, LRU of block graphs, progress), UrlSource / MapSource
src/pipeline.ts    Pipeline: prepare / forward / denoise / decodeVideo (VAE or TAEH3) / decodeAudio /
                   generate + events; T2VA and FL2VA (keyframes through the encoder graphs)
src/plan.ts        plan / formatPlan: the memory plan
src/timestep.ts    sinusoid (pinned by the tests; the table itself is precomputed by the exporter)
demo/              index.html + studio.ts: the browser studio
tests/             host.test.ts (vitest), the browser fixture check, a static server
```

## Build and test

```bash
cd web
npm ci                     # or npm install
npx tsc --noEmit           # typecheck src/, demo/, tests/fixture-page.ts
npx vitest run             # tests/host.test.ts
npm run test:fixture       # npm run build && node tests/run-fixture.mjs
npm test                   # all of the above
```

CI runs exactly `npm ci`, `npx tsc --noEmit && npx vitest run`, then
`npx playwright install --with-deps chromium && npm run test:fixture`;
`playwright` is pinned as an exact devDependency (1.63.0) for that.

- `vitest run` — `tests/host.test.ts`, every case of
  `python/tests/test_host.py`: the V2 ladder, the uniform grid and the Pika
  knots, the LongLive-Plug grid's f32 bit patterns (`0x3F800000 0x3F7914C2
  0x3F6C4EC5 0x3F4CCCCD 0` at shift 12, `0x3F800000 0x3F666666 0x3F400000
  0x3F000000 0` at shift 3), the PDD plan, 124 → 37 and 362 → 107 latent
  frames, 124 → 207 and 243 → 405 audio latents, the canvas rules (16:9 →
  768×1344), the 768p T2VA layout, the FL2VA anchors (first = 3.0, last = 3 +
  5/3·22 − 5/3), row timesteps and the modulation runs `[(2,0,1), (1,0,0),
  (12,2,0), (40,1,2), (42,0,0)]`, the Ref2VA segment order, runs and the patch
  rows round trip, the RoPE tables, the VAE's derived numbers `(16, 4, 3, 5, 2,
  5)`, the TAEH3 kept frames `(17, 124, 5)`, SplitMix64
  (`Rng(0x9E3779B97F4A7C15).nextU64() === 0xE220A8397B1DCDAF`), the sinusoid;
  plus `splitTiles(768, 256, 64, 16) = [[0,160,336,512], [256×4], [96,80,80]]`,
  `temporalPlan(h3, 37) = [0, 7, 124]`, `(…, 7) = [0, 1, 22]`, `(…, 2) =
  [5, 1, 5]`, the blend / stitch arithmetic, the float32 flow step, the f16
  round trip, the AdaLN table layout, the manifest validation and the plan.
- `node tests/run-fixture.mjs` — launches headless Chromium through Playwright,
  serves `web/`, `fixtures/tiny` and `node_modules/@litertjs/core/wasm` from a
  local static server (`tests/serve.mjs`), loads the tiny export on the
  **Wasm** backend and runs the fixture's T2VA request (seed 7, the 6-row
  prompt tagged `[1, 1, 0, 1, 1, 1]`, `96x64x22`): `denoise`, `decodeVideo`,
  `decodeAudio`, compared with `expected.safetensors` the way `h3-litert
  check` does (`rel = max|Δ| / max|expected|`; for an f32 export on Wasm the
  video latent and the audio rows at 1e-3, the frames at 2/255, the waveform at
  1e-2; f16 / int8 / int4 exports and GPU delegates at 2e-2, 16/255, 2e-1). It
  prints the numbers and exits non-zero on failure. Measured here on the
  committed f32 fixture (Chromium 143 headless, Linux):

  ```
  fixture 96x64x22 seed 7 (122 rows, 124 with the text pad) [f32] on wasm:
    video latent rel 1.72e-5 (max |Δ| 7.92e-5, tol 1.00e-3)
    audio rows   rel 2.00e-5 (max |Δ| 7.22e-5, tol 1.00e-3)
    frames 22/22 96x64 max |Δ| 1/255 (mean 0.001, tol 2)
    waveform 370 samples/channel at 320 Hz rel 2.13e-4 (max |Δ| 2.13e-4, tol 1.00e-2)
    21 file downloads, prepare 63 ms, steps 159/60 ms, decode 70 ms, audio 30 ms, 0.6 s total: OK
  ```

  The check runs with `resident: 1`, so the downloads show the LRU evicting
  and re-fetching block graphs between steps, as a 50-block model would.
  Flags: `--preview` also decodes through the TAEH3 graph, `--fl2va` also runs
  an FL2VA denoise with a synthetic first keyframe through `vae_enc_image`
  (both reported, not judged: the fixture has no reference for them),
  `--fixture DIR` (or `$H3_FIXTURE`) points at another export,
  `--accelerator webgpu`, `--jspi`, `--threads`, `--timeout S`.

## Use the package

```ts
import { loadLiteRt } from "@litertjs/core";
import { AdalnTables, Geometry, GraphStore, Pipeline, UrlSource, litertCompiler, loadManifest, plan } from "@h3/litert";

await loadLiteRt("/wasm/");                              // the @litertjs/core wasm directory
const m = await loadManifest("/graphs/fasth3-trim-int4/"); // format must be h3-litert/1
const source = new UrlSource("/graphs/fasth3-trim-int4/", m.files);
const store = new GraphStore(m, source, litertCompiler({ accelerator: "webgpu" }), {
  resident: 2,                                           // block graphs kept alive (LRU)
  onProgress: (p) => console.log(p.file, p.loaded, p.total),
});
const adaln = AdalnTables.parse(await source.fetch(m.adaln.file));
const pipe = new Pipeline(store, adaln);
const g = Geometry.parse("1344x768x124");
const p = plan(m, { geometry: g, textRows: 40, keyframes: 0, preview: false, resident: 2, budgetBytes: 64 * 2 ** 30 });
if (!p.fits) throw new Error(formatPlan(p));
const { video, audio } = await pipe.generate(
  { embeds /* Float32Array [rows·5120] */, textTags /* [rows]: 1 text, 0 vision */, geometry: g, seed: 7, keyframes: [], preview: false },
  (e) => {
    if (e.type === "frames") ctx.putImageData(new ImageData(e.rgba[0], e.width, e.height), 0, 0);
  },
);
```

Events are the contract's: `prepared {ms, tokens}` (packed rows before the
text padding), one `step {index, total, ms}` per sampler step, `frames {first,
count, width, height, rgba}` per decoder chunk (`rgba` is one
`Uint8ClampedArray` per frame, alpha 255, ready for `ImageData`), and `audio
{sampleRate, stereo}` with the two waveforms in `[−1, 1]`. `denoise` returns
the video latent `[24, T, Hl, Wl]` and the target audio rows `[2A, 32]`;
`decodeVideo(latent, preview)` turns the latent into frames through the ViT
VAE (256 px tiles, 7-latent chunks, the blends) or TAEH3; `decodeAudio` runs
the BigVGAN graph per channel. Keyframes (`{anchor: "first" | "last", pixels:
Float32Array [3, H, W] in [0, 1]}`) go through `vae_enc_image`, the posterior
at seed 42, an f16 round, normalisation and the 0.999 augmentation, as the
reference. Graph inputs are bound by name and checked against the signature;
outputs come back by position (`output_0`, `output_1`, …). A size, prompt
length or audio length the export has no graphs for, a manifest of another
format, a step whose times are not in the AdaLN table, or a mismatched
signature throws with the mismatch in the message; nothing falls back.

## The studio (`demo/`)

```bash
npm run build
npm run serve                      # http://127.0.0.1:8765/demo/?base=/fixtures/tiny/
```

`npm run serve` serves `web/` (so `demo/`, `dist/`, `node_modules/` are
reachable) and `../fixtures/`. The page takes a graph base URL (default
`./graphs/`, i.e. `web/demo/graphs/`; `?base=` prefills it), a gallery URL
(default `./gallery/`, `?gallery=`), a prompt, a size from the manifest's
`sizes`, a seed, the number of resident block graphs, an optional first / last
keyframe image (FL2VA; scaled to the canvas when it is another size) and the
TAEH3 preview toggle. It shows the memory plan and refuses a run that does not
fit (on a phone it says so: a phone cannot hold H3; only the tiny self-test
export runs there), shows download progress per block (from the manifest's
`files` sizes), per-step timings, plays the frames on a canvas at 24 fps as
they arrive and the audio through WebAudio when the clip is done (the tiny
fixture's 320 Hz is sample-held up to 32 kHz because browsers refuse rates
below a few kHz). WebGPU when the browser has a GPU adapter, else Wasm.

**Gallery.** There is no text encoder in the browser (Qwen3-VL-32B is not
on-device work). `gallery/index.json` lists prompt files, either
`["fox.safetensors", …]` or `[{"name": "a red fox…", "file":
"fox.safetensors"}, …]`; each file is a safetensors with `prompt_embeds
[rows, 5120]` f32 and `text_tags [rows]` (1 text, 0 vision; all text when
absent), from `h3-litert encode`.

### Hosting

- Serve `dist/`, `demo/`, the graphs and `adaln.safetensors` statically. Block
  graphs are plain files (two per block), so any static host or CDN works;
  `files[*].bytes` drives the progress bars and is checked after download.
- `@litertjs/core`'s `dist/index.js` imports `@litertjs/wasm-utils` as a bare
  specifier; both are mapped in `demo/index.html`'s import map. Without
  `node_modules` on the server, point the import map at jsDelivr and the
  `data-wasm-dir` attribute (or `?wasm=`) at
  `https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/`.
- The threaded Wasm build needs COOP/COEP headers (`tests/serve.mjs` sends
  them). Nothing else needs special headers.

## What is not verified

- **No GPU here.** Everything ran on the Wasm (XNNPACK) backend in headless
  Chromium on Linux. `accelerator: 'webgpu'`, `precision: 'fp16'`,
  WebGPU/Wasm partitioning under JSPI and the 2e-2 GPU tolerance are untested;
  the studio's WebGPU path is the same code with another `CompileOptions`.
- **The real model.** Only the tiny export (2 blocks, 122 rows) has run.
  Speed and memory at 1344×768×124 with the 42/50-block graphs (streaming
  blocks, the 1.1 GB `q/k/v/o` buffers, the 36-block ViT decoder over 3×4 tiles
  × 7 chunks) are from `plan`, not measured. The plan counts f32 row buffers
  at `S × 7168`; nothing is quantised on the host.
- **FL2VA and the TAEH3 preview** run (the fixture check's `--fl2va` and
  `--preview` flags, and the studio) but have no reference tensors in the
  fixture; the Python tests judge them against PyTorch. Ref2VA has the layout
  only (`buildRef2vaLayout`); the pipeline builds T2VA / FL2VA layouts, as the
  Python runtime.
- **PDD and renoise samplers** are ported (`flowStep`, the `weights` head
  input, the `seed + 1` generator) but the tiny fixture is Euler; they are
  exercised by the unit tests, not by a graph run.
- **Phones.** The phone message is a user-agent check and `plan`'s budget
  (`navigator.deviceMemory`, 2 GiB when unknown on a phone); no device was
  available.
