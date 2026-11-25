// The browser fixture check: serves `web/` + `fixtures/tiny`, drives
// `tests/fixture.html` in headless Chromium through Playwright, runs T2VA on
// the tiny export, prints the numbers and exits non-zero when they are off.
// Options:
//   --accelerator wasm|webgpu   (default wasm; WebGPU needs a GPU and
//                                `--enable-unsafe-webgpu`, not available headless here)
//   --jspi        load the JSPI Wasm build (`loadLiteRt(dir, {jspi: true})`)
//   --threads     load the threaded build (needs COOP/COEP, which serve.mjs sends)
//   --fixture DIR the tiny export (default ../fixtures/tiny, or $H3_FIXTURE)
//   --preview     also run the TAEH3 preview decoder (reported, not judged)
//   --fl2va       also run an FL2VA denoise with a synthetic first keyframe (reported, not judged)
//   --timeout S   give up after S seconds (default 300)
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { defaultMounts, fixtureDir, startServer } from "./serve.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const flag = (n) => args.includes(n);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const accelerator = opt("--accelerator", "wasm");
const jspi = flag("--jspi");
const threads = flag("--threads");
const preview = flag("--preview");
const fl2va = flag("--fl2va");
const timeoutMs = Number(opt("--timeout", "300")) * 1000;

const here = dirname(fileURLToPath(import.meta.url));
if (!existsSync(join(here, "..", "dist", "src", "index.js"))) {
  console.error("web/dist is missing: run `npm run build` first (or `npm run test:fixture`, which builds)");
  process.exit(2);
}
const fixture = fixtureDir(args);
if (!existsSync(join(fixture, "manifest.json"))) {
  console.error(`no tiny fixture at ${fixture} (no manifest.json): export it with \`h3-litert export --model tiny --out fixtures/tiny\` or pass --fixture DIR`);
  process.exit(2);
}

const { server, port } = await startServer({ quiet: true, mounts: defaultMounts(fixture) });
const launch = { args: [] };
if (accelerator === "webgpu") launch.args.push("--enable-unsafe-webgpu", "--enable-features=Vulkan");
const browser = await chromium.launch(launch);
let code = 1;
const logs = [];
try {
  const page = await browser.newPage();
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
  const q = new URLSearchParams({ manual: "1" });
  await page.goto(`http://127.0.0.1:${port}/tests/fixture.html?${q}`);
  await page.waitForFunction(() => typeof window.runFixture === "function", null, { timeout: 60_000 });
  const t0 = Date.now();
  const run = page.evaluate((o) => window.runFixture(o), {
    base: "/fixtures/tiny/",
    wasmDir: "/node_modules/@litertjs/core/wasm/",
    accelerator,
    jspi,
    threads,
    preview,
    fl2va,
  });
  const timer = new Promise((_, reject) => setTimeout(() => reject(new Error(`no result after ${timeoutMs / 1000} s`)), timeoutMs));
  const r = await Promise.race([run, timer]);
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  for (const e of r.events) console.log("  " + e);
  const ex = (x) => x.toExponential(2);
  console.log(
    `fixture ${r.geometry} seed ${r.seed} (${r.tokens} rows, ${r.paddedTokens} with the text pad) [${r.weights}] on ${r.accelerator}${jspi ? "+jspi" : ""}${threads ? "+threads" : ""}:\n` +
      `  video latent rel ${ex(r.latent.rel)} (max |Δ| ${ex(r.latent.maxAbs)}, tol ${ex(r.latent.tol)})\n` +
      `  audio rows   rel ${ex(r.audioRows.rel)} (max |Δ| ${ex(r.audioRows.maxAbs)}, tol ${ex(r.audioRows.tol)})\n` +
      `  frames ${r.frames.count}/${r.frames.expected} ${r.frames.width}x${r.frames.height} max |Δ| ${r.frames.maxDelta}/255 (mean ${r.frames.meanDelta.toFixed(3)}, tol ${r.frames.tol})\n` +
      `  waveform ${r.audio.samples} samples/channel at ${r.audio.sampleRate} Hz rel ${ex(r.audio.rel)} (max |Δ| ${ex(r.audio.maxAbs)}, tol ${ex(r.audio.tol)})\n` +
      (r.preview ? `  TAEH3 preview: ${r.preview.count} frames ${r.preview.width}x${r.preview.height}, max |Δ| vs the VAE frames ${r.preview.maxDeltaVsVae}/255 (not judged)\n` : "") +
      (r.fl2va ? `  FL2VA (first keyframe through the encoder): ${r.fl2va.tokens} rows (${r.fl2va.paddedTokens} padded, ${r.fl2va.conditionRows} condition rows), latent max |Δ| vs T2VA ${r.fl2va.latentDeltaVsT2va.toFixed(3)} (not judged)\n` : "") +
      `  ${r.downloads} file downloads, prepare ${r.timings.prepareMs.toFixed(0)} ms, steps ${r.timings.stepsMs.map((m) => m.toFixed(0)).join("/")} ms, ` +
      `decode ${r.timings.decodeMs.toFixed(0)} ms, audio ${r.timings.audioMs.toFixed(0)} ms, ${secs} s total: ${r.ok ? "OK" : "FAIL"}`,
  );
  // LiteRT prints its INFO lines through console.error; keep the real warnings only.
  const warn = logs.filter((l) => /^\[(warning|error|pageerror)\]/.test(l) && !/\] INFO:/.test(l) && !/WebGPU device|No available adapters/.test(l));
  if (warn.length) console.log(warn.join("\n"));
  code = r.ok ? 0 : 1;
} catch (e) {
  console.error("fixture check failed:", e);
  if (logs.length) console.error("page log:\n" + logs.join("\n"));
  code = 1;
} finally {
  await browser.close();
  server.close();
}
process.exit(code);
