// A tiny static server for the fixture check and the demo: serves `web/` at
// `/`, the tiny fixture at `/fixtures/tiny/` (the repository's `fixtures/`
// directory, or the directory in `H3_FIXTURE` / `--fixture`) read-only. Sends
// COOP/COEP so the threaded Wasm build can be used when a page asks for it.
// No dependencies.
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
export const WEB_ROOT = resolve(here, "..");
export const REPO_ROOT = resolve(WEB_ROOT, "..");

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".map": "application/json",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".tflite": "application/octet-stream",
  ".safetensors": "application/octet-stream",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

/** The tiny fixture directory: `--fixture DIR`, `$H3_FIXTURE`, or `<repo>/fixtures/tiny`. */
export function fixtureDir(args = process.argv.slice(2)) {
  const i = args.indexOf("--fixture");
  if (i >= 0 && args[i + 1]) return resolve(args[i + 1]);
  if (process.env.H3_FIXTURE) return resolve(process.env.H3_FIXTURE);
  return join(REPO_ROOT, "fixtures", "tiny");
}

/** Mounts: URL prefix → directory. The first matching prefix wins. */
export function defaultMounts(fixture = fixtureDir()) {
  return [
    ["/fixtures/tiny/", fixture],
    ["/fixtures/", join(REPO_ROOT, "fixtures")],
    ["/", WEB_ROOT],
  ];
}

function resolvePath(url, mounts) {
  const path = decodeURIComponent(new URL(url, "http://x").pathname);
  for (const [prefix, dir] of mounts) {
    if (!path.startsWith(prefix)) continue;
    const rel = normalize(path.slice(prefix.length)).replace(/^(\.\.(\/|\\|$))+/, "");
    const full = resolve(dir, rel);
    if (full !== dir && !full.startsWith(dir + sep)) return null;
    return full;
  }
  return null;
}

export function startServer({ port = 0, mounts = defaultMounts(), quiet = true } = {}) {
  const server = createServer((req, res) => {
    const file = resolvePath(req.url ?? "/", mounts);
    const headers = {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin",
      "Cache-Control": "no-cache",
    };
    let target = file;
    try {
      if (target && statSync(target).isDirectory()) target = join(target, "index.html");
      const st = target ? statSync(target) : null;
      if (!st || !st.isFile()) throw new Error("not found");
      res.writeHead(200, {
        ...headers,
        "Content-Type": TYPES[extname(target)] ?? "application/octet-stream",
        "Content-Length": st.size,
      });
      createReadStream(target).pipe(res);
    } catch {
      if (!quiet) console.error("404", req.url);
      res.writeHead(404, headers);
      res.end("not found");
    }
  });
  return new Promise((ok) => {
    server.listen(port, "127.0.0.1", () => ok({ server, port: server.address().port }));
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const port = Number(args[args.indexOf("--port") + 1] || 8765);
  const { port: p } = await startServer({ port, quiet: false });
  console.log(`serving ${WEB_ROOT} on http://127.0.0.1:${p}/  (demo: /demo/, fixture: /fixtures/tiny/ -> ${fixtureDir(args)})`);
}
