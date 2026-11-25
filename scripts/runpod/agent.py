"""The on-pod command agent: a stdlib HTTP server behind RunPod's proxy
(`https://<pod>-8000.proxy.runpod.net/<token>/...`), so a sandbox that can
reach the proxy but not the pod drives the pod without SSH or a bucket.

    GET  /T/log               the boot log
    POST /T/run/N  (script)   run the script as command N (bash, cwd /root/work), async
    GET  /T/out/N             command N's output so far (+ "\n== exit K" when done)
    GET  /T/file?p=PATH       a file from the pod
    POST /T/put?p=PATH        store the body at PATH
    GET  /T/ps                running commands

T is a per-pod token (the first argument); a request without it is 404.
Everything runs as root inside the pod; the pod holds no account key.
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

TOKEN = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
WORK = "/root/work"
os.makedirs(WORK, exist_ok=True)
procs = {}
lock = threading.Lock()


def run_cmd(n, script):
    path = f"{WORK}/cmd-{n}.sh"
    out = f"{WORK}/out-{n}.log"
    with open(path, "w") as f:
        f.write(script)
    log = open(out, "wb")
    p = subprocess.Popen(["bash", path], cwd=WORK, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CMD": n})
    with lock:
        procs[n] = p

    def wait():
        code = p.wait()
        log.write(f"\n== exit {code}\n".encode())
        log.flush()
        log.close()

    threading.Thread(target=wait, daemon=True).start()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self):
        u = urlparse(self.path)
        parts = u.path.strip("/").split("/")
        if not parts or parts[0] != TOKEN:
            self._send(404, b"not found")
            return None, None
        return parts[1:], parse_qs(u.query)

    def do_GET(self):
        parts, q = self._route()
        if parts is None:
            return
        if parts == ["log"]:
            self._send(200, open("/root/boot.log", "rb").read() if os.path.exists("/root/boot.log") else b"")
        elif len(parts) == 2 and parts[0] == "out":
            p = f"{WORK}/out-{parts[1]}.log"
            self._send(200, open(p, "rb").read() if os.path.exists(p) else b"(not yet)")
        elif parts == ["file"]:
            p = q.get("p", [""])[0]
            if os.path.isfile(p):
                self._send(200, open(p, "rb").read(), "application/octet-stream")
            else:
                self._send(404, b"no such file")
        elif parts == ["ps"]:
            with lock:
                st = {n: p.poll() for n, p in procs.items()}
            self._send(200, json.dumps(st).encode(), "application/json")
        else:
            self._send(404, b"not found")

    def do_POST(self):
        parts, q = self._route()
        if parts is None:
            return
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        if len(parts) == 2 and parts[0] == "run":
            run_cmd(parts[1], body.decode())
            self._send(200, b"started")
        elif parts == ["put"]:
            p = q.get("p", [""])[0]
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            open(p, "wb").write(body)
            self._send(200, f"{len(body)} bytes".encode())
        else:
            self._send(404, b"not found")


ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
