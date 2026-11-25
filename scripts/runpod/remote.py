"""Drive a pod's agent through RunPod's proxy. `H3_POD=<id> H3_TOKEN=<token>`.

    remote.py run N FILE          queue FILE as command N
    remote.py wait N [SECONDS]    poll until command N exits (prints its output), default 3 h
    remote.py out N | log | ps
    remote.py get PATH DEST       download a file from the pod
    remote.py put SRC PATH        upload a file to the pod
"""
import os
import sys
import time
import urllib.request

POD, TOKEN = os.environ["H3_POD"], os.environ["H3_TOKEN"]
BASE = f"https://{POD}-8000.proxy.runpod.net/{TOKEN}"


def req(path, data=None, timeout=120):
    r = urllib.request.Request(BASE + path, data=data, method="POST" if data is not None else "GET",
                               headers={"Content-Type": "application/octet-stream", "User-Agent": "h3-litert-remote/1"})
    return urllib.request.urlopen(r, timeout=timeout).read()


op = sys.argv[1]
if op == "run":
    print(req(f"/run/{sys.argv[2]}", open(sys.argv[3], "rb").read()).decode())
elif op == "wait":
    n, limit = sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3 * 3600
    t0, shown = time.time(), 0
    while time.time() - t0 < limit:
        try:
            out = req(f"/out/{n}").decode(errors="replace")
        except Exception as e:  # proxy hiccups while the pod is busy
            sys.stdout.write(f"(proxy: {e})\n")
            time.sleep(15)
            continue
        if len(out) > shown:
            sys.stdout.write(out[shown:])
            sys.stdout.flush()
            shown = len(out)
        if "\n== exit " in out:
            sys.exit(0 if out.rstrip().endswith("exit 0") else 1)
        time.sleep(20)
    sys.exit(f"timeout after {limit} s")
elif op in ("out", "log", "ps"):
    print(req("/" + (f"out/{sys.argv[2]}" if op == "out" else op)).decode(errors="replace"))
elif op == "get":
    data = req(f"/file?p={urllib.request.quote(sys.argv[2])}", timeout=900)
    open(sys.argv[3], "wb").write(data)
    print(f"{sys.argv[3]}: {len(data)} bytes")
elif op == "put":
    print(req(f"/put?p={urllib.request.quote(sys.argv[3])}", open(sys.argv[2], "rb").read(), timeout=900).decode())
else:
    sys.exit(__doc__)
