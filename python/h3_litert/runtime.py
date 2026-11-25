"""Running `.tflite` graphs on the desktop through `ai-edge-litert` (h3-litert's manifest; the Graph / GraphStore classes are fastwan-litert's).

`Graph` wraps one graph: the classic `Interpreter` (XNNPACK, CPU) or the
`CompiledModel` API (`--accelerator gpu`). Inputs are bound by name, outputs
returned in signature order (`docs/graphs.md`). `GraphStore` opens graphs of a
manifest on demand and keeps at most `resident` block graphs alive, so a
30-block model streams through memory one block at a time, like the browser
build of fastwan-rs.
"""

from __future__ import annotations

import json
import os
import re
import threading
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

FORMAT = "h3-litert/1"


def _output_order(names: list[str]) -> list[str]:
    def key(n: str):
        m = re.search(r"(\d+)$", n)
        return int(m.group(1)) if m else 0

    return sorted(names, key=key)


class Graph:
    """One compiled `.tflite` graph."""

    def __init__(self, path: Path | str, accelerator: str = "cpu", threads: int | None = None):
        self.path = str(path)
        self.accelerator = accelerator
        self._lock = threading.Lock()
        if accelerator == "cpu":
            from ai_edge_litert import interpreter as li

            self._it = li.Interpreter(model_path=self.path, num_threads=threads or os.cpu_count() or 4)
            sig = self._it.get_signature_list()
            self.signature = next(iter(sig))
            self.input_names = list(sig[self.signature]["inputs"])
            self.output_names = _output_order(sig[self.signature]["outputs"])
            self._runner = self._it.get_signature_runner(self.signature)
            self._cm = None
        elif accelerator == "gpu":
            from ai_edge_litert import compiled_model as cm

            try:
                self._cm = cm.CompiledModel.from_file(self.path, cm.HardwareAccelerator.GPU)
            except RuntimeError as e:
                # the accelerator refuses some graphs (Conv3D, grouped 1-D convs, large f16 blocks
                # on the WebGPU backend): that graph runs on the CPU, the rest stay on the GPU
                warnings.warn(f"{os.path.basename(self.path)}: GPU compile failed ({str(e).strip().splitlines()[-1][:80]}); running it on the CPU")
                self.__init__(path, "cpu", threads)
                return
            sig = self._cm.get_signature_list()
            self.signature = next(iter(sig))
            self.input_names = list(sig[self.signature]["inputs"])
            self.output_names = _output_order(sig[self.signature]["outputs"])
            self._in = {n: self._cm.create_input_buffer_by_name(self.signature, n) for n in self.input_names}
            self._out = {n: self._cm.create_output_buffer_by_name(self.signature, n) for n in self.output_names}
            # CompiledModel has no output-shape query; a metadata-only interpreter answers it
            from ai_edge_litert import interpreter as li

            det = li.Interpreter(model_path=self.path).get_signature_runner(self.signature).get_output_details()
            self._out_shape = {n: tuple(int(x) for x in det[n]["shape"]) for n in self.output_names}
            self._it = None
        else:
            raise ValueError(f"accelerator `{accelerator}` is not cpu or gpu")

    def run(self, **inputs: np.ndarray) -> list[np.ndarray]:
        missing = set(self.input_names) - set(inputs)
        extra = set(inputs) - set(self.input_names)
        if missing or extra:
            raise ValueError(f"{os.path.basename(self.path)}: inputs {sorted(self.input_names)}, got {sorted(inputs)}")
        feed = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in inputs.items()}
        with self._lock:
            if self._it is not None:
                out = self._runner(**feed)
                return [np.asarray(out[n]) for n in self.output_names]
            for n, v in feed.items():
                self._in[n].write(v)
            self._cm.run_by_name(self.signature, self._in, self._out)
            return [
                np.asarray(self._out[n].read(int(np.prod(self._out_shape[n])), np.float32), dtype=np.float32).reshape(self._out_shape[n])
                for n in self.output_names
            ]

    def input_shape(self, name: str) -> tuple[int, ...]:
        if self._it is not None:
            for d in self._it.get_input_details():
                if d["name"].split(":")[0].endswith(name) or d["name"] == name:
                    return tuple(int(x) for x in d["shape"])
            det = self._runner.get_input_details()[name]
            return tuple(int(x) for x in det["shape"])
        det = self._cm.get_input_tensor_details(self.signature)[name]
        return tuple(int(x) for x in det["shape"])


class Manifest:
    def __init__(self, directory: Path | str):
        self.dir = Path(directory)
        self.raw = json.loads((self.dir / "manifest.json").read_text())
        if self.raw.get("format") != FORMAT:
            raise ValueError(f"{self.dir / 'manifest.json'}: format {self.raw.get('format')!r}, expected {FORMAT!r}")
        from .config import AudioVaeConfig, H3Config, TaeH3Config, VideoVaeConfig
        

        self.cfg = H3Config.from_json(self.raw["config"])
        self.vae = VideoVaeConfig.from_json(self.raw["vae"])
        self.audio_vae = AudioVaeConfig.from_json(self.raw["audio_vae"])
        self.taeh3 = TaeH3Config.from_json(self.raw["taeh3"]) if self.raw.get("taeh3") else None
        self.sampler: dict = self.raw["sampler"]
        self.chunk: int = int(self.raw["chunk"])
        self.head_group: int = int(self.raw["head_group"])
        self.weights: str = self.raw["weights"]
        self.graphs = self.raw["graphs"]

    def attention_for(self, tokens: int) -> dict:
        for a in self.raw["attention"]:
            if a["tokens"] == tokens:
                return a
        have = sorted({a["tokens"] for a in self.raw["attention"]})
        raise ValueError(f"this export has no attention graph for {tokens} rows (it has {have}); re-export with --size")

    def text_refine_for(self, rows: int) -> dict:
        for t in sorted(self.graphs["text_refine"], key=lambda t: t["rows"]):
            if t["rows"] >= rows:
                return t
        raise ValueError(f"the prompt has {rows} rows; this export's largest text_refine bucket is {max(t['rows'] for t in self.graphs['text_refine'])}")

    def audio_for(self, latents: int) -> dict:
        for a in self.raw["audio"]:
            if a["latents"] == latents:
                return a
        raise ValueError(f"this export has no audio VAE graphs for {latents} latents (it has {sorted(a['latents'] for a in self.raw['audio'])})")

    def taeh3_for(self, latent_h: int, latent_w: int) -> dict:
        for t in self.raw.get("taeh3_graphs", []):
            if t["latent_h"] == latent_h and t["latent_w"] == latent_w:
                return t
        raise ValueError(f"this export has no TAEH3 graph for latent {latent_h}x{latent_w}")

    def path(self, file: str) -> Path:
        return self.dir / file


class GraphStore:
    """Graphs of one manifest, opened on demand. Block graphs beyond
    `resident` are closed least-recently-used first."""

    def __init__(self, manifest: Manifest, accelerator: str = "cpu", resident: int = 4, threads: int | None = None):
        self.m = manifest
        self.accelerator = accelerator
        self.threads = threads
        self.resident = max(1, resident)
        self._fixed: dict[str, Graph] = {}
        self._blocks: OrderedDict[str, Graph] = OrderedDict()

    def fixed(self, file: str) -> Graph:
        g = self._fixed.get(file)
        if g is None:
            g = self._fixed[file] = Graph(self.m.path(file), self.accelerator, self.threads)
        return g

    def block(self, file: str) -> Graph:
        g = self._blocks.get(file)
        if g is not None:
            self._blocks.move_to_end(file)
            return g
        while len(self._blocks) >= self.resident:
            self._blocks.popitem(last=False)
        g = self._blocks[file] = Graph(self.m.path(file), self.accelerator, self.threads)
        return g

    def drop_blocks(self) -> None:
        self._blocks.clear()
