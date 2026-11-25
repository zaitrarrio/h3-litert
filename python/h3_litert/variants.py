"""The released variants as presets (h3-core `variant.rs`): checkpoint, sampler,
shifts. VSA numbers are recorded but every clip runs dense here."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import H3Config
from .schedule import (FASTH3_V1_RUNGS, FASTH3_V2_RUNGS, H3_BASE_POINTS, PIKA_KNOTS, PLUG4_KNOTS, SHIFTS_FASTH3_V2, SHIFTS_H3,
                       PddPlan, Schedule, Shifts, step_plan)

BASE = "MiniMaxAI/MiniMax-H3"


@dataclass(frozen=True)
class Variant:
    name: str
    repo: str
    transformer_dir: str
    config: H3Config
    sampler: str  # "uniform" | "rungs" | "renoise" | "pdd"
    shifts: Shifts
    rungs: tuple[int, ...] = ()
    points: int = 0
    pdd_indices: tuple[int, ...] = ()
    adapters: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ("t2va", "fl2va")
    summary: str = ""

    def rule(self) -> str:
        return {"uniform": "euler", "rungs": "euler", "renoise": "renoise", "pdd": "pdd"}[self.sampler]

    def schedules(self) -> tuple[Schedule, Schedule, PddPlan | None]:
        if self.sampler == "uniform":
            return Schedule.uniform(self.points, self.shifts.video), Schedule.uniform(self.points, self.shifts.audio), None
        if self.sampler == "rungs":
            return Schedule.from_rungs(list(self.rungs), self.shifts.video), Schedule.from_rungs(list(self.rungs), self.shifts.audio), None
        if self.sampler == "renoise":
            return (Schedule.from_rungs(list(self.rungs), self.shifts.video, "renoise"),
                    Schedule.from_rungs(list(self.rungs), self.shifts.audio, "renoise"), None)
        p = PddPlan.new(list(self.pdd_indices), self.shifts)
        v, a = p.schedules()
        return v, a, p

    def sampler_json(self) -> dict:
        v, a, p = self.schedules()
        return {
            "rule": self.rule(),
            "shifts": {"video": self.shifts.video, "audio": self.shifts.audio},
            "steps": step_plan(v, a, p, self.config.pdd_steps),
            "cond_video_t": 0.999,
            "cond_audio_t": 1.0,
        }

    def forwards(self) -> int:
        return len(self.sampler_json()["steps"])


def all_variants() -> list[Variant]:
    h3 = H3Config.h3()
    return [
        Variant("h3", BASE, "transformer", h3, "uniform", SHIFTS_H3, points=H3_BASE_POINTS, tasks=("t2va", "fl2va", "ref2va"),
                summary="MiniMax-H3 base: 49 Euler forwards, dense"),
        Variant("h3-plug4", BASE, "transformer", h3, "renoise", SHIFTS_H3, rungs=tuple(PLUG4_KNOTS), adapters=("generator_lora.safetensors",),
                tasks=("t2va", "fl2va", "ref2va"), summary="LongLive-Plug few-step LoRA over the base, 4 renoise forwards"),
        Variant("fasth3-v1", "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree", "transformer", h3, "rungs", SHIFTS_H3,
                rungs=tuple(FASTH3_V1_RUNGS), summary="FastH3 V1: 4 DMD rungs (VSA 0.9 in the reference; dense here)"),
        Variant("fasth3-v2", "FastVideo/FastVideo-FastH3-8-Step-V2", "transformer", h3, "rungs", SHIFTS_FASTH3_V2,
                rungs=tuple(FASTH3_V2_RUNGS), summary="FastH3 V2: 8 DMD rungs, shifts 10 / 3"),
        Variant("fasth3-trim", "FastVideo/FastVideo-FastH3-Trim-8-Step", "transformer", H3Config.fasth3_trim(), "rungs", SHIFTS_FASTH3_V2,
                rungs=tuple(FASTH3_V2_RUNGS), summary="FastH3 Trim: 42 blocks, rank-16 AdaLN, 26-block decoder"),
        Variant("fasth3-omni", "FastVideo/FastVideo-FastH3-Omni-8-Step-V1", "transformer_ref", H3Config.fasth3_omni(), "pdd", SHIFTS_H3,
                pdd_indices=(0, 4, 8, 12, 16, 20, 24, 28, 32), tasks=("ref2va",), summary="FastH3 Omni: PDD, 8 fused-head forwards, Ref2VA"),
        Variant("pika-flex", "Pika-Labs/Pika-H3-Flex", "transformer", h3, "renoise", SHIFTS_H3, rungs=tuple(PIKA_KNOTS),
                adapters=("fl2va", "ref2va"), tasks=("fl2va", "ref2va"), summary="Pika H3 Flex: LoRA per task, 6 renoise knots"),
        Variant("tiny", "", "", H3Config.tiny(), "rungs", SHIFTS_FASTH3_V2, rungs=(999, 500), summary="the self-test model"),
    ]


def by_name(name: str) -> Variant:
    aliases = {"v1": "fasth3-v1", "v2": "fasth3-v2", "trim": "fasth3-trim", "omni": "fasth3-omni", "pika": "pika-flex", "plug4": "h3-plug4"}
    name = aliases.get(name, name)
    for v in all_variants():
        if v.name == name:
            return v
    raise ValueError(f"unknown variant `{name}`; one of {', '.join(v.name for v in all_variants())}")
