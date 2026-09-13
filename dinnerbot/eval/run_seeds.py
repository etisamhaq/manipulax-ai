"""Evaluate the dinner-table system across randomised seeds and render the video.

The same script produces the metrics table and the demonstration video, so the
video can never drift out of sync with the numbers it is supposed to evidence.
Every frame is annotated with the natural-language command, the sub-task the
system believes it is executing, which arm owns it, and the live inference
device + latency, so a viewer can verify the claims without reading the code.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

class _NpEncoder(json.JSONEncoder):
    """numpy scalars and arrays are not JSON-serialisable by default."""

    def default(self, o):
        import numpy as _np
        if isinstance(o, _np.bool_):
            return bool(o)
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        return super().default(o)


from ..sim.env import DinnerTableEnv
from ..expert.primitives import Executor
from ..expert.scripted_task import SUBTASK_ARM, SUBTASK_LANG, run_expert

COMMAND = ("Open the drawer, put the fork on the right of the mat with a hand-off, "
           "centre the plate, then hold the mug and pour water into it.")

FLAGS = [("drawer_open", "drawer"), ("handoff_done", "hand-off"),
         ("fork_placed", "fork"), ("plate_placed", "plate"),
         ("mug_placed", "mug"), ("water_poured", "pour")]


def hud(frame, *, seed, subtask, arm, task, device, latency_ms, step, extra=""):
    import cv2
    f = np.ascontiguousarray(frame)
    h, w = f.shape[:2]
    pad = 78
    canvas = np.zeros((h + pad + 34, w, 3), np.uint8)
    canvas[pad:pad + h] = f
    cv2.rectangle(canvas, (0, 0), (w, pad), (18, 18, 22), -1)
    cv2.rectangle(canvas, (0, pad + h), (w, pad + h + 34), (18, 18, 22), -1)

    def put(txt, xy, scale=0.44, col=(235, 235, 235), th=1):
        cv2.putText(canvas, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)

    words, line, lines = COMMAND.split(), "", []
    for wd in words:
        if len(line) + len(wd) > 78:
            lines.append(line); line = ""
        line += wd + " "
    lines.append(line)
    put('"' + lines[0].strip(), (10, 18), 0.42, (120, 210, 255))
    put((lines[1] if len(lines) > 1 else "").strip() + '"', (10, 34), 0.42, (120, 210, 255))
    put(f"seed {seed:>3d}   step {step:>5d}", (10, 54), 0.42, (170, 170, 175))
    put(f"subtask: {subtask}  [{arm}]", (10, 70), 0.46, (255, 225, 140))
    right = (f"controller: {device}" if latency_ms <= 0
             else f"{device} | {latency_ms:.2f} ms/inference")
    put((right + " " + extra).strip(), (w - 330, 70), 0.42, (150, 255, 170))

    x = 10
    for key, label in FLAGS:
        ok = bool(task.get(key, False))
        col = (90, 220, 120) if ok else (70, 70, 78)
        cv2.circle(canvas, (x + 6, pad + h + 17), 6, col, -1)
        put(label, (x + 18, pad + h + 22), 0.42, (235, 235, 235) if ok else (130, 130, 136))
        x += 22 + 9 * len(label)
    return canvas


class HudRecorder:
    """Wraps an Executor so every simulator tick lands in the video."""

    def __init__(self, env, ex, seed, cam="front", size=(720, 540), every=3):
        self.env, self.ex, self.seed = env, ex, seed
        self.cam, self.size, self.every = cam, size, every
        self.frames = []
        self.device, self.latency, self.extra = "scripted expert", 0.0, ""
        self._n = 0

    def tick(self):
        self._n += 1
        if self._n % self.every:
            return
        img = self.env.render(self.cam, self.size)
        self.frames.append(hud(img, seed=self.seed, subtask=self.ex.subtask,
                               arm=self.ex.arm_of_subtask, task=self.env.task.score(),
                               device=self.device, latency_ms=self.latency,
                               step=self.env.t, extra=self.extra))


def run_seed(seed, dr=1.0, cam="front", size=(720, 540), every=3, video=True,
             runner=None):
    env = DinnerTableEnv(seed=seed, dr_level=dr, img_size=(96, 96))
    ex = Executor(env, record=False, frame_cams=())
    rec = HudRecorder(env, ex, seed, cam, size, every) if video else None
    if rec is not None:
        orig_log = ex._log
        def logged(a):
            orig_log(a); rec.tick()
        ex._log = logged
    t0 = time.time()
    rep = (runner or run_expert)(ex)
    rep["seed"] = seed
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["steps"] = env.t
    frames = rec.frames if rec else []
    env.close()
    return rep, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--dr", type=float, default=1.0)
    ap.add_argument("--cam", default="front")
    ap.add_argument("--out", default="artifacts/eval")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    if "-" in a.seeds:
        lo, hi = a.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(a.out, exist_ok=True)

    import imageio.v2 as imageio
    reports = []
    # The combined reel is written incrementally.  Holding every frame of every
    # seed in a list to concatenate at the end costs gigabytes -- a 3000-step
    # episode is ~1000 frames at 720x540x3, and ten of those will be OOM-killed
    # partway through the run.
    reel = None
    if not a.no_video:
        reel = imageio.get_writer(os.path.join(a.out, "demo_all_seeds.mp4"),
                                  fps=a.fps, quality=7, macro_block_size=1)
    try:
        for s in seeds:
            rep, frames = run_seed(s, a.dr, a.cam, every=a.every, video=not a.no_video)
            reports.append(rep)
            print(f"  seed {s:>3d}  {rep['n_done']}/6 subtasks  water={rep['water_in_mug']}  "
                  f"{rep['steps']} steps  {rep['wall_s']}s  {rep['task']}"
                  + (f"  ABORTED: {rep['aborted']}" if rep.get("aborted") else ""),
                  flush=True)
            if frames:
                imageio.mimsave(os.path.join(a.out, f"seed_{s:03d}.mp4"), frames,
                                fps=a.fps, quality=7, macro_block_size=1)
                if reel is not None:
                    for fr in frames:
                        reel.append_data(fr)
                    for _ in range(a.fps):        # hold the final state
                        reel.append_data(frames[-1])
            del frames
    finally:
        if reel is not None:
            reel.close()
            print("wrote", os.path.join(a.out, "demo_all_seeds.mp4"))

    n = len(reports)
    agg = {k: sum(1 for r in reports if r["task"][k]) for k, _ in FLAGS}
    full = sum(1 for r in reports if r["n_done"] >= 6)
    mean_sub = float(np.mean([r["n_done"] for r in reports]))
    summary = dict(seeds=seeds, n=n, per_subtask=agg, full_success=full,
                   mean_subtasks=round(mean_sub, 2),
                   mean_subtask_rate=round(mean_sub / 6, 3), reports=reports)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=_NpEncoder)

    lines = [f"# Evaluation over {n} randomised seeds", "",
             f"- domain-randomisation level: `{a.dr}`",
             f"- seeds: `{seeds}`", "",
             "| sub-task | arm | successes | rate |", "|---|---|---|---|"]
    keymap = {"drawer_open": "open_drawer", "handoff_done": "handoff_fork",
              "fork_placed": "place_fork", "plate_placed": "place_plate",
              "mug_placed": "place_mug", "water_poured": "pour"}
    for k, label in FLAGS:
        lines.append(f"| {label} | {SUBTASK_ARM.get(keymap[k],'-')} | "
                     f"{agg[k]}/{n} | {agg[k]/n:.0%} |")
    lines += ["", f"- **mean sub-tasks completed: {mean_sub:.2f} / 6 "
                  f"({mean_sub/6:.0%})**",
              f"- fully complete episodes: {full}/{n}", "",
              "| seed | sub-tasks | water in mug | steps |", "|---|---|---|---|"]
    for r in reports:
        lines.append(f"| {r['seed']} | {r['n_done']}/6 | {r['water_in_mug']} | {r['steps']} |")
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nmean {mean_sub:.2f}/6 sub-tasks | full success {full}/{n}")
    print("wrote", os.path.join(a.out, "report.md"))


if __name__ == "__main__":
    main()
