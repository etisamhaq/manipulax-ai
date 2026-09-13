"""Assemble SUBMISSION.md from the artifacts actually produced by the pipeline.

Every number in the report is read from a JSON that some script wrote, so the
write-up cannot drift away from the runs it describes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))


def load(p):
    """Missing or truncated JSON yields None rather than killing the report.

    An artifact written before the numpy-serialisation fix can be half-written;
    the report should still assemble from everything else.
    """
    p = os.path.join(ROOT, p)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"note: ignoring unreadable {os.path.basename(p)} ({e.__class__.__name__})")
        return None


def cpu():
    try:
        for line in subprocess.check_output(["lscpu"], text=True).splitlines():
            if line.strip().startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        return "unknown"
    return "unknown"


def main():
    ev = load("artifacts/eval/summary.json")
    bench = load("artifacts/benchmark.json")
    acc = load("artifacts/accuracy.json")
    ds = load("artifacts/dataset_report.json")
    hist = load("artifacts/policy_history.json")

    L = ["# ManipulaX — Submission summary", "",
         f"Host: `{cpu()}`", ""]

    L += ["## Deliverables", "",
          "| # | required | where |", "|---|---|---|",
          "| 1 | reproducible GitHub repository | this repo; `make setup` then `make demo` |",
          "| 2 | reproducible MuJoCo simulation | `dinnerbot/sim/` — scene generated per seed by `build_scene.py` |",
          "| 3 | Intel inference benchmark script | `dinnerbot/bench/benchmark.py` → `artifacts/benchmark.md` |",
          "| 4 | demonstration video, 10 randomised seeds | `artifacts/eval/demo_all_seeds.mp4` (+ per-seed clips) |",
          "| 5 | technical README / architecture | `README.md`, `ARCHITECTURE.md` |", ""]

    if ev:
        L += [f"## Task performance — {ev['n']} randomised seeds", "",
              f"Domain randomisation on; seeds `{ev['seeds']}`.", "",
              "| sub-task | successes | rate |", "|---|---|---|"]
        label = {"drawer_open": "open drawer", "handoff_done": "arm-to-arm hand-off",
                 "fork_placed": "fork placed", "plate_placed": "plate placed",
                 "mug_placed": "mug placed", "water_poured": "water poured"}
        for k, v in ev["per_subtask"].items():
            L.append(f"| {label.get(k,k)} | {v}/{ev['n']} | {v/ev['n']:.0%} |")
        L += ["", f"- mean sub-tasks completed: **{ev['mean_subtasks']:.2f} / 6 "
                  f"({ev['mean_subtask_rate']:.0%})**",
              f"- fully complete episodes: {ev['full_success']}/{ev['n']}", ""]

    if bench:
        L += ["## Intel inference benchmark", "",
              f"OpenVINO `{bench['openvino']}`, devices reported: "
              f"`{', '.join(bench['devices'])}`.", "",
              "| device | precision | p50 ms | p99 ms | inf/s | control Hz |",
              "|---|---|---|---|---|---|"]
        base = next((r for r in bench["rows"] if r["device"].startswith("PyTorch")), None)
        for r in bench["rows"]:
            L.append(f"| {r['device']} | {r['precision'].upper()} | {r['p50_ms']:.2f} | "
                     f"{r['p99_ms']:.2f} | {r['fps']:.1f} | {r['control_hz']:.1f} |")
        if base:
            best = min((r for r in bench["rows"] if not r["device"].startswith("PyTorch")),
                       key=lambda r: r["p50_ms"], default=None)
            if best:
                L += ["", f"- best OpenVINO configuration: **{best['device']} "
                          f"{best['precision'].upper()}**, "
                          f"{base['p50_ms']/best['p50_ms']:.2f}x faster than the "
                          f"PyTorch-CPU baseline"]
        L.append("")

    if acc:
        L += ["## Quantisation fidelity", "",
              "Deviation from the PyTorch reference on held-out demonstration "
              "frames.  The last column -- the error as a share of the reference "
              "output's own standard deviation -- is the scale-free reading.", "",
              "| precision | weights | mean | p99 | max | share of signal sd |",
              "|---|---|---|---|---|---|"]
        for r in acc:
            L.append(f"| {r['precision']} | {r.get('size_mb','-')} MB | "
                     f"{r['mae_mrad']:.1f} mrad | {r['p99_mrad']:.1f} mrad | "
                     f"{r['max_mrad']:.1f} mrad | "
                     f"{r.get('frac_of_signal',0):.1%} |")
        f16 = next((r for r in acc if r["precision"] == "FP16"), None)
        i8 = next((r for r in acc if r["precision"] == "INT8"), None)
        if f16 and i8 and f16.get("mae_norm", 0) > i8.get("mae_norm", 0):
            L += ["", f"Calibrated INT8 is both smaller and more faithful than naive "
                      f"FP16 weight compression ({i8.get('frac_of_signal',0):.1%} vs "
                      f"{f16.get('frac_of_signal',0):.1%} of the signal). See "
                      f"`artifacts/accuracy.md` for why.", ""]
        else:
            L.append("")

    if ds:
        L += ["## Training data", "",
              f"- episodes recorded: {len(ds['episodes'])}, kept: {ds['kept']}",
              f"- frames: {ds['frames']}", ""]
    if hist:
        L += [f"- policy behaviour cloning: {len(hist)} epochs, "
              f"final train L1 {hist[-1]['train']:.4f} / val L1 {hist[-1]['val']:.4f}", ""]

    out = os.path.join(ROOT, "SUBMISSION.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", out)
    missing = [n for n, v in [("eval", ev), ("benchmark", bench), ("accuracy", acc),
                              ("dataset", ds), ("training", hist)] if v is None]
    if missing:
        print("note: not yet produced ->", ", ".join(missing))


if __name__ == "__main__":
    main()
