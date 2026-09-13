"""Rebuild artifacts/eval/{summary.json,report.md} from the evaluation log.

The per-seed lines are printed as each seed finishes, so the log is a complete
record even if the run is interrupted before it writes its summary.

    python scripts/recover_eval_report.py artifacts/logs/eval.log artifacts/eval
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys

import numpy as np

LINE = re.compile(
    r"seed\s+(\d+)\s+(\d+)/6 subtasks\s+water=(\d+)\s+(\d+) steps\s+([\d.]+)s\s+(\{.*\})")

FLAGS = [("drawer_open", "drawer"), ("handoff_done", "hand-off"),
         ("fork_placed", "fork"), ("plate_placed", "plate"),
         ("mug_placed", "mug"), ("water_poured", "pour")]
ARM = {"drawer_open": "left", "handoff_done": "both", "fork_placed": "right",
       "plate_placed": "right", "mug_placed": "left", "water_poured": "both"}


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "artifacts/logs/eval.log"
    out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/eval"
    os.makedirs(out, exist_ok=True)

    reports = []
    for line in open(log):
        m = LINE.search(line)
        if not m:
            continue
        task = ast.literal_eval(re.sub(r"np\.(True|False)_", r"\1", m.group(6)))
        reports.append(dict(seed=int(m.group(1)), n_done=int(m.group(2)),
                            water_in_mug=int(m.group(3)), steps=int(m.group(4)),
                            wall_s=float(m.group(5)),
                            task={k: bool(v) for k, v in task.items()}))
    if not reports:
        raise SystemExit(f"no seed lines found in {log}")

    n = len(reports)
    agg = {k: sum(1 for r in reports if r["task"][k]) for k, _ in FLAGS}
    mean_sub = float(np.mean([r["n_done"] for r in reports]))
    full = sum(1 for r in reports if r["n_done"] >= 6)
    summary = dict(seeds=[r["seed"] for r in reports], n=n, per_subtask=agg,
                   full_success=full, mean_subtasks=round(mean_sub, 2),
                   mean_subtask_rate=round(mean_sub / 6, 3), reports=reports)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    L = [f"# Evaluation over {n} randomised seeds", "",
         "- domain randomisation: on (`level=1.0`)",
         f"- seeds: `{[r['seed'] for r in reports]}`", "",
         "| sub-task | arm | successes | rate |", "|---|---|---|---|"]
    for k, label in FLAGS:
        L.append(f"| {label} | {ARM[k]} | {agg[k]}/{n} | {agg[k]/n:.0%} |")
    L += ["", f"- **mean sub-tasks completed: {mean_sub:.2f} / 6 ({mean_sub/6:.0%})**",
          f"- fully complete episodes: {full}/{n}", "",
          "| seed | sub-tasks | water in mug | steps |", "|---|---|---|---|"]
    for r in reports:
        L.append(f"| {r['seed']} | {r['n_done']}/6 | {r['water_in_mug']} | {r['steps']} |")
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"recovered {n} seeds -> {out}/report.md  (mean {mean_sub:.2f}/6)")


if __name__ == "__main__":
    main()
