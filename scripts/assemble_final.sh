#!/usr/bin/env bash
# Final assembly: merge the evaluation logs into one report, stitch the demo
# reel from the per-seed clips, benchmark on an idle machine, and regenerate
# SUBMISSION.md from the artifacts.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export MUJOCO_GL=egl

echo "== merging evaluation logs =="
cat artifacts/logs/eval_final.log artifacts/logs/eval_rest.log > artifacts/logs/eval_all.log 2>/dev/null || \
  cp artifacts/logs/eval_final.log artifacts/logs/eval_all.log
$PY scripts/recover_eval_report.py artifacts/logs/eval_all.log artifacts/eval

echo "== stitching the demo reel from all per-seed clips =="
$PY scripts/stitch_demo.py artifacts/eval 30

echo "== benchmark (machine should be idle now) =="
$PY -m dinnerbot.bench.benchmark --devices auto --iters 200 --out artifacts/benchmark

echo "== submission summary =="
$PY scripts/make_submission_report.py
