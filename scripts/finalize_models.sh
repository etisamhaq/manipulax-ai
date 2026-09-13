#!/usr/bin/env bash
# Export -> benchmark -> fidelity -> report, using the trained checkpoint.
set -x
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export MUJOCO_GL=egl
$PY -m dinnerbot.bench.export --ckpt artifacts/policy.pt --calib artifacts/calib.npz --out artifacts/ov
$PY -m dinnerbot.bench.benchmark --devices auto --iters 120 --out artifacts/benchmark
$PY -m dinnerbot.bench.accuracy --ir-dir artifacts/ov --data artifacts/dataset.npz \
    --ckpt artifacts/policy.pt --n 128 --out artifacts/accuracy
$PY scripts/make_submission_report.py
