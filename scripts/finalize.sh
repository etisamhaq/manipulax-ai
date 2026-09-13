#!/usr/bin/env bash
# Train -> export -> benchmark -> fidelity -> submission report.
# Run after the evaluation has finished (they compete for CPU and RAM).
set -x
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export MUJOCO_GL=egl

$PY -m dinnerbot.policy.train --data artifacts/dataset.npz \
    --epochs "${EPOCHS:-2}" --batch 48 --threads "${THREADS:-16}" \
    --workers 0 --stride "${STRIDE:-2}"

$PY -m dinnerbot.bench.export --ckpt artifacts/policy.pt \
    --calib artifacts/calib.npz --out artifacts/ov

$PY -m dinnerbot.bench.benchmark --devices auto --iters 120 --out artifacts/benchmark

$PY -m dinnerbot.bench.accuracy --ir-dir artifacts/ov --data artifacts/dataset.npz \
    --ckpt artifacts/policy.pt --n 128 --out artifacts/accuracy

$PY scripts/make_submission_report.py
