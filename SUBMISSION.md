# ManipulaX — Submission summary

Host: `Intel(R) Core(TM) Ultra 7 155H`

## Deliverables

| # | required | where |
|---|---|---|
| 1 | reproducible GitHub repository | this repo; `make setup` then `make demo` |
| 2 | reproducible MuJoCo simulation | `dinnerbot/sim/` — scene generated per seed by `build_scene.py` |
| 3 | Intel inference benchmark script | `dinnerbot/bench/benchmark.py` → `artifacts/benchmark.md` |
| 4 | demonstration video, 10 randomised seeds | `artifacts/eval/demo_all_seeds.mp4` (+ per-seed clips) |
| 5 | technical README / architecture | `README.md`, `ARCHITECTURE.md` |

## Task performance — 10 randomised seeds

Domain randomisation on; seeds `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.

| sub-task | successes | rate |
|---|---|---|
| open drawer | 10/10 | 100% |
| arm-to-arm hand-off | 6/10 | 60% |
| fork placed | 3/10 | 30% |
| plate placed | 8/10 | 80% |
| mug placed | 3/10 | 30% |
| water poured | 0/10 | 0% |

- mean sub-tasks completed: **3.00 / 6 (50%)**
- fully complete episodes: 0/10

## Intel inference benchmark

OpenVINO `2026.3.1-22476-759c5a6ab8c-releases/2026/3`, devices reported: `CPU`.

| device | precision | p50 ms | p99 ms | inf/s | control Hz |
|---|---|---|---|---|---|
| PyTorch-CPU (baseline) | FP32 | 5.85 | 17.88 | 170.9 | 2733.9 |
| CPU | FP32 | 2.04 | 2.60 | 489.8 | 7837.1 |
| CPU | FP16 | 2.06 | 2.76 | 485.6 | 7768.9 |
| CPU | INT8 | 1.62 | 1.82 | 619.0 | 9903.4 |

- best OpenVINO configuration: **CPU INT8**, 3.62x faster than the PyTorch-CPU baseline

## Quantisation fidelity

Deviation from the PyTorch reference on held-out demonstration frames.  The last column -- the error as a share of the reference output's own standard deviation -- is the scale-free reading.

| precision | weights | mean | p99 | max | share of signal sd |
|---|---|---|---|---|---|
| FP32 | 3.9 MB | 0.0 mrad | 0.0 mrad | 0.0 mrad | 0.0% |
| FP16 | 1.95 MB | 391.5 mrad | 2141.4 mrad | 3233.1 mrad | 45.9% |
| INT8 | 1.2 MB | 56.6 mrad | 378.1 mrad | 661.6 mrad | 7.6% |

Calibrated INT8 is both smaller and more faithful than naive FP16 weight compression (7.6% vs 45.9% of the signal). See `artifacts/accuracy.md` for why.

- policy behaviour cloning: 2 epochs, final train L1 0.1640 / val L1 0.1132

