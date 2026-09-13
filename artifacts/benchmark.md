# Intel inference benchmark

- host: `Intel(R) Core(TM) Ultra 7 155H`
- OpenVINO: `2026.3.1-22476-759c5a6ab8c-releases/2026/3`
- devices reported: `CPU`
- policy emits a 16-step action chunk per inference, so the sustainable control rate is 16 x inferences/s

| device | precision | p50 ms | p90 ms | p99 ms | inf/s | control Hz | speedup vs PyTorch | weights MB |
|---|---|---|---|---|---|---|---|---|
| PyTorch-CPU (baseline) | FP32 | 5.85 | 10.73 | 17.88 | 170.9 | 2733.9 | 1.00x | - |
| CPU | FP32 | 2.04 | 2.15 | 2.60 | 489.8 | 7837.1 | 2.87x | 3.9 |
| CPU | FP16 | 2.06 | 2.26 | 2.76 | 485.6 | 7768.9 | 2.84x | 1.95 |
| CPU | INT8 | 1.62 | 1.68 | 1.82 | 619.0 | 9903.4 | 3.62x | 1.2 |
