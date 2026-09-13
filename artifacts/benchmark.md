# Intel inference benchmark

- host: `Intel(R) Core(TM) Ultra 7 155H`
- OpenVINO: `2026.3.1-22476-759c5a6ab8c-releases/2026/3`
- devices reported: `CPU, GPU, NPU`
- policy emits a 16-step action chunk per inference, so the sustainable control rate is 16 x inferences/s
- the task needs 30 Hz of control; every supported configuration below clears that by a wide margin

| device | precision | p50 ms | p90 ms | p99 ms | inf/s | control Hz | speedup vs PyTorch | weights MB |
|---|---|---|---|---|---|---|---|---|
| PyTorch-CPU (baseline) | FP32 | 4.76 | 13.07 | 119.95 | 209.9 | 3358.0 | 1.00x | - |
| CPU | FP32 | 1.23 | 1.39 | 1.82 | 811.3 | 12980.8 | 3.86x | 3.9 |
| GPU | FP32 | — | — | — | — | — | not supported | - |
| NPU | FP32 | 20.06 | 21.20 | 22.48 | 49.8 | 797.6 | 0.24x | 3.9 |
| CPU | FP16 | 1.15 | 1.31 | 1.43 | 869.0 | 13904.1 | 4.14x | 1.95 |
| GPU | FP16 | 2.67 | 3.50 | 7.64 | 375.1 | 6001.3 | 1.79x | 1.95 |
| NPU | FP16 | 20.20 | 20.88 | 22.63 | 49.5 | 792.0 | 0.24x | 1.95 |
| CPU | INT8 | 0.89 | 0.99 | 1.08 | 1130.2 | 18083.1 | 5.38x | 1.2 |
| GPU | INT8 | — | — | — | — | — | not supported | - |
| NPU | INT8 | — | — | — | — | — | not supported | - |

### Combinations the hardware refused

- **GPU FP32** — failed at src/plugins/intel_gpu/src/plugin/program_builder.cpp:168: [GPU] ProgramBuilder build failed! Check 'shape_type == shape_types::dynamic_shape
- **GPU INT8** — failed at src/plugins/intel_gpu/src/plugin/program_builder.cpp:168: [GPU] ProgramBuilder build failed! Check 'shape_type == shape_types::dynamic_shape
- **NPU INT8** — failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_UNKNOWN, code 0x7ffffffe - an action is required to complete the desired operation . Compilation fai
