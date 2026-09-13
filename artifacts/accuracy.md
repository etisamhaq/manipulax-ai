# Quantisation fidelity

Deviation of each OpenVINO precision from the PyTorch reference over 128 held-out demonstration frames.  Reported both in milliradians at the joint (the policy's output unit) and as a fraction of the reference output's own standard deviation, which is the scale-free way to read it.

| precision | weights | mean abs error | p99 | max | share of signal sd |
|---|---|---|---|---|---|
| FP32 | 3.9 MB | 0.0 mrad | 0.0 mrad | 0.0 mrad | 0.0% |
| FP16 | 1.95 MB | 391.5 mrad | 2141.4 mrad | 3233.1 mrad | 45.9% |
| INT8 | 1.2 MB | 56.6 mrad | 378.1 mrad | 661.6 mrad | 7.6% |

## Reading this

- **FP32 IR is exact.** It reproduces PyTorch to ~3e-6; the conversion itself loses nothing.
- **Naive FP16 weight compression is the worst option here, not the safe one.** It drifts 46% of the output's own standard deviation -- and it is both larger (1.95 MB) and less faithful than INT8 (1.2 MB, 8%). The difference is calibration: INT8 post-training quantisation saw real recorded frames and joint states and placed its ranges accordingly, while FP16 just rounds every weight blindly. A 2-layer pre-norm transformer at d_model=128 has little headroom for that.
- **INT8 is the precision to ship, with a caveat.** At 57 mrad mean deviation it is the same order as the SO-ARM100's own servo tracking error under load, not comfortably below it. It is defensible for this closed-loop task -- the policy re-plans every few control ticks -- but it is not free, and a model trained quantisation-aware would be the honest next step.

The end-to-end task numbers in `artifacts/eval/report.md` were produced by the scripted expert, so they do not yet isolate the effect of precision on task success; that comparison needs a policy good enough to close the loop on its own.
