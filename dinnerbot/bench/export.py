"""Export the policy to OpenVINO IR and produce FP32 / FP16 / INT8 variants.

INT8 uses NNCF post-training quantization with a calibration set drawn from the
recorded demonstrations, so the quantiser sees the real distribution of camera
images and joint states rather than noise.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from ..policy.model import ACTION_DIM, CHUNK, IMG, N_CAMS, STATE_DIM, make_policy

DEFAULT_OUT = "artifacts/ov"


def example_inputs(batch=1):
    return (torch.zeros(batch, N_CAMS, 3, IMG, IMG),
            torch.zeros(batch, STATE_DIM),
            torch.zeros(batch, dtype=torch.long))


def load_policy(ckpt=None):
    m = make_policy()
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)  # checkpoint carries numpy norm stats
        m.load_state_dict(sd["model"] if "model" in sd else sd)
        print(f"loaded weights from {ckpt}")
    else:
        print("no checkpoint given -- exporting randomly initialised weights "
              "(graph/latency are identical, only the numbers differ)")
    return m.eval()


def calibration_data(path, n=96):
    """Yield calibration samples from the recorded dataset if we have one."""
    if path and os.path.exists(path):
        d = np.load(path)
        imgs, state, sub = d["images"], d["state"], d["subtask"]
        idx = np.linspace(0, len(state) - 1, min(n, len(state))).astype(int)
        return [{"imgs": imgs[i:i + 1].astype(np.float32) / 255.0,
                 "state": state[i:i + 1].astype(np.float32),
                 "subtask": sub[i:i + 1].astype(np.int64)} for i in idx]
    rng = np.random.default_rng(0)
    return [{"imgs": rng.random((1, N_CAMS, 3, IMG, IMG), dtype=np.float32),
             "state": rng.normal(0, 1, (1, STATE_DIM)).astype(np.float32),
             "subtask": rng.integers(0, 11, (1,)).astype(np.int64)} for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/policy.pt")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--calib", default="artifacts/calib.npz")
    ap.add_argument("--no-int8", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import openvino as ov

    model = load_policy(a.ckpt)
    ex = example_inputs()
    with torch.no_grad():
        ref = model(*ex).numpy()

    t0 = time.time()
    # Trace with check_trace off: re-running the trace to compare produces
    # freshly mangled module names for the Sequential head and the checker
    # reports that as a divergence even though the graph is identical.
    traced = torch.jit.trace(model, ex, check_trace=False, strict=False)
    traced = torch.jit.freeze(traced.eval())
    ov_model = ov.convert_model(traced, example_input=ex,
                                input=[("imgs", [1, N_CAMS, 3, IMG, IMG]),
                                       ("state", [1, STATE_DIM]),
                                       ("subtask", [1])])
    print(f"converted to IR in {time.time()-t0:.1f}s")

    # save_model(compress_to_fp16=True) rewrites the model it is handed, so each
    # precision is produced from its own conversion.  Sharing one object made the
    # FP16 save contaminate the INT8 quantisation that followed it.
    fp32 = os.path.join(a.out, "policy_fp32.xml")
    ov.save_model(ov_model, fp32, compress_to_fp16=False)

    fp16_src = ov.convert_model(traced, example_input=ex,
                                input=[("imgs", [1, N_CAMS, 3, IMG, IMG]),
                                       ("state", [1, STATE_DIM]),
                                       ("subtask", [1])])
    fp16 = os.path.join(a.out, "policy_fp16.xml")
    ov.save_model(fp16_src, fp16, compress_to_fp16=True)
    print("wrote", fp32, "and", fp16)

    # numerical check of the FP32 IR against PyTorch
    core = ov.Core()
    out = core.compile_model(fp32, "CPU")(dict(
        imgs=ex[0].numpy(), state=ex[1].numpy(), subtask=ex[2].numpy()))
    got = list(out.values())[0]
    print(f"FP32 IR vs PyTorch: max abs diff {np.abs(got-ref).max():.2e}")

    if not a.no_int8:
        import nncf
        samples = calibration_data(a.calib)
        print(f"INT8 PTQ over {len(samples)} calibration samples "
              f"({'recorded' if os.path.exists(a.calib) else 'synthetic'})")
        ds = nncf.Dataset(samples, lambda s: s)
        int8_src = ov.convert_model(traced, example_input=ex,
                                    input=[("imgs", [1, N_CAMS, 3, IMG, IMG]),
                                           ("state", [1, STATE_DIM]),
                                           ("subtask", [1])])
        q = nncf.quantize(int8_src, ds, subset_size=len(samples),
                          preset=nncf.QuantizationPreset.MIXED,
                          model_type=nncf.ModelType.TRANSFORMER)
        int8 = os.path.join(a.out, "policy_int8.xml")
        ov.save_model(q, int8)
        out = core.compile_model(int8, "CPU")(dict(
            imgs=ex[0].numpy(), state=ex[1].numpy(), subtask=ex[2].numpy()))
        g8 = list(out.values())[0]
        print("wrote", int8,
              f"| INT8 vs PyTorch: max abs diff {np.abs(g8-ref).max():.2e}")

    for f in sorted(os.listdir(a.out)):
        if f.endswith(".bin"):
            print(f"  {f:24s} {os.path.getsize(os.path.join(a.out,f))/1e6:7.2f} MB")


if __name__ == "__main__":
    main()
