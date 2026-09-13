"""Intel inference benchmark for the dinner-table policy.

Sweeps {device} x {precision}, reporting compile time, latency percentiles,
throughput, and the control rate the policy can sustain.  Because the policy
emits an action chunk of CHUNK steps per inference, the sustainable control
rate is CHUNK / latency -- that is the number that actually matters for whether
the robot runs in real time, so it is reported alongside raw latency.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np

from ..policy.model import ACTION_DIM, CHUNK, IMG, N_CAMS, STATE_DIM

PRECISIONS = ("fp32", "fp16", "int8")


def cpu_name():
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        for line in out.splitlines():
            if line.strip().startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def make_inputs(rng):
    return dict(imgs=rng.random((1, N_CAMS, 3, IMG, IMG), dtype=np.float32),
                state=rng.normal(0, 1, (1, STATE_DIM)).astype(np.float32),
                subtask=rng.integers(0, 11, (1,)).astype(np.int64))


def bench_one(core, xml, device, iters=120, warmup=20, hint="LATENCY"):
    import openvino as ov
    rng = np.random.default_rng(0)
    cfg = {"PERFORMANCE_HINT": hint}
    t0 = time.perf_counter()
    compiled = core.compile_model(xml, device, cfg)
    compile_s = time.perf_counter() - t0
    req = compiled.create_infer_request()
    x = make_inputs(rng)
    for _ in range(warmup):
        req.infer(x)
    lat = []
    for _ in range(iters):
        x = make_inputs(rng)
        t = time.perf_counter()
        req.infer(x)
        lat.append((time.perf_counter() - t) * 1000.0)
    lat = np.array(lat)
    return dict(device=device, compile_s=round(compile_s, 3),
                p50_ms=round(float(np.percentile(lat, 50)), 3),
                p90_ms=round(float(np.percentile(lat, 90)), 3),
                p99_ms=round(float(np.percentile(lat, 99)), 3),
                mean_ms=round(float(lat.mean()), 3),
                fps=round(1000.0 / float(np.percentile(lat, 50)), 1),
                control_hz=round(CHUNK * 1000.0 / float(np.percentile(lat, 50)), 1))


def torch_baseline(iters=60):
    import torch
    from ..policy.model import make_policy
    m = make_policy().eval()
    torch.set_num_threads(os.cpu_count() or 4)
    x = (torch.zeros(1, N_CAMS, 3, IMG, IMG), torch.zeros(1, STATE_DIM),
         torch.zeros(1, dtype=torch.long))
    with torch.no_grad():
        for _ in range(10):
            m(*x)
        lat = []
        for _ in range(iters):
            t = time.perf_counter()
            m(*x)
            lat.append((time.perf_counter() - t) * 1000.0)
    lat = np.array(lat)
    return dict(device="PyTorch-CPU (baseline)", compile_s=0.0,
                p50_ms=round(float(np.percentile(lat, 50)), 3),
                p90_ms=round(float(np.percentile(lat, 90)), 3),
                p99_ms=round(float(np.percentile(lat, 99)), 3),
                mean_ms=round(float(lat.mean()), 3),
                fps=round(1000.0 / float(np.percentile(lat, 50)), 1),
                control_hz=round(CHUNK * 1000.0 / float(np.percentile(lat, 50)), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir-dir", default="artifacts/ov")
    ap.add_argument("--devices", default="auto",
                    help="comma list, or 'auto' for every device OpenVINO reports")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--out", default="artifacts/benchmark")
    ap.add_argument("--skip-torch", action="store_true")
    a = ap.parse_args()

    import openvino as ov
    core = ov.Core()
    avail = core.available_devices
    devices = avail if a.devices == "auto" else [d.strip() for d in a.devices.split(",")]
    devices = [d for d in devices if d.split(".")[0] in {x.split(".")[0] for x in avail}]

    print(f"host        : {cpu_name()}")
    print(f"openvino    : {ov.__version__}")
    print(f"devices     : {avail}")
    for d in avail:
        try:
            print(f"  {d:6s} -> {core.get_property(d, 'FULL_DEVICE_NAME')}")
        except Exception:
            pass
    print()

    rows = []
    if not a.skip_torch:
        rows.append(dict(precision="fp32", **torch_baseline()))

    for prec in PRECISIONS:
        xml = os.path.join(a.ir_dir, f"policy_{prec}.xml")
        if not os.path.exists(xml):
            continue
        for dev in devices:
            try:
                r = bench_one(core, xml, dev, iters=a.iters)
                r["precision"] = prec
                r["size_mb"] = round(os.path.getsize(xml.replace(".xml", ".bin")) / 1e6, 2)
                rows.append(r)
                print(f"  {dev:5s} {prec:5s}  p50 {r['p50_ms']:7.2f} ms  "
                      f"p99 {r['p99_ms']:7.2f} ms  {r['fps']:6.1f} inf/s  "
                      f"{r['control_hz']:7.1f} Hz control")
            except Exception as e:
                print(f"  {dev:5s} {prec:5s}  UNSUPPORTED: {str(e).splitlines()[0][:90]}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    meta = dict(host=cpu_name(), openvino=ov.__version__, devices=avail,
                chunk=CHUNK, image=IMG, cameras=N_CAMS, rows=rows)
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    base = next((r for r in rows if r["device"].startswith("PyTorch")), None)
    lines = ["# Intel inference benchmark", "",
             f"- host: `{meta['host']}`", f"- OpenVINO: `{meta['openvino']}`",
             f"- devices reported: `{', '.join(avail)}`",
             f"- policy emits a {CHUNK}-step action chunk per inference, so the "
             f"sustainable control rate is {CHUNK} x inferences/s", "",
             "| device | precision | p50 ms | p90 ms | p99 ms | inf/s | control Hz | speedup vs PyTorch | weights MB |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sp = f"{base['p50_ms']/r['p50_ms']:.2f}x" if base and r["p50_ms"] > 0 else "-"
        lines.append(f"| {r['device']} | {r['precision'].upper()} | {r['p50_ms']:.2f} | "
                     f"{r['p90_ms']:.2f} | {r['p99_ms']:.2f} | {r['fps']:.1f} | "
                     f"{r['control_hz']:.1f} | {sp} | {r.get('size_mb','-')} |")
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}.json and {a.out}.md")


if __name__ == "__main__":
    main()
