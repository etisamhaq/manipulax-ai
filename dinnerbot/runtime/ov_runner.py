"""Closed-loop execution of the policy through OpenVINO.

The policy emits a 16-step action chunk, so one inference covers 16 control
ticks.  The runner executes the chunk open-loop and then re-infers, which is the
standard ACT deployment pattern and is what keeps the inference budget on an
edge CPU comfortable.
"""
from __future__ import annotations

import time

import numpy as np

from ..policy.model import CHUNK, IMG, N_CAMS
from ..sim.env import CAMERAS


class OVPolicyRunner:
    def __init__(self, xml, device="AUTO", ckpt=None, exec_frac=0.5):
        import openvino as ov
        core = ov.Core()
        t0 = time.perf_counter()
        self.model = core.compile_model(xml, device, {"PERFORMANCE_HINT": "LATENCY"})
        self.compile_s = time.perf_counter() - t0
        self.req = self.model.create_infer_request()
        self.device = device
        self.xml = xml
        self.latencies = []
        # execute only the first half of each chunk before re-planning: the tail
        # of a chunk is the least accurate part of it
        self.n_exec = max(1, int(CHUNK * exec_frac))
        self.s_mean = np.zeros(24, np.float32)
        self.s_std = np.ones(24, np.float32)
        self.a_mean = np.zeros(12, np.float32)
        self.a_std = np.ones(12, np.float32)
        if ckpt:
            import torch
            # weights_only=False: the checkpoint carries numpy normalisation stats
            n = torch.load(ckpt, map_location="cpu", weights_only=False)["norm"]
            self.s_mean, self.s_std = n["s_mean"].astype(np.float32), n["s_std"].astype(np.float32)
            self.a_mean, self.a_std = n["a_mean"].astype(np.float32), n["a_std"].astype(np.float32)

    @property
    def mean_latency_ms(self):
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    def infer(self, obs, subtask_id):
        imgs = np.stack([obs["images"][c].transpose(2, 0, 1) for c in CAMERAS])
        imgs = imgs[None].astype(np.float32) / 255.0
        state = ((obs["state"] - self.s_mean) / self.s_std)[None].astype(np.float32)
        sub = np.array([subtask_id], np.int64)
        t0 = time.perf_counter()
        out = self.req.infer(dict(imgs=imgs, state=state, subtask=sub))
        self.latencies.append((time.perf_counter() - t0) * 1000.0)
        chunk = list(out.values())[0][0]
        return chunk * self.a_std[None, :] + self.a_mean[None, :]

    def run_subtask(self, env, subtask_id, max_ticks=240, on_step=None):
        obs = env.obs()
        for _ in range(0, max_ticks, self.n_exec):
            chunk = self.infer(obs, subtask_id)
            for k in range(self.n_exec):
                obs = env.step(chunk[k])
                if on_step is not None:
                    on_step()
        return obs
