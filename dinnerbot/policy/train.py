"""Behaviour cloning for the language-conditioned action-chunk policy.

Trained on the scripted expert's demonstrations.  Loss is L1 over the whole
16-step chunk (ACT's choice -- L1 tolerates the multi-modality of human-ish
demonstrations far better than L2, which averages two valid trajectories into an
invalid one between them).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .model import ACTION_DIM, CHUNK, IMG, N_CAMS, STATE_DIM, make_policy


class ChunkDataset(torch.utils.data.Dataset):
    """Each item is (3 images, state, subtask id) -> the next CHUNK actions."""

    def __init__(self, path, chunk=CHUNK, stride=1):
        d = np.load(path)
        self.img = d["images"]              # (N, 3, 3, H, W) uint8
        self.state = d["state"].astype(np.float32)
        self.action = d["action"].astype(np.float32)
        self.sub = d["subtask"].astype(np.int64)
        self.ep = d["episode"]
        self.chunk = chunk
        # never let a chunk run off the end of its episode
        ok = []
        for i in range(len(self.state)):
            j = min(i + chunk, len(self.state))
            if self.ep[j - 1] == self.ep[i]:
                ok.append(i)
        self.index = np.array(ok)[::max(1, stride)]
        self.s_mean = self.state.mean(0)
        self.s_std = self.state.std(0) + 1e-3
        self.a_mean = self.action.mean(0)
        self.a_std = self.action.std(0) + 1e-3

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = int(self.index[k])
        a = np.zeros((self.chunk, ACTION_DIM), np.float32)
        seg = self.action[i:i + self.chunk]
        a[:len(seg)] = seg
        if len(seg) < self.chunk:
            a[len(seg):] = seg[-1]
        img = self.img[i].astype(np.float32) / 255.0
        return (torch.from_numpy(img),
                torch.from_numpy((self.state[i] - self.s_mean) / self.s_std),
                torch.tensor(self.sub[i]),
                torch.from_numpy((a - self.a_mean) / self.a_std))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/dataset.npz")
    ap.add_argument("--out", default="artifacts/policy.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth chunk start; trades epochs for wall time")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers; 0 avoids forking a multi-GB process")
    a = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(a.threads or (os.cpu_count() or 8))

    ds = ChunkDataset(a.data, stride=a.stride)
    n_val = max(1, int(len(ds) * a.val_frac))
    g = torch.Generator().manual_seed(0)
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True,
                                     num_workers=a.workers, drop_last=True)
    vl = torch.utils.data.DataLoader(va, batch_size=a.batch, num_workers=a.workers)
    print(f"{len(ds)} chunks ({len(tr)} train / {len(va)} val) from {a.data}")

    model = make_policy()
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(1, a.epochs * len(dl)))
    lossfn = nn.L1Loss()
    print(f"policy {n/1e6:.2f}M params | {torch.get_num_threads()} CPU threads")

    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        model.train()
        run = k = 0
        for img, st, sub, act in dl:
            pred = model(img, st, sub)
            loss = lossfn(pred, act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item(); k += 1
            if k % 25 == 0:
                print(f"  ep{ep+1} step {k}/{len(dl)} loss {run/k:.4f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        model.eval()
        vs = vk = 0
        with torch.no_grad():
            for img, st, sub, act in vl:
                vs += lossfn(model(img, st, sub), act).item(); vk += 1
        hist.append(dict(epoch=ep + 1, train=run / max(k, 1), val=vs / max(vk, 1)))
        print(f"epoch {ep+1}/{a.epochs}  train {run/max(k,1):.4f}  "
              f"val {vs/max(vk,1):.4f}  [{time.time()-t0:.0f}s]", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(dict(model=model.state_dict(),
                    norm=dict(s_mean=ds.s_mean, s_std=ds.s_std,
                              a_mean=ds.a_mean, a_std=ds.a_std),
                    history=hist, params=n), a.out)
    with open(a.out.replace(".pt", "_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"saved {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB, "
          f"{time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
