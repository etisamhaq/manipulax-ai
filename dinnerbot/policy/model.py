"""A compact ACT-style language-conditioned visuomotor policy.

Design constraints for this project: it must train on a CPU in a reasonable
time and it must export to a *static-shape* OpenVINO IR (the NPU plugin will
not take dynamic shapes), so everything below is fixed-size and free of
control flow that depends on tensor values.

  inputs   3 x RGB 96x96   (overhead, left wrist, right wrist)
           state           24-D  (12 joint positions + 12 velocities)
           subtask id      int   (which of the 11 language sub-tasks is active)
  output   action chunk    K x 12 joint position targets

Action chunking (predicting K steps at once and executing them open-loop) is
what makes ACT robust to the compounding error of single-step behaviour
cloning, and it also means one inference call covers K control ticks -- which
is exactly what makes the latency budget on an edge CPU comfortable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

N_CAMS = 3
STATE_DIM = 24
ACTION_DIM = 12
N_SUBTASKS = 11
CHUNK = 16
IMG = 96


class CamEncoder(nn.Module):
    """Small shared CNN trunk; each camera gets its own learned view embedding."""

    def __init__(self, d_model=128):
        super().__init__()
        c = [3, 32, 64, 96, 128]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1),
                       nn.GroupNorm(8, c[i + 1]), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)          # 96 -> 6
        self.proj = nn.Conv2d(c[-1], d_model, 1)
        self.view_emb = nn.Parameter(torch.zeros(N_CAMS, 1, d_model))
        nn.init.normal_(self.view_emb, std=0.02)

    def forward(self, imgs):                        # (B, N_CAMS, 3, H, W)
        b, n = imgs.shape[0], imgs.shape[1]
        x = self.net(imgs.reshape(b * n, 3, IMG, IMG))
        x = self.proj(x)                            # (B*N, d, 6, 6)
        d = x.shape[1]
        x = x.flatten(2).transpose(1, 2)            # (B*N, 36, d)
        x = x.reshape(b, n, -1, d) + self.view_emb[None, :, :, :]
        return x.reshape(b, -1, d)                  # (B, N*36, d)


def sincos(n, d):
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class DinnerPolicy(nn.Module):
    def __init__(self, d_model=128, nhead=4, enc_layers=2, dec_layers=2,
                 chunk=CHUNK, dim_ff=384):
        super().__init__()
        self.chunk = chunk
        self.cams = CamEncoder(d_model)
        self.state_proj = nn.Linear(STATE_DIM, d_model)
        self.lang_emb = nn.Embedding(N_SUBTASKS, d_model)
        ntok = N_CAMS * 36 + 2
        self.register_buffer("pos", sincos(ntok, d_model).unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.query = nn.Parameter(torch.zeros(1, chunk, d_model))
        nn.init.normal_(self.query, std=0.02)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ACTION_DIM))

    def forward(self, imgs, state, subtask):
        """imgs (B,3,3,96,96) float 0..1 | state (B,24) | subtask (B,) int64."""
        tok = self.cams(imgs)
        s = self.state_proj(state).unsqueeze(1)
        l = self.lang_emb(subtask).unsqueeze(1)
        x = torch.cat([tok, s, l], dim=1) + self.pos
        mem = self.encoder(x)
        q = self.query.expand(x.shape[0], -1, -1)
        y = self.decoder(q, mem)
        return self.head(y)                          # (B, chunk, 12)


class ExportWrapper(nn.Module):
    """Fixed batch-1 wrapper with int64 subtask folded to a float index, so the
    graph contains no embedding-lookup op that some plugins dislike."""

    def __init__(self, policy):
        super().__init__()
        self.p = policy

    def forward(self, imgs, state, subtask):
        return self.p(imgs, state, subtask.to(torch.long).reshape(-1))


def make_policy(**kw):
    return DinnerPolicy(**kw)


if __name__ == "__main__":
    m = make_policy().eval()
    n = sum(p.numel() for p in m.parameters())
    x = torch.zeros(1, N_CAMS, 3, IMG, IMG)
    s = torch.zeros(1, STATE_DIM)
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        y = m(x, s, t)
    print(f"params {n/1e6:.2f}M   output {tuple(y.shape)}")
