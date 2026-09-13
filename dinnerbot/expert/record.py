"""Record scripted-expert demonstrations into a LeRobot-style flat dataset.

Each frame carries the three camera images, the proprioceptive state, the
action the expert commanded, and the *language* annotation of the sub-task in
progress -- that last field is what makes the policy language-conditioned
rather than a blind trajectory replayer.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

class _NpEncoder(json.JSONEncoder):
    """numpy scalars and arrays are not JSON-serialisable by default."""

    def default(self, o):
        import numpy as _np
        if isinstance(o, _np.bool_):
            return bool(o)
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        return super().default(o)


from ..sim.env import CAMERAS, DinnerTableEnv
from .primitives import Executor
from .scripted_task import SUBTASK_ID, SUBTASK_LANG, run_expert


def record_episode(seed, dr_level=1.0, img=96):
    env = DinnerTableEnv(seed=seed, dr_level=dr_level, img_size=(img, img))
    ex = Executor(env, record=True, frame_cams=())
    rep = run_expert(ex)
    n = len(ex.traj)
    imgs = np.zeros((n, len(CAMERAS), 3, img, img), np.uint8)
    state = np.zeros((n, 24), np.float32)
    action = np.zeros((n, 12), np.float32)
    sub = np.zeros(n, np.int64)
    for i, t in enumerate(ex.traj):
        for c, cam in enumerate(CAMERAS):
            imgs[i, c] = t["images"][cam].transpose(2, 0, 1)
        state[i] = t["state"]
        action[i] = t["action"]
        sub[i] = SUBTASK_ID.get(t["subtask"], SUBTASK_ID["idle"])
    env.close()
    return dict(images=imgs, state=state, action=action, subtask=sub,
                episode=np.full(n, seed, np.int64)), rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--dr", type=float, default=1.0)
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--out", default="artifacts/dataset.npz")
    ap.add_argument("--min-subtasks", type=int, default=3,
                    help="drop episodes where the expert achieved fewer than this")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    keep, reports, t0 = [], [], time.time()
    for k in range(a.episodes):
        seed = a.start_seed + k
        ep, rep = record_episode(seed, a.dr, a.img)
        reports.append({"seed": seed, "n_done": rep["n_done"], **rep["task"]})
        if rep["n_done"] >= a.min_subtasks:
            keep.append(ep)
            tag = "keep"
        else:
            tag = "drop"
        print(f"  ep {k+1:3d}/{a.episodes} seed {seed} n_done={rep['n_done']} "
              f"frames={len(ep['state'])} {tag}  [{time.time()-t0:.0f}s]", flush=True)

    if not keep:
        raise SystemExit("no usable episodes recorded")
    data = {k: np.concatenate([e[k] for e in keep]) for k in keep[0]}
    np.savez_compressed(a.out, **data)
    # a small calibration slice for NNCF, kept separate so quantisation never
    # sees the frames it will later be evaluated on
    idx = np.linspace(0, len(data["state"]) - 1, min(96, len(data["state"]))).astype(int)
    np.savez_compressed(a.out.replace("dataset", "calib"),
                        images=data["images"][idx], state=data["state"][idx],
                        subtask=data["subtask"][idx])
    with open(a.out.replace(".npz", "_report.json"), "w") as f:
        json.dump({"episodes": reports, "kept": len(keep),
                   "frames": int(len(data["state"])),
                   "subtask_lang": SUBTASK_LANG}, f, indent=2, cls=_NpEncoder)
    print(f"\nkept {len(keep)}/{a.episodes} episodes, {len(data['state'])} frames "
          f"-> {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
