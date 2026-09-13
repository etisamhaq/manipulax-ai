"""Concatenate the per-seed clips into the single demonstration video.

Insurance: run_seeds writes each seed's clip as it finishes but only writes the
combined video at the very end, so an interrupted run leaves the clips without
the reel.  This rebuilds it from whatever clips exist.
"""
from __future__ import annotations

import glob
import os
import sys

import imageio.v2 as imageio


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "artifacts/eval"
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    clips = sorted(glob.glob(os.path.join(d, "seed_*.mp4")))
    if not clips:
        raise SystemExit(f"no seed_*.mp4 in {d}")
    out = os.path.join(d, "demo_all_seeds.mp4")
    w = imageio.get_writer(out, fps=fps, quality=7, macro_block_size=1)
    total = 0
    for c in clips:
        last = None
        for frame in imageio.get_reader(c):
            w.append_data(frame)
            last = frame
            total += 1
        for _ in range(fps):          # hold the final state of each seed
            w.append_data(last)
            total += 1
        print(f"  + {os.path.basename(c)}")
    w.close()
    print(f"wrote {out} ({len(clips)} clips, {total} frames, "
          f"{os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
