"""Run the scripted expert on one or more seeds and print the sub-task report."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dinnerbot.sim.env import DinnerTableEnv
from dinnerbot.expert.primitives import Executor
from dinnerbot.expert.scripted_task import run_expert
from dinnerbot.sim.build_scene import FORK_TARGET

for seed in [int(s) for s in sys.argv[1:]] or [0]:
    t0 = time.time()
    env = DinnerTableEnv(seed=seed, dr_level=1.0, img_size=(64, 64))
    ex = Executor(env, record=False, frame_cams=())
    rep = run_expert(ex, verbose=True)
    fork = env.body_pos("fork")
    print(f"seed {seed}: {rep['n_done']}/6  water={rep['water_in_mug']}  "
          f"fork={np.round(fork,3)} target={FORK_TARGET} "
          f"err={np.linalg.norm(fork[:2]-np.array(FORK_TARGET))*1000:.0f}mm  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print("  ", rep["task"], flush=True)
    env.close()
