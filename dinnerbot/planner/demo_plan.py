"""Run the planner on a randomised scene and show the plan it produces.

    python -m dinnerbot.planner.demo_plan --backend vlm --device CPU
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..sim.env import DinnerTableEnv
from .schema import validate
from .vlm_planner import (Planner, describe_scene, rule_plan,
                          scene_sentence)

COMMANDS = [
    "Open the drawer, put the fork on the right of the mat, centre the plate, "
    "then hold the mug and pour water into it.",
    "Set the table: plate in the middle, fork on the right, mug on the left.",
    "Take the fork out of the drawer and lay it on the right side.",
    "Pick up the mug and pour some water into it.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="rules", choices=("rules", "vlm"))
    ap.add_argument("--model", default="artifacts/vlm/smolvlm-500m-int8")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--command", default=None)
    a = ap.parse_args()

    p = Planner(a.backend, a.model, a.device)
    print(f"backend={p.backend} device={a.device}")
    if p.vlm is not None:
        print(f"VLM load time {p.vlm.load_s:.1f}s")

    cmds = [a.command] if a.command else COMMANDS
    all_plans, rule_plans = [], []
    for seed in [int(s) for s in a.seeds.split(",")]:
        env = DinnerTableEnv(seed=seed, dr_level=1.0, img_size=(96, 96))
        img = env.render("overhead", (384, 384))
        print(f"\n=== seed {seed} | scene: {scene_sentence(describe_scene(env))}")
        for c in cmds:
            t0 = time.perf_counter()
            plan, raw = p.plan(env, c, img)
            dt = (time.perf_counter() - t0) * 1000
            all_plans.append(plan)
            rule_plans.append(validate(rule_plan(c, describe_scene(env))))
            print(f'\n  command: "{c}"')
            if raw is not None:
                print(f"  raw VLM reply: {raw[:220]!r}")
            print(f"  plan ({len(plan)} steps, {dt:.0f} ms):")
            for s in plan:
                tgt = f" -> {s['target']}" if s["target"] else ""
                print(f"    {s['skill']:<11} {str(s['object']):<7} [{s['arm']:<5}]{tgt}"
                      f"   ({s['why']})")
        env.close()

    # Does the planner actually respond to the command, or is it echoing the
    # exemplar in the prompt?  A small VLM will happily do the latter, and a
    # "valid JSON" success count hides it completely.
    if len(cmds) > 1 and all_plans:
        sigs = {json.dumps([(s["skill"], s["object"], s["arm"]) for s in pl])
                for pl in all_plans}
        print(f"\nplan distinctiveness: {len(sigs)} distinct plans for "
              f"{len(all_plans)} different commands"
              + ("  <-- IDENTICAL: the model is echoing the prompt exemplar, "
                 "not reading the command" if len(sigs) == 1 else ""))
        rule_match = sum(1 for pl, rp in zip(all_plans, rule_plans)
                         if [(x["skill"], x["object"]) for x in pl]
                         == [(y["skill"], y["object"]) for y in rp])
        print(f"agreement with the deterministic rule planner: "
              f"{rule_match}/{len(all_plans)}")

    if p.stats["latency_ms"]:
        l = np.array(p.stats["latency_ms"])
        print(f"\nVLM latency: mean {l.mean():.0f} ms  p90 {np.percentile(l,90):.0f} ms "
              f"over {len(l)} calls")
    print(f"stats: {json.dumps({k:v for k,v in p.stats.items() if k!='latency_ms'})}")


if __name__ == "__main__":
    main()
