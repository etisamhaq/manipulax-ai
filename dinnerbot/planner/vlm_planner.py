"""Language + vision -> bimanual plan.

Two backends, same interface:

  "vlm"        a small vision-language model (SmolVLM) compiled to OpenVINO IR
               and run on the Intel CPU / iGPU / NPU.  It sees the overhead
               camera frame and the operator's sentence and emits a JSON plan.
  "rules"      a deterministic parser over the same sentence.  It is the
               fallback when no VLM is present, and it is also the reference
               the VLM is scored against.

Whichever produced it, the plan goes through schema.validate() before it can
move a joint, and a failed sub-task triggers replan() with the updated scene.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from .schema import (OBJECTS, PROMPT, PlanError, parse_plan, reachable,
                     validate)

# Where each item belongs once the table is set.
HOME_OF = {"fork": "right of the mat", "plate": "centre of the mat",
           "mug": "left of the mat", "spoon": "right of the mat"}


def describe_scene(env):
    """A compact textual scene state.

    This reads object poses from the simulator.  On real hardware it would come
    from a detector on the same overhead frame the VLM already receives; it is
    privileged information here and is flagged as such in the README.
    """
    d = {}
    for o in ("plate", "mug", "fork", "spoon", "bottle"):
        p = env.body_pos(o)
        side = "left" if p[0] < -0.06 else ("right" if p[0] > 0.06 else "centre")
        d[o] = dict(xy=[round(float(p[0]), 3), round(float(p[1]), 3)], side=side)
    d["drawer"] = dict(open=bool(env.task.drawer_open), side="left")
    return d


def scene_sentence(state):
    bits = []
    for k, v in state.items():
        if k == "drawer":
            bits.append(f"the drawer is {'open' if v['open'] else 'closed'} on the left")
        else:
            bits.append(f"the {k} is on the {v['side']} at {v['xy']}")
    return "; ".join(bits)


# --------------------------------------------------------------------- rules
def rule_plan(command, state):
    """Deterministic parse of the command into the skill vocabulary."""
    c = command.lower()
    plan = []
    if "drawer" in c:
        plan.append(dict(skill="open_drawer", object="drawer", arm="left",
                         target=None, why="the cutlery is inside it"))
    for obj in ("fork", "spoon"):
        if obj in c:
            src = state.get(obj, {}).get("side", "left")
            dst = "right"                      # cutlery belongs right of the mat
            pick_arm = "left" if src in ("left", "centre") else "right"
            plan.append(dict(skill="pick", object=obj, arm=pick_arm, target=None,
                             why=f"the {obj} starts on the {src}"))
            if pick_arm != dst:
                plan.append(dict(skill="handoff", object=obj, arm="both", target=None,
                                 why=f"the {obj} must cross to the {dst} half"))
            plan.append(dict(skill="place", object=obj, arm=dst,
                             target=HOME_OF[obj], why="set the table"))
    if "plate" in c:
        plan.append(dict(skill="pick", object="plate", arm="right", target=None,
                         why="the plate starts on the right"))
        plan.append(dict(skill="place", object="plate", arm="right",
                         target=HOME_OF["plate"], why="set the table"))
    if "mug" in c or "cup" in c:
        plan.append(dict(skill="pick", object="mug", arm="left", target=None,
                         why="the mug starts on the left"))
    if "pour" in c or "water" in c:
        plan.append(dict(skill="pick", object="bottle", arm="right", target=None,
                         why="the bottle starts on the right"))
        plan.append(dict(skill="pour", object="bottle", arm="both", target="mug",
                         why="left holds the mug while right tips the bottle"))
    if "mug" in c or "cup" in c:
        plan.append(dict(skill="place", object="mug", arm="left",
                         target=HOME_OF["mug"], why="set the table"))
    return plan


# ----------------------------------------------------------------------- VLM
class OpenVinoVLM:
    """SmolVLM compiled to OpenVINO IR, running on an Intel device."""

    def __init__(self, model_dir, device="AUTO", max_new_tokens=340, image_px=224):
        from optimum.intel import OVModelForVisualCausalLM
        from transformers import AutoProcessor
        t0 = time.perf_counter()
        # SmolVLM tiles large images into many patches; each tile costs ~64
        # image tokens and dominates the latency, so feed it one modest tile.
        self.proc = AutoProcessor.from_pretrained(model_dir)
        if hasattr(self.proc, "image_processor"):
            self.proc.image_processor.do_image_splitting = False
        self.model = OVModelForVisualCausalLM.from_pretrained(model_dir, device=device)
        self.load_s = time.perf_counter() - t0
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.image_px = image_px
        self.last_latency_ms = 0.0

    def __call__(self, image, command, scene):
        from PIL import Image
        img = Image.fromarray(np.asarray(image)).convert("RGB").resize(
            (self.image_px, self.image_px))
        text = PROMPT.format(scene=scene_sentence(scene), command=command)
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": text}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.proc(text=prompt, images=[img], return_tensors="pt")
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.proc.batch_decode(gen, skip_special_tokens=True)[0]


# -------------------------------------------------------------------- facade
class Planner:
    def __init__(self, backend="rules", model_dir=None, device="AUTO"):
        self.backend = backend
        self.vlm = None
        self.stats = dict(calls=0, vlm_ok=0, vlm_invalid=0, fallbacks=0,
                          replans=0, latency_ms=[])
        if backend == "vlm":
            try:
                self.vlm = OpenVinoVLM(model_dir, device)
            except Exception as e:
                print(f"[planner] VLM unavailable ({type(e).__name__}: {e}); "
                      f"falling back to rules")
                self.backend = "rules"

    def plan(self, env, command, image=None):
        self.stats["calls"] += 1
        state = describe_scene(env)
        if self.vlm is not None and image is not None:
            try:
                raw = self.vlm(image, command, state)
                self.stats["latency_ms"].append(self.vlm.last_latency_ms)
                p = validate(parse_plan(raw), {k: v["xy"] for k, v in state.items()
                                               if "xy" in v})
                self.stats["vlm_ok"] += 1
                return p, raw
            except (PlanError, TypeError, ValueError, KeyError) as e:
                self.stats["vlm_invalid"] += 1
                self.stats["fallbacks"] += 1
                print(f"[planner] VLM plan rejected ({e}); using the rule plan")
        p = validate(rule_plan(command, state),
                     {k: v["xy"] for k, v in state.items() if "xy" in v})
        return p, None

    def replan(self, env, command, failed_step, image=None):
        """Re-derive the remainder of the plan after a sub-task failed.

        The scene has changed -- an object may have been knocked somewhere new,
        or the drawer may already be open -- so the new plan is built from the
        *current* state rather than resuming a stale script.
        """
        self.stats["replans"] += 1
        plan, raw = self.plan(env, command, image)
        return [s for s in plan if not (s["skill"] == failed_step.get("skill")
                                        and s["object"] == failed_step.get("object"))], raw
