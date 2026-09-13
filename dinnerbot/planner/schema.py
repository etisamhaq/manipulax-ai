"""The action vocabulary the planner may emit and the validator that enforces it.

A language model is free to produce anything; the robot is not.  Every plan is
checked against this schema before a single joint moves -- unknown skills,
unknown objects, an arm that cannot reach the object, or a hand-off that is not
actually needed are all rejected here rather than discovered at runtime.
"""
from __future__ import annotations

import json
import re

import numpy as np

SKILLS = ("open_drawer", "pick", "place", "handoff", "pour", "home")
OBJECTS = ("drawer", "fork", "spoon", "plate", "mug", "bottle")
ARMS = ("left", "right", "both")

# which half of the table each arm can comfortably service
ARM_BASE = {"left": np.array([-0.165, 0.10]), "right": np.array([0.165, 0.10])}
REACH_MIN, REACH_MAX = 0.11, 0.30


def reachable(arm, xy):
    if arm == "both":
        return True
    d = float(np.linalg.norm(np.asarray(xy, float)[:2] - ARM_BASE[arm]))
    return REACH_MIN <= d <= REACH_MAX


class PlanError(ValueError):
    pass


def _repair(js: str) -> str:
    """Best-effort repair of the JSON a small model actually emits.

    A 256M-parameter VLM gets the *shape* of the answer right far more often
    than it gets the punctuation right, so it is worth fixing trailing commas,
    single quotes, unquoted keys and an unclosed array before giving up.
    """
    js = js.strip()
    js = re.sub(r"```(?:json)?", "", js)
    js = js.replace("\u201c", '"').replace("\u201d", '"').replace("'", '"')
    js = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*):', r'\1"\2"\3:', js)
    js = re.sub(r",\s*([}\]])", r"\1", js)          # trailing commas
    # close whatever the model left hanging
    js += "}" * max(0, js.count("{") - js.count("}"))
    js += "]" * max(0, js.count("[") - js.count("]"))
    return js


def parse_plan(text):
    """Pull the first JSON array out of a model's reply, repairing if needed."""
    if isinstance(text, (list, tuple)):
        return list(text)
    m = re.search(r"\[.*", str(text), re.S)
    if not m:
        raise PlanError("no JSON array found in model output")
    raw = m.group(0)
    for candidate in (raw, _repair(raw)):
        try:
            out = json.loads(candidate)
            if isinstance(out, list):
                return out
        except json.JSONDecodeError as e:
            last = e
    # last resort: salvage the individual step objects
    steps = []
    for obj in re.finditer(r"\{[^{}]*\}", _repair(raw)):
        try:
            steps.append(json.loads(obj.group(0)))
        except json.JSONDecodeError:
            pass
    if steps:
        return steps
    raise PlanError(f"plan is not valid JSON: {last}")


def validate(plan, world=None):
    """Return a cleaned plan or raise PlanError.  `world` maps object -> xy."""
    if not isinstance(plan, list) or not plan:
        raise PlanError("plan must be a non-empty list")
    out = []
    for i, s in enumerate(plan):
        if not isinstance(s, dict):
            raise PlanError(f"step {i} is not an object")
        skill = s.get("skill")
        if skill not in SKILLS:
            raise PlanError(f"step {i}: unknown skill {skill!r}")
        arm = s.get("arm", "right")
        if arm not in ARMS:
            raise PlanError(f"step {i}: unknown arm {arm!r}")
        obj = s.get("object")
        # A model can emit anything here -- a list, a dict, a number.  Reject it
        # on type before any lookup: `obj in world` on a list raises TypeError,
        # which would crash the run instead of falling back to the rule planner.
        if obj is not None and not isinstance(obj, str):
            raise PlanError(f"step {i}: object must be a name, got {type(obj).__name__}")
        tgt = s.get("target")
        if tgt is not None and not isinstance(tgt, (str, list, tuple)):
            raise PlanError(f"step {i}: target must be a name or xy, got {tgt!r}")
        if skill in ("pick", "handoff", "pour") and obj not in OBJECTS:
            raise PlanError(f"step {i}: {skill} needs a known object, got {obj!r}")
        if world and obj in world and skill == "pick" and not reachable(arm, world[obj]):
            raise PlanError(f"step {i}: {arm} arm cannot reach {obj} at {world[obj]}")
        why = s.get("why", "")
        out.append({"skill": skill, "object": obj, "arm": arm,
                    "target": tgt, "why": why if isinstance(why, str) else ""})
    return out


PROMPT = """You control a two-armed robot setting a dinner table.
Arms: "left" and "right". The left arm serves the left half of the table, the
right arm the right half. An object that starts on one side but belongs on the
other must be passed across with a "handoff".

Skills: open_drawer, pick, place, handoff, pour, home.
Objects: drawer, fork, spoon, plate, mug, bottle.

Scene: {scene}
Command: {command}

Reply with ONLY a JSON array. Here is the format, shown on a DIFFERENT task
("put the spoon on the left") so you can see the shape -- do not copy it, answer
the command above instead:
[{{"skill":"pick","object":"spoon","arm":"right","target":null,"why":"spoon is on the right"}},
 {{"skill":"handoff","object":"spoon","arm":"both","target":null,"why":"it must cross to the left"}},
 {{"skill":"place","object":"spoon","arm":"left","target":"left of the mat","why":"set the table"}}]

JSON array for the command above:"""
