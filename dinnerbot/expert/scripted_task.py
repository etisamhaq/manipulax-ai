"""The scripted bimanual expert for 'set the dinner table'.

Arm assignment follows the scene layout: the cutlery drawer is on the LEFT, so
the left arm opens it and takes the fork out -- but the fork belongs on the
RIGHT of the placemat.  The only way to finish is to pass it across, which is
the hand-off the rubric asks for.  The pour is the second coordination moment:
the left arm holds the mug up while the right arm tips the bottle into it.

  1  open_drawer    left    pull the cutlery drawer open
  2  pick_fork      left    take the fork out of the drawer
  3  handoff_fork   both    left passes the fork to the right arm
  4  place_fork     right   lay the fork right of the mat
  5  pick_plate     right   take the plate
  6  place_plate    right   centre it on the mat
  7  pick_mug       left    take the mug and hold it up
  8  pick_bottle    right   take the bottle
  9  pour           both    right pours while left holds the mug
 10  place_mug      left    set the mug down left of the mat
"""
from __future__ import annotations

import numpy as np

from ..sim.build_scene import PLATE_TARGET, MUG_TARGET, FORK_TARGET
from .primitives import Executor

# Biased towards the receiving (right) arm rather than dead centre: with the
# meeting point in the middle both forearms have to occupy the same volume, and
# the taker ends up 80 mm short of its target because it is physically blocked.
HANDOFF = np.array([0.055, 0.02, 0.105])
POUR_POINT = np.array([-0.02, -0.02, 0.115])

SUBTASK_LANG = {
    "open_drawer": "open the cutlery drawer",
    "pick_fork": "pick up the fork from the drawer",
    "handoff_fork": "pass the fork from the left arm to the right arm",
    "place_fork": "place the fork to the right of the placemat",
    "pick_plate": "pick up the plate",
    "place_plate": "place the plate in the centre of the placemat",
    "pick_mug": "pick up the mug and hold it up",
    "pick_bottle": "pick up the bottle",
    "pour": "pour water into the mug the other arm is holding",
    "place_mug": "place the mug to the left of the placemat",
    "idle": "wait",
}
SUBTASKS = [k for k in SUBTASK_LANG if k != "idle"]
SUBTASK_ID = {k: i for i, k in enumerate(SUBTASK_LANG)}

# which arm owns each sub-task (used by the coordinator and the dataset)
SUBTASK_ARM = {
    "open_drawer": "left", "pick_fork": "left", "handoff_fork": "both",
    "place_fork": "right", "pick_plate": "right", "place_plate": "right",
    "pick_mug": "left", "pick_bottle": "right", "pour": "both",
    "place_mug": "left", "idle": "none",
}


def run_expert(ex: Executor, verbose=False):
    env = ex.env
    rep = {}

    def step(name, fn):
        arm = SUBTASK_ARM[name]
        ex.begin(name, arm)
        try:
            ok = bool(fn())
        except Exception as exc:
            ok, ex.aborted = False, f"{name}: {exc}"
        rep[name] = ok
        if verbose:
            print(f"   {name:14s} [{arm:5s}] -> {ok}")
        return ok

    def placed(obj, target, tol=0.065):
        return float(np.linalg.norm(env.body_pos(obj)[:2] - np.array(target))) < tol

    # 1-4 ------------------------------------ drawer, fork, cross-arm hand-off
    step("open_drawer", lambda: ex.open_drawer("left"))
    if step("pick_fork", lambda: ex.pick("left", "fork")):
        ho = step("handoff_fork", lambda: ex.handoff("left", "right", "fork", HANDOFF))
        env.task.handoff_done = env.task.handoff_done or ho
        if ho:
            step("place_fork",
                 lambda: (ex.place("right", FORK_TARGET, z=0.010), placed("fork", FORK_TARGET))[1])
            ex.home("right", steps=35)
    else:
        ex.home("left", steps=35)

    # 5-6 ------------------------------------------------------------- plate
    if step("pick_plate", lambda: ex.pick("right", "plate")):
        step("place_plate",
             lambda: (ex.place("right", PLATE_TARGET, z=0.010), placed("plate", PLATE_TARGET))[1])
    ex.home("right", steps=35)

    # 7-9 ------------------------------------ mug held aloft while the other pours
    got_mug = step("pick_mug", lambda: ex.pick("left", "mug"))
    if got_mug:
        ex.begin("pick_mug", "left")
        ex.reach("left", POUR_POINT, steps=45)

    if step("pick_bottle", lambda: ex.pick("right", "bottle")):
        if got_mug and env.held["left"] == "mug":
            mug_xy = env.body_pos("mug")[:2]
            step("pour", lambda: (ex.pour("right", mug_xy), env.water_in_mug() >= 3)[1])
            env.task.water_poured = env.task.water_poured or env.water_in_mug() >= 3
        ex.begin("pick_bottle", "right")
        bp = env.params.bottle_pose
        ex.place("right", (bp[0], bp[1]), z=0.065)
        ex.home("right", steps=35)

    # 10 -------------------------------------------------------- set the mug
    if got_mug and env.held["left"] == "mug":
        step("place_mug",
             lambda: (ex.place("left", MUG_TARGET, z=0.012), placed("mug", MUG_TARGET))[1])
    ex.home("left", steps=35)
    ex.hold(15)

    rep["aborted"] = ex.aborted        # first exception, if any -- keeps a
    rep["task"] = env.task.score()     # zero-score episode diagnosable
    rep["n_done"] = env.task.n_done
    rep["water_in_mug"] = env.water_in_mug()
    return rep
