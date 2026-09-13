"""Bimanual dinner-table environment (two SO-ARM100 arms in MuJoCo).

Observation = 3 RGB cameras (overhead + one wrist camera per arm) plus the
24-D proprioceptive vector (12 joint positions, 12 joint velocities).
Action     = 12 joint position targets (6 per arm, the 6th being the jaw).

Grasping uses a *pinch detector + weld* model: when an arm's jaw is commanded
closed and a graspable body lies between the finger pads, a weld equality is
activated.  This is a deliberate simplification -- see README "Simplifications".
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

from .build_scene import (SceneParams, randomize, write_scene, GRASPABLE,
                          PLATE_TARGET, MUG_TARGET, FORK_TARGET, DRAWER_TRAVEL)

ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
ARMS = ("left", "right")
HOME = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])
JAW_OPEN, JAW_CLOSED = 1.30, -0.17
CAMERAS = ("overhead", "left_wrist", "right_wrist")


@dataclass
class TaskState:
    drawer_open: bool = False
    fork_placed: bool = False
    plate_placed: bool = False
    mug_placed: bool = False
    water_poured: bool = False
    handoff_done: bool = False

    def score(self):
        # plain bools, not numpy bools: json.dump refuses np.bool_, and these
        # dicts end up in every report the pipeline writes
        return dict(drawer_open=bool(self.drawer_open),
                    fork_placed=bool(self.fork_placed),
                    plate_placed=bool(self.plate_placed),
                    mug_placed=bool(self.mug_placed),
                    water_poured=bool(self.water_poured),
                    handoff_done=bool(self.handoff_done))

    @property
    def n_done(self):
        return int(sum(self.score().values()))


class DinnerTableEnv:
    SUBTASKS = 6

    def __init__(self, seed=0, dr_level=1.0, img_size=(128, 128), n_substeps=10,
                 grasp_assist=True, cameras=CAMERAS):
        self.img_size = img_size
        self.n_substeps = n_substeps
        self.grasp_assist = grasp_assist
        self.cameras = list(cameras)
        self.dr_level = dr_level
        self._renderers = {}
        self._rend_model = None
        self.reset(seed, dr_level)

    # ------------------------------------------------------------------ setup
    def reset(self, seed=None, dr_level=None):
        if seed is not None:
            self.seed = int(seed)
        if dr_level is not None:
            self.dr_level = dr_level
        self.params: SceneParams = randomize(self.seed, self.dr_level)
        path = write_scene(self.params)
        self.model = mujoco.MjModel.from_xml_path(path)
        try:
            os.remove(path)
        except OSError:
            pass
        self.data = mujoco.MjData(self.model)

        m = self.model
        self.act_id = {a: np.array([m.actuator(f"{a}_{j}").id for j in ARM_JOINTS])
                       for a in ARMS}
        self.qadr = {a: np.array([m.jnt_qposadr[m.joint(f"{a}_{j}").id] for j in ARM_JOINTS])
                     for a in ARMS}
        self.dadr = {a: np.array([m.jnt_dofadr[m.joint(f"{a}_{j}").id] for j in ARM_JOINTS])
                     for a in ARMS}
        self.eq_id = {(a, o): m.equality(f"{a}__{o}").id for a in ARMS for o in GRASPABLE}
        self.drawer_qadr = m.jnt_qposadr[m.joint("drawer_slide").id]
        # Named MuJoCo lookups cost a string hash each; the grasp and pour
        # checks run every sub-step, so every id they need is resolved once here.
        self._tcp_sid = {a: m.site(f"{a}_tcp").id for a in ARMS}
        self._jaw_bid = {a: m.body(f"{a}_Fixed_Jaw").id for a in ARMS}
        self._obj_bid = {o: m.body(o).id for o in self.WELDABLE}
        self._spout_sid = m.site("spout").id
        self._bottle_bid = m.body("bottle").id
        self._mug_bid = m.body("mug").id
        self._water_adr = [m.jnt_qposadr[m.joint(f"water{i}_j").id]
                           for i in range(self.params.n_water)]
        self._water_dadr = [m.jnt_dofadr[m.joint(f"water{i}_j").id]
                            for i in range(self.params.n_water)]
        self._jaw_act = {a: self.act_id[a][5] for a in ARMS}
        self.held = {a: None for a in ARMS}
        self.task = TaskState()
        self._water_free = list(range(self.params.n_water))
        self._water_out = []
        self._pour_cool = 0
        self.t = 0

        for a in ARMS:
            self.data.qpos[self.qadr[a]] = HOME
            self.data.ctrl[self.act_id[a]] = HOME
        self.data.qpos[self.qadr["left"][5]] = JAW_OPEN
        self.data.qpos[self.qadr["right"][5]] = JAW_OPEN
        self.data.ctrl[self.act_id["left"][5]] = JAW_OPEN
        self.data.ctrl[self.act_id["right"][5]] = JAW_OPEN
        mujoco.mj_forward(self.model, self.data)
        for _ in range(300):
            mujoco.mj_step(self.model, self.data)
        return self.obs()

    # ----------------------------------------------------------------- helpers
    def body_pos(self, name):
        return self.data.body(name).xpos.copy()

    def tcp(self, arm):
        return self.data.site_xpos[self._tcp_sid[arm]].copy()

    def jaw_gap(self, arm):
        f = self.data.geom(f"{arm}_fixed_jaw_pad_3").xpos
        g = self.data.geom(f"{arm}_moving_jaw_pad_3").xpos
        return float(np.linalg.norm(f - g))

    def qarm(self, arm):
        return self.data.qpos[self.qadr[arm]].copy()

    # ------------------------------------------------------------ grasp model
    # The drawer is deliberately excluded: it is pulled by real finger contact.
    # Welding a 6-DoF constraint onto its 1-DoF slide joint over-constrains the
    # solver and the drawer simply refuses to move.
    WELDABLE = tuple(o for o in GRASPABLE if o != "drawer")

    # Where on each object the fingers are meant to close.  A mug's body origin
    # sits at its base, so measuring the pinch from there reports the gripper as
    # a whole mug-height too far away.
    def grasp_point(self, obj):
        p = self.data.xpos[self._obj_bid[obj]].copy()
        if obj == "mug":
            p[2] += 0.45 * self.params.mug_h
        return p

    # Capture volume of the pinch detector: a short cylinder running along the
    # gripper's approach axis, not a sphere.  The jaws are ~100 mm long and open
    # to 104 mm, so an object a few centimetres ahead of the tool point really is
    # between the fingers, while one the same distance off to the side is not.
    # Pinch capture: the object must lie within CAPTURE_R of the tool point.
    # For an elongated object the distance is measured to the nearest point of
    # its graspable span, not to its body origin -- a fork may legitimately be
    # pinched anywhere along its handle, and during a hand-off it deliberately
    # is not pinched at its centre.
    CAPTURE_R = 0.052

    def approach_axis(self, arm):
        R = self.data.xmat[self._jaw_bid[arm]].reshape(3, 3)
        a = R @ np.array([-0.2108, -0.9775, 0.0])
        return a / np.linalg.norm(a)

    def _grasp_span(self, obj):
        """(unit long axis in world, half-length of the graspable span)."""
        if obj in ("fork", "spoon"):
            R = self.data.xmat[self._obj_bid[obj]].reshape(3, 3)
            return R @ np.array([0.0, 1.0, 0.0]), 0.5 * self.params.fork_len
        if obj == "bottle":
            R = self.data.xmat[self._obj_bid[obj]].reshape(3, 3)
            return R @ np.array([0.0, 0.0, 1.0]), 0.3 * self.params.bottle_h
        return np.array([0.0, 0.0, 1.0]), 0.0

    def nearest_grasp_point(self, arm, obj):
        centre = self.grasp_point(obj)
        axis_o, half = self._grasp_span(obj)
        if half > 0:
            tcp = self.tcp(arm)
            t = float(np.clip(np.dot(tcp - centre, axis_o), -half, half))
            centre = centre + t * axis_o
        return centre

    def capture_distance(self, arm, obj):
        return float(np.linalg.norm(self.nearest_grasp_point(arm, obj) - self.tcp(arm)))

    def in_capture_volume(self, arm, obj):
        return self.capture_distance(arm, obj) <= self.CAPTURE_R

    def _graspable_near(self, arm):
        best, bd = None, self.CAPTURE_R
        for o in self.WELDABLE:
            dist = self.capture_distance(arm, o)
            if dist < bd:
                best, bd = o, dist
        return best

    def _attach(self, arm, obj):
        m, d = self.model, self.data
        b1 = m.body(f"{arm}_Fixed_Jaw").id
        b2 = m.body(obj).id
        p1, q1 = d.xpos[b1], d.xquat[b1]
        p2, q2 = d.xpos[b2], d.xquat[b2]
        nq1 = np.zeros(4); mujoco.mju_negQuat(nq1, q1)
        rel = np.zeros(3); mujoco.mju_sub3(rel, p2, p1)
        relpos = np.zeros(3); mujoco.mju_rotVecQuat(relpos, rel, nq1)
        relq = np.zeros(4); mujoco.mju_mulQuat(relq, nq1, q2)
        eid = self.eq_id[(arm, obj)]
        m.eq_data[eid][:3] = 0.0
        m.eq_data[eid][3:6] = relpos
        m.eq_data[eid][6:10] = relq
        m.eq_data[eid][10] = 1.0
        d.eq_active[eid] = 1
        self.held[arm] = obj

    def transfer(self, giver, taker, obj):
        """Hand `obj` from one arm to the other in a single tick.

        Going through the normal pinch detector cannot work here: it refuses to
        grab anything the other arm already holds (otherwise an arm brushing
        past would steal objects).  A hand-off is the one legitimate exception,
        so it re-welds and releases atomically -- the two stiff welds never
        coexist for a simulation step.
        """
        if self.held[giver] != obj:
            return False
        # the taker must actually have the object between its fingers -- welding
        # regardless would "succeed" with the object dangling 10 cm away, and the
        # place that follows would then aim the wrong point at the mat
        if not self.in_capture_volume(taker, obj):
            return False
        self._attach(taker, obj)
        self.data.eq_active[self.eq_id[(giver, obj)]] = 0
        self.held[giver] = None
        return True

    def _detach(self, arm):
        if self.held[arm] is not None:
            self.data.eq_active[self.eq_id[(arm, self.held[arm])]] = 0
            self.held[arm] = None

    def _update_grasp(self):
        if not self.grasp_assist:
            return
        for a in ARMS:
            cmd_close = self.data.ctrl[self._jaw_act[a]] < 0.55
            if self.held[a] is None:
                if cmd_close:
                    o = self._graspable_near(a)
                    # do not steal an object the other arm is already holding
                    if o is not None and o not in self.held.values():
                        self._attach(a, o)
            else:
                if not cmd_close:
                    self._detach(a)

    # -------------------------------------------------------------- pour model
    def _update_pour(self):
        """Release particles from the bottle spout once it is tipped far enough."""
        if "bottle" not in self.held.values() or not self._water_free:
            return
        spout = self.data.site_xpos[self._spout_sid].copy()
        R = self.data.xmat[self._bottle_bid].reshape(3, 3)
        up = R @ np.array([0.0, 0.0, 1.0])
        tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
        self._pour_cool = max(0, self._pour_cool - 1)
        if tilt > 45.0 and spout[2] > 0.03 and self._water_free and self._pour_cool == 0:
            i = self._water_free.pop(0)
            adr, dadr = self._water_adr[i], self._water_dadr[i]
            self.data.qpos[adr:adr + 3] = spout + np.array([0, 0, -0.006])
            self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
            self.data.qvel[dadr:dadr + 6] = 0.0
            self.data.qvel[dadr + 2] = -0.05
            self._water_out.append(i)
            self._pour_cool = 6

    def water_in_mug(self):
        if not self._water_out:
            return 0
        mug = self.data.xpos[self._mug_bid]
        r = self.params.mug_r
        n = 0
        for i in self._water_out:
            adr = self._water_adr[i]
            p = self.data.qpos[adr:adr + 3]
            if (np.linalg.norm(p[:2] - mug[:2]) < r * 0.95
                    and mug[2] - 0.01 < p[2] < mug[2] + self.params.mug_h + 0.01):
                n += 1
        return n

    # ----------------------------------------------------------------- scoring
    ON_MAT_TOL = 0.055

    def _on_target(self, obj, target, tol=None):
        p = self.body_pos(obj)
        tol = tol or self.ON_MAT_TOL
        return (np.linalg.norm(p[:2] - np.array(target)) < tol
                and p[2] < 0.06 and self.held["left"] != obj and self.held["right"] != obj)

    def _update_task(self):
        t = self.task
        t.drawer_open = t.drawer_open or bool(
            self.data.qpos[self.drawer_qadr] > 0.6 * DRAWER_TRAVEL)
        t.plate_placed = self._on_target("plate", PLATE_TARGET, 0.06)
        t.mug_placed = self._on_target("mug", MUG_TARGET, 0.06)
        t.fork_placed = self._on_target("fork", FORK_TARGET, 0.07)
        t.water_poured = t.water_poured or (self.water_in_mug() >= 3)

    # -------------------------------------------------------------------- step
    def step(self, ctrl=None):
        if ctrl is not None:
            ctrl = np.asarray(ctrl, float).reshape(12)
            self.data.ctrl[self.act_id["left"]] = ctrl[:6]
            self.data.ctrl[self.act_id["right"]] = ctrl[6:]
        for _ in range(self.n_substeps):
            self._update_grasp()
            self._update_pour()
            mujoco.mj_step(self.model, self.data)
        self._update_task()
        self.t += 1
        return self.obs()

    # ---------------------------------------------------------------- observe
    def obs(self, render=True):
        q = np.concatenate([self.qarm(a) for a in ARMS])
        dq = np.concatenate([self.data.qvel[self.dadr[a]] for a in ARMS])
        o = {"qpos": q.astype(np.float32), "qvel": dq.astype(np.float32),
             "state": np.concatenate([q, dq]).astype(np.float32)}
        if render:
            o["images"] = self.render_all()
        return o

    def _get_renderer(self, size=None):
        """Renderers are cached per size.  Building one costs an EGL context, so
        creating a fresh renderer per video frame -- which is what happens if you
        forget this -- makes recording an order of magnitude slower than the
        physics it is recording."""
        size = size or self.img_size
        if getattr(self, "_rend_model", None) != id(self.model):
            for r in getattr(self, "_renderers", {}).values():
                try:
                    r.close()
                except Exception:
                    pass
            self._renderers = {}
            self._rend_model = id(self.model)
        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(self.model, size[1], size[0])
        return self._renderers[size]

    def render(self, cam, size=None):
        r = self._get_renderer(size)
        r.update_scene(self.data, camera=cam)
        return r.render()

    def render_all(self):
        return {c: self.render(c) for c in self.cameras}

    def close(self):
        for r in getattr(self, "_renderers", {}).values():
            try:
                r.close()
            except Exception:
                pass
        self._renderers = {}
        self._rend_model = None
