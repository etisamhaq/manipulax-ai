"""Damped-least-squares IK for a single SO-ARM100 arm inside the bimanual scene.

The arm has 5 positioning DoF (the 6th joint drives the jaw), so a full 6-DoF
pose is not reachable in general.  We therefore solve for
    3 position residuals  +  3 (rank-2) approach-axis alignment residuals
which is exactly what a parallel-jaw pick needs: where the fingers are, and
which way they point.  Roll about the approach axis is left free and handled
separately by the wrist-roll heuristic in primitives.py.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
JAW_JOINT = "Jaw"


class ArmIK:
    def __init__(self, model, arm: str):
        self.m = model
        self.arm = arm
        self.jnt_ids = [model.joint(f"{arm}_{j}").id for j in ARM_JOINTS]
        self.dof_ids = np.array([model.jnt_dofadr[j] for j in self.jnt_ids])
        self.qpos_ids = np.array([model.jnt_qposadr[j] for j in self.jnt_ids])
        self.jaw_qpos = model.jnt_qposadr[model.joint(f"{arm}_{JAW_JOINT}").id]
        self.jaw_act = model.actuator(f"{arm}_{JAW_JOINT}").id
        self.act_ids = np.array([model.actuator(f"{arm}_{j}").id for j in ARM_JOINTS])
        self.site = model.site(f"{arm}_tcp").id
        self.body = model.body(f"{arm}_Fixed_Jaw").id
        self.lo = model.jnt_range[self.jnt_ids, 0].copy()
        self.hi = model.jnt_range[self.jnt_ids, 1].copy()
        # approach axis of the gripper expressed in the Fixed_Jaw body frame
        self.approach_local = np.array([-0.2108, -0.9775, 0.0])
        self.approach_local /= np.linalg.norm(self.approach_local)

    # ------------------------------------------------------------------
    def tcp(self, data):
        return data.site_xpos[self.site].copy()

    def approach(self, data):
        R = data.xmat[self.body].reshape(3, 3)
        return R @ self.approach_local

    def q(self, data):
        return data.qpos[self.qpos_ids].copy()

    # ------------------------------------------------------------------
    def solve(self, data, target_pos, target_approach=None, q_init=None,
              iters=260, tol=1.0e-3, damping=0.04, w_rot=0.22):
        """Return joint angles for the 5 arm joints.  Works on a scratch MjData
        copy so the live simulation is never disturbed."""
        m = self.m
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = data.qpos
        q = (np.asarray(q_init, float).copy() if q_init is not None
             else data.qpos[self.qpos_ids].copy())
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        best_q, best_err = q.copy(), np.inf

        if target_approach is not None:
            a_des = np.asarray(target_approach, float)
            a_des = a_des / np.linalg.norm(a_des)

        for _ in range(iters):
            scratch.qpos[self.qpos_ids] = q
            mujoco.mj_kinematics(m, scratch)
            mujoco.mj_comPos(m, scratch)

            e_pos = np.asarray(target_pos, float) - scratch.site_xpos[self.site]
            rows = [e_pos]
            mujoco.mj_jacSite(m, scratch, jacp, jacr, self.site)
            J = [jacp[:, self.dof_ids]]

            if target_approach is not None:
                R = scratch.xmat[self.body].reshape(3, 3)
                a_cur = R @ self.approach_local
                e_rot = np.cross(a_cur, a_des)
                ang = np.arctan2(np.linalg.norm(e_rot), float(np.dot(a_cur, a_des)))
                if np.linalg.norm(e_rot) > 1e-9:
                    e_rot = e_rot / np.linalg.norm(e_rot) * ang
                rows.append(w_rot * e_rot)
                # d(a_cur)/dq = -[a_cur]_x * Jr
                ax = np.array([[0, -a_cur[2], a_cur[1]],
                               [a_cur[2], 0, -a_cur[0]],
                               [-a_cur[1], a_cur[0], 0]])
                J.append(w_rot * (-ax @ jacr[:, self.dof_ids]))

            err = np.concatenate(rows)
            pos_err = float(np.linalg.norm(e_pos))
            score = pos_err + (0.03 * float(np.linalg.norm(err[3:])) if len(err) > 3 else 0)
            if score < best_err:
                best_err, best_q = score, q.copy()
            if pos_err < tol and (target_approach is None or np.linalg.norm(err[3:]) < 0.12):
                break

            Jm = np.vstack(J)
            JT = Jm.T
            dq = JT @ np.linalg.solve(Jm @ JT + (damping ** 2) * np.eye(Jm.shape[0]), err)
            n = np.linalg.norm(dq)
            if n > 0.25:
                dq *= 0.25 / n
            q = np.clip(q + dq, self.lo + 1e-3, self.hi - 1e-3)

        return best_q, best_err


ARM_BASE = {"left": np.array([-0.165, 0.10]), "right": np.array([0.165, 0.10])}


def approach_for(arm: str, target_xy, tilt_deg: float = 22.0):
    """A mostly-downward approach that leans away from the arm's own base.

    The 5-DoF SO-ARM100 cannot hold a perfectly vertical gripper across the
    whole workspace; letting it lean outward the way a human wrist does keeps
    the IK residual under a degree or two instead of ~15 deg.
    """
    v = np.asarray(target_xy, float)[:2] - ARM_BASE[arm]
    n = np.linalg.norm(v)
    lat = v / n if n > 1e-6 else np.zeros(2)
    t = np.radians(tilt_deg)
    a = np.array([lat[0] * np.sin(t), lat[1] * np.sin(t), -np.cos(t)])
    return a / np.linalg.norm(a)


def _fk_err(self, data, q, pos, approach, want_deg=None):
    """Score a candidate: position error (m) plus small orientation penalties.

    The downward-preference term exists because a gripper that rolls over drops
    what it is holding.  But when a large tilt is *deliberately* requested --
    tipping a bottle to pour -- that same term overrules the request, so the
    threshold follows what was asked for rather than being fixed at 45 deg.
    """
    s = mujoco.MjData(self.m)
    s.qpos[:] = data.qpos
    s.qpos[self.qpos_ids] = q
    mujoco.mj_kinematics(self.m, s)
    p = s.site_xpos[self.site]
    a = s.xmat[self.body].reshape(3, 3) @ self.approach_local
    ang = np.degrees(np.arccos(np.clip(float(a @ approach), -1, 1)))
    down = np.degrees(np.arccos(np.clip(float(-a[2]), -1, 1)))
    allow = 45.0 if want_deg is None else max(45.0, float(want_deg) + 12.0)
    return float(np.linalg.norm(p - pos)) + 0.0004 * ang + 0.0006 * max(0.0, down - allow)


def _down_angle(self, data, q):
    """Angle (deg) between the gripper's approach axis and straight down."""
    s = mujoco.MjData(self.m)
    s.qpos[:] = data.qpos
    s.qpos[self.qpos_ids] = q
    mujoco.mj_kinematics(self.m, s)
    a = s.xmat[self.body].reshape(3, 3) @ self.approach_local
    return float(np.degrees(np.arccos(np.clip(float(-a[2]), -1, 1))))


ArmIK.fk_err = _fk_err
ArmIK.down_angle = _down_angle
