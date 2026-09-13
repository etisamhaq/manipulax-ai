"""Scripted bimanual motion primitives.

These are the *expert* used to generate imitation-learning demonstrations.  Each
primitive is a generator of joint-space waypoints; the Executor interpolates
between them, steps the simulator, and records (observation, action) pairs
together with the language annotation of the sub-task being performed.
"""
from __future__ import annotations

import numpy as np

from ..sim.env import ARMS, ARM_JOINTS, HOME, JAW_OPEN, JAW_CLOSED
from .ik import ArmIK, approach_for

TILTS = (12.0, 22.0, 32.0, 42.0, 55.0)
# A hand-off happens in mid-air with the other arm already occupying the space,
# so the taker needs to be able to come in almost horizontally.  Restricting it
# to near-vertical approaches leaves the IK 70 mm short.
HANDOFF_TILTS = (15.0, 30.0, 45.0, 60.0, 75.0)

# Per-object grasp recipes.  A tall thin bottle has to be approached almost
# vertically or the tilted gripper clips its neck and topples it; a flat plate
# tolerates -- and at the edge of the workspace needs -- a much bigger lean.
GRASP_CFG = {
    "plate":  dict(hover=0.085, dz=0.020, lift=0.145, tilts=TILTS),
    "mug":    dict(hover=0.085, dz=0.024, lift=0.135, tilts=TILTS),
    "fork":   dict(hover=0.075, dz=0.018, lift=0.145, tilts=TILTS),
    "spoon":  dict(hover=0.075, dz=0.018, lift=0.145, tilts=TILTS),
    "bottle": dict(hover=0.115, dz=0.030, lift=0.170, tilts=(6.0, 12.0, 20.0)),
}


def solve_best(ik: ArmIK, data, arm, pos, q_init=None, tilts=TILTS,
               max_down_deg=None):
    """Try several wrist tilts from several seeds; keep the cleanest solution.

    Multi-start matters after a hand-off: the taker ends up in a contorted
    configuration, and a damped-least-squares solver seeded only from there sits
    in a local minimum and reports a solution 200 mm short of the target.
    Re-seeding from the home pose (and from a mirrored elbow) gets it out.
    """
    seeds = []
    if q_init is not None:
        seeds.append(np.asarray(q_init, float))
    seeds.append(HOME[:5].copy())
    alt = HOME[:5].copy()
    alt[1] *= 0.6
    alt[2] *= 0.6
    seeds.append(alt)

    # `max_down_deg` is a hard gate, not a penalty.  Carrying an open bottle,
    # any solution that rolls the gripper past ~40 deg from vertical tips the
    # bottle -- and because the bottle is welded to the jaw, it empties itself
    # in mid-air before the arm ever arrives.  Multi-start IK makes those
    # elbow-flipped solutions *more* likely, so they have to be excluded
    # outright rather than merely discouraged.
    best, bscore = None, np.inf
    gated, gscore = None, np.inf
    for seed in seeds:
        for t in tilts:
            ap = approach_for(arm, pos, t)
            q, _ = ik.solve(data, pos, ap, q_init=seed)
            sc = ik.fk_err(data, q, pos, ap, want_deg=t)
            if sc < bscore:
                best, bscore = q, sc
            if max_down_deg is not None and ik.down_angle(data, q) <= max_down_deg:
                if sc < gscore:
                    gated, gscore = q, sc
        if max_down_deg is None and bscore < 0.004:
            break
        if max_down_deg is not None and gscore < 0.004:
            break
    if max_down_deg is not None and gated is not None:
        return gated, gscore
    return best, bscore


class Executor:
    """Runs primitives against a DinnerTableEnv while logging a trajectory."""

    def __init__(self, env, record=True, frame_cams=("front",), fps_every=2):
        self.env = env
        self.ik = {a: ArmIK(env.model, a) for a in ARMS}
        self.record = record
        self.frame_cams = frame_cams
        self.fps_every = fps_every
        self.traj = []       # list of dicts: state / images / action / subtask
        self.frames = []
        self.subtask = "idle"
        self.arm_of_subtask = "none"
        self._ctrl = np.concatenate([env.data.ctrl[env.act_id[a]].copy() for a in ARMS])
        self.aborted = None

    # ------------------------------------------------------------------ util
    def ctrl_of(self, arm):
        return self._ctrl[:6] if arm == "left" else self._ctrl[6:]

    def set_arm(self, arm, q6):
        if arm == "left":
            self._ctrl[:6] = q6
        else:
            self._ctrl[6:] = q6

    def _log(self, action):
        e = self.env
        rec = {"state": np.concatenate([e.qarm("left"), e.qarm("right"),
                                        e.data.qvel[e.dadr["left"]],
                                        e.data.qvel[e.dadr["right"]]]).astype(np.float32),
               "action": action.astype(np.float32).copy(),
               "subtask": self.subtask, "arm": self.arm_of_subtask}
        if self.record:
            rec["images"] = e.render_all()
        self.traj.append(rec)

    def _grab_frame(self):
        if self.frame_cams and self.env.t % self.fps_every == 0:
            self.frames.append({c: self.env.render(c, (640, 480)) for c in self.frame_cams})

    # ------------------------------------------------------------ trajectory
    def hold(self, steps):
        for _ in range(steps):
            self._log(self._ctrl)
            self.env.step(self._ctrl)
            self._grab_frame()

    def move(self, targets: dict, steps=40, settle=4):
        """Cosine-interpolate each named arm from its current ctrl to targets."""
        start = {a: self.ctrl_of(a).copy() for a in ARMS}
        goal = {a: np.asarray(targets.get(a, start[a]), float).copy() for a in ARMS}
        for i in range(1, steps + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / steps)
            for a in ARMS:
                self.set_arm(a, start[a] + s * (goal[a] - start[a]))
            self._log(self._ctrl)
            self.env.step(self._ctrl)
            self._grab_frame()
        self.hold(settle)

    def set_jaw(self, arm, value, steps=12):
        q = self.ctrl_of(arm).copy()
        q[5] = value
        self.move({arm: q}, steps=steps, settle=3)

    def reach(self, arm, pos, steps=45, jaw=None, refine=1, tol=0.008, tilts=TILTS,
              max_down_deg=None):
        """Move the TCP to `pos`.  The position servos droop by a few centimetres
        under load, so after the first move we measure the real TCP and re-aim at
        an over-corrected target.  One refinement pass removes most of the error."""
        pos = np.asarray(pos, float)
        target = pos.copy()
        err = None
        for k in range(refine + 1):
            q0 = self.ctrl_of(arm)
            q5, err = solve_best(self.ik[arm], self.env.data, arm, target,
                                 q_init=q0[:5], tilts=tilts,
                                 max_down_deg=max_down_deg)
            q = np.concatenate([q5, [q0[5] if jaw is None else jaw]])
            self.move({arm: q}, steps=steps if k == 0 else max(12, steps // 3))
            resid = pos - self.env.tcp(arm)
            if np.linalg.norm(resid) < tol:
                break
            target = target + np.clip(resid, -0.05, 0.05)
        return float(np.linalg.norm(pos - self.env.tcp(arm)))

    def home(self, arm, steps=45, jaw=JAW_OPEN):
        q = HOME.copy(); q[5] = jaw
        self.move({arm: q}, steps=steps)

    # ------------------------------------------------------------ primitives
    def begin(self, name, arm):
        self.subtask, self.arm_of_subtask = name, arm

    def pick(self, arm, obj, hover=None, grasp_dz=None, lift=None):
        e = self.env
        cfg = GRASP_CFG.get(obj, dict(hover=0.085, dz=0.020, lift=0.14, tilts=TILTS))
        hover = cfg["hover"] if hover is None else hover
        grasp_dz = cfg["dz"] if grasp_dz is None else grasp_dz
        lift = cfg["lift"] if lift is None else lift
        tl = cfg["tilts"]
        for attempt in range(2):
            p = e.grasp_point(obj)
            self.set_jaw(arm, JAW_OPEN)
            self.reach(arm, p + [0, 0, hover], steps=50 if attempt == 0 else 34, tilts=tl)
            # stop just short of the object: driving the tool point all the way
            # to the object centre simply knocks it across the table
            p = e.grasp_point(obj)
            dz = grasp_dz + (0.010 if attempt else 0.0)
            self.reach(arm, p + [0, 0, dz], steps=38, refine=2, tol=0.010,
                       tilts=tl if attempt == 0 else TILTS)
            self.set_jaw(arm, JAW_CLOSED, steps=16)
            self.hold(8)
            if e.held[arm] == obj:
                break
            self.set_jaw(arm, JAW_OPEN, steps=10)   # missed -- back off and retry
        if e.held[arm] != obj:
            return False
        up = e.body_pos(obj)
        # an open bottle must stay upright once it leaves the table
        self.reach(arm, [up[0], up[1], max(lift, up[2] + 0.08)], steps=36, tilts=tl,
                   max_down_deg=35.0 if obj == "bottle" else None)
        return e.held[arm] == obj

    def grasp_offset(self, arm):
        """Vector from the TCP to the held object -- a plate pinched at its rim
        hangs several centimetres from the tool point, so every place target has
        to be corrected by this or the object lands off the mat."""
        obj = self.env.held[arm]
        if obj is None or obj == "drawer":
            return np.zeros(3)
        return self.env.body_pos(obj) - self.env.tcp(arm)

    def place(self, arm, xy, z=0.045, hover=0.10, retreat=True):
        off = self.grasp_offset(arm)
        tx, ty = xy[0] - off[0], xy[1] - off[1]
        tz = z - off[2]
        self.reach(arm, [tx, ty, tz + hover], steps=50)
        off = self.grasp_offset(arm)          # re-measure after the transit
        tx, ty, tz = xy[0] - off[0], xy[1] - off[1], z - off[2]
        self.reach(arm, [tx, ty, tz], steps=34)
        # closed-loop correction on the *object*, not the tool point
        obj = self.env.held[arm]
        if obj is not None:
            for _ in range(2):
                e = np.asarray(xy, float) - self.env.body_pos(obj)[:2]
                if np.linalg.norm(e) < 0.012:
                    break
                tx, ty = tx + float(np.clip(e[0], -0.05, 0.05)), ty + float(np.clip(e[1], -0.05, 0.05))
                self.reach(arm, [tx, ty, tz], steps=18, refine=0)
        self.hold(6)                       # let the object settle on the table
        self.set_jaw(arm, JAW_OPEN, steps=16)
        self.hold(10)
        if retreat:
            self.reach(arm, [tx, ty, tz + hover + 0.03], steps=28)

    def open_drawer(self, arm, travel=0.078, attempts=3):
        """Pull the drawer open by its post, re-gripping if it slips.

        Under randomisation the drawer's damping varies by ~3x and the post's
        friction with it, so a single pull that works at one setting stalls at
        another.  Each attempt re-measures where the handle actually is now --
        a partly-open drawer has moved -- and pulls the remaining distance.
        """
        e = self.env
        for attempt in range(attempts):
            opened = float(e.data.qpos[e.drawer_qadr])
            if opened > 0.62 * travel:
                break
            h = e.data.site("handle_site").xpos.copy()
            self.set_jaw(arm, JAW_OPEN)
            self.reach(arm, h + [0, 0, 0.075], steps=46 if attempt == 0 else 32)
            h = e.data.site("handle_site").xpos.copy()
            self.reach(arm, h, steps=38, refine=2, tol=0.006)
            self.set_jaw(arm, JAW_CLOSED, steps=16)
            self.hold(8)
            # pull the *remaining* travel in small increments so the slide joint
            # is dragged rather than yanked out of the fingers
            remaining = max(0.0, travel - float(e.data.qpos[e.drawer_qadr]))
            base = e.data.site("handle_site").xpos.copy()
            steps_n = 5 + 2 * attempt
            for k in range(1, steps_n + 1):
                self.reach(arm, base + np.array([0, remaining * k / steps_n, 0]),
                           steps=20, refine=0)
            self.set_jaw(arm, JAW_OPEN, steps=14)
            self.hold(6)
            self.reach(arm, e.tcp(arm) + [0, 0.015, 0.085], steps=28)
            if float(e.data.qpos[e.drawer_qadr]) > 0.62 * travel:
                break
        ok = bool(e.data.qpos[e.drawer_qadr] > 0.6 * travel)
        self.home(arm, steps=35)
        return ok

    def handoff(self, giver, taker, obj, site=(0.055, 0.02, 0.105)):
        """Pass an object from one arm to the other across the shared workspace.

        The taker must grab a *different* part of the object than the giver is
        holding -- aim both grippers at the same point and the jaws collide, the
        pinch lands centimetres off, and everything downstream inherits a bogus
        grasp offset.  So the taker targets the far end of the object's long
        axis, choosing whichever end is further from the giver's tool point.
        """
        e = self.env
        site = np.asarray(site, float)
        self.reach(giver, site, steps=55, tilts=HANDOFF_TILTS)
        self.hold(8)

        gp = e.body_pos(obj)
        R = e.data.body(obj).xmat.reshape(3, 3)
        long_axis = R @ np.array([0.0, 1.0, 0.0])       # cutlery run along local y
        extent = 0.30 * getattr(e.params, "fork_len", 0.07)
        giver_tcp = e.tcp(giver)
        cands = [gp + extent * long_axis, gp - extent * long_axis]
        target = max(cands, key=lambda c: np.linalg.norm(c - giver_tcp))

        self.set_jaw(taker, JAW_OPEN)
        self.reach(taker, target + [0, 0, 0.075], steps=48, tilts=HANDOFF_TILTS)
        self.reach(taker, target, steps=34, refine=2, tol=0.008, tilts=HANDOFF_TILTS)
        # Servo onto the object itself.  Aiming at a precomputed mid-air point
        # is not enough: the giver is still holding the object and it swings a
        # little as the taker arrives, so close the loop on the capture geometry.
        for _ in range(3):
            if e.in_capture_volume(taker, obj):
                break
            resid = e.nearest_grasp_point(taker, obj) - e.tcp(taker)
            self.reach(taker, e.tcp(taker) + np.clip(resid, -0.06, 0.06),
                       steps=16, refine=0, tilts=HANDOFF_TILTS)
        self.set_jaw(taker, JAW_CLOSED, steps=16)
        ok = e.transfer(giver, taker, obj)              # atomic re-weld + release
        self.hold(8)
        self.set_jaw(giver, JAW_OPEN, steps=14)         # giver opens its fingers
        self.hold(8)
        ok = ok and e.held[taker] == obj
        self.reach(giver, np.array(site) + [0, 0.07, 0.04], steps=30)
        self.home(giver, steps=40)
        return ok

    def bottle_tilt(self):
        R = self.env.data.body("bottle").xmat.reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip((R @ np.array([0, 0, 1.0]))[2], -1, 1))))

    def spout(self):
        return self.env.data.site("spout").xpos.copy()

    def aim_spout(self, arm, target, iters=3, tol=0.010, tilts=(8.0, 16.0, 28.0, 40.0)):
        """Servo the bottle's spout to a 3-D point by moving the tool point.

        The spout is ~10 cm from the tool frame and swings as the wrist pitches,
        so commanding the jaw to a position says very little about where the
        liquid will actually go.  This closes the loop on the spout itself.
        """
        env = self.env
        for _ in range(iters):
            err = np.asarray(target, float) - self.spout()
            if np.linalg.norm(err) < tol:
                return True
            tcp = env.tcp(arm)
            self.reach(arm, tcp + np.clip(err, -0.07, 0.07), steps=14, refine=1,
                       tol=0.008, tilts=tilts)
        return float(np.linalg.norm(np.asarray(target, float) - self.spout())) < tol * 2

    def pour_target(self, extra_height=0.045):
        """Where the spout should sit: just above the mug's rim, centred."""
        mug = self.env.body_pos("mug")
        return np.array([mug[0], mug[1], mug[2] + self.env.params.mug_h + extra_height])

    def catch_with_mug(self, mug_arm, target_xy, tol=0.010):
        """Bring the held mug under a point, using the arm that holds it.

        Aiming the bottle is a 5-DoF problem tangled up with the tilt; moving
        the mug is a plain position move with a free arm, so when the stream is
        off target it is much cheaper to move the cup than the bottle.
        """
        env = self.env
        if env.held[mug_arm] != "mug":
            return False
        err = np.asarray(target_xy, float)[:2] - env.body_pos("mug")[:2]
        if np.linalg.norm(err) < tol:
            return True
        tcp = env.tcp(mug_arm)
        self.reach(mug_arm, [tcp[0] + float(np.clip(err[0], -0.06, 0.06)),
                             tcp[1] + float(np.clip(err[1], -0.06, 0.06)), tcp[2]],
                   steps=12, refine=0)
        return float(np.linalg.norm(np.asarray(target_xy, float)[:2]
                                    - env.body_pos("mug")[:2])) < tol * 2

    def pour_hold(self, arm, steps, mug_tol=0.010):
        """Hold the pour, re-aiming the spout over the mug as it empties.

        Particles leave the spout one at a time over the whole hold, so aiming
        once at the start wastes most of them -- the bottle drifts as it lightens
        and the arm settles.  Re-centring during the hold is what turns a
        one-drop pour into most of the bottle.
        """
        env = self.env
        other = "left" if arm == "right" else "right"
        chunk = 10
        for _ in range(max(1, steps // chunk)):
            self.hold(chunk)
            sp = self.spout()
            # cheapest correction first: slide the cup under the stream
            if not self.catch_with_mug(other, sp[:2], tol=0.012):
                tgt = self.pour_target()
                if np.linalg.norm(tgt - sp) > mug_tol:
                    self.aim_spout(arm, tgt, iters=1, tol=mug_tol)

    def pour(self, arm, over_xy, hold_steps=180, tilt_target=55.0):
        """Tip the held bottle so its spout empties into the mug.

        Three things fought each other here and the order of operations is the
        whole trick:

        * Wrist *roll* spins the bottle about its own axis and pours nothing --
          the tilt has to come from the gripper's approach angle.
        * Positioning and tilting compete for the same five joints, so servoing
          the spout onto the mug re-solves the wrist and undoes any pitch that
          was commanded directly.  Asking IK for a tilted *approach* lets it
          solve both together instead.
        * A tipped bottle pours wherever it points, so it must arrive over the
          mug before it is tilted, and the spout -- not the jaw, which is ~10 cm
          away -- is what has to be aimed.

        So: transit upright-ish, then walk the requested approach angle up until
        the bottle passes the pouring angle with its spout still on target.
        """
        env = self.env
        over_xy = np.asarray(over_xy, float)[:2]
        if not env._water_free:
            return False                      # nothing left to pour

        # 1 ---------------------------------- arrive above the mug, still upright
        rim = env.body_pos("mug")[2] + env.params.mug_h
        self.reach(arm, [over_xy[0], over_xy[1], rim + 0.13], steps=50,
                   tilts=(10.0, 20.0, 30.0))

        # 2 ------------- walk the approach angle up until the bottle actually tips
        base = self.ctrl_of(arm).copy()
        best = (-1.0, None, 1.0)
        for want in (30.0, 40.0, 50.0, 60.0, 75.0, 88.0):
            self.aim_spout(arm, self.pour_target(), iters=4, tol=0.010,
                           tilts=(want,))
            tilt = self.bottle_tilt()
            miss = float(np.linalg.norm(self.pour_target() - self.spout()))
            if tilt > best[0] and miss < 0.035:
                best = (tilt, self.ctrl_of(arm).copy(), miss)
            if tilt > tilt_target and miss < 0.025:
                self.pour_hold(arm, hold_steps, mug_tol=0.012)
                if env.water_in_mug() >= 1:
                    self.move({arm: base}, steps=24)
                    return True

        if best[1] is not None:
            self.move({arm: best[1]}, steps=14)
            self.aim_spout(arm, self.pour_target(), iters=2, tol=0.012)
        self.pour_hold(arm, hold_steps, mug_tol=0.012)
        ok = env.water_in_mug() >= 1
        self.move({arm: base}, steps=24)
        return ok
