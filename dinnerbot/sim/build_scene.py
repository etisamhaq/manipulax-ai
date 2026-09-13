"""Procedural MuJoCo scene builder for the bimanual dinner-table task.

The whole scene is generated from a seed so that every episode is reproducible
and so that domain randomisation can touch quantities (geom sizes, masses,
frictions) that cannot be safely mutated on an already-compiled model.

World frame: the table top is the z=0 plane, +y points from the table towards
the robots, +x is to the robots' right.  Both SO-ARM100 arms are mounted on the
table at y=+0.10 and reach in the -y direction (this is the arm's natural
forward direction, measured from the menagerie model).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# ---------------------------------------------------------------- layout ----
# Layout is tuned so that every scripted reach lands between 0.14 m and 0.28 m
# from the acting arm's base -- the SO-ARM100's comfortable annulus.  Closer
# than that it cannot fold; further and the IK residual blows up.
ARM_Y = 0.10          # both arm bases sit at this y
ARM_DX = 0.165        # half the distance between the two arm bases
TABLE_HALF_X = 0.45
TABLE_Y0, TABLE_Y1 = -0.32, 0.26

# The drawer lives on the LEFT.  The fork it contains belongs on the RIGHT of
# the mat, so the only way to finish the task is an arm-to-arm hand-off.
CABINET_X = -0.250
CABINET_Y = -0.140
DRAWER_TRAVEL = 0.085 # how far the drawer slides towards the robots

# placemat targets (where things must end up)
PLATE_TARGET = (0.0, -0.03)
FORK_TARGET = (0.105, -0.03)
MUG_TARGET = (-0.105, -0.03)


@dataclass
class SceneParams:
    """Every quantity the domain randomiser is allowed to touch."""
    seed: int = 0
    # geometry
    plate_r: float = 0.038
    plate_h: float = 0.005
    mug_r: float = 0.021
    mug_h: float = 0.045
    mug_wall: float = 0.0028
    fork_len: float = 0.070
    fork_w: float = 0.020
    bottle_r: float = 0.018
    bottle_h: float = 0.090
    # dynamics
    plate_mass: float = 0.040
    mug_mass: float = 0.035
    fork_mass: float = 0.012
    bottle_mass: float = 0.090
    slide_fric: float = 1.0
    tors_fric: float = 0.006
    drawer_damping: float = 1.8
    # initial poses  (x, y, yaw)
    plate_pose: tuple = (0.235, -0.06, 0.0)
    mug_pose: tuple = (-0.265, -0.01, 0.0)
    bottle_pose: tuple = (0.275, 0.035, 0.0)
    fork_in_drawer: float = 0.016      # lateral offset of the fork inside drawer
    spoon_in_drawer: float = -0.016
    # appearance / sensing
    light_pos: tuple = (0.0, 0.1, 0.9)
    light_diffuse: tuple = (0.75, 0.75, 0.75)
    light_ambient: tuple = (0.25, 0.25, 0.25)
    table_rgba: tuple = (0.72, 0.60, 0.45, 1.0)
    floor_rgba: tuple = (0.30, 0.32, 0.35, 1.0)
    plate_rgba: tuple = (0.92, 0.92, 0.95, 1.0)
    mug_rgba: tuple = (0.25, 0.45, 0.80, 1.0)
    cam_jitter: tuple = (0.0, 0.0, 0.0)
    n_water: int = 10
    water_r: float = 0.0045


def randomize(seed: int, level: float = 1.0) -> SceneParams:
    """Draw a randomised SceneParams.  level=0 reproduces the nominal scene."""
    p = SceneParams(seed=seed)
    if level <= 0:
        return p
    rng = np.random.default_rng(seed)
    u = lambda a, b: float(rng.uniform(a, b))
    s = level

    # --- object shape -------------------------------------------------
    p.plate_r = 0.038 * (1 + s * u(-0.13, 0.13))
    p.mug_r = 0.021 * (1 + s * u(-0.12, 0.16))
    p.mug_h = 0.045 * (1 + s * u(-0.15, 0.15))
    p.fork_len = 0.070 * (1 + s * u(-0.12, 0.12))
    p.bottle_r = 0.018 * (1 + s * u(-0.10, 0.10))

    # --- mass & friction ----------------------------------------------
    p.plate_mass = 0.040 * (1 + s * u(-0.5, 1.4))
    p.mug_mass = 0.035 * (1 + s * u(-0.5, 1.6))
    p.fork_mass = 0.012 * (1 + s * u(-0.5, 1.2))
    p.bottle_mass = 0.090 * (1 + s * u(-0.4, 0.9))
    p.slide_fric = float(np.clip(1.0 * (1 + s * u(-0.45, 0.6)), 0.35, 2.2))
    p.tors_fric = 0.006 * (1 + s * u(-0.5, 1.0))
    p.drawer_damping = 1.8 * (1 + s * u(-0.4, 1.1))

    # --- initial placement --------------------------------------------
    # Spawn zones are chosen so each arm works mostly inside its own reachable
    # half; the fork is the exception -- it starts on the right but belongs on
    # the left of the mat, which is what forces the arm-to-arm hand-off.
    p.plate_pose = (u(0.200, 0.270), u(-0.10, -0.02), u(-0.5, 0.5) * s)
    p.mug_pose = (u(-0.300, -0.235), u(-0.05, 0.03), u(-3.1, 3.1) * s)
    p.bottle_pose = (u(0.250, 0.305), u(0.005, 0.065), u(-3.1, 3.1) * s)
    p.fork_in_drawer = u(0.008, 0.024)
    p.spoon_in_drawer = u(-0.024, -0.008)

    # --- lighting & background ----------------------------------------
    p.light_pos = (u(-0.5, 0.5), u(-0.2, 0.5), u(0.65, 1.15))
    g = u(0.5, 1.0)
    p.light_diffuse = (g * u(0.9, 1.1), g * u(0.9, 1.1), g * u(0.9, 1.1))
    a = u(0.12, 0.38)
    p.light_ambient = (a, a, a * u(0.9, 1.15))
    p.table_rgba = (u(0.35, 0.85), u(0.30, 0.72), u(0.25, 0.62), 1.0)
    p.floor_rgba = (u(0.15, 0.6), u(0.15, 0.6), u(0.18, 0.65), 1.0)
    p.plate_rgba = (u(0.75, 0.98), u(0.75, 0.98), u(0.78, 0.99), 1.0)
    p.mug_rgba = (u(0.1, 0.9), u(0.1, 0.7), u(0.2, 0.9), 1.0)

    # --- camera jitter (metres / radians) -----------------------------
    p.cam_jitter = (u(-0.012, 0.012) * s, u(-0.012, 0.012) * s, u(-0.02, 0.02) * s)
    return p


# ------------------------------------------------------------ xml pieces ----
def _ring(name, r, h, wall, n=14, rgba="0.3 0.5 0.8 1", mass=0.004):
    """A hollow cylinder approximated by n thin boxes -- used for the mug."""
    out = []
    for i in range(n):
        a = 2 * np.pi * i / n
        cx, cy = r * np.cos(a), r * np.sin(a)
        tang = np.pi * r / n  # half-width of each plank
        out.append(
            f'<geom name="{name}_w{i}" type="box" pos="{cx:.5f} {cy:.5f} {h/2:.5f}" '
            f'euler="0 0 {a:.5f}" size="{wall:.5f} {tang:.5f} {h/2:.5f}" '
            f'rgba="{rgba}" mass="{mass:.5f}" class="obj"/>'
        )
    return "\n        ".join(out)


def _water(p: SceneParams):
    """Water particles: parked under the floor until a pour is triggered."""
    out = []
    for i in range(p.n_water):
        out.append(
            f'<body name="water{i}" pos="{-0.6 + 0.02*i:.3f} 0.6 -0.5">'
            f'<freejoint name="water{i}_j"/>'
            f'<geom name="water{i}_g" type="sphere" size="{p.water_r:.5f}" '
            f'mass="0.002" rgba="0.25 0.55 0.95 0.85" '
            f'friction="0.2 0.002 0.0001" solimp="0.9 0.95 0.002" condim="4"/></body>'
        )
    return "\n      ".join(out)


GRASPABLE = ("plate", "mug", "fork", "spoon", "bottle", "drawer")


def _welds():
    """One (arm, object) weld per pair -- the runtime flips eq_active to model a
    successful pinch.  See runtime/grasp.py for the activation conditions."""
    out = []
    for arm in ("left", "right"):
        for obj in GRASPABLE:
            out.append(
                f'    <weld name="{arm}__{obj}" body1="{arm}_Fixed_Jaw" body2="{obj}" '
                f'active="false" solimp="0.98 0.9999 0.001 0.5 2" solref="0.004 1" '
                f'torquescale="1"/>'
            )
    return "\n".join(out) + "\n"


def build_xml(p: SceneParams) -> str:
    welds = _welds()
    fr = f"{p.slide_fric:.4f} {p.tors_fric:.5f} 0.0001"
    lr, lg, lb = p.light_diffuse
    ar, ag, ab = p.light_ambient
    rgba = lambda t: " ".join(f"{v:.4f}" for v in t)

    # cutlery sits inside the drawer; local coords are relative to the drawer
    # Cutlery are TOP-LEVEL bodies (a freejoint may not live under the sliding
    # drawer body).  They start resting inside the drawer tray; when the drawer
    # is pulled the rear wall pushes them along, which is what really happens.
    cut = []
    for nm, dx, rg in (("fork", p.fork_in_drawer, "0.85 0.85 0.88 1"),
                       ("spoon", p.spoon_in_drawer, "0.80 0.80 0.84 1")):
        cut.append(f"""
    <body name="{nm}" pos="{CABINET_X + dx:.4f} {CABINET_Y:.4f} 0.0245">
      <freejoint name="{nm}_j"/>
      <geom name="{nm}_handle" type="box" size="{p.fork_w/2:.5f} {p.fork_len/2:.5f} 0.003"
            rgba="{rg}" mass="{p.fork_mass:.5f}" class="obj"/>
      <geom name="{nm}_head" type="box" pos="0 {p.fork_len/2+0.010:.5f} 0"
            size="{p.fork_w/2*1.1:.5f} 0.010 0.0025" rgba="{rg}"
            mass="{p.fork_mass*0.4:.5f}" class="obj"/>
    </body>""")

    return f"""<mujoco model="dinner_table_bimanual_seed{p.seed}">
  <compiler angle="radian" meshdir="so_arm100_meshes/" autolimits="true"/>
  <option timestep="0.002" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <size memory="64M"/>

  <asset>
    <model name="so_arm" file="so_arm100_arm.xml"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.5 0.6 0.7" rgb2="0.1 0.12 0.16" width="256" height="256"/>
    <texture name="tabletex" type="2d" builtin="checker" rgb1="{rgba(p.table_rgba[:3])}"
             rgb2="{rgba(tuple(c*0.82 for c in p.table_rgba[:3]))}" width="300" height="300"/>
    <material name="tablemat" texture="tabletex" texrepeat="6 5" specular="0.2" shininess="0.3"/>
    <material name="floormat" rgba="{rgba(p.floor_rgba)}"/>
    <material name="cabmat" rgba="0.45 0.33 0.24 1"/>
  </asset>

  <default>
    <default class="obj">
      <geom friction="{fr}" condim="4" solimp="0.95 0.99 0.001" solref="0.008 1"/>
    </default>
  </default>

  <visual>
    <headlight ambient="{ar:.3f} {ag:.3f} {ab:.3f}" diffuse="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <quality shadowsize="0" offsamples="0"/>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <worldbody>
    <light name="key" pos="{rgba(p.light_pos)}" dir="0 -0.2 -1" diffuse="{lr:.3f} {lg:.3f} {lb:.3f}"
           specular="0.2 0.2 0.2" castshadow="false"/>
    <light name="fill" pos="0 0.6 0.7" dir="0 -0.5 -1" diffuse="0.25 0.25 0.28" castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 -0.40" size="3 3 0.1" material="floormat"/>

    <!-- table top: the z=0 plane -->
    <geom name="table" type="box" material="tablemat"
          pos="0 {(TABLE_Y0+TABLE_Y1)/2:.4f} -0.01"
          size="{TABLE_HALF_X} {(TABLE_Y1-TABLE_Y0)/2:.4f} 0.01"
          friction="{fr}" condim="4"/>
    <geom name="leg0" type="box" pos="-0.39 -0.26 -0.21" size="0.02 0.02 0.19" material="cabmat"/>
    <geom name="leg1" type="box" pos="0.39 -0.26 -0.21" size="0.02 0.02 0.19" material="cabmat"/>
    <geom name="leg2" type="box" pos="-0.39 0.20 -0.21" size="0.02 0.02 0.19" material="cabmat"/>
    <geom name="leg3" type="box" pos="0.39 0.20 -0.21" size="0.02 0.02 0.19" material="cabmat"/>

    <!-- placemat: purely visual targets -->
    <geom name="placemat" type="box" pos="0 -0.03 0.0005" size="0.175 0.070 0.0005"
          rgba="0.85 0.28 0.25 0.55" contype="0" conaffinity="0"/>
    <site name="plate_target" pos="{PLATE_TARGET[0]} {PLATE_TARGET[1]} 0.004" size="0.006"
          rgba="0 1 0 0.35"/>
    <site name="mug_target" pos="{MUG_TARGET[0]} {MUG_TARGET[1]} 0.004" size="0.006" rgba="0 1 0 0.35"/>
    <site name="fork_target" pos="{FORK_TARGET[0]} {FORK_TARGET[1]} 0.004" size="0.006" rgba="0 1 0 0.35"/>
    <site name="handoff" pos="0 0.03 0.135" size="0.005" rgba="1 0.6 0 0.30"/>

    <!-- ------------------------------------------------ cabinet + drawer -->
    <body name="cabinet" pos="{CABINET_X} {CABINET_Y} 0">
      <geom name="cab_back" type="box" pos="0 -0.050 0.030" size="0.088 0.005 0.030" material="cabmat"/>
      <geom name="cab_left" type="box" pos="-0.083 0 0.030" size="0.005 0.050 0.030" material="cabmat"/>
      <geom name="cab_right" type="box" pos="0.083 0 0.030" size="0.005 0.050 0.030" material="cabmat"/>
      <geom name="cab_top" type="box" pos="0 -0.006 0.064" size="0.088 0.050 0.005" material="cabmat"/>
      <body name="drawer" pos="0 0 0.010">
        <joint name="drawer_slide" type="slide" axis="0 1 0" range="0 {DRAWER_TRAVEL}"
               damping="{p.drawer_damping:.3f}" frictionloss="0.06" armature="0.005"/>
        <geom name="drawer_bottom" type="box" pos="0 0 0.004" size="0.072 0.046 0.004"
              rgba="0.62 0.46 0.32 1" mass="0.12" class="obj"/>
        <geom name="drawer_back" type="box" pos="0 -0.044 0.010" size="0.072 0.004 0.010"
              rgba="0.62 0.46 0.32 1" mass="0.02" class="obj"/>
        <geom name="drawer_l" type="box" pos="-0.068 0 0.003" size="0.004 0.046 0.003"
              rgba="0.62 0.46 0.32 1" mass="0.02" class="obj"/>
        <geom name="drawer_r" type="box" pos="0.068 0 0.003" size="0.004 0.046 0.003"
              rgba="0.62 0.46 0.32 1" mass="0.02" class="obj"/>
        <geom name="drawer_front" type="box" pos="0 0.059 0.020" size="0.078 0.005 0.024"
              rgba="0.55 0.40 0.27 1" mass="0.05" class="obj"/>
        <!-- handle: a vertical post.  A top-down approach drops it straight
             between the jaws, which a flat bar at table height cannot do. -->
        <geom name="drawer_post" type="cylinder" pos="0 0.073 0.050" size="0.0092 0.024"
              rgba="0.82 0.82 0.86 1" mass="0.012" friction="2.2 0.02 0.001"
              solimp="0.97 0.99 0.001" solref="0.006 1" condim="4"/>
        <geom name="drawer_handle" type="cylinder" pos="0 0.073 0.076" euler="0 1.5708 0"
              size="0.005 0.022" rgba="0.86 0.86 0.90 1" mass="0.010" class="obj"/>
        <site name="handle_site" pos="0 0.073 0.050" size="0.004" rgba="1 0 0 0.4"/>
      </body>
    </body>

    {''.join(cut)}

    <!-- ------------------------------------------------------- tableware -->
    <body name="plate" pos="{p.plate_pose[0]:.4f} {p.plate_pose[1]:.4f} {p.plate_h/2+0.001:.4f}"
          euler="0 0 {p.plate_pose[2]:.4f}">
      <freejoint name="plate_j"/>
      <geom name="plate_g" type="cylinder" size="{p.plate_r:.5f} {p.plate_h/2:.5f}"
            rgba="{rgba(p.plate_rgba)}" mass="{p.plate_mass:.5f}" class="obj"/>
    </body>

    <body name="mug" pos="{p.mug_pose[0]:.4f} {p.mug_pose[1]:.4f} 0.001"
          euler="0 0 {p.mug_pose[2]:.4f}">
      <freejoint name="mug_j"/>
      <geom name="mug_base" type="cylinder" pos="0 0 0.004" size="{p.mug_r:.5f} 0.004"
            rgba="{rgba(p.mug_rgba)}" mass="{p.mug_mass*0.5:.5f}" class="obj"/>
      {_ring('mug', p.mug_r, p.mug_h, p.mug_wall, 14, rgba(p.mug_rgba), p.mug_mass*0.5/14)}
      <site name="mug_site" pos="0 0 {p.mug_h*0.55:.5f}" size="0.004" rgba="1 1 0 0.3"/>
    </body>

    <body name="bottle" pos="{p.bottle_pose[0]:.4f} {p.bottle_pose[1]:.4f} {p.bottle_h/2+0.001:.4f}"
          euler="0 0 {p.bottle_pose[2]:.4f}">
      <freejoint name="bottle_j"/>
      <geom name="bottle_body" type="cylinder" size="{p.bottle_r:.5f} {p.bottle_h/2:.5f}"
            rgba="0.20 0.70 0.35 0.9" mass="{p.bottle_mass:.5f}" class="obj"/>
      <geom name="bottle_neck" type="cylinder" pos="0 0 {p.bottle_h/2+0.010:.5f}"
            size="{p.bottle_r*0.45:.5f} 0.010" rgba="0.85 0.85 0.85 1"
            mass="{p.bottle_mass*0.08:.5f}" class="obj"/>
      <site name="spout" pos="0 0 {p.bottle_h/2+0.022:.5f}" size="0.003" rgba="1 0 0 0.4"/>
    </body>

    {_water(p)}

    <!-- ------------------------------------------------------------ arms -->
    <body name="left_mount" pos="{-ARM_DX} {ARM_Y} 0">
      <attach model="so_arm" body="Base" prefix="left_"/>
    </body>
    <body name="right_mount" pos="{ARM_DX} {ARM_Y} 0">
      <attach model="so_arm" body="Base" prefix="right_"/>
    </body>

    <!-- ------------------------------------------------------- cameras -->
    <camera name="overhead" pos="{p.cam_jitter[0]:.4f} {-0.04+p.cam_jitter[1]:.4f} 0.78"
            euler="0 0 {p.cam_jitter[2]:.4f}" fovy="52"/>
    <camera name="front" pos="{p.cam_jitter[0]*0.5:.4f} -0.70 0.52"
            xyaxes="1 0 0 0 0.62 0.79" fovy="50"/>
    <camera name="side" pos="0.72 -0.30 0.44" xyaxes="0.45 0.89 0 -0.50 0.25 0.83" fovy="48"/>
  </worldbody>

  <equality>
{welds}  </equality>

</mujoco>
"""


def write_scene(p: SceneParams, path: str | None = None) -> str:
    path = path or os.path.join(ASSETS, f"_scene_seed{p.seed}.xml")
    with open(path, "w") as f:
        f.write(build_xml(p))
    return path


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    lvl = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    p = randomize(seed, lvl)
    path = write_scene(p)
    print("wrote", path)
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    print("compiled ok: nq=%d nu=%d nbody=%d ngeom=%d" % (m.nq, m.nu, m.nbody, m.ngeom))
