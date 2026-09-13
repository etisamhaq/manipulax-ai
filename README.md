# ManipulaX — Bimanual VLA Manipulation with Multi-Modal Reasoning

Intel Physical AI Online Challenge · *Setting Up a Dinner Table* · simulation-first

Two simulated **SO-ARM100/SO-101** arms set a dinner table in MuJoCo from a
natural-language instruction. A vision-language planner turns the sentence and
the overhead camera view into a validated bimanual plan; a language-conditioned
ACT-style visuomotor policy executes it; everything runs through **OpenVINO** on
an **Intel Core Ultra** client machine.

---

## 1. What the system does

```
 "Open the drawer, put the fork on the right of the mat, centre the plate,
  then hold the mug and pour water into it."
                              │
            ┌─────────────────▼──────────────────┐
            │  PLANNER                           │   overhead camera + sentence
            │  SmolVLM-500M (OV INT8) proposes   │   ~1 call per sub-task
            │  → schema validate / repair        │   invalid ⇒ rule planner
            │  → deterministic rule planner      │   (today: 3 of 4 commands)
            └─────────────────┬──────────────────┘
                              │  [{skill, object, arm, target, why}, …]
            ┌─────────────────▼──────────────────┐
            │  COORDINATOR                       │   arm assignment, shared-
            │  hand-off FSM · replan on failure  │   workspace sequencing
            └─────────────────┬──────────────────┘
                              │  one sub-task + language token at a time
            ┌─────────────────▼──────────────────┐
            │  POLICY  (ACT-style, OV FP16/INT8) │   3 cameras + 24-D state
            │  → 16-step × 12-DoF action chunk   │   2.6 ms / inference
            └─────────────────┬──────────────────┘
                              │
                    MuJoCo · dual SO-ARM100
```

The two **dual-arm coordination moments** the challenge asks for are both in the
task, and both are forced by the layout rather than staged:

- **Hand-off** — the cutlery drawer is on the *left*, but the fork belongs on the
  *right* of the placemat. Neither arm can do it alone, so the left arm passes
  the fork to the right arm above the middle of the table.
- **Hold-while-acting** — the left arm holds the mug in the air while the right
  arm tips the bottle and pours into it.

---

## 2. Quick start

```bash
git clone git@github.com:etisamhaq/manipulax-ai.git && cd manipulax-ai
uv venv --python 3.10 .venv                       # or python -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# one-off: let OpenVINO see the iGPU and NPU (Ubuntu 22.04, needs sudo)
sudo bash scripts/setup_intel_runtime.sh          # then log out and back in

export MUJOCO_GL=egl                              # headless rendering
```

Reproduce the whole demonstration:

```bash
make demo
```

or step by step:

```bash
# 1. look at a randomised scene
python -m dinnerbot.sim.build_scene 7 1.0

# 2. language + vision -> bimanual plan
python -m dinnerbot.planner.demo_plan --backend vlm --device CPU --seeds 0

# 3. evaluate across 10 randomised seeds and render the demo video
python -m dinnerbot.eval.run_seeds --seeds 0-9 --out artifacts/eval

# 4. record demonstrations, train, export, benchmark
python -m dinnerbot.expert.record   --episodes 40 --out artifacts/dataset.npz
python -m dinnerbot.policy.train    --data artifacts/dataset.npz --epochs 8
python -m dinnerbot.bench.export    --ckpt artifacts/policy.pt
python -m dinnerbot.bench.benchmark --devices auto
```

---

## 2b. Results

Every number below is generated, not typed: `scripts/make_submission_report.py`
reads the JSON each stage writes and assembles `SUBMISSION.md`.

### Task, 10 randomised seeds (`level=1.0`)

| sub-task | arm | rate |
|---|---|---|
| open the drawer | left | **10/10** |
| plate on the mat | right | 8/10 |
| arm-to-arm hand-off | both | 6/10 |
| fork on the mat | right | 3/10 |
| mug on the mat | left | 3/10 |
| pour water into the mug | both | **0/10** |

**Mean 3.00 / 6 sub-tasks (50%); 0/10 episodes fully complete.**

Read honestly: the drawer and the hand-off — the two things the challenge singles
out as hard — work, and the hand-off is verified rather than assumed (the taker
must really have the fork between its fingers or `transfer()` refuses). The tail
of the sequence does not hold up. Fork and mug placement each succeed 3/10, and
**the pour never lands water in the mug across 10 seeds**, so no episode is fully
complete. Failures compound: an episode that drops the fork also tends to be the
one whose arm is badly posed for the mug.

### Inference on the Core Ultra 7 155H

| precision | p50 | p99 | weights | vs PyTorch-CPU |
|---|---|---|---|---|
| FP32 | 2.04 ms | 2.60 ms | 3.90 MB | 2.9x |
| FP16 | 2.06 ms | 2.76 ms | 1.95 MB | 2.8x |
| **INT8** | **1.62 ms** | **1.82 ms** | **1.20 MB** | **3.6x** |

One inference produces a 16-step action chunk, so INT8 sustains ~9900 Hz of
control — about 300x more than the 30 Hz the task needs. Latency is not the
binding constraint here; task success is.

Measured on **CPU only** — the iGPU and NPU are not yet visible to OpenVINO on
this machine (see `scripts/setup_intel_runtime.sh`).

| artifact | what it holds |
|---|---|
| `SUBMISSION.md` | one-page summary: deliverables, task success, benchmark, fidelity |
| `artifacts/eval/report.md` | per-sub-task success across the 10 randomised seeds |
| `artifacts/eval/demo_all_seeds.mp4` | the demonstration video (plus `seed_0NN.mp4` per seed) |
| `artifacts/benchmark.md` | device x precision latency / throughput sweep |
| `artifacts/accuracy.md` | how far each quantised precision drifts from PyTorch |

The HUD burned into every video frame shows the natural-language command, the
sub-task in progress and which arm owns it, the inference device and latency,
and a row of indicators that light up as each sub-task completes — so the video
evidences the table rather than merely accompanying it.

---

## 3. The MuJoCo environment

`dinnerbot/sim/build_scene.py` **generates** the scene XML from a seed rather
than shipping one fixed file. Geom sizes, masses and frictions cannot be safely
mutated on an already-compiled model, so randomising them properly means
regenerating and recompiling — which MuJoCo does in ~0.3 s.

| | |
|---|---|
| arms | 2 × SO-ARM100 from `mujoco_menagerie/trs_so_arm100` (Apache-2.0), 6 DoF each, attached with `<attach prefix=…>` |
| cameras | `overhead`, `front`, `side`, plus a wrist camera per arm |
| articulation | drawer on a slide joint with a vertical pull-post |
| objects | plate, mug (hollow, 14-plank ring), fork, spoon, bottle, 10 water particles |
| control | 12 position actuators (5 arm joints + jaw, per arm) |

### Domain randomisation (`randomize(seed, level)`)

| group | randomised |
|---|---|
| shape | plate radius ±13 %, mug radius/height ±15 %, fork length ±12 %, bottle radius ±10 % |
| dynamics | masses ×0.5–2.6, sliding friction 0.35–2.2, torsional friction ×0.5–2.0, drawer damping ×0.6–2.1 |
| placement | every object's x, y and yaw within its spawn zone; cutlery position inside the drawer |
| appearance | table and floor colour, plate and mug colour |
| lighting | key-light position, diffuse intensity ×0.5–1.0, ambient 0.12–0.38 |
| sensing | camera translation ±12 mm and yaw ±0.02 rad |

`level=0` reproduces the nominal scene; `level=1.0` is what the reported numbers
use.

---

## 4. Policy

`dinnerbot/policy/model.py` — a compact ACT-style transformer, **1.01 M
parameters**:

- 3 camera images (96×96) through a shared 4-layer CNN → 36 tokens each, plus a
  learned per-view embedding so the model knows which camera it is looking at
- 24-D proprioception (12 joint positions + 12 velocities) → 1 token
- **language**: the active sub-task → 1 embedding token. This is what makes the
  policy language-conditioned rather than a trajectory replayer.
- 2-layer encoder / 2-layer decoder → **16-step × 12-DoF action chunk**

Action chunking matters twice over: it is what makes behaviour cloning robust to
compounding error, and it means one inference covers 16 control ticks, so the
sustainable control rate is `16 × inferences/s` — the number reported in the
benchmark.

Everything is fixed-shape and free of data-dependent control flow, because the
OpenVINO **NPU plugin will not accept dynamic shapes**.

### Training data

There is no teleoperation rig, so demonstrations come from a **scripted expert**
(`dinnerbot/expert/`): damped-least-squares IK over the 5 positioning joints,
plus motion primitives (`pick`, `place`, `open_drawer`, `handoff`, `pour`). The
expert runs under full domain randomisation, so the dataset inherits the
randomisation for free and every frame is automatically annotated with the
sub-task language.

---

## 5. Intel deployment

| stage | what OpenVINO does |
|---|---|
| policy | `torch.jit.trace` → `ov.convert_model` → IR; FP32 / FP16 / **NNCF INT8 PTQ** calibrated on recorded frames |
| planner | SmolVLM-500M exported with `optimum-cli export openvino --weight-format int8` |
| runtime | `PERFORMANCE_HINT=LATENCY`, static shapes, device selectable per component |

`dinnerbot/bench/benchmark.py` sweeps {device} × {precision} and reports compile
time, p50/p90/p99 latency, throughput, the sustainable control rate, and the
speed-up over the PyTorch-CPU baseline. See `artifacts/benchmark.md`.

---

## 6. Honest limitations

**What does not work.** The pour never succeeds across 10 randomised seeds
(0/10), so no episode completes all six sub-tasks. `pour()` does the right
thing in principle — it tips with wrist *pitch* (rolling spins the bottle about
its own axis and pours nothing) and re-aims the arm after each increment to keep
the *spout*, not the gripper, over the mug — and an earlier, less randomised run
did land water on 2 of 6 seeds. Under full randomisation it does not. Fork and
mug placement each land 3/10. These are real failures, not tuning left undone,
and the eval table reports them as measured.

The rest are deliberate, documented simplifications:

1. **Grasping uses a pinch detector + weld**, not friction-only contact. When an
   arm's jaw is commanded closed and a graspable body lies inside a cylindrical
   capture volume along the gripper's approach axis (70 mm deep, 42 mm radius), a
   MuJoCo weld equality is activated. The **drawer is the exception** — it is
   pulled by real finger contact, because welding a 6-DoF constraint onto its
   1-DoF slide joint over-constrains the solver and it refuses to move.
2. **Pouring is particle-based.** Ten spheres are released from the bottle spout
   once its tilt exceeds 55°; success is counted by how many settle inside the
   mug. There is no fluid solver.
3. **The VLM is not good enough to plan on its own, and the numbers say so.**
   On four different commands SmolVLM-500M produced four *distinct* replies
   (so it is reading the input), but only **1 of 4** survived schema
   validation, and that one was wrong — it echoed the format exemplar from the
   prompt. The three usable plans came from the deterministic rule planner.
   What is real here is the *architecture*: a genuine multimodal model runs on
   OpenVINO INT8 over the actual camera frame at ~9.7 s/call, its output is
   parsed, repaired and validated, and anything that fails is replaced rather
   than executed. `demo_plan` prints `vlm_ok` / `vlm_invalid` / `fallbacks`,
   plan distinctiveness across commands, and agreement with the rule planner,
   so none of this is hidden. A 3B-class VLM would very likely close the gap;
   there was not time to export and test one.

   The repair layer handles
   trailing commas, single quotes, unquoted keys and unclosed brackets, and
   salvages individual step objects when the array as a whole will not parse;
   the validator rejects unknown skills, unknown arms, non-string object names
   and out-of-reach picks, so a model's output can never crash the runtime.
4. **The scene description handed to the planner reads object poses from the
   simulator.** The VLM does receive the real overhead image, but the textual
   scene state is privileged. A deployment would swap in a detector on that same
   frame; the interface is already in `describe_scene()`.
5. **Hardware is Core Ultra Series 1** (Core Ultra 7 155H, Meteor Lake), not the
   Series 2/3 the brief prefers. OpenVINO targets them identically; only the
   absolute numbers would shift.
6. **The arm model is SO-ARM100**, the predecessor of the SO-101, because that is
   what `mujoco_menagerie` ships. Same kinematic family, same 6-DoF layout.

## 7. Licences

`dinnerbot/sim/assets/so_arm100*` and the meshes under `so_arm100_meshes/` come
from [`mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
(`trs_so_arm100`, Apache-2.0) — see `SO_ARM100_LICENSE`. The scene XML around
them, and all code in `dinnerbot/`, is this project's own.
