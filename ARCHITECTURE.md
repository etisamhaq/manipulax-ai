# ManipulaX — Architecture

## Why this shape

A single end-to-end VLA driving a ten-step drawer → hand-off → plate → mug →
pour sequence, trained without a GPU and without any teleoperation data, does
not work. The system is therefore **hierarchical**, and each level is chosen for
what it is actually good at:

| level | rate | job | why not the level above/below |
|---|---|---|---|
| planner (SmolVLM-500M on OpenVINO INT8, with a deterministic fallback) | ~1 call / sub-task | sentence + overhead image → validated JSON plan | too slow for control, and it does not need to be fast: one call covers hundreds of control ticks |
| coordinator | per sub-task | arm assignment, hand-off FSM, shared-workspace sequencing, replan on failure | this is discrete logic with hard safety constraints; a network is the wrong tool |
| policy (ACT-style, OV FP16/INT8) | 16 control ticks / inference | pixels + proprioception + language token → action chunk | it cannot hold a ten-step plan in its head, and it does not have to |

The split is also what makes the reasoning *legible*: the plan is a JSON array a
judge can read, with a `why` field on every step.

**What the VLM actually contributes today.** Measured over four commands on one
scene, SmolVLM-500M returned four distinct replies but only one passed schema
validation, and that one echoed the prompt's format exemplar. The deterministic
planner supplied the rest. So the multimodal path is real and measured — a
genuine VLM, on the real camera frame, quantised to INT8 and timed on Intel
silicon — but it is not yet what decides the robot's behaviour, and the
fallback is doing the work. The honest summary is that the *plumbing* around a
small VLM (parse, repair, validate, reject, replan, and count all four) is the
contribution; the model itself needs to be perhaps 3B parameters to carry it.

## Bimanual coordination

Coordination is not decoration here — the layout makes it mandatory.

- The cutlery drawer is on the **left**; the fork belongs on the **right** of the
  mat. Each arm's comfortable reach is an annulus of roughly 0.11–0.30 m about
  its own base, and neither annulus covers both. So the left arm must pass the
  fork to the right arm above the middle of the table.
- The hand-off itself is atomic (`env.transfer`): the taker's weld is created and
  the giver's released inside a single simulation tick. The normal pinch detector
  cannot do this, because it deliberately refuses to grab anything the other arm
  is already holding — otherwise an arm brushing past would steal objects.
- The pour is the complementary case: the left arm holds the mug aloft while the
  right arm tips the bottle. Both arms occupy the shared centre of the workspace
  at the same time.

## Two things that were not obvious

**Wrist roll does not pour.** Rolling the wrist spins the bottle about its own
long axis and tips nothing; the tilt has to come from wrist *pitch*. But pitching
swings the spout several centimetres sideways, off the mug. So `pour()` pitches
in small increments and after each one re-aims the arm to bring the **spout** —
not the gripper — back over the mug, watching the bottle's own tilt angle until
it crosses the pouring threshold.

**A weld is the wrong model for a drawer.** Welding the gripper to the drawer
constrains all 6 DoF of a body that has exactly 1 (a slide joint). The solver
fights itself and the drawer does not move. The drawer is therefore pulled by
genuine finger contact on a thick, high-friction vertical post, and it is the one
object excluded from the grasp-assist path.

## Robustness

Randomisation is applied at **scene-generation** time, not by poking a compiled
model, because geom sizes, masses and frictions cannot be safely mutated after
compilation. `randomize(seed, level)` covers object shape, mass, friction,
placement, table and floor colour, key-light position and intensity, ambient
level, and camera pose. The expert runs under the same randomisation that the
evaluation uses, so the demonstrations carry the variation into the dataset.

On top of that the expert is closed-loop rather than open-loop:

- `reach()` measures the achieved tool-point position and re-aims, because the
  position servos droop by centimetres under load;
- `pick()` retries up to three times, backing off and approaching a little higher
  each time;
- `place()` corrects against the **object's** measured position, not the tool
  point, since a plate pinched at its rim hangs several centimetres from the tool
  frame;
- a failed sub-task triggers `Planner.replan()` against the *current* scene, so a
  knocked-over object produces a new plan rather than a stale script.

## Intel mapping

| component | precision | device | rationale |
|---|---|---|---|
| policy vision + transformer | INT8 (NNCF PTQ) | **CPU** | measured fastest of the three by 3x; see below |
| policy (alternative) | FP16 | iGPU | the only precision the iGPU accepts for this graph |
| planner VLM (SmolVLM-500M) | INT8 weights | iGPU / CPU | called ~once per sub-task, so latency is off the control path |
| MuJoCo physics + rendering | — | CPU + iGPU (EGL) | rendering goes through the Intel Arc iGPU |

Calibration data for the INT8 policy comes from the recorded demonstrations, so
the quantiser sees the real distribution of camera frames and joint states.

**Why the CPU and not the NPU.** The model was built to NPU constraints — static
shapes, fixed batch 1, no data-dependent control flow — and it does compile and
run there. It is simply slower: 20.1 ms on the NPU against 0.89 ms on the CPU,
which is slower even than unoptimised PyTorch. At 1.01 M parameters the fixed
dispatch cost dominates, and an accelerator built for sustained large models has
nothing to amortise it against. The iGPU sits between the two at 2.67 ms and
accepts only FP16 for this graph. The NPU additionally refuses the INT8 IR:
NNCF emits 128 per-axis scales against a quantised dimension of 32 and
`vpux-compiler` rejects it, so per-tensor quantisation or a QAT retrain would be
needed to try INT8 there. Measuring all three is what makes "ship it on the CPU"
an answer rather than an assumption.
