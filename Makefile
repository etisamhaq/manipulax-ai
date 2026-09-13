# ManipulaX -- reproduce the whole demonstration.
#   make setup   once, then  make demo
PY ?= .venv/bin/python
export MUJOCO_GL := egl
SEEDS ?= 0-9
EPISODES ?= 40
EPOCHS ?= 8

.PHONY: help setup intel scene plan eval dataset train export bench demo clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:              ## create the venv and install dependencies
	uv venv --python 3.10 .venv
	uv pip install --python $(PY) -r requirements.txt

intel:              ## install the Intel iGPU/NPU userspace runtimes (needs sudo)
	sudo bash scripts/setup_intel_runtime.sh

scene:              ## generate and compile a randomised scene
	$(PY) -m dinnerbot.sim.build_scene 7 1.0

plan:               ## language + vision -> bimanual plan
	$(PY) -m dinnerbot.planner.demo_plan --backend vlm --device CPU --seeds 0,1

eval:               ## run $(SEEDS) randomised seeds and render the demo video
	$(PY) -m dinnerbot.eval.run_seeds --seeds $(SEEDS) --out artifacts/eval

dataset:            ## record scripted-expert demonstrations
	$(PY) -m dinnerbot.expert.record --episodes $(EPISODES) --out artifacts/dataset.npz

train:              ## behaviour-clone the action-chunk policy
	$(PY) -m dinnerbot.policy.train --data artifacts/dataset.npz --epochs $(EPOCHS)

export:             ## PyTorch -> OpenVINO IR (FP32 / FP16 / INT8)
	$(PY) -m dinnerbot.bench.export --ckpt artifacts/policy.pt --calib artifacts/calib.npz

bench:              ## sweep device x precision on Intel hardware
	$(PY) -m dinnerbot.bench.benchmark --devices auto --out artifacts/benchmark

demo: scene plan eval bench   ## the full demonstration path
	@echo "artifacts/eval/demo_all_seeds.mp4, artifacts/eval/report.md, artifacts/benchmark.md"

clean:
	rm -rf artifacts/eval artifacts/ov artifacts/benchmark.* dinnerbot/sim/assets/_scene_seed*.xml
