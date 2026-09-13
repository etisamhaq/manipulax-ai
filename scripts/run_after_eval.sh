#!/usr/bin/env bash
# Wait for the evaluation process to exit, then finalize.
cd "$(dirname "$0")/.."
while pgrep -f "dinnerbot.eval.run_seeds" > /dev/null; do sleep 15; done
# the running eval predates the numpy-JSON fix, so rebuild its report from the log
.venv/bin/python scripts/recover_eval_report.py artifacts/logs/eval.log artifacts/eval || true
exec bash scripts/finalize.sh
