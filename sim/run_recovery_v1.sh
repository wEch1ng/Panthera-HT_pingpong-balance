#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
recovery_python=${PYTHON:-python}
recovery_run=${RECOVERY_RUN:-panthera_d405_recovery_v1}
recovery_output=outputs/controller/$recovery_run
if [[ ! -f assets/panthera/scene_balance.xml ]]; then
  "$recovery_python" -s -m sim.build_model
fi
"$recovery_python" -s -u -m sim.train_controller \
  --name "$recovery_run" --recovery-task \
  --init weights/d405_rgb_balance.pt \
  --iters 1500 --steps 128 --workers 2 --envs-per-worker 1 \
  --device cpu --act-hist 3 --max-latency 2 --lr 0.00005 --save-every 10 \
  --target-height 0.527232155 --target-u 0 --target-w 0 --energy-weight 0.02
echo 'Training finished; starting independent RGB-D baseline and final evaluations.'
"$recovery_python" -s -u -m sim.eval_recovery \
  --controller "$recovery_output/initial.pt" --episodes 5 --level 1 --latency 2 \
  --json "$recovery_output/initial_eval.json"
"$recovery_python" -s -u -m sim.eval_recovery \
  --controller "$recovery_output/last.pt" --episodes 5 --level 1 --latency 2 \
  --json "$recovery_output/final_eval.json"
echo 'Recovery training and independent evaluations complete.'
