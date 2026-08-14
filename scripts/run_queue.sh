#!/usr/bin/env bash
# Unattended queue for the replication study and the reward-metric comparison.
#
#   stage 1  seed 43, accuracy reward   (launched separately, this script waits for it)
#   stage 2  seed 44, accuracy reward
#   stage 3  seed 42, MCC reward        -- the reward-metric counterfactual
#   stage 4  emit committee circuits from every new policy and summarise the structure
#
# Three runs per stage, one per GPU. Checkpoints land every 25 steps because the mains
# supply on this box trips; a coarser interval throws away the whole segment on a cut.
#
# The script is idempotent. A finished run is detected from its trainer_state, and a stage
# already recorded in the state file is skipped, so after a power cut you rerun the same
# command and it resumes from the last checkpoint.
#
# STAGGER: the three cards are brought up 90s apart. Three A6000s hitting full draw at the
# same moment is an inrush spike, and this supply has already tripped twice under the
# three-way load.
#
#   setsid tmux -S ~/.tmux-taqcc.sock new-session -d -s queue \
#     'bash scripts/run_queue.sh 2>&1 | tee logs/queue.log; sleep 604800'

set -uo pipefail
cd /home/chibuike/task-aware-qcc

PY=/home/chibuike/miniconda/envs/taqcc-grpo/bin/python
SOCK=/home/chibuike/.tmux-taqcc.sock
BASE=models/sft_compress_e2_merged
STATE=logs/queue_state.txt
mkdir -p logs results
touch "$STATE"

log(){ echo "[queue $(date '+%m-%d %H:%M:%S')] $*"; }

# A run is finished when its trainer_state records the final step.
finished(){
  local out=$1
  [ -f "models/$out/checkpoint-250/trainer_state.json" ] && return 0
  grep -qs "model saved under models/$out" "logs/$out.log" && return 0
  return 1
}

# True when a training process for this run is alive right now.
alive(){ pgrep -f "output models/$1( |$)" >/dev/null 2>&1; }

launch(){ # name lr seed gpu metric
  local name=$1 lr=$2 seed=$3 gpu=$4 metric=$5
  if finished "$name"; then log "skip $name (already complete)"; return; fi
  # Do not disturb a run that is already going. This makes the supervisor safe to restart
  # at any moment, which matters because editing this script while bash is part-way
  # through it is not safe and a restart is the clean way to pick up a change.
  if alive "$name"; then log "attach $name (already running)"; return; fi
  tmux -S "$SOCK" kill-session -t "$name" 2>/dev/null
  log "launch $name  lr=$lr seed=$seed metric=$metric gpu=$gpu"
  setsid tmux -S "$SOCK" new-session -d -s "$name" -c /home/chibuike/task-aware-qcc \
    "export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=16 \
       PYTHONPATH=src:/home/chibuike/quantum-cirq-opt/src; \
     $PY scripts/train_taskaware_grpo.py \
       --base-model $BASE --num-qubits 6 --max-steps 250 --gate-mode or \
       --lr $lr --seed $seed --util-metric $metric --save-steps 25 --auto-resume \
       --output models/$name 2>&1 | tee logs/$name.log; echo EXIT=\$?; sleep 604800"
}

wait_for(){ # names...
  local pending=1 waited=0
  while [ $pending -eq 1 ]; do
    pending=0
    for n in "$@"; do finished "$n" || pending=1; done
    if [ $pending -eq 1 ]; then
      # A run whose tmux session has died without finishing is relaunched by the
      # stage loop on the next pass rather than blocking the queue forever.
      sleep 120; waited=$((waited+2))
      if [ $((waited % 20)) -eq 0 ]; then
        log "waiting (${waited}m): $(for n in "$@"; do finished "$n" && echo -n "$n=done " || echo -n "$n=running "; done)"
      fi
      if [ $waited -gt 600 ]; then log "TIMEOUT after ${waited}m; continuing"; return 1; fi
    fi
  done
  return 0
}

stage(){ # tag metric seed  name1 name2 name3
  local tag=$1 metric=$2 seed=$3; shift 3
  if grep -qx "$tag" "$STATE"; then log "stage $tag already recorded, skipping"; return; fi
  local lrs=(5e-6 7.5e-6 1e-5) i=0
  for n in "$@"; do
    launch "$n" "${lrs[$i]}" "$seed" "$i" "$metric"
    i=$((i+1))
    [ $i -lt 3 ] && sleep 90   # stagger the spin-up, see STAGGER note at the top
  done
  wait_for "$@"
  # one retry pass for anything the reboot or an OOM killed
  local retry=0
  for n in "$@"; do finished "$n" || retry=1; done
  if [ $retry -eq 1 ]; then
    log "stage $tag: relaunching unfinished runs once"
    i=0; for n in "$@"; do finished "$n" || launch "$n" "${lrs[$i]}" "$seed" "$i" "$metric"; i=$((i+1)); done
    wait_for "$@"
  fi
  echo "$tag" >> "$STATE"
  log "stage $tag complete"
}

log "queue started"

# ---- stage 1: seed 43, accuracy reward ----
stage stage1 accuracy 43 rep_lr5_s43 rep_lr75_s43 rep_lr10_s43

# ---- bank the stage-1 result immediately: emission is cheap and the supply is unreliable
emit(){
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=src:/home/chibuike/quantum-cirq-opt/src
  $PY scripts/emit_replicate_circuits.py --models "$@" \
      --output results/replicates_structure.json 2>&1
}
if grep -qx stage1 "$STATE" && ! grep -qx emit1 "$STATE"; then
  log "banking stage-1 circuits"
  emit models/grpo_fix_lr5 models/grpo_fix_lr75 models/grpo_fix_lr10 \
       models/rep_lr5_s43 models/rep_lr75_s43 models/rep_lr10_s43
  echo emit1 >> "$STATE"
  log "stage-1 circuits banked in results/replicates_structure.json"
fi

# Stages 2 and 3 are off by default. Two seeds per learning rate already answer the
# question qualitatively (do repeated runs emit the same circuits?), and on a supply that
# cuts every few hours the marginal third seed is not worth the risk of landing nothing.
# Re-enable with:  RUN_FULL=1 bash scripts/run_queue.sh
if [ "${RUN_FULL:-0}" = "1" ]; then
  # ---- stage 2: seed 44, accuracy reward ----
  stage stage2 accuracy 44 rep_lr5_s44 rep_lr75_s44 rep_lr10_s44

  # ---- stage 3: seed 42, MCC reward (the reward-metric counterfactual) ----
  stage stage3 mcc 42 mcc_lr5_s42 mcc_lr75_s42 mcc_lr10_s42
else
  log "stages 2 and 3 skipped (set RUN_FULL=1 to enable)"
fi

# ---- stage 4: emit committee circuits and summarise structure ----
if ! grep -qx stage4 "$STATE"; then
  log "stage 4: emitting committee circuits from every new policy"
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=src:/home/chibuike/quantum-cirq-opt/src
  # emit_replicate_circuits.py skips adapters that do not exist, so this covers whichever
  # stages actually ran.
  emit models/grpo_fix_lr5 models/grpo_fix_lr75 models/grpo_fix_lr10 \
       models/rep_lr5_s43 models/rep_lr75_s43 models/rep_lr10_s43 \
       models/rep_lr5_s44 models/rep_lr75_s44 models/rep_lr10_s44 \
       models/mcc_lr5_s42 models/mcc_lr75_s42 models/mcc_lr10_s42
  echo stage4 >> "$STATE"
  log "stage 4 complete"
fi

log "QUEUE FINISHED -- results/replicates_structure.json is ready"
