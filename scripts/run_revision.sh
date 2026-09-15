#!/usr/bin/env bash
# Three-GPU job queue for the SNCS-D-26-07200 revision campaign.
#
# Jobs are listed in priority order with a completion marker and optional prerequisites.
# The loop assigns the first eligible, unfinished, not-running job to any free GPU, so
# two GPUs start on training at once while the third works through the short
# evaluations, then all three train, then the ten-qubit evaluations follow.
#
# Restart-safe. Every job is idempotent: training auto-resumes from its last checkpoint
# (every 25 steps), fusion evaluations resume per noise cell, the emitter skips models it
# has already summarised. After a power cut, rerun the same launch command:
#
#   setsid tmux -S ~/.tmux-taqcc.sock new-session -d -s revision \
#     'bash scripts/run_revision.sh 2>&1 | tee -a logs/revision_queue.log'
#
# A job whose tmux session dies without producing its marker is relaunched up to three
# times, then recorded as FAILED in the state file and left in place; nothing is hidden.
set -uo pipefail
cd /home/chibuike/task-aware-qcc
PY=/home/chibuike/miniconda/envs/taqcc-grpo/bin/python
SOCK=/home/chibuike/.tmux-taqcc.sock
OUT=results/corrected
STATE=$OUT/queue_state.txt
mkdir -p logs $OUT $OUT/circuits
touch "$STATE"
export PYTHONPATH=src:/home/chibuike/quantum-cirq-opt/src
DATA=/home/chibuike/quantum-ml-iot-nid
BASE=models/sft_compress_e2_merged
SEEDS_EVAL=0,1,2,3,4
GRID6=0.0,0.01,0.03,0.05,0.1
PAIRS_REAL=0.0:0.0,0.0005:0.01,0.0005:0.02

log(){ echo "[revision $(date '+%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------- job table
# name | marker | prereq globs (space separated, may be empty) | command
declare -a NAME MARK PREQ CMD
add(){ NAME+=("$1"); MARK+=("$2"); PREQ+=("$3"); CMD+=("$4"); }

TRAIN_ARGS="--base-model $BASE --num-qubits 6 --max-steps 250 --gate-mode or \
 --util-metric mcc --util-reference clean_source --require-effective \
 --train-size 48 --test-size 24 --pool-size 4000 --noise-p1 0.01 \
 --save-steps 25 --auto-resume"
train(){ # name lr seed
  add "$1" "models/$1/checkpoint-250/trainer_state.json" "" \
    "$PY scripts/train_taskaware_grpo.py $TRAIN_ARGS --lr $2 --seed $3 \
       --reward-log logs/$1.rewards.jsonl --output models/$1"
}
ds_tag(){ case "$1" in IoT_Original_Distribution.csv) echo IoT;; UNSW_NB15.csv) echo UNSW;; *) echo Bot;; esac; }
DSETS="IoT_Original_Distribution.csv UNSW_NB15.csv UNSW_2018_IoT_Botnet_Final_10_Best.csv"

# The first two training jobs go out immediately so the critical path starts at once.
train corr_lr5_s42 5e-6 42
train corr_lr5_s43 5e-6 43

# E2: pre-correction seed-43 circuits at the full protocol, same file layout as Table 3.
for ds in $DSETS; do t=$(ds_tag $ds)
  add "cmp_pre_$t" "$OUT/compression_matched_precorrection_$t.done" "" \
    "$PY scripts/eval_compression_matched.py --datasets $ds --data-dir $DATA --num-qubits 6 \
       --train-size 200 --test-size 400 --seeds $SEEDS_EVAL --noise-grid $GRID6 \
       --extra-models rep_lr5_s43,rep_lr75_s43,rep_lr10_s43,sft_warmup \
       --cache-dirs results/replicate_circuits \
       --output $OUT/compression_matched_precorrection_$t.json \
     && touch $OUT/compression_matched_precorrection_$t.done"
done
# E4: SVM C sweep on the same Gram matrices at the transition regime and the coupled p1=0.01.
for ds in $DSETS; do t=$(ds_tag $ds)
  add "csweep_$t" "$OUT/c_sweep_$t.done" "" \
    "$PY scripts/eval_fusion_full.py --datasets $ds --data-dir $DATA --num-qubits 6 \
       --train-size 200 --test-size 400 --seeds $SEEDS_EVAL --noise-grid 0.0,0.002,0.005,0.01 \
       --no-ablation --c-values 0.1,1,10,100 --resume --output $OUT/c_sweep_$t.json \
     && touch $OUT/c_sweep_$t.done"
done
# E3 eight qubits on the two datasets the article did not cover.
for ds in IoT_Original_Distribution.csv UNSW_2018_IoT_Botnet_Final_10_Best.csv; do t=$(ds_tag $ds)
  add "real8q_$t" "$OUT/fusion_realistic_8q_$t.done" "" \
    "$PY scripts/eval_fusion_full.py --datasets $ds --data-dir $DATA --num-qubits 8 \
       --train-size 200 --test-size 400 --seeds $SEEDS_EVAL --noise-pairs $PAIRS_REAL \
       --no-ablation --resume --output $OUT/fusion_realistic_8q_$t.json \
     && touch $OUT/fusion_realistic_8q_$t.done"
done
# E1: the remaining thirteen training runs. Seeds fixed in advance: 42 43 44 45 46.
for s in 44 45 46; do train corr_lr5_s$s 5e-6 $s; done
for s in 42 43 44 45 46; do train corr_lr75_s$s 7.5e-6 $s; done
for s in 42 43 44 45 46; do train corr_lr10_s$s 1e-5 $s; done
# E3 ten qubits, sixteen hours each, resumable per noise cell.
for ds in IoT_Original_Distribution.csv UNSW_2018_IoT_Botnet_Final_10_Best.csv; do t=$(ds_tag $ds)
  add "real10q_$t" "$OUT/fusion_realistic_10q_$t.done" "" \
    "$PY scripts/eval_fusion_full.py --datasets $ds --data-dir $DATA --num-qubits 10 \
       --train-size 200 --test-size 400 --seeds $SEEDS_EVAL --noise-pairs $PAIRS_REAL \
       --no-ablation --resume --output $OUT/fusion_realistic_10q_$t.json \
     && touch $OUT/fusion_realistic_10q_$t.done"
done
# Post-training pipeline. Prerequisites are globs over job names; a job is eligible once
# every matching job is DONE or FAILED (a failed run is skipped by the emitter, not hidden).
CORR_MODELS=$(for lr in lr5 lr75 lr10; do for s in 42 43 44 45 46; do echo -n "models/corr_${lr}_s$s "; done; done)
add emit_corr "$OUT/emit_corr.done" "corr_*" \
  "$PY scripts/emit_replicate_circuits.py --models $CORR_MODELS --base $BASE --num-qubits 6 \
     --cache-dir $OUT/circuits --output $OUT/structure.json && touch $OUT/emit_corr.done"
add audit_corr "$OUT/audit_corr.done" "emit_corr" \
  "$PY scripts/audit_effective_params.py --circuits $OUT/circuits --output $OUT/effective_params.json \
   && touch $OUT/audit_corr.done"
CORR_NAMES=$(for lr in lr5 lr75 lr10; do for s in 42 43 44 45 46; do echo -n "corr_${lr}_s$s,"; done; done)
for ds in $DSETS; do t=$(ds_tag $ds)
  add "cmp_corr_$t" "$OUT/compression_matched_corrected_$t.done" "audit_corr" \
    "$PY scripts/eval_compression_matched.py --datasets $ds --data-dir $DATA --num-qubits 6 \
       --train-size 200 --test-size 400 --seeds $SEEDS_EVAL --noise-grid $GRID6 \
       --extra-models ${CORR_NAMES}rep_lr5_s43,rep_lr75_s43,rep_lr10_s43,sft_warmup \
       --cache-dirs $OUT/circuits,results/replicate_circuits \
       --output $OUT/compression_matched_corrected_$t.json \
     && touch $OUT/compression_matched_corrected_$t.done"
done
add rescore_corr "$OUT/rescore_corr.done" "emit_corr" \
  "$PY scripts/rescore_reward_loop.py --circuits $OUT/circuits --models ${CORR_NAMES%,} \
       --seeds 42,43,44,45,46 --train-size 48 --test-size 24 --util-metric mcc \
       --util-reference clean_source --require-effective \
       --output $OUT/rescoring_corrected_circuits_corrected_objective.json \
   && $PY scripts/rescore_reward_loop.py --circuits $OUT/circuits --models ${CORR_NAMES%,} \
       --seeds 42,43,44,45,46 --train-size 16 --test-size 8 --util-metric accuracy \
       --util-reference noisy_source \
       --output $OUT/rescoring_corrected_circuits_original_objective.json \
   && touch $OUT/rescore_corr.done"
add campaign_complete "$OUT/campaign_complete.done" "*" "date > $OUT/campaign_complete.done"

# ---------------------------------------------------------------- scheduler
N=${#NAME[@]}
declare -A ATT GPUOF
for ((i=0;i<N;i++)); do ATT[$i]=0; done
declare -a GPUJOB=("" "" "")
terminal(){ [ -f "${MARK[$1]}" ] && return 0; grep -qx "${NAME[$1]} FAILED" "$STATE"; }
running(){ tmux -S "$SOCK" has-session -t "${NAME[$1]}" 2>/dev/null; }
prereqs_met(){ local i=$1 pat j ok=1; [ -z "${PREQ[$i]}" ] && return 0
  for pat in ${PREQ[$i]}; do for ((j=0;j<N;j++)); do
    [ $j -eq $i ] && continue
    case "${NAME[$j]}" in $pat) terminal $j || ok=0;; esac
  done; done; [ $ok -eq 1 ]; }
launch(){ local i=$1 g=$2
  log "launch ${NAME[$i]} on GPU $g (attempt $((ATT[$i]+1)))"
  tmux -S "$SOCK" kill-session -t "${NAME[$i]}" 2>/dev/null
  setsid tmux -S "$SOCK" new-session -d -s "${NAME[$i]}" -c /home/chibuike/task-aware-qcc \
    "export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=12 PYTHONPATH=src:/home/chibuike/quantum-cirq-opt/src; \
     ( ${CMD[$i]} ) 2>&1 | tee -a logs/${NAME[$i]}.log; echo EXIT=\$? >> logs/${NAME[$i]}.log"
  ATT[$i]=$((ATT[$i]+1)); GPUJOB[$g]=$i; GPUOF[$i]=$g
  echo "${NAME[$i]} LAUNCHED gpu=$g attempt=${ATT[$i]} $(date '+%m-%d %H:%M')" >> "$STATE"
}

log "queue started with $N jobs"
# Adopt sessions that are already alive (supervisor restarted while jobs ran).
for ((i=0;i<N;i++)); do if running $i; then
  g=$(grep "^${NAME[$i]} LAUNCHED gpu=" "$STATE" | tail -1 | sed 's/.*gpu=\([0-9]\).*/\1/')
  [ -z "$g" ] && g=0; GPUJOB[$g]=$i; GPUOF[$i]=$g; ATT[$i]=$(grep -c "^${NAME[$i]} LAUNCHED" "$STATE"); log "adopt ${NAME[$i]} on GPU $g"; fi; done

tick=0
while true; do
  all_terminal=1
  for ((i=0;i<N;i++)); do
    if [ -f "${MARK[$i]}" ]; then
      grep -qx "${NAME[$i]} DONE" "$STATE" || { echo "${NAME[$i]} DONE $(date '+%m-%d %H:%M')" >> "$STATE"; log "done ${NAME[$i]}"; }
      continue
    fi
    grep -qx "${NAME[$i]} FAILED" "$STATE" && continue
    all_terminal=0
    running $i && continue
    if [ ${ATT[$i]} -ge 3 ]; then
      echo "${NAME[$i]} FAILED $(date '+%m-%d %H:%M')" >> "$STATE"; log "FAILED ${NAME[$i]} after 3 attempts; see logs/${NAME[$i]}.log"; continue
    fi
    prereqs_met $i || continue
    for g in 0 1 2; do
      j=${GPUJOB[$g]}
      if [ -z "$j" ] || ! running $j; then launch $i $g; sleep 90; break; fi
    done
  done
  [ $all_terminal -eq 1 ] && { log "ALL JOBS TERMINAL"; break; }
  tick=$((tick+1))
  if [ $((tick % 15)) -eq 0 ]; then
    log "status: $(for ((i=0;i<N;i++)); do if [ -f "${MARK[$i]}" ]; then :; elif running $i; then echo -n "${NAME[$i]}=run "; fi; done) done=$(grep -c ' DONE' "$STATE") failed=$(grep -c ' FAILED' "$STATE")"
  fi
  sleep 120
done
log "campaign finished; state in $STATE"
