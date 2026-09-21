#!/usr/bin/env bash
# Conservative job queue for the SNCS-D-26-07200 supplement (protocol: revision/14_EXPERIMENT_DECISION.md).
#
# One GPU-intensive job at a time by default (MAX_CONCURRENT=1). The power strip feeding
# this machine has tripped under three concurrent A6000 loads, so the schedule trades
# wall-clock time for stability. Before every launch the target GPU must be idle and
# cool, the load average, free memory and free disk must be within limits; after every
# job the GPU must release its memory before the next launch. A resource logger runs
# alongside. Every launch and every terminal state is appended to a durable JSONL of job
# records. Jobs are idempotent and resumable; rerun the same launch command after any
# interruption:
#
#   setsid tmux -S ~/.tmux-taqcc.sock new-session -d -s supplement \
#     'bash scripts/run_supplement.sh 2>&1 | tee -a logs/supplement_queue.log'
set -uo pipefail
cd /home/chibuike/task-aware-qcc
PY=/home/chibuike/miniconda/envs/taqcc-grpo/bin/python
SOCK=/home/chibuike/.tmux-taqcc.sock
OUT=results/supplement
STATE=$OUT/queue_state.txt
RECORDS=$OUT/job_records.jsonl
mkdir -p logs $OUT $OUT/circuits
touch "$STATE"
export PYTHONPATH=src:/home/chibuike/quantum-cirq-opt/src
DATA=/home/chibuike/quantum-ml-iot-nid
BASE=models/sft_compress_e2_merged
SEEDS_EVAL=0,1,2,3,4
GRID6=0.0,0.01,0.03,0.05,0.1
PAIRS_REAL=0.0:0.0,0.0005:0.01,0.0005:0.02
MAX_CONCURRENT=${MAX_CONCURRENT:-1}
GPUS=(0 1 2)
THREADS=${OMP_THREADS:-16}          # of 128 logical cores
# launch limits
MIN_GPU_FREE_MB=40000; MAX_GPU_TEMP=75; MAX_LOAD1=40; MIN_RAM_FREE_MB=200000; MIN_DISK_FREE_GB=25
COMMIT=$(git rev-parse --short HEAD)

log(){ echo "[supplement $(date '+%m-%d %H:%M:%S')] $*"; }
record(){ # job status gpu attempt extra-json
  printf '{"job":"%s","status":"%s","gpu":%s,"attempt":%s,"time":"%s","commit":"%s","host":"%s"%s}\n' \
    "$1" "$2" "${3:-null}" "${4:-0}" "$(date '+%Y-%m-%d %H:%M:%S')" "$COMMIT" "$(hostname)" "${5:-}" >> "$RECORDS"
}

# ---------------------------------------------------------------- job table
declare -a NAME MARK PREQ CMD CKPT
add(){ NAME+=("$1"); MARK+=("$2"); PREQ+=("$3"); CMD+=("$4"); CKPT+=("${5:-}"); }
TRAIN_ARGS="--base-model $BASE --num-qubits 6 --max-steps 250 --gate-mode or \
 --util-metric mcc --util-reference clean_source --require-effective \
 --train-size 48 --test-size 24 --pool-size 4000 --noise-p1 0.01 \
 --save-steps 25 --save-total-limit 2 --auto-resume"
train(){ add "$1" "models/$1/checkpoint-250/trainer_state.json" "" \
  "$PY scripts/train_taskaware_grpo.py $TRAIN_ARGS --lr $2 --seed $3 --reward-log logs/$1.rewards.jsonl --output models/$1" "models/$1"; }
ds_tag(){ case "$1" in IoT_Original_Distribution.csv) echo IoT;; UNSW_NB15.csv) echo UNSW;; *) echo Bot;; esac; }
DSETS="IoT_Original_Distribution.csv UNSW_NB15.csv UNSW_2018_IoT_Botnet_Final_10_Best.csv"

# S1: six-qubit fusion grid on data splits 5 to 14 (Reviewer 1, comment 6).
SEEDS_EXT=5,6,7,8,9,10,11,12,13,14
GRID7=0.0,0.002,0.005,0.01,0.03,0.05,0.1
for ds in $DSETS; do t=$(ds_tag $ds)
  add "s1_6q_ext_$t" "$OUT/fusion_6q_splits5to14_$t.done" "" \
    "$PY scripts/eval_fusion_full.py --datasets $ds --data-dir $DATA --num-qubits 6 --train-size 200 --test-size 400 --seeds $SEEDS_EXT --noise-grid $GRID7 --no-ablation --resume --output $OUT/fusion_6q_splits5to14_$t.json && touch $OUT/fusion_6q_splits5to14_$t.done"
done
# S2: device-derived noise models, all six seven-qubit snapshots (Reviewer 1, comment 10).
for dev in fake_casablanca fake_jakarta fake_lagos fake_nairobi fake_oslo fake_perth; do
  for ds in $DSETS; do t=$(ds_tag $ds)
    add "s2_${dev}_$t" "$OUT/fusion_6q_${dev}_$t.done" "s1_*" \
      "$PY scripts/eval_fusion_full.py --datasets $ds --data-dir $DATA --num-qubits 6 --train-size 200 --test-size 400 --seeds $SEEDS_EVAL --device-noise $dev --no-ablation --resume --output $OUT/fusion_6q_${dev}_$t.json && touch $OUT/fusion_6q_${dev}_$t.done"
  done
done
# S3: reward subsample seeded by the training seed, confirmatory rate only.
trainrs(){ add "$1" "models/$1/checkpoint-250/trainer_state.json" "s2_*" \
  "$PY scripts/train_taskaware_grpo.py $TRAIN_ARGS --lr 5e-6 --seed $2 --reward-seed $2 --reward-log logs/$1.rewards.jsonl --output models/$1" "models/$1"; }
for s in 43 44 45 46; do trainrs rs_lr5_s$s $s; done
RS_MODELS=$(for s in 43 44 45 46; do echo -n "models/rs_lr5_s$s "; done)
RS_NAMES=rs_lr5_s43,rs_lr5_s44,rs_lr5_s45,rs_lr5_s46
add emit_rs "$OUT/emit_rs.done" "rs_*" \
  "$PY scripts/emit_replicate_circuits.py --models $RS_MODELS --base $BASE --num-qubits 6 --cache-dir $OUT/circuits --output $OUT/structure_rs.json && touch $OUT/emit_rs.done"
add audit_rs "$OUT/audit_rs.done" "emit_rs" \
  "$PY scripts/audit_effective_params.py --circuits $OUT/circuits --output $OUT/effective_params_rs.json && touch $OUT/audit_rs.done"
for ds in $DSETS; do t=$(ds_tag $ds)
  add "cmp_rs_$t" "$OUT/compression_matched_rs_$t.done" "audit_rs" \
    "$PY scripts/eval_compression_matched.py --datasets $ds --data-dir $DATA --num-qubits 6 --train-size 200 --test-size 400 --seeds $SEEDS_EVAL --noise-grid $GRID6 --extra-models $RS_NAMES --cache-dirs $OUT/circuits --output $OUT/compression_matched_rs_$t.json && touch $OUT/compression_matched_rs_$t.done"
done
add supplement_complete "$OUT/supplement_complete.done" "*" "date > $OUT/supplement_complete.done"

# ---------------------------------------------------------------- resource checks
gpu_q(){ nvidia-smi -i "$1" --query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | tr -d ' '; }
gpu_ready(){ # exit 0 if GPU $1 is idle, cool and free enough to launch
  local q; q=$(gpu_q "$1"); local used=${q%%,*}; local rest=${q#*,}; local free=${rest%%,*}; rest=${rest#*,}; local temp=${rest%%,*}; local util=${rest#*,}
  [ "$used" -lt 2000 ] && [ "$free" -gt $MIN_GPU_FREE_MB ] && [ "$temp" -lt $MAX_GPU_TEMP ] && [ "$util" -lt 10 ]
}
host_ready(){ # exit 0 if load, RAM and disk are within limits; prints the reason otherwise
  local l1; l1=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
  local ram; ram=$(awk '/MemAvailable/{printf "%d",$2/1024}' /proc/meminfo)
  local disk; disk=$(df -BG /home/chibuike | awk 'NR==2{gsub("G","",$4);print $4}')
  [ "$l1" -gt $MAX_LOAD1 ] && { echo "load $l1 > $MAX_LOAD1"; return 1; }
  [ "$ram" -lt $MIN_RAM_FREE_MB ] && { echo "ram free ${ram}MB < $MIN_RAM_FREE_MB"; return 1; }
  [ "$disk" -lt $MIN_DISK_FREE_GB ] && { echo "disk free ${disk}GB < $MIN_DISK_FREE_GB"; return 1; }
  return 0
}
# ---------------------------------------------------------------- scheduler
N=${#NAME[@]}
declare -A ATT; for ((i=0;i<N;i++)); do ATT[$i]=$(grep -c "^${NAME[$i]} LAUNCHED" "$STATE"); done
declare -A GPUJOB
done_(){ [ -f "${MARK[$1]}" ]; }
failed(){ grep -q "^${NAME[$1]} FAILED" "$STATE"; }
terminal(){ done_ $1 || failed $1; }
running(){ tmux -S "$SOCK" has-session -t "${NAME[$1]}" 2>/dev/null; }
n_running(){ local c=0 i; for ((i=0;i<N;i++)); do running $i && c=$((c+1)); done; echo $c; }
prereqs_met(){ local i=$1 pat j ok=1; [ -z "${PREQ[$i]}" ] && return 0
  for pat in ${PREQ[$i]}; do for ((j=0;j<N;j++)); do [ $j -eq $i ] && continue
    case "${NAME[$j]}" in $pat) terminal $j || ok=0;; esac; done; done; [ $ok -eq 1 ]; }
free_gpu(){ local g; for g in "${GPUS[@]}"; do local j=${GPUJOB[$g]:-}; if [ -z "$j" ] || ! running "$j"; then gpu_ready "$g" && { echo "$g"; return 0; }; fi; done; return 1; }
launch(){ local i=$1 g=$2
  log "launch ${NAME[$i]} on GPU $g (attempt $((ATT[$i]+1)))  gpu[$g]=$(gpu_q $g) load=$(cut -d' ' -f1 /proc/loadavg)"
  tmux -S "$SOCK" kill-session -t "${NAME[$i]}" 2>/dev/null
  setsid tmux -S "$SOCK" new-session -d -s "${NAME[$i]}" -c /home/chibuike/task-aware-qcc \
    "export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTHONPATH=src:/home/chibuike/quantum-cirq-opt/src; \
     ( ${CMD[$i]} ) 2>>logs/${NAME[$i]}.err | tee -a logs/${NAME[$i]}.log; echo EXIT=\${PIPESTATUS[0]} >> logs/${NAME[$i]}.log"
  ATT[$i]=$((ATT[$i]+1)); GPUJOB[$g]=$i
  echo "${NAME[$i]} LAUNCHED gpu=$g attempt=${ATT[$i]} $(date '+%m-%d %H:%M')" >> "$STATE"
  record "${NAME[$i]}" "launched" "$g" "${ATT[$i]}" ",\"checkpoint\":\"${CKPT[$i]}\",\"stdout\":\"logs/${NAME[$i]}.log\",\"stderr\":\"logs/${NAME[$i]}.err\""
}

log "queue started with $N jobs, MAX_CONCURRENT=$MAX_CONCURRENT, threads=$THREADS, commit $COMMIT"
tmux -S "$SOCK" has-session -t resmon 2>/dev/null || { setsid tmux -S "$SOCK" new-session -d -s resmon 'bash scripts/resource_monitor.sh 30'; log "resource monitor started (logs/resource_usage.csv)"; }
for ((i=0;i<N;i++)); do if running $i; then
  g=$(grep "^${NAME[$i]} LAUNCHED gpu=" "$STATE" | tail -1 | sed 's/.*gpu=\([0-9]\).*/\1/'); [ -z "$g" ] && g=0
  GPUJOB[$g]=$i; log "adopt ${NAME[$i]} on GPU $g"; fi; done
for ((i=0;i<N;i++)); do done_ $i && ! grep -q "^${NAME[$i]} DONE" "$STATE" && { echo "${NAME[$i]} DONE $(date '+%m-%d %H:%M')" >> "$STATE"; record "${NAME[$i]}" "done-before-start"; }; done

tick=0; paused=0
while true; do
  all_terminal=1
  for ((i=0;i<N;i++)); do
    if done_ $i; then
      if ! grep -q "^${NAME[$i]} DONE" "$STATE"; then
        echo "${NAME[$i]} DONE $(date '+%m-%d %H:%M')" >> "$STATE"; log "done ${NAME[$i]}"
        record "${NAME[$i]}" "done" "null" "${ATT[$i]}" ",\"checkpoint\":\"${CKPT[$i]}\""
        sleep 60   # let the card release memory and cool before anything else launches
      fi
      continue
    fi
    failed $i && continue
    all_terminal=0
    running $i && continue
    if [ ${ATT[$i]} -ge 3 ]; then
      echo "${NAME[$i]} FAILED $(date '+%m-%d %H:%M')" >> "$STATE"; log "FAILED ${NAME[$i]} after 3 attempts; see logs/${NAME[$i]}.err"
      record "${NAME[$i]}" "failed" "null" "${ATT[$i]}"; continue
    fi
    [ ${ATT[$i]} -gt 0 ] && log "note: ${NAME[$i]} was interrupted (attempt ${ATT[$i]} ended without its marker); will resume"
    prereqs_met $i || continue
    [ "$(n_running)" -ge "$MAX_CONCURRENT" ] && continue
    reason=$(host_ready) || { [ $paused -eq 0 ] && log "PAUSED: host not ready ($reason)"; paused=1; continue; }
    g=$(free_gpu) || { [ $paused -eq 0 ] && log "PAUSED: no GPU idle and cool (gpu0=$(gpu_q 0) gpu1=$(gpu_q 1) gpu2=$(gpu_q 2))"; paused=1; continue; }
    [ $paused -eq 1 ] && { log "resuming launches"; paused=0; }
    launch $i $g; sleep 90
  done
  [ $all_terminal -eq 1 ] && { log "ALL JOBS TERMINAL"; break; }
  tick=$((tick+1))
  [ $((tick % 15)) -eq 0 ] && log "status: running=[$(for ((i=0;i<N;i++)); do running $i && echo -n "${NAME[$i]} "; done)] done=$(grep -c ' DONE' "$STATE") failed=$(grep -c ' FAILED' "$STATE") gpu0=$(gpu_q 0) load=$(cut -d' ' -f1 /proc/loadavg)"
  sleep 120
done
tmux -S "$SOCK" kill-session -t resmon 2>/dev/null
log "campaign finished; state in $STATE, records in $RECORDS"
