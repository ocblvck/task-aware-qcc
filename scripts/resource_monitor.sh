#!/usr/bin/env bash
# Append one line every INTERVAL seconds to logs/resource_usage.csv: GPU utilisation,
# memory, temperature and power per card, load average, RAM used, disk free.
# Started by run_revision.sh; safe to run on its own.
INTERVAL=${1:-30}
OUT=/home/chibuike/task-aware-qcc/logs/resource_usage.csv
[ -f "$OUT" ] || echo "time,gpu0_util,gpu0_mem_mb,gpu0_temp,gpu0_w,gpu1_util,gpu1_mem_mb,gpu1_temp,gpu1_w,gpu2_util,gpu2_mem_mb,gpu2_temp,gpu2_w,load1,ram_used_mb,disk_free_gb" > "$OUT"
while true; do
  g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ',' )
  l=$(cut -d' ' -f1 /proc/loadavg)
  r=$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "%d",(t-a)/1024}' /proc/meminfo)
  d=$(df -BG /home/chibuike | awk 'NR==2{gsub("G","",$4);print $4}')
  echo "$(date '+%Y-%m-%d %H:%M:%S'),${g}${l},${r},${d}" >> "$OUT"
  sleep "$INTERVAL"
done
