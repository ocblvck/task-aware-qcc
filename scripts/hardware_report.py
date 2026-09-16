#!/usr/bin/env python
"""Snapshot of the machine before a campaign stage: GPUs, CPU, memory, disk, processes,
software. Written atomically to results/corrected/hardware_report_<stamp>.json and
printed as a short table."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e: return f"ERR {e}"
rep = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": sh("hostname"), "uptime": sh("uptime -p"),
       "boot": sh("uptime -s"), "kernel": sh("uname -r")}
gq = sh("nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits")
rep["gpus"] = [dict(zip(["index","name","mem_total_mb","mem_used_mb","mem_free_mb","util_pct","temp_c","power_w","power_limit_w"], [x.strip() for x in l.split(",")])) for l in gq.splitlines() if l.strip()]
rep["gpu_processes"] = sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader")
rep["driver"] = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
rep["cpu"] = {"model": sh("lscpu | grep 'Model name' | sed 's/.*: *//'"), "cores_logical": os.cpu_count(),
              "cores_physical": sh("lscpu | grep '^Core(s) per socket' | awk '{print $NF}'"),
              "load_1_5_15": os.getloadavg()}
mem = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal","MemAvailable","SwapTotal","SwapFree")}
rep["memory_mb"] = mem
du = shutil.disk_usage("/home/chibuike"); rep["disk_home_gb"] = {"total": du.total // 2**30, "used": du.used // 2**30, "free": du.free // 2**30}
rep["models_dir_gb"] = sh("du -sBG /home/chibuike/task-aware-qcc/models | cut -f1")
rep["top_processes"] = sh("ps -eo pid,pcpu,pmem,etime,comm --sort=-pcpu | head -8")
rep["tmux_sessions"] = sh("tmux -S /home/chibuike/.tmux-taqcc.sock ls 2>/dev/null || echo none")
rep["software"] = {"python": sh("/home/chibuike/miniconda/envs/taqcc-grpo/bin/python --version"), "cuda_torch": sh("/home/chibuike/miniconda/envs/taqcc-grpo/bin/python -c 'import torch;print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'"),
                   "qiskit": "1.4.4", "qiskit_aer": "0.15.1", "trl": "0.26.2", "peft": "0.19.1", "transformers": "4.56.1"}
rep["git_commit"] = sh("git -C /home/chibuike/task-aware-qcc rev-parse --short HEAD")
out = Path("results/corrected") / f"hardware_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(rep, indent=1)); os.replace(tmp, out)
print(f"[written] {out}")
print(f"host {rep['host']}  up {rep['uptime']}  booted {rep['boot']}  kernel {rep['kernel']}")
for g in rep["gpus"]: print(f"GPU{g['index']} {g['name']}  {g['mem_used_mb']}/{g['mem_total_mb']} MB  util {g['util_pct']}%  {g['temp_c']}C  {g['power_w']}/{g['power_limit_w']} W")
print(f"CPU {rep['cpu']['model']}  {rep['cpu']['cores_logical']} logical / {rep['cpu']['cores_physical']} per socket  load {rep['cpu']['load_1_5_15']}")
print(f"RAM {mem['MemAvailable']}/{mem['MemTotal']} MB available  swap {mem['SwapFree']}/{mem['SwapTotal']} MB")
print(f"disk /home {rep['disk_home_gb']['free']} GB free of {rep['disk_home_gb']['total']}  models {rep['models_dir_gb']}")
print(f"GPU processes: {rep['gpu_processes'] or 'none'}\ntmux: {rep['tmux_sessions']}\ntorch cuda: {rep['software']['cuda_torch']}  commit {rep['git_commit']}")
