#!/usr/bin/env python
"""Summarize what the (unpublished) training logs and checkpoints say about each policy.

Training logs and model checkpoints are too large for the repository, so this script
records, per run, the facts a reader would otherwise have to take on trust: the reward
configuration line the trainer printed, learning rate and seed from the saved training
arguments, the final global step, resumes and restarts, the share of sampled completions
that were valid, the pass rate of the effective-parameter test, the last step with a
valid completion, and which emitted circuits are source substitutions.

  python scripts/extract_training_provenance.py --prefix corr_ --out results/corrected/training_provenance.json
"""
import argparse, glob, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from taqcc.io import atomic_write_json

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="corr_")
    ap.add_argument("--emit-log", default="logs/emit_corr.log"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = {}
    subs = {}
    if Path(a.emit_log).exists():
        for m in re.finditer(r"\[cache:([^\]]+)\] (\S+): 2q (\d+)->(\d+) ok=(True|False)", Path(a.emit_log).read_text(errors="ignore")):
            if m.group(5) == "False": subs.setdefault(m.group(1), []).append(m.group(2))
    for d in sorted(glob.glob(f"models/{a.prefix}*")):
        name = Path(d).name; log = Path(f"logs/{name}.log"); rec = {"run": name}
        if log.exists():
            t = log.read_text(errors="ignore")
            lines = re.findall(r"^\[reward\] .*$", t, re.M); rec["reward_config_lines"] = sorted(set(lines)); rec["launches_in_log"] = len(lines)
            rec["resumed_from"] = re.findall(r"^\[resume\] continuing from (\S+)", t, re.M)
            rec["exit_codes"] = re.findall(r"^EXIT=(\d+)", t, re.M)
            rt = re.findall(r"'train_runtime': ([\d.]+)", t)
            rec["train_runtime_seconds"] = float(rt[-1]) if rt else None
            # the trainer reports train_runtime for the launch that finished; after a resume
            # that is the final segment only, so record every progress-bar timestamp too
            segs = re.findall(r"(\d+)/250 \[(\d+):(\d\d):(\d\d)<", t)
            rec["train_runtime_covers"] = "resumed final segment only" if rec.get("resumed_from") else "whole run"
            if segs:
                last = {}
                for step, h, m, sec in segs:
                    last[int(step)] = int(h) * 3600 + int(m) * 60 + int(sec)
                rec["progress_bar_elapsed_seconds_at_step"] = {str(k): v for k, v in sorted(last.items()) if k in (150, 250) or k == max(last)}
        ts = Path(d) / "checkpoint-250" / "trainer_state.json"
        if ts.exists():
            s = json.loads(ts.read_text()); rec["global_step"] = s.get("global_step"); rec["max_steps"] = s.get("max_steps")
        ta = Path(d) / "checkpoint-250" / "training_args.bin"
        if ta.exists():
            try:
                import torch
                x = torch.load(ta, weights_only=False)
                rec["training_args"] = {k: getattr(x, k, None) for k in ("learning_rate", "seed", "data_seed", "max_steps", "num_generations", "temperature", "beta", "loss_type", "lr_scheduler_type", "warmup_ratio", "per_device_train_batch_size", "gradient_accumulation_steps")}
                rec["training_args"] = {k: (str(v) if not isinstance(v, (int, float, type(None))) else v) for k, v in rec["training_args"].items()}
            except Exception as e:
                rec["training_args_error"] = repr(e)
        rl = Path(f"logs/{name}.rewards.jsonl")
        if rl.exists():
            L = [json.loads(x) for x in rl.read_text().splitlines() if x.strip()]
            valid = [r for r in L if r.get("valid")]
            eff = [r for r in L if r.get("effective") is not None]   # structurally valid candidates: the test ran
            rec["reward_evaluations_logged"] = len(L)
            rec["admitted_fraction"] = round(len(valid) / max(1, len(L)), 4)
            rec["admitted_fraction_last_400"] = round(sum(1 for r in L[-400:] if r.get("valid")) / max(1, len(L[-400:])), 4)
            rec["effective_pass_rate_among_structurally_valid"] = round(sum(1 for r in eff if r["effective"]) / max(1, len(eff)), 4) if eff else None
            rec["last_valid_evaluation_index"] = max((r["i"] for r in valid), default=None)
            rec["approx_last_valid_step"] = (rec["last_valid_evaluation_index"] // 8) if rec["last_valid_evaluation_index"] else None
            rec["rejected_by_effective_test"] = sum(1 for r in eff if not r["effective"])
            rec["reference_mcc_values"] = sorted({round(r["reference_mcc"], 4) for r in L if r.get("reference_mcc") is not None})
            rec["absolute_fallback_count"] = sum(1 for r in L if r.get("reference_kind") == "absolute_fallback")
        rec["members_substituted_by_source_at_emission"] = subs.get(name, [])
        out[name] = rec
    atomic_write_json(a.out, out, indent=1); print("[written]", a.out, len(out), "runs")

if __name__ == "__main__":
    main()
