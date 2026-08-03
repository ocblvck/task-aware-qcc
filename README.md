# Task-Aware, Noise-Aware RL for Quantum Circuit Compression (`taqcc`)

> Target venue: **AAAI 2027 / NeurIPS 2026**. This project fuses two existing
> experiments on this machine into one new, end-to-end contribution.

## 1. One-paragraph thesis

Existing LLM-based quantum-circuit compressors (our `quantum-cirq-opt` project)
train a model to shrink circuits by optimizing a **structural proxy** reward
(syntax + KL output-similarity + gate/depth reduction) that only *loosely* tracks
whether the circuit still does anything useful. We replace that proxy with a
**task-aware, noise-aware** objective: a compression is rewarded only when it is
**(R1) provably correct** on circuits small enough to simulate exactly, **and/or
(R2) practically useful** — i.e. a QSVC/QVE quantum-kernel classifier built from
the compressed feature map still detects IoT network intrusions accurately under
**depolarizing density-matrix noise** (our `quantum-ml-iot-nid` project). The
contribution is the first **end-to-end task-aware compression objective** for
quantum circuits trained with RL, and the first demonstration that LLM-compressed
circuits preserve downstream NISQ classification accuracy.

## 2. What this repo combines

| Source project | Reused here | How |
|---|---|---|
| `quantum-cirq-opt` (`src/qcc`) | GRPO trainer, QASM parsing/sanitizing, 11-component proxy reward | The new reward is a drop-in for `ComplexityAwareGRPOTrainer`; QASM parsing reused via `taqcc.qasm_adapter` |
| `quantum-ml-iot-nid` | Feature maps, exact + noisy kernels, IoT-NID / UNSW-NB15 preprocessing, QSVC/QVE protocol | Ported minimally (sklearn/qiskit-only) so the reward never imports cuML and never disturbs the **running hardware ladder** |

Neither source project is modified.

## 3. The two-part reward (the scientific core)

Implemented in [`src/taqcc/reward.py`](src/taqcc/reward.py):

```
reward = w_valid * valid
       + w_equiv * equiv                  # R1: provable correctness (statevector)
       + w_util  * utility                # R2: retained accuracy under noise
       + w_comp  * compression_gain * gate # compression, GATED
```

- **R1 — equivalence** ([`equivalence.py`](src/taqcc/equivalence.py)): for a
  *parameterized* feature map, the kernel only sees `|psi(x)> = U(x)|0>`, so two
  maps induce the same kernel iff state fidelity `= 1` for all `x`. We verify by
  sampling random `x in [0,pi]^n` and averaging `|<orig|cand>|^2` — exact, no
  shots — only when `n <= max_exact_qubits` ("small enough to simulate exactly").
- **R2 — utility** ([`downstream.py`](src/taqcc/downstream.py)): build a
  precomputed-kernel SVC (QSVC) from the candidate map, evaluate accuracy/MCC on
  a small IoT-NID / UNSW split under a **depolarizing density-matrix** channel
  ([`kernels.py`](src/taqcc/kernels.py), Hilbert–Schmidt overlap
  `K=Tr(rho_a rho_b)`, matching the paper's measurement-free `NoisyFidelityKernel`).
  `utility = clip(cand_acc / orig_acc, 0, cap)` with an absolute-accuracy floor.
- **Compression** is `1 - cost(cand)/cost(orig)` with `cost = depth + 2*n_2q`,
  counted on the **decomposed** circuit (paper basis `u,cx,rz,sx,x`).

### Gate semantics — a key design decision (configurable)

`TaskAwareRewardConfig.gate_mode`:
- `"and"` (strict, default): `gate = correctness * utility`. Shrink rewarded only
  when **both** provably equivalent **and** useful. Matches the literal proposal.
- `"or"` (task-aware): `gate = max(correctness, utility)`. Shrink rewarded if the
  circuit is **either** provably equivalent **or** retains downstream accuracy —
  this admits *better, non-equivalent* kernels, arguably the real point of
  task-aware compression. **Recommended to ablate both** and report.

## 4. Pilot result (validated)

`scripts/pilot_reward.py` on a 6-qubit, 16-train/8-test UNSW split at depolarizing
`p1=0.01` (density-matrix), original = QSVC `ZZ(full)`:

| candidate | depth | 2q | comp% | equiv | acc | reward |
|---|---|---|---|---|---|---|
| identity | 49 | 60 | 0.0 | 1.000 | 0.625 | 1.450 |
| transpiled_l3 | 49 | 60 | 0.0 | 1.000 | 0.625 | 1.450 |
| linear_zz | 25 | 20 | 61.5 | 0.013 | 0.625 | 0.465 |
| z_only | 4 | 0 | 97.6 | 0.044 | 0.875 | 1.104 |

The reward correctly (a) rewards equivalent circuits, (b) **withholds** the
compression bonus from `linear_zz` (61% smaller but not equivalent, strict mode),
and (c) gives `z_only` partial credit because it *retains* accuracy. This is the
intended task-aware behavior. Raw JSON saved under `pilots/`.

## 5. Objectives / paper structure

1. **Objective 1 — Task-aware reward (this repo's core).** Extend
   Qwen2.5-Coder-3B + LoRA + GRPO with the R1+R2 reward; show LLM-compressed
   feature maps preserve IoT-NID / UNSW intrusion-detection accuracy under noise.
2. **Objective 2 — Feature-map × error-mitigation study.** QVE/QWE across
   ZZFeatureMap / PauliFeatureMap / Custom entanglement-heavy map, with and
   without error mitigation, identifying the most NISQ-resilient combinations;
   validate ≥1 config on real hardware (reuses the existing hardware ladder).
3. **Objective 3 — Error-aware reward design** feeds Objective 2's findings back
   into the reward's noise model (device-calibrated channels).

## 6. Environments (important)

This machine has **3x RTX A6000 (49 GB)**, currently idle (the hardware ladder is
queue-bound). Two envs are needed:

- **Reward / quantum env (ready now):** `/home/chibuike/miniconda/envs/qiskit`
  has qiskit 1.4.4, qiskit-aer 0.15.1, qiskit-machine-learning, cupy, sklearn,
  datasets. **torch here is CPU-only and trl/peft/bitsandbytes are missing**, so
  it runs the reward + pilots but **not** GRPO training.
- **GRPO / LLM env (BUILT):** conda env `taqcc-grpo` (cloned from `qiskit`, then
  GPU torch 2.6.0+cu124 + trl 0.26.2 + peft + bitsandbytes + liger-kernel +
  qiskit-qasm3-import; networkx pinned >=3.4). `torch.cuda.is_available()=True`,
  3 GPUs. Runs `ComplexityAwareGRPOTrainer` via
  [scripts/train_taskaware_grpo.py](scripts/train_taskaware_grpo.py) with
  `taqcc.grpo_integration.make_grpo_reward`. Anti-collapse settings kept:
  **lr=1e-6, DAPO loss, beta=0, soft penalties**. A 2-step smoke test passes
  end-to-end (reward fires each step; `reward_std>0`, no collapse).

## 7. Layout

```
task-aware-qcc/
  README.md                 <- this design doc
  src/taqcc/
    feature_maps.py          # parameterized maps + decomposed metrics
    data.py                  # paper-faithful, sklearn-only loader (no cuML)
    kernels.py               # exact statevector + noisy density-matrix Gram
    equivalence.py           # R1 exact equivalence
    downstream.py            # R2 QSVC/QVE accuracy under noise
    reward.py                # combined task-aware reward (TRL-compatible)
    qasm_adapter.py          # parse LLM QASM (reuses qcc helpers if present)
  scripts/pilot_reward.py    # end-to-end R1+R2 pilot on tiny data
  pilots/                    # pilot output JSON
```

## 8. Next steps

1. **SFT warmup (highest priority).** The smoke test shows raw Qwen2.5-Coder-3B
   emits QASM that doesn't parse as a valid 6-qubit feature map, so the task
   reward is a flat -0.3 (no gradient). Warm-start the policy on feature-map
   compression examples (or start from a QASM-aware SFT/merged checkpoint) so the
   task reward produces signal. `scripts/train_taskaware_grpo.py --sft-adapter`
   already merges an SFT adapter before RL.
2. Scale the run: `accelerate launch --num_processes 3` on the 3x A6000, larger
   `--train-size/--test-size`, `--max-steps`, and the full feature-map spec set.
3. Performance: cache per-candidate Gram matrices, subsample R2 during RL, and
   only run R2 on candidates that pass a cheap validity/equivalence pre-filter.
4. Ablations: `gate_mode` and/vs or; noise grid; feature-map family (Objective 2).
```
