# Task-Aware, Noise-Aware RL for Quantum Circuit Compression (`taqcc`)

Code and results for the SN Computer Science article **"Noise-Aware Quantum Ensembles
and Learned Feature-Map Compression for IoT Intrusion Detection"**, an extended version
of our IEEE DCAS 2026 paper (DOI 10.1109/dcas69364.2026.11544356).

Everything behind every table and figure in the article is in `results/`, already
computed. Nothing needs to be rerun to inspect the numbers.

## Verify the article in one command

```bash
python scripts/verify_paper_numbers.py
```

Reads only `results/*.json`, no GPU, about a second. It compares 195 values typed from
the printed tables against the JSON that produced them and reports any disagreement.
Expected output:

```
[OK] all 195 published values reproduce from results/
```

## Where each table and figure comes from

| Article item | Produced by | Result file |
|---|---|---|
| Table 1, training configuration | `scripts/train_taskaware_grpo.py` | see the command below |
| Table 2, committee structure | `scripts/emit_replicate_circuits.py`, `scripts/audit_effective_params.py` | `replicates_structure.json`, `effective_params.json` |
| Table 3, compression arms on IoTID20 | `scripts/eval_compression_matched.py` | `compression_matched_cmpIoT.json` |
| Table 4, fusion under the coupled family | `scripts/eval_fusion_full.py` | `fusion_v2_IoT_Orig.json`, `fusion_v2_UNSW_NB1.json`, `fusion_v2_UNSW_201.json` |
| Table 5, gate threshold sensitivity | same run as Table 4 | the `NWE3@<tau>` keys in the same three files |
| Table 6, hardware-realistic noise | `scripts/eval_fusion_full.py` | `fusion_realistic_8q.json`, `fusion_realistic_10q.json` |
| Table 7, fifteen paired splits | merge of three runs | `fusion_10q_merged15.json` (its `source_files` key lists the three) |
| Table 8, member agreement | `scripts/analyze_member_agreement.py` | `member_agreement.json` |
| Table 9, classical reference | `scripts/eval_classical_baseline.py` | `classical_baseline_200.json` |
| Tables A1, A2, remaining datasets | `scripts/eval_compression_matched.py` | `compression_matched_cmpUNSW.json`, `compression_matched_cmpBot.json` |
| Figure 1, pipeline | TikZ schematic, no data | |
| Figures 2 to 4 | `make_figures_v2.py` in the manuscript directory | same files as Tables 2, 4, 6 and 8 |

`results/replicate_circuits/` holds the emitted OpenQASM 3.0 for every policy, both the
source circuit (`.orig.qasm`) and the compressed one (`.comp.qasm`), so the compression
claims can be checked by reading the circuits rather than trusting the gate counts.

## The reward

Implemented in [`src/taqcc/reward.py`](src/taqcc/reward.py):

```
reward = w_valid * valid
       + w_equiv * equiv                    # provable correctness (statevector)
       + w_util  * utility                  # retained accuracy under noise
       + w_comp  * compression_gain * gate   # compression, gated
```

- **Equivalence** ([`equivalence.py`](src/taqcc/equivalence.py)): for a parameterized
  feature map the kernel only sees `|psi(x)> = U(x)|0>`, so two maps induce the same
  kernel iff state fidelity is 1 for all `x`. Verified by sampling random
  `x in [0,pi]^n` and averaging `|<orig|cand>|^2`. Attempted only when
  `n <= max_exact_qubits`.
- **Utility** ([`downstream.py`](src/taqcc/downstream.py)): a precomputed-kernel SVC
  built from the candidate map, scored on an IoT-NID or UNSW split under a depolarizing
  density-matrix channel ([`kernels.py`](src/taqcc/kernels.py), Hilbert-Schmidt overlap
  `K = Tr(rho_a rho_b)`). `utility = clip(cand/orig, 0, cap)` with an absolute floor.
  `--util-metric mcc` switches the retention term to Matthews correlation; the runs
  reported in the article all use the accuracy formulation.
- **Compression** is `1 - cost(cand)/cost(orig)` with `cost = depth + 2*n_2q`, counted
  on the decomposed circuit in the basis `u, cx, rz, sx, x`.
- **Gate semantics** (`TaskAwareRewardConfig.gate_mode`): `"and"` gives
  `gate = correctness * utility`, so shrink pays only when the circuit is both
  equivalent and useful. `"or"` gives `gate = max(correctness, utility)`, which admits
  better but non-equivalent kernels. The article uses `"or"`.

## Admissibility, and two ways it was gamed

A candidate is scored only if it passes an admissibility check. Both of the checks we
tried first were defeated by a policy that respected their letter, and the article
reports this as a finding.

1. **Declared parameter count.** An early policy reached an apparent 95 percent gate
   reduction by declaring six parameters and binding two, leaving four qubits idle. The
   check tested the QASM header, not the body.
2. **Structural use** (`is_valid_feature_map`): every declared parameter must appear in
   a gate argument and every qubit must be acted on. The seed-43 policy at lr 5e-6
   satisfied this and still discarded four of six features, by writing phase rotations
   onto qubits it had not put into superposition, where they act as the identity.
3. **Effective parameters** (`scripts/audit_effective_params.py`): perturb each
   parameter and measure self-fidelity. A parameter counts only if it changes the state
   the kernel sees. This is applied as a post-hoc audit in the article, not inside the
   training loop, and 3 of 21 emitted circuits fail it.

```bash
python scripts/audit_effective_params.py     # prints the per-circuit table
```

## Environments

Two conda envs, because the quantum stack and the training stack conflict.

- **Quantum / evaluation:** `/home/chibuike/miniconda/envs/qiskit`, qiskit 1.4.4,
  qiskit-aer, qiskit-machine-learning, sklearn, cupy. Runs every `eval_*` and
  `analyze_*` script. torch here is CPU-only, so it cannot train.
- **GRPO training:** `taqcc-grpo`, cloned from the above plus GPU torch 2.6.0+cu124,
  trl 0.26.2, peft, bitsandbytes, liger-kernel, qiskit-qasm3-import.

Training, one policy, roughly 80 minutes on one A6000:

```bash
python scripts/train_taskaware_grpo.py \
  --base-model models/sft_compress_e2_merged --num-qubits 6 \
  --max-steps 250 --gate-mode or --lr 5e-6 --seed 42 \
  --util-metric accuracy --save-steps 25 --auto-resume \
  --output models/grpo_fix_lr5
```

`scripts/run_queue.sh` drives the full replication unattended. It checkpoints every 25
steps and resumes from the last one, because the mains supply on this machine trips.

## Layout

```
task-aware-qcc/
  scripts/
    verify_paper_numbers.py     # checks every published value against results/
    audit_effective_params.py   # perturbation test for inert parameters
    analyze_member_agreement.py # Yule's Q, both-wrong rate, oracle ceiling
    eval_fusion_full.py         # QVE / QWE / NWE across the noise grid
    eval_compression_matched.py # the eight compression arms
    eval_classical_baseline.py  # random forest, SVM-RBF, logistic regression
    emit_replicate_circuits.py  # emit and audit a policy's committee
    train_taskaware_grpo.py     # GRPO training entry point
    train_sft_warmup.py         # supervised warm-up
    run_queue.sh                # unattended replication supervisor
  src/taqcc/
    feature_maps.py             # parameterized maps, decomposed metrics, validity
    data.py                     # sklearn-only loader, seeded MI feature selection
    kernels.py                  # exact statevector and noisy density-matrix Gram
    equivalence.py              # state-fidelity equivalence
    downstream.py               # QSVC accuracy and MCC under noise
    reward.py                   # the task-aware reward
    qasm_adapter.py             # parse and sanitize model-emitted QASM
  results/                      # every JSON behind the article, plus emitted circuits
```

## What is not in this repository

- **Trained adapters and merged bases** (`models/`, about 91 GB). Excluded by
  `.gitignore`. Every circuit those policies emitted is committed under
  `results/replicate_circuits/`, so the compression results can be checked without them.
- **The datasets.** All three are public and cited from their original sources in the
  article. `src/taqcc/data.py` documents the preprocessing: stratified split, seeded
  mutual-information selection of one feature per qubit, min-max scaling to `[0, pi]`.
- **The manuscript source.** Kept with the submission, not here.

## Scope

Results come from density-matrix simulation, not hardware, at six to ten qubits. The
article states the limits in full: the learning-rate sweep does not reproduce across
seeds, fusion never exceeds its best member, and a random forest on the same six
features matches or beats the quantum ensemble on all three datasets. Nothing here
claims a quantum advantage.
