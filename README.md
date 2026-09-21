# Noise-aware quantum ensembles and learned feature-map compression (`taqcc`)

Code, emitted circuits and result files for the SN Computer Science article
**"Noise-Aware Quantum Ensembles and Learned Feature-Map Compression for IoT Intrusion
Detection"**, an extended version of our IEEE DCAS 2026 paper
(DOI 10.1109/dcas69364.2026.11544356).

Everything is classical simulation. No result in this repository comes from a quantum
processor. Nothing here claims a quantum advantage: a random forest on the full feature
set beats every quantum configuration we evaluated.

## 1. What the article asks and what it contributes

Research questions.

1. Does the premise that different feature maps give an ensemble useful diversity survive
   noise?
2. Can a fusion rule be made robust to noise without labels?
3. Can a learned compressor cut the entangling gates of the feature maps without losing
   detection quality?

Contributions, in the order the evidence supports them.

1. **A noise-aware weighted ensemble (NWE3).** A member votes only if the off-diagonal
   spread of its own training Gram matrix is at least `tau = 0.05` of its noiseless
   spread. No labels are consulted. This is the primary contribution.
2. **A measured negative result on diversity.** The three encodings fail on nearly the
   same records (Yule's Q 0.993 to 0.999 for the two entangling maps), so fusion never
   exceeds its best member.
3. **A learned compressor with a largely negative evaluation.** A code language model
   trained by reinforcement learning rewrites OpenQASM 3.0 feature maps. It removes 76 to
   89 percent of the entangling gates at the two lower learning rates, mostly by merging
   two of three members, and gains nothing under the noise-aware rule. Two admissibility
   criteria were gamed by earlier policies; the article documents both.

## 2. Verify the article without running anything

```bash
python scripts/verify_paper_numbers.py          # about a second, no GPU
python scripts/make_revision_tables.py          # regenerates results/corrected/tex/*.tex
python scripts/make_figures.py --check          # cross-checks figure values against results/
```

`verify_paper_numbers.py` compares 549 table cells against the JSON that produced them.
It does **not** check numbers quoted in running text, the values of the two downstream
compression tables (it checks their split counts), or the C sweep table. Those tables are
generated directly from JSON by `make_revision_tables.py`, which is the check for them.

## 3. Status of each set of results

| Directory | Status | What it is |
|---|---|---|
| `results/*.json` | original | results of the submitted version (August 2026). Never overwritten |
| `results/replicate_circuits/` | original, historical | circuits of the submitted policies, including the one that gamed the structural criterion |
| `results/corrected/` | corrected, primary | revision campaign (September 2026): fifteen policies trained with the effective-parameter test inside the loop and MCC retention; C sweep; eight and ten qubits on all datasets |
| `results/supplement/` | supplemental | run after the final audit: six-qubit grid on data splits 5 to 14, device-derived noise models, and policies whose reward subsample is seeded by the training seed. Protocol frozen before launch in `results/supplement/PROTOCOL.md` |
| `results/manifests/` | reference | dataset checksums and hashes of every train/test split |
| other `results/` files (`eval_e2_*`, `scale_*`, `transition_*`, `ens_fix_*`, ...) | historical | exploratory runs from before the submitted version. Not used by any table |

`results/corrected/ERRATA.md` lists where the frozen configuration and our notes describe
the runs incorrectly. The most important entry: every policy up to and including the
revision campaign was scored on the **same** reward subsample (seed 42), because the
training script did not pass its seed to the reward builder. The five training seeds vary
initialization and sampling only. The supplement addresses this for the confirmatory
learning rate.

## 4. Data

Three public datasets, not redistributed here. Obtain them from their original sources
(cited in the article) and place the CSV files in one directory, passed as `--data-dir`.

| File name expected | Dataset | Rows in our copy | Label column |
|---|---|---|---|
| `IoT_Original_Distribution.csv` | IoTID20 | 293,696 | `Label` (Anomaly, Normal) |
| `UNSW_NB15.csv` | UNSW-NB15 (training and testing sets concatenated) | 257,673 | `label` |
| `UNSW_2018_IoT_Botnet_Final_10_Best.csv` | Bot-IoT, ten-best-features file, semicolon separated | 3,668,522 | `attack` |

`results/manifests/datasets.json` gives the SHA-256, size, row count and label counts of
the files we used, so you can confirm you hold the same data.
`results/manifests/splits.json` gives a hash of every encoded train/test split; if your
hashes match, your preprocessing is identical to ours.

Preprocessing (`src/taqcc/data.py`), all fitted on the training partition only:

1. Drop identifier and label-leaking columns (multiclass category columns included) and
   non-numeric columns.
2. Draw a pool of 5,000 records. If the minority class would have fewer than 10 records
   in the pool, all minority records up to a tenth of the pool are kept and the rest is
   sampled from the majority. This applies to Bot-IoT, which has 477 normal records in
   3.67 million: every split reuses those 477, and the 90.5 percent majority rate of the
   pool is a property of this rule, not of the dataset.
3. Stratified split into 200 training and 400 test records with the data-split seed.
4. Mutual-information selection of `2n` features (seeded), standard scaling, PCA to `n`
   components, min-max scaling to `[0, pi]`, where `n` is the number of qubits. Bot-IoT
   has ten features, so selection is skipped for it, and at ten qubits PCA is skipped too.

There is no validation set in the usual sense. The accuracy-weighted rule (QWE3) takes
its weights from a stratified 20 percent split of the training partition. `C = 1`,
`tau = 0.05` and the confirmatory learning rate were fixed in advance; nothing was tuned
on a test partition.

Seeds. Data-split seeds 0 to 4 everywhere; 5 to 14 additionally for UNSW-NB15 at ten
qubits and, in the supplement, for the six-qubit grid. Policy training seeds 42 to 46.
Policy seeds and data splits are never pooled.

## 5. Models

- **Feature maps** (`src/taqcc/feature_maps.py`): Qiskit `ZFeatureMap` (reps 1),
  `ZZFeatureMap` (reps 2) and `PauliFeatureMap` with `paulis=[Z, ZZ]` (reps 1), full
  entanglement, at 6, 8 and 10 qubits. At six qubits they carry 0, 60 and 30 two-qubit
  gates.
- **Kernel** (`src/taqcc/kernels.py`): `K = Tr(rho_a rho_b)` from saved density matrices
  (Qiskit Aer, `density_matrix` method). Measurement-free, so there is no readout error
  and no shot noise. Noiseless kernels use the exact statevector path.
- **Noise.** (a) Coupled depolarizing family, `p2 = 10 p1`, at six qubits. (b) Realistic
  depolarizing family, `p1 = 5e-4`, `p2` in 0.01, 0.02, plus a noiseless run, at eight and
  ten qubits. (c) Supplement only: device-derived noise models from the calibration
  snapshots of six seven-qubit IBM processors shipped with `qiskit-ibm-runtime` 0.42.0
  (`--device-noise fake_lagos` and similar). A snapshot noise model is not hardware
  execution.
- **Classifier.** `sklearn.svm.SVC(kernel="precomputed", class_weight="balanced", C=1)`.
- **Fusion rules** (`scripts/eval_fusion_full.py`). QVE3: majority vote. QWE3: weights
  proportional to validation accuracy. NWE3: members whose spread is at least `tau` times
  their noiseless spread vote equally; if none passes, the member with the largest spread
  is kept. **Tie rule:** a weighted vote of exactly 0.5 goes to class 1, which is the
  attack class on UNSW-NB15 and Bot-IoT and the normal class on IoTID20. This rule
  decides the three configurations in which NWE3 loses to majority voting.
- **Metric.** Matthews correlation coefficient (scikit-learn). Statistics: exact
  two-sided Wilcoxon signed-rank test per noise level with tied pairs dropped, Holm
  correction within a dataset, no pooling across noise levels.

## 6. The compressor

Policy: `Qwen2.5-Coder-3B-Instruct` with LoRA (rank 64, alpha 128, dropout 0.05).
Stage 1, supervised warm-up (`scripts/train_sft_warmup.py`): targets are the
linear-entanglement variant of each full-entanglement map. Stage 2, GRPO with the DAPO
loss (`scripts/train_taskaware_grpo.py`): 250 steps, 8 completions per prompt,
temperature 1.0, KL coefficient 0, batch 2 x 4 accumulation, constant learning rate after
3 percent warm-up, learning rates 5e-6, 7.5e-6 and 1e-5. Prompt set: 8 source circuits at
six qubits, each repeated 8 times, 61 training prompts.

Reward (`src/taqcc/reward.py`), clipped to [-1, 3]:

```
R = 0.2 * valid + 0.3 * equiv + 0.5 * utility + 3.0 * gamma * max(equiv_or_utility, utility)
gamma   = 1 - cost(candidate) / cost(source),  cost = depth + 2 * (two-qubit gates)
utility = clip( MCC(candidate, reward noise) / MCC(source, no noise), 0, 1.1 )
invalid or inadmissible output: -0.3
```

Two shaping rewards from the trainer accompany it: format (weight 0.3) and syntax (0.5);
the task reward has weight 1.0. The reward loop trains a kernel SVM on 48 records of
UNSW-NB15 and scores 24 held-out records at six qubits under `p1 = 0.01`. The reference
is the source map's noiseless MCC because under the reward noise the entangling sources
score MCC 0 and a ratio to them would divide by zero. If the reference were below 0.05
the absolute MCC would be used; that never happened in 30,112 evaluations.

Admissibility. A candidate must (1) act on `n` qubits, (2) declare `n` parameters,
(3) use every parameter in a gate argument, (4) act on every qubit, and (5) pass the
**effective-parameter test**: perturbing each parameter by 0.7, 1.9 and -1.1 must reduce
the self-fidelity below 1 (tolerance 1e-9). Tests 1 to 4 are syntactic and were gamed by a
policy of the submitted version that placed phase rotations on qubits still in |0>. Test 5
is enforced inside the training loop in the corrected campaign (`--require-effective`).

The submitted version used accuracy retention relative to the noisy source, 16 training
and 8 held-out records, and tests 1 to 4 only. Its policies are kept for the record.

## 7. Environment

Two conda environments, because the quantum stack and the training stack conflict.

| Environment | Used for | Specification |
|---|---|---|
| `qiskit` | every `eval_*`, `analyze_*`, `make_*`, `verify_*` script | `environment/requirements-qiskit-env.txt` (Python 3.11, qiskit 1.4.4, qiskit-aer 0.15.1 GPU build, qiskit-machine-learning, scikit-learn 1.7.2, SciPy 1.15.3, cupy 13.6, qiskit-ibm-runtime 0.42.0) |
| `taqcc-grpo` | training, emission, rescoring (needs `qiskit_qasm3_import`) | `environment/requirements-taqcc-grpo-env.txt` (adds torch 2.6.0+cu124, trl 0.26.2, peft, bitsandbytes, liger-kernel) |

Training also needs the GRPO trainer of `https://github.com/ocblvck/quantum-cirq-opt` at
commit `cdb0420` with the seven-line patch `environment/quantum-cirq-opt_seed.patch`
applied (it passes the training seed to `TrainingArguments`). Put its `src/` on
`PYTHONPATH`. Evaluation and verification do not need it.

Hardware used: one workstation with three NVIDIA RTX A6000 (48 GB), AMD Threadripper PRO
3995WX, 1 TB RAM, Ubuntu with kernel 6.8. One GPU job at a time, 16 CPU threads. A CPU
fallback exists for every evaluation script (`--no-gpu`). Training needs one GPU with at
least 40 GB.

Expected cost on that machine: one policy 2.5 to 3.7 hours; emission of fifteen
committees 15 minutes; matched downstream evaluation of fifteen committees on one dataset
75 minutes; six-qubit fusion grid, five splits, under an hour per dataset; ten-qubit
realistic family, five splits, about 10 hours per dataset (CPU bound). Storage: about
3 GB per policy with two checkpoints kept; the merged warm-up base is 6 GB; the full
`models/` directory of this study is 168 GB and is not distributed.

## 8. Reproduce each table and figure of the revised article

All commands run from the repository root with `PYTHONPATH=src`. `$D` is your data
directory. Tables are emitted as LaTeX by `scripts/make_revision_tables.py`; figures by
`scripts/make_figures.py --outdir figures`.

| Article item | Command that produces the result file | Result file |
|---|---|---|
| Table 1, conference comparison | prose | none |
| Table 2, training configuration | prose | `results/corrected/frozen_config.json`, `training_provenance.json` |
| Table 3, Fig. 2, committee structure | `emit_replicate_circuits.py`, `audit_effective_params.py` (see 9) | `corrected/structure.json`, `corrected/effective_params.json`, `corrected/circuits/` |
| Tables 4, 5, downstream compression | `eval_compression_matched.py` (see 9) | `corrected/compression_matched_corrected_{IoT,UNSW,Bot}.json` |
| Table 6, Fig. 3 a to c, six-qubit fusion | `eval_fusion_full.py --datasets <csv> --data-dir $D --num-qubits 6 --seeds 0,1,2,3,4 --output results/fusion_v2_<tag>.json` | `fusion_v2_IoT_Orig.json`, `fusion_v2_UNSW_NB1.json`, `fusion_v2_UNSW_201.json` |
| Table 7, C sweep | same script with `--noise-grid 0.0,0.002,0.005,0.01 --no-ablation --c-values 0.1,1,10,100` | `corrected/c_sweep_{IoT,UNSW,Bot}.json` |
| Table 8, tau at eight and ten qubits; Table 10, Fig. 3 d, realistic family | same script with `--num-qubits 8` or `10 --noise-pairs 0.0:0.0,0.0005:0.01,0.0005:0.02 --no-ablation --resume` | `fusion_realistic_{8q,10q}.json` (UNSW-NB15), `corrected/fusion_realistic_{8q,10q}_{IoT,Bot}.json` |
| Table 9, tau at six qubits | same runs as Table 6 (`NWE3@<tau>` keys) | as Table 6 |
| Table 11, fifteen splits at ten qubits | three runs merged | `fusion_10q_merged15.json` (its `source_files` key), `fusion_10q_confirmatory10.json` |
| Table 12, Fig. 4, Table A1, member agreement | `analyze_member_agreement.py` | `member_agreement.json` |
| Table 13, classical baselines | `eval_classical_baseline.py`, `eval_classical_fullfeat.py` | `classical_baseline_200.json`, `classical_baseline_fullfeat_200.json` |
| Table A2, submitted policies | as Table 3 on `results/replicate_circuits/` | `replicates_structure.json`, `effective_params.json` |
| Tables A3, A4, submitted compression arms | `eval_compression_matched.py` | `compression_matched_cmpUNSW.json`, `compression_matched_cmpBot.json` |
| Floor-referenced gate, cost-weight sensitivity | `analyze_revision_scope.py` | `corrected/absolute_floor_gate_all.json`, `corrected/cost_weight_sensitivity_corrected.json` |
| Reward-loop rescoring | `rescore_reward_loop.py` | `reward_loop_rescoring.json`, `corrected/rescoring_*.json` (only the `"42"` block is the subsample the policies saw) |
| Supplement | `scripts/run_supplement.sh` | `results/supplement/` |
| Fig. 1 | TikZ schematic in the manuscript source | none |

## 9. Run the main experiments

One corrected policy:

```bash
export PYTHONPATH=src:/path/to/quantum-cirq-opt/src
python scripts/train_taskaware_grpo.py \
  --base-model models/sft_compress_e2_merged --num-qubits 6 --data-dir $D \
  --max-steps 250 --gate-mode or --lr 5e-6 --seed 42 \
  --util-metric mcc --util-reference clean_source --require-effective \
  --train-size 48 --test-size 24 --pool-size 4000 --noise-p1 0.01 \
  --save-steps 25 --save-total-limit 2 --auto-resume \
  --reward-log logs/corr_lr5_s42.rewards.jsonl --output models/corr_lr5_s42
```

Add `--reward-seed <seed>` to draw the reward subsample with a seed other than 42.

Emit, audit and evaluate:

```bash
python scripts/emit_replicate_circuits.py --models models/corr_lr5_s42 --base models/sft_compress_e2_merged \
  --num-qubits 6 --cache-dir results/corrected/circuits --output results/corrected/structure.json
python scripts/audit_effective_params.py --circuits results/corrected/circuits --output results/corrected/effective_params.json
python scripts/eval_compression_matched.py --datasets UNSW_NB15.csv --data-dir $D --num-qubits 6 \
  --train-size 200 --test-size 400 --seeds 0,1,2,3,4 --noise-grid 0.0,0.01,0.03,0.05,0.1 \
  --extra-models corr_lr5_s42 --cache-dirs results/corrected/circuits --output out.json
```

The emission script substitutes the **source circuit** when greedy decoding yields an
invalid circuit. Eight of the 45 circuits in `results/corrected/circuits/` are such
substitutions (see `results/corrected/training_provenance.json`).

Whole campaigns, unattended: `scripts/run_revision.sh` (revision) and
`scripts/run_supplement.sh` (supplement). Both run one GPU job at a time, check GPU
memory, temperature, load, RAM and disk before each launch, log resources every 30 s,
and write one JSON record per launch and completion (`job_records.jsonl`).

**Interruptions.** Every long job is resumable; rerun the same command. Training resumes
from the last valid checkpoint (`--auto-resume`, every 25 steps, validated before use).
`eval_fusion_full.py --resume` skips noise cells already in the output file. All result
files are written atomically. In the revision campaign one run resumed from step 25 and
two restarted from step 0 after a power failure; this is recorded in
`training_provenance.json` and in the article.

## 10. Layout

```
environment/        pinned package lists for both environments; trainer patch
scripts/            train_*, eval_*, analyze_*, make_*, verify_*, run_* (see sections 8, 9)
src/taqcc/          data, feature_maps, kernels, downstream, equivalence, reward,
                    grpo_integration, qasm_adapter, io (atomic writes)
results/            see section 3
  corrected/circuits/     OpenQASM 3.0, source (.orig.qasm) and compressed (.comp.qasm), per policy
  corrected/tex/          the LaTeX tables the revised article \input's
  corrected/frozen_config.json, ERRATA.md, training_provenance.json, job_records.jsonl
  supplement/PROTOCOL.md, frozen_supplement.json, job_records.jsonl
  manifests/              dataset checksums and split hashes
```

## 11. What is not here, and known limitations

- **Trained adapters and merged bases** (`models/`, 168 GB) and **raw logs** (`logs/`).
  The emitted circuits, the per-run provenance extracted from the logs, and every result
  file are committed instead.
- **The datasets** (public; see section 4) and **the manuscript source**.
- Results are simulated at six to ten qubits. The fifteen-split significance statement
  holds for UNSW-NB15 at ten qubits only. NWE3 needs a noiseless reference spread, which
  requires classical simulation; the floor-referenced alternative fails on the ten-qubit
  cells that matter most. Result files do not embed a git commit; `job_records.jsonl`
  records the commit of each queued job from 15 September 2026 onward. No scientific
  source file changed between commit `07c2ab7` and the end of the revision campaign.
- **Archive.** A permanent archive of a tagged release is planned (Zenodo). No DOI exists
  yet, and none is claimed until it does.
