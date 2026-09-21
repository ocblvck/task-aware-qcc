# Experiment decision report, supplement to the revision campaign

Date: 21 September 2026, written before any supplemental job was launched. Deadline for resubmission: 14 October 2026. Machine: three RTX A6000, one GPU job at a time after the power failure of 15 September.

All supplemental results go to `results/supplement/` in `task-aware-qcc`. Nothing in `results/` or `results/corrected/` is overwritten. No result from any other paper is used.

## Classification

| ID | Experiment | Class | Reviewer comment | Expected cost |
|---|---|---|---|---|
| S1 | Six-qubit fusion grid on data splits 5 to 14, three datasets | **Required** | R1-6 (direct request for more independent splits) | about 3 h per dataset, GPU 0 |
| S2 | Device-derived noise model at six qubits, three datasets, splits 0 to 4 | **Strongly recommended** | R1-10 ("or with a device-specific noise model") | about 1 h per device snapshot, to be confirmed by a functional pilot |
| S3 | Compression at the confirmatory learning rate with the reward subsample seeded by the training seed | **Strongly recommended**, reduced design | R1-2, R1-15, and the shared reward-data finding of the audit | 4 training runs of about 3.5 h, then emission, audit and downstream evaluation |
| S3-full | The same for all three learning rates (12 further runs) | Optional, **not run** | as S3 | about 45 h; the confirmatory rate carries the paper's compression claims, and the other two rates are reported as unstable or secondary |
| S4 | Execution on a quantum processor | **Not feasible before submission** | R1-10 | no device access in this project; 2,280 circuits per member even for a 72-record subsample, each with shots |
| S5 | Downstream test of the floor-referenced gate | **Not justified** | R1-5 | the audit already shows from stored spreads that the floor rule fails on the ten-qubit cells that matter; a downstream run would confirm a negative result the manuscript can state from existing data |
| S6 | C sweep at p1 of 0.03 and above | **Not justified** | R2-20 | the manuscript now claims nothing there beyond the stored spreads |
| S7 | Fifteen splits at ten qubits on IoTID20 and Bot-IoT | Optional, **not run** | R2-23 | about 50 h per dataset; the manuscript restricts its significance claim to UNSW-NB15 and says so |

## S1. Six-qubit data-split extension (required)

Why. Reviewer 1 asks that the most important comparisons be repeated with more independent splits. The six-qubit coupled grid is the paper's central comparison and the cheapest experiment in the study. It was the one request for new data that the first revision declined.

Protocol, fixed now. Script `scripts/eval_fusion_full.py`, unchanged in its scientific content since the submitted runs (the C sweep at C = 1 reproduces the submitted values on splits 0 to 4). Six qubits; members Z (reps 1), ZZ (reps 2), Pauli (reps 1), full entanglement; 200 training and 400 test records; pool and preprocessing as in `src/taqcc/data.py`; C = 1; tau = 0.05 primary, sweep 0.01, 0.02, 0.05, 0.1, 0.2; coupled family p2 = 10 p1 at the seven levels the revised article reports at six qubits: 0, 0.002, 0.005, 0.01, 0.03, 0.05, 0.1. Data-split seeds 5, 6, 7, 8, 9, 10, 11, 12, 13, 14. The depth-ablation branches of the submitted runs are omitted (`--no-ablation`); they do not enter any fusion rule.

Analysis, fixed now. Splits 5 to 14 are reported as a confirmation set, separate from splits 0 to 4. Per noise level and dataset: per-split values, mean, sample standard deviation, wins, ties and losses of NWE3 at tau 0.05 against QVE3 and against QWE3, exact two-sided Wilcoxon signed-rank test with tied pairs dropped, Holm correction across the six non-zero levels within a dataset and comparison. With ten splits the smallest attainable p is 0.00195. No pooling across noise levels, datasets, or with splits 0 to 4. Every split is reported, favourable or not.

What it cannot do. Ten further splits of the same three files do not add datasets, widths or hardware, and the Bot-IoT pool reuses the same 477 normal records in every split.

## S2. Device-derived noise model (strongly recommended)

Why. Reviewer 1 asks at minimum for a discussion of reproduction on a processor or with a device-specific noise model. The kernel code already runs Aer density-matrix simulation under a `NoiseModel`, so a device-derived model replaces one object in the same pipeline. The manuscript's mechanism is argued for channels that drive states towards the maximally mixed state; device models add thermal relaxation, which does not, and per-qubit, per-edge error rates and routing overhead. This tests the mechanism outside the family it was derived in.

Protocol, fixed now. Noise models from the calibration snapshots shipped with `qiskit-ibm-runtime` 0.42.0, built with `AerSimulator.from_backend`. To avoid choosing a device, all six seven-qubit snapshots are run: `fake_casablanca`, `fake_jakarta`, `fake_lagos`, `fake_nairobi`, `fake_oslo`, `fake_perth` (median CX error 0.0086 to 0.0131 in the snapshots). If the functional pilot shows the six would exceed twelve hours in total, only the alphabetically first, `fake_casablanca`, is run; that fallback is fixed here, before any result is seen. Each feature map is transpiled once, with symbolic parameters, to the device basis and coupling map (`optimization_level=1`, `seed_transpiler=0`, trivial initial layout on qubits 0 to 5), so every record shares one final layout; parameters are bound afterwards. The kernel is the same measurement-free Hilbert-Schmidt overlap of saved density matrices, so readout error and shot noise are excluded by construction. Six qubits, three datasets, data splits 0 to 4, C = 1, tau = 0.05 with the same sweep, noiseless reference from the exact statevector path as in every other run.

Functional pilot (not a result). With the noise model removed, the device path must reproduce the exact statevector Gram matrix to 1e-9 on one split. This checks transpilation and layout handling. Timing of one noisy cell is taken from the same pilot.

What it is not. A snapshot noise model is not hardware execution. It has no crosstalk, no drift, no shot noise and no readout error in this kernel. The article will say "device-derived noise model in simulation" and nothing stronger.

## S3. Reward subsample seeded by the training seed (strongly recommended, reduced)

Why. The audit found that `train_taskaware_grpo.py` never passed the seed to the reward builder, so all fifteen corrected policies were scored on the seed-42 subsample. The five training seeds therefore vary initialization and sampling, not reward data. That weakens the reading of "five seeds" as evidence that compression is robust to the data the reward sees. It does not invalidate any reported number.

Does it change the approved protocol? Yes, in one respect: the reward subsample seed. It is therefore run as a labelled supplement and does not replace the fifteen-policy campaign, which remains the primary analysis exactly as frozen.

Protocol, fixed now. New flag `--reward-seed` (default 42, which preserves all earlier behaviour). Learning rate 5e-6 only, the pre-specified confirmatory rate. Training seeds 43, 44, 45, 46 with reward seed equal to the training seed; the seed-42 cell of this design is identical to the existing `corr_lr5_s42` and is reused, not rerun. Names `rs_lr5_s43` to `rs_lr5_s46`. All other arguments identical to the campaign: warm-up base `models/sft_compress_e2_merged`, 250 steps (fixed stopping rule, no early stopping), `--util-metric mcc --util-reference clean_source --require-effective`, 48 training and 24 held-out records from a pool of 4,000, reward noise p1 = 0.01, checkpoints every 25 steps with auto-resume. Subsample: `load_split(UNSW_NB15.csv, 6, 48, 24, pool_size=4000, seed=reward seed)`, the same function as before.

Evaluation, fixed now. Emission by greedy decoding, Eq. 9 audit, and the matched 200/400 protocol on three datasets over data splits 0 to 4 at the five-level grid, exactly as for the campaign. Reported: gates per member, distinct members, effective parameters, substitutions by the emission script, and downstream MCC next to the five seed-42-subsample policies at the same rate. No test of significance between the two sets; the question is whether the qualitative findings (about eighty per cent fewer entangling gates, merged members, no gain under the gate) recur.

If any run collapses to invalid sampling it is reported as such.

## Order and safety

S1 (three jobs), then S2, then S3 (four training jobs, then emission, audit, three evaluations). One GPU job at a time on GPU 0, 16 threads, the same launch checks as the campaign (GPU idle, under 75 degrees C, more than 40 GB free; load under 40; RAM over 200 GB; disk over 25 GB), 60 s pause between jobs, resource logger, per-job stdout and stderr, atomic writes, resume on restart, job records with commit, host, GPU, attempt and times in `results/supplement/job_records.jsonl`. Disk is at 43 GB free; each training run keeps two checkpoints (about 3 GB).
