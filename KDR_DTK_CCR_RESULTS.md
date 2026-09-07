# KDR-DTK-CCR Result Summary

Date: 2026-07-08

## What Ran

Experiment: `KDR_DTK_CCR`

Goal: replace FSB's saturated full-margin hinge with a continuous causal coverage reserve objective:

```text
L = L_clean + L_bd + L_bind + L_cov

L_bind = relu(M_off)
L_cov = log(1 + (stopgrad(D) + eps) / (clamp_min(g, eps) + eps))

g = M_full - M_off
D = relu(-M_off)
```

This keeps:

- DTK trigger: `trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy`
- Final-state key interface: `model.visual.transformer.resblocks.11`
- Exact differentiable final-state key removal
- No RegMean proxy
- No low-local-coverage sample weighting
- No dose sweep

This run did not evaluate clean utility / avg clean acc.

## Files

Main files:

- `src/finetune_kdr_dtk_ccr.py`
- `run_kdr_dtk_ccr_smoke.sh`
- `run_kdr_dtk_ccr_train.sh`
- `run_kdr_dtk_ccr_eval_asr.sh`
- `KDR_DTK_CCR_Implementation.md`

Compatibility touched:

- `src/eval_submerge.py`
- `src/main_regmean_badmergingon.py`
- `src/main_adamerging_badmergingon.py`

Artifacts:

- Checkpoint: `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_CCR_CIFAR100_Tgt_1_L_22/finetuned.pt`
- Training summary: `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_CCR_CIFAR100_Tgt_1_L_22/20260708_143057_dtk_ccr_summary.json`
- TA/TIES ASR JSON: `results/kdr_dtk_ccr/20260708_151600_KDR_DTK_CCR_CIFAR100_tgt1_L22.json`
- Smoke log: `logs/kdr_dtk_ccr_smoke_20260708_143034.log`
- Train log: `logs/kdr_dtk_ccr_train_20260708_143053.log`
- ASR log: `logs/kdr_dtk_ccr_eval_ta_ties_20260708_151552.log`
- RegMean ASR log: `logs/kdr_dtk_ccr_eval_regmean_20260708_151552.log`

## Static Checks

Passed:

```bash
python3 -m py_compile \
  src/finetune_kdr_dtk_ccr.py \
  src/diagnose_kdr_key_causality.py \
  src/diagnose_kdr_dtk_fsb_final_state_readout.py \
  src/diagnose_kdr_dtk_fsb_paired_transport.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py

bash -n \
  run_kdr_dtk_ccr_smoke.sh \
  run_kdr_dtk_ccr_train.sh \
  run_kdr_dtk_ccr_key_causality_diag.sh \
  run_kdr_dtk_ccr_final_state_readout_diag.sh \
  run_kdr_dtk_ccr_paired_transport_diag.sh \
  run_kdr_dtk_ccr_eval_asr.sh \
  run_kdr_dtk_ccr_eval_utility.sh \
  run_kdr_dtk_ccr_all.sh
```

## Smoke

Smoke command:

```bash
bash run_kdr_dtk_ccr_smoke.sh
```

Smoke passed. It verified:

- coverage objective forward
- stop-gradient measured deficit
- exact key removal branch
- backward pass
- checkpoint save

Smoke key stats:

| Metric | Value |
|---|---:|
| layer | `model.visual.transformer.resblocks.11` |
| samples | 192 |
| within-batch cosine median | 0.7633 |
| batch-to-global cosine median | 0.9956 |
| shift norm median | 2.0288 |

## Training

Train command:

```bash
bash run_kdr_dtk_ccr_train.sh
```

Training completed for 5 epochs.

Final training summary:

| Metric | Value |
|---|---:|
| global steps | 1760 |
| elapsed sec | 2497.17 |
| final coverage loss | 0.6159 |
| final off deficit | 51.0299 |
| final coverage mean | 1.1758 |
| final coverage median | 1.1729 |
| final coverage q10 | 1.1624 |
| final coverage below-one rate | 0.0000 |
| final key gain | 59.9177 |
| final full margin | 8.8877 |
| final key-removed margin | -51.0299 |
| final atk target rate | 100.00% |
| final key-removed target rate | 0.00% |
| final key cosine | 0.7355 |
| final key energy ratio | 0.5413 |
| final residual target gain | 5.4072 |

Training interpretation:

- The local training proxy fully satisfies causal coverage reserve.
- Key-removed target rate remains 0%, so the training proxy did not obviously learn a key-independent target branch.
- Coverage q10 is above 1 in the final epoch.

## ASR Results

ASR command:

```bash
bash run_kdr_dtk_ccr_eval_asr.sh
```

Important note: an earlier first attempt produced invalid TA/TIES numbers because `eval_submerge.py` accepted `KDR_DTK_CCR` as an attack type but had not yet routed the adversary task to the CCR checkpoint. That invalid run used the clean CIFAR100 checkpoint with the KDR_DTK trigger. I fixed the checkpoint and trigger compatibility, verified the resolved paths, and reran ASR. The valid results are below.

Valid resolved paths:

```text
checkpoint: ./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_CCR_CIFAR100_Tgt_1_L_22/finetuned.pt
trigger:    ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy
```

Results:

| Merge | ASR | Backdoored Count | Non-target Count | Patched Top-1 |
|---|---:|---:|---:|---:|
| TA | 100.00% | 9900 | 9900 | 1.00% |
| TIES | 100.00% | 9900 | 9900 | 1.00% |
| RegMean | 97.97% | 9699 | 9900 | 2.84% |

No clean utility / avg clean acc was run.

## Comparison

| Method | TA ASR | TIES ASR | RegMean ASR |
|---|---:|---:|---:|
| BIND-EXACT | 99.96% | 99.79% | 92.25% |
| SCB | 99.92% | 99.86% | 93.72% |
| FSB | 100.00% | 99.71% | 91.66% |
| CCR | 100.00% | 100.00% | 97.97% |

## Takeaway

CCR substantially improves RegMean ASR while preserving TA/TIES saturation:

```text
FSB RegMean: 91.66%
CCR RegMean: 97.97%
```

This supports the causal coverage reserve hypothesis:

> FSB closed the key-independent bypass, but endpoint feasibility did not guarantee enough causal coverage reserve under RegMean's differential transport. Optimizing continuous key gain relative to measured off-key deficit closes most of the remaining RegMean gap.

The next diagnostic should verify CCR key causality under RegMean:

```bash
bash run_kdr_dtk_ccr_key_causality_diag.sh
```

The critical check is that RegMean ASR(-key) remains low. If ASR improved by recreating a key-independent target path, the method story fails. If ASR(-key) remains low, CCR is the first strong version of the mechanism.

## Follow-up Diagnostics

Per request, I then ran:

```bash
bash run_kdr_dtk_ccr_key_causality_diag.sh
bash run_kdr_dtk_ccr_final_state_readout_diag.sh
bash run_kdr_dtk_ccr_paired_transport_diag.sh
```

No utility / clean acc was run.

Artifacts:

- Key causality JSON: `analysis/kdr_dtk_ccr_key_causality/20260708_160702_kdr_key_causality.json`
- Final-state readout JSON: `analysis/kdr_dtk_ccr_final_state_readout/20260708_161400_kdr_dtk_fsb_final_state_readout.json`
- Final-state readout CSV: `analysis/kdr_dtk_ccr_final_state_readout/20260708_161400_kdr_dtk_fsb_final_state_readout_samples.csv`
- Paired transport summary: `analysis/kdr_dtk_ccr_paired_transport/20260708_161411_fsb_paired_transport_summary.md`

### Key Causality

This was the one-vote veto experiment.

| Method | ASR | ASR(-key) | drop key | random drop | specificity | margin drop key | margin drop random |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean local | 0.05% | 0.05% | 0.00 | 0.00 | 0.00 | 0.8629 | -0.0861 |
| local attack | 100.00% | 0.00% | 100.00 | 0.00 | 100.00 | 74.2320 | 4.1202 |
| TA | 100.00% | 0.00% | 100.00 | 0.00 | 100.00 | 34.1042 | 1.6262 |
| TIES | 100.00% | 0.00% | 100.00 | 0.00 | 100.00 | 32.1616 | 2.1250 |
| RegMean | 97.79% | 0.05% | 97.74 | 0.37 | 97.37 | 19.4369 | 1.2267 |

Key emission:

| Method | key cosine median | key energy median | \|a\| median |
|---|---:|---:|---:|
| clean local | 0.2449 | 0.0600 | 0.3025 |
| local attack | 0.7379 | 0.5445 | 15.4863 |
| TA | 0.6989 | 0.4885 | 7.4407 |
| TIES | 0.6996 | 0.4895 | 6.2553 |
| RegMean | 0.6662 | 0.4439 | 4.2160 |

Interpretation:

CCR passes the veto. RegMean ASR rises to about 98%, while `ASR(-key)` stays near zero. The improvement is not from recreating a key-independent target path.

### Final-State Readout

| Method | Group | N | ASR | \|a\| | g_key | g_spec | r | r_spec | M_off | deficit | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local attack | all | 1901 | 100.00% | 15.4863 | 74.2320 | 67.9500 | 4.7946 | 4.3914 | -58.6223 | 58.6223 | 1.2669 |
| TA | all | 1901 | 100.00% | 7.4407 | 34.1042 | 30.8526 | 4.5862 | 4.1421 | -28.4843 | 28.4843 | 1.1983 |
| TIES | all | 1901 | 100.00% | 6.2553 | 32.1616 | 28.4414 | 5.1378 | 4.5515 | -24.6056 | 24.6056 | 1.3084 |
| RegMean | all | 1901 | 97.84% | 4.2641 | 19.6617 | 17.6787 | 4.5702 | 4.1077 | -15.0211 | 15.0211 | 1.2939 |
| RegMean | success | 1860 | 100.00% | 4.2763 | 19.7408 | 17.7368 | 4.5763 | 4.1081 | -15.0366 | 15.0366 | 1.2976 |
| RegMean | failure | 41 | 0.00% | 3.1969 | 11.9250 | 12.2188 | 3.8910 | 3.8026 | -13.7412 | 13.7412 | 0.9602 |

RegMean success-dose counterfactual:

| Metric | Value |
|---|---:|
| reference \|a\| | 4.2763 |
| original full ASR | 97.84% |
| equalized all-sample ASR | 98.74% |
| original failure count | 41 |
| original failure rescue rate | 80.49% |
| failure full margin median | -0.5668 |
| failure equalized margin median | 1.4338 |
| failure current \|a\| median | 3.1969 |
| failure added \|a\| median | 1.0794 |

Compared with FSB:

| Metric | FSB | CCR |
|---|---:|---:|
| RegMean all coverage | 1.7837 | 1.2939 |
| RegMean failure count | 150 | 41 |
| RegMean failure coverage | 0.8321 | 0.9602 |
| RegMean failure \|a\| | 2.3902 | 3.1969 |
| RegMean failure deficit | 10.0812 | 13.7412 |
| failure rescue rate by success dose | 52.67% | 80.49% |

The all-sample coverage median is lower under CCR because the objective drives both gain and deficit upward; the important change is that far fewer samples remain failures and the remaining failure margins are close to the boundary.

### Paired Transport

Reference context: `local_attack`.

Future RegMean success: 1860 samples.

Future RegMean failure: 41 samples.

Local metrics grouped by future RegMean outcome:

| Local metric | Future success median | Future failure median | fail/succ |
|---|---:|---:|---:|
| `|a|` | 15.4938 | 14.6728 | 0.9470 |
| `r` | 4.7910 | 5.0044 | 1.0445 |
| `r_spec` | 4.3883 | 4.6051 | 1.0494 |
| `M_off` | -58.6253 | -57.9912 | 0.9892 |
| `off_deficit` | 58.6253 | 57.9912 | 0.9892 |
| `coverage` | 1.2668 | 1.2686 | 1.0014 |

Local predictability:

| Local metric | orientation-free AUC | failure direction |
|---|---:|---|
| `local_a_abs` | 0.7874 | lower |
| `local_g_key` | 0.7295 | lower |
| `local_g_spec` | 0.6873 | lower |
| `local_r` | 0.7923 | higher |
| `local_r_spec` | 0.7888 | higher |
| `local_m_off` | 0.7412 | higher |
| `local_off_deficit` | 0.7412 | lower |
| `local_coverage` | 0.6505 | higher |

Paired RegMean transport:

| Metric | Future success median | Future failure median |
|---|---:|---:|
| `A_ratio` | 0.2764 | 0.2173 |
| `r_ratio` | 0.9539 | 0.7701 |
| `r_spec_ratio` | 0.9336 | 0.8451 |
| `deficit_ratio` | 0.2569 | 0.2367 |
| `coverage_ratio` | 1.0224 | 0.7544 |
| `required_dose` | 3.2764 | 3.5384 |
| `dose_reserve` | 0.9858 | -0.1540 |
| `dose_shortfall` | 0.0000 | 0.1540 |

Mean log-transport decomposition:

| Metric | Future success mean | Future failure mean |
|---|---:|---:|
| `log_a_transport` | -1.302499 | -1.512776 |
| `log_r_transport` | -0.068279 | -0.312069 |
| `log_deficit_transport` | -1.404969 | -1.484343 |
| `log_coverage_transport` | 0.034192 | -0.340503 |
| `dose_pressure` | 1.302499 | 1.512776 |
| `readout_pressure` | 0.068279 | 0.312069 |
| `deficit_pressure` | -1.404969 | -1.484343 |
| `log_coverage_reconstruction_error` | 0.000000 | 0.000000 |

Factor replacement:

| Scenario | Success C median | Success C>1 | Failure C median | Failure C>1 |
|---|---:|---:|---:|---:|
| `factor_local` | 1.2668 | 100.00% | 1.2686 | 100.00% |
| `factor_a_only` | 0.3502 | 0.00% | 0.2758 | 0.00% |
| `factor_r_only` | 1.2092 | 91.77% | 0.9772 | 46.34% |
| `factor_d_only` | 4.9308 | 100.00% | 5.3523 | 100.00% |
| `factor_a_r` | 0.3371 | 0.00% | 0.2067 | 0.00% |
| `factor_a_d` | 1.3727 | 97.79% | 1.1858 | 80.49% |
| `factor_r_d` | 4.7135 | 100.00% | 4.1376 | 100.00% |
| `factor_full` | 1.2976 | 100.00% | 0.9602 | 0.00% |

### Mechanism Takeaway

CCR does not improve `A_RM / A_local`; it is actually lower than FSB:

| Metric on future RegMean failures | FSB | CCR |
|---|---:|---:|
| `A_ratio` | 0.2741 | 0.2173 |
| `deficit_ratio` | 1.7849 | 0.2367 |
| `coverage_ratio` | 0.3069 | 0.7544 |
| `dose_shortfall` | 0.4965 | 0.1540 |

The main repair is not better dose transport. It is that CCR changes the local causal geometry so much that, after RegMean transport, the off-key deficit is no longer adversarially expanded. In fact, `D_RM / D_local` drops far below 1.

This supports the stronger interpretation:

> CCR pre-builds enough causal coverage reserve so the channel remains feasible after adverse merge transport. It does not preserve the representation dose; it makes the transported causal gain sufficient relative to the transported off-key deficit.

The one-vote veto also passes:

```text
RegMean ASR      = 97.79%
RegMean ASR(-k) = 0.05%
specificity     = 97.37 pp
```

So CCR is currently the strongest version of the method.
