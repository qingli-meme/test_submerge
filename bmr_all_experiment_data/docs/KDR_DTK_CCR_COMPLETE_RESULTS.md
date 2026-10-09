# KDR-DTK-CCR Complete Results

Generated: 2026-07-08 18:06 CST

## Scope

This file consolidates the completed CCR experiment and the currently available baseline table from existing logs/results.

Main method:

`KDR_DTK_CCR`

Checkpoint:

`checkpoints/ViT-B-32/CIFAR100_KDR_DTK_CCR_CIFAR100_Tgt_1_L_22/finetuned.pt`

Trigger:

`trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy`

Evaluation setting:

- model: `ViT-B-32`
- adversary task: `CIFAR100`
- target task: `CIFAR100`
- target class: `1`
- patch size: `22`
- scaling coefficient: `0.3`
- datasets: `CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS`
- merge methods: `TA`, `TIES`, `RegMean`, `AdaMerging`

## CCR Result

| Merge | ASR | Avg Clean Acc | Source |
| --- | ---: | ---: | --- |
| TA | 100.00% | 74.74% | `logs/kdr_dtk_ccr_utility_ta_ties_20260708_162235.log` |
| TIES | 100.00% | 72.72% | `logs/kdr_dtk_ccr_utility_ta_ties_20260708_162235.log` |
| RegMean | 97.92% | 75.20% | `logs/kdr_dtk_ccr_utility_regmean_20260708_165611.log` |
| AdaMerging | 100.00% | 80.40% | `logs/kdr_dtk_ccr_eval_adamerging_20260708_170052.log` |

CCR passes the current clean-utility veto. The very negative `M_off` observed in diagnostics does not come from collapsed merged clean utility.

## CCR Diagnostics

| Check | Key number |
| --- | ---: |
| RegMean ASR | 97.79% |
| RegMean ASR after key removal | 0.05% |
| RegMean key specificity | 97.37 pp |
| RegMean final-state readout coverage | 1.2939 |
| RegMean future-failure count | 41 / 1901 |
| RegMean failure coverage | 0.9602 |

Interpretation:

CCR improves RegMean mainly by building enough causal coverage reserve before merge. The improvement is not caused by a key-independent shortcut, because removing the final-state key still collapses RegMean ASR to near zero.

## Baseline ASR Table

`N/R` means no same-method result was found in current logs/results for that merge method.

| Method | TA | TIES | RegMean | AdaMerging |
| --- | ---: | ---: | ---: | ---: |
| BadMerging-On | 99.99% | 99.98% | 99.85% | 99.99% |
| KDR | 99.98% | 99.93% | 66.09% | 99.98% |
| DTK | 99.97% | 99.47% | 89.92% | 99.97% |
| BIND-EXACT | 99.96% | 99.79% | 92.25% | N/R |
| FSB | 100.00% | 99.71% | 91.66% | N/R |
| CCR | 100.00% | 100.00% | 97.92% | 100.00% |

Notes:

- KDR RegMean ASR is `66.09%` from the ASR-only all-merge run. The full-utility KDR RegMean run recorded `65.42%` ASR with `75.54%` Avg Clean Acc.
- CCR RegMean ASR is `97.92%` from the clean-utility run. The ASR-only RegMean run recorded `97.97%`.
- BIND-EXACT and FSB AdaMerging were not found as completed same-method logs in the current workspace.

## Baseline Avg Clean Acc Table

Only values with same-method utility logs are filled.

| Method | TA | TIES | RegMean | AdaMerging |
| --- | ---: | ---: | ---: | ---: |
| BadMerging-On | 75.41% | 72.91% | 75.42% | 80.99% |
| KDR | 75.01% | 72.85% | 75.54% | 80.55% |
| DTK | N/R | N/R | 75.47% | 75.10% |
| BIND-EXACT | N/R | N/R | N/R | N/R |
| FSB | N/R | N/R | N/R | N/R |
| CCR | 74.74% | 72.72% | 75.20% | 80.40% |

## Raw Result Sources

CCR:

- `logs/kdr_dtk_ccr_utility_ta_ties_20260708_162235.log`
- `logs/kdr_dtk_ccr_utility_regmean_20260708_165611.log`
- `logs/kdr_dtk_ccr_eval_adamerging_20260708_170052.log`
- `results/kdr_dtk_ccr_utility/20260708_162243_KDR_DTK_CCR_CIFAR100_tgt1_L22.json`
- `results/kdr_dtk_ccr/20260708_151600_KDR_DTK_CCR_CIFAR100_tgt1_L22.json`

Baselines:

- BadMerging-On: `logs/badmergingon_full_utility_20260630_134603.log`
- KDR: `logs/kdr_full_utility_20260630_134603.log`, `logs/kdr_all_merge_asr_then_diag_20260630_134603.log`
- DTK: `logs/kdr_dtk_eval_20260706_190141.log`, `logs/kdr_dtk_eval_regmean_retry_20260706_190325.log`, `logs/kdr_dtk_eval_adamerging_retry_20260706_191103.log`
- BIND-EXACT: `logs/kdr_dtk_bind_exact_eval_ta_ties_20260707_185943.log`, `logs/kdr_dtk_bind_exact_eval_regmean_20260707_185943.log`
- FSB: `logs/kdr_dtk_fsb_eval_ta_ties_20260708_121211.log`, `logs/kdr_dtk_fsb_eval_regmean_20260708_121211.log`

