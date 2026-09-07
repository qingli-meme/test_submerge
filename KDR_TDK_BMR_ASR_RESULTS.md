# KDR-TDK-BMR ASR Results

## Setup

- Variant: `KDR_TDK_BMR`
- Change from BMR: removed dormant-key constraint in trigger Stage 1; used target-directed key trigger.
- Model: `ViT-B-32`
- Target dataset: `CIFAR100`
- Target class: `1`
- Patch size: `22`
- Scaling coefficient: `0.3`
- Checkpoint: `checkpoints/ViT-B-32/CIFAR100_KDR_TDK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt`
- Trigger: `trigger/ViT-B-32/KDR_TDK_CIFAR100_Tgt_1_L_22.npy`

## Trigger Optimization Summary

- `cos_t` improved from `0.2635` to `0.6186` over 3 epochs.
- `target_projection` improved from `4.6548` to `14.3835`.
- `shift_norm` improved from `17.4203` to `23.0848`.

## Training Summary

Final epoch summary:

- `final_avg_attack_target_rate`: `1.0000`
- `final_avg_attack_margin`: `23.2418`
- `final_avg_background_target_rate`: `0.6500`
- `final_avg_background_margin`: `0.2956`
- `final_avg_margin_gain`: `22.9462`
- `final_avg_reserve_ratio_median`: `17.4993`
- `final_avg_reserve_below_one_rate`: `0.0000`
- `final_avg_clean_acc` in training batches: `0.9954`

## ASR Results

| Merge method | ASR | Backdoored count |
|---|---:|---:|
| TA | 100.00% | 9900 / 9900 |
| TIES | 99.99% | 9899 / 9900 |
| RegMean | 99.44% | 9845 / 9900 |

## Clean-Merged Trigger Prior / Stealth Check

This measures clean merged models with no malicious checkpoint in the merge, using the same `KDR_TDK` trigger.

| Clean merge method | Trigger ASR | Backdoored count | Patched top-1 |
|---|---:|---:|---:|
| Clean TA | 13.57% | 1343 / 9900 | 65.19% |
| Clean TIES | 35.80% | 3544 / 9900 | 50.70% |
| Clean RegMean | 25.90% | 2564 / 9900 | 57.68% |

Interpretation: removing dormancy substantially improves RegMean attack ASR, but it also creates a much stronger trigger prior on clean merged models. This is worse for stealth than the dormant-trigger version and means the no-dormancy variant is not a clean stealth improvement.

## Comparison To Dormant BMR

Previous `KDR_DTK_BMR` ASR:

| Method | TA | TIES | RegMean |
|---|---:|---:|---:|
| KDR_DTK_BMR | 100.00% | 100.00% | 97.52% |
| KDR_TDK_BMR, no dormancy | 100.00% | 99.99% | 99.44% |

## Takeaway

Removing dormancy did not hurt TA/TIES, and RegMean improved from `97.52%` to `99.44%` under the same ASR-only evaluation path. This suggests the dormancy constraint was not necessary for attack strength in this BMR setting, and may have been limiting RegMean robustness.

## Source Logs

- Trigger optimization log: `logs/kdr_tdk_bmr_optimize_trigger_20260709_092932.log`
- Training log: `logs/kdr_tdk_bmr_train_20260709_101900.log`
- ASR eval logs: latest `logs/kdr_tdk_bmr_eval_ta_ties_*.log` and `logs/kdr_tdk_bmr_eval_regmean_*.log`
- Training summary: `checkpoints/ViT-B-32/CIFAR100_KDR_TDK_BMR_CIFAR100_Tgt_1_L_22/20260709_101903_dtk_bmr_summary.json`
