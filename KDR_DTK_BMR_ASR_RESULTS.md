# KDR-DTK-BMR ASR Results

Run date: 2026-07-08

## Setup

- Method: `KDR_DTK_BMR`
- Model: `ViT-B-32`
- Adversary task: `CIFAR100`
- Target task: `CIFAR100`
- Target class: `1`
- Patch size: `22`
- Trigger: `trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy`
- Init checkpoint: `checkpoints/ViT-B-32/CIFAR100/finetuned.pt`
- Output checkpoint: `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt`

Only ASR was evaluated. Utility / clean accuracy was not run.

## ASR

| Merge method | ASR | Count |
|---|---:|---:|
| TA | 100.00% | 9900 / 9900 |
| TIES | 100.00% | 9900 / 9900 |
| RegMean | 97.52% | 9654 / 9900 |

## Training Summary

Final epoch metrics:

| Metric | Value |
|---|---:|
| `final_avg_attack_target_rate` | 1.0000 |
| `final_avg_attack_margin` | 28.6913 |
| `final_avg_background_margin` | -2.7492 |
| `final_avg_background_deficit` | 2.8380 |
| `final_avg_margin_gain` | 31.4405 |
| `final_avg_margin_gain_min` | 26.0297 |
| `final_avg_reserve_ratio_median` | 11.8842 |
| `final_avg_reserve_ratio_q10` | 5.4715 |
| `final_avg_reserve_below_one_rate` | 0.0000 |
| `final_avg_reserve_loss` | 0.0830 |

## Interpretation

BMR is a strong simplification result. Removing final-state key estimation, key deletion, bind loss, and key-specific coverage still gives RegMean ASR close to CCR:

| Method | TA | TIES | RegMean |
|---|---:|---:|---:|
| CCR reference | 100.00% | 100.00% | 97.92% |
| BMR | 100.00% | 100.00% | 97.52% |

This supports the hypothesis that the main effective component can be simplified to background margin reserve:

`malicious margin gain > background target deficit`

CCR remains useful as a mechanism-discovery and diagnostic path, but BMR appears sufficient for the ASR objective under TA/TIES/RegMean in this run.

## Logs

- Consolidated run log: `logs/kdr_dtk_bmr_all_20260708_225616.log`
- Training log: `logs/kdr_dtk_bmr_train_20260708_225629.log`
- TA/TIES eval log: `logs/kdr_dtk_bmr_eval_ta_ties_20260708_232814.log`
- RegMean eval log: `logs/kdr_dtk_bmr_eval_regmean_20260708_232814.log`
- TA/TIES result JSON: `results/kdr_dtk_bmr/20260708_232822_KDR_DTK_BMR_CIFAR100_tgt1_L22.json`
