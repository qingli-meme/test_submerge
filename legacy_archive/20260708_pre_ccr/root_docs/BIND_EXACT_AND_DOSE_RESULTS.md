# BIND-EXACT and Key Dose-Response Results

## Source Files

- `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BIND_EXACT_CIFAR100_Tgt_1_L_22/20260707_181134_dtk_bind_exact_summary.json`
- `results/kdr_dtk_bind_exact/20260707_185952_KDR_DTK_BIND_EXACT_CIFAR100_tgt1_L22.json`
- `analysis/kdr_dtk_bind_exact_key_causality/20260707_185942_kdr_key_causality.json`
- `analysis/kdr_dtk_bind_key_dose_response/20260707_210141_kdr_key_dose_response.json`

## BIND-EXACT Training Summary

| Metric | Value |
|---|---:|
| `resolved_layer` | `model.visual.transformer.resblocks.11.ln_2` |
| `within_batch_cosine_median` | 0.806508 |
| `batch_to_global_cosine_median` | 0.997595 |
| `shift_norm_median` | 14.129754 |
| `final_avg_atk_target_rate` | 1.000000 |
| `final_avg_key_removed_target_rate` | 0.006436 |
| `final_avg_full_margin` | 7.433076 |
| `final_avg_key_removed_margin` | -3.129856 |
| `final_avg_key_gain` | 10.562932 |
| `final_avg_bind_loss` | 0.002637 |

## BIND-EXACT ASR

| Merge | ASR (%) | Patched Top-1 (%) | Backdoored / Non-target |
|---|---:|---:|---:|
| TA | 99.96 | 1.04 | 9896 / 9900 |
| TIES | 99.79 | 1.19 | 9879 / 9900 |
| RegMean | 92.25 | 6.02 | 9133 / 9900 |

## BIND-EXACT Key Causality Audit

Stage-1 key layer: `model.visual.transformer.resblocks.11.ln_2`
Stage-1 within-batch cosine median: `0.809560`
Stage-1 batch-to-global cosine median: `0.995586`

| Method | ASR (%) | ASR after key removal (%) | Drop key (pp) | Drop random (pp) | Specificity (pp) | Median margin drop key | Median margin drop random |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_local | 0.05 | 0.05 | 0.00 | 0.00 | 0.00 | -0.1208 | -0.0235 |
| local_attack | 100.00 | 0.00 | 100.00 | 0.00 | 100.00 | 15.8529 | -0.4077 |
| ta | 99.95 | 1.16 | 98.79 | -0.05 | 98.84 | 6.8035 | -1.1133 |
| ties | 99.68 | 10.42 | 89.27 | -0.16 | 89.43 | 6.4001 | -1.0883 |
| regmean | 92.43 | 32.14 | 60.28 | -1.79 | 62.07 | 3.0702 | -1.1105 |

## Existing BIND Key Dose-Response Audit

Audit checkpoint: `./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BIND_CIFAR100_Tgt_1_L_22/finetuned.pt`
Trigger: `./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy`
Key layer: `model.visual.transformer.resblocks.11.ln_2`
Within-batch cosine median: `0.809560`
Batch-to-global cosine median: `0.995586`

### local_attack

| Lambda | Key ASR (%) | Random ASR (%) | Key margin median | Random margin median | Key target-logit median | Random target-logit median |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.47 | 100.00 | -3.9761 | 9.3930 | 24.4423 | 30.1767 |
| 0.25 | 10.73 | 100.00 | -1.3695 | 9.1895 | 25.5407 | 30.2353 |
| 0.50 | 95.74 | 100.00 | 1.6306 | 8.8870 | 27.0494 | 30.2859 |
| 0.75 | 99.84 | 100.00 | 4.8847 | 8.5467 | 28.8490 | 30.3530 |
| 1.00 | 100.00 | 100.00 | 8.1706 | 8.1706 | 30.4030 | 30.4030 |

- `key_auc_target_rate`: 64.1373
- `random_auc_target_rate`: 100.0000
- `key_interior_target_rate_mean`: 68.7708
- `random_interior_target_rate_mean`: 100.0000
- `key_normalized_target_rate_recovery`: `{'0.0000': 0.0, '0.2500': 0.10306553969071391, '0.5000': 0.9571881585646594, '0.7500': 0.9984144006420222, '1.0000': 1.0}`
- `key_normalized_margin_recovery`: `{'0.0000': 0.0, '0.2500': 0.21459712025968414, '0.5000': 0.4615880215163986, '0.7500': 0.7294808766852083, '1.0000': 1.0}`

### ta

| Lambda | Key ASR (%) | Random ASR (%) | Key margin median | Random margin median | Key target-logit median | Random target-logit median |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.21 | 99.95 | -2.1970 | 5.7555 | 20.6856 | 23.9862 |
| 0.25 | 14.62 | 99.95 | -1.0381 | 5.6355 | 20.8424 | 23.8251 |
| 0.50 | 77.28 | 99.95 | 0.6757 | 5.3359 | 21.7573 | 23.7461 |
| 0.75 | 99.47 | 99.95 | 2.6513 | 4.9189 | 23.2635 | 23.8064 |
| 1.00 | 99.89 | 99.89 | 4.4371 | 4.4371 | 23.9620 | 23.9620 |

- `key_auc_target_rate`: 60.4813
- `random_auc_target_rate`: 99.9408
- `key_interior_target_rate_mean`: 63.7910
- `random_interior_target_rate_mean`: 99.9474
- `key_normalized_target_rate_recovery`: `{'0.0000': 0.0, '0.2500': 0.13592750283834956, '0.5000': 0.7707889443912247, '0.7500': 0.9957355925359241, '1.0000': 1.0}`
- `key_normalized_margin_recovery`: `{'0.0000': 0.0, '0.2500': 0.1746894130912982, '0.5000': 0.4330293519040025, '0.7500': 0.7308167247575091, '1.0000': 1.0}`

### ties

| Lambda | Key ASR (%) | Random ASR (%) | Key margin median | Random margin median | Key target-logit median | Random target-logit median |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 11.94 | 99.79 | -1.3273 | 6.3807 | 25.5846 | 26.6690 |
| 0.25 | 42.45 | 99.74 | -0.2155 | 6.3743 | 25.6896 | 26.5517 |
| 0.50 | 93.06 | 99.74 | 1.5471 | 6.0941 | 26.6934 | 26.5260 |
| 0.75 | 99.32 | 99.74 | 3.4830 | 5.6128 | 27.5597 | 26.5841 |
| 1.00 | 99.63 | 99.63 | 4.9942 | 4.9942 | 26.7799 | 26.7799 |

- `key_auc_target_rate`: 72.6526
- `random_auc_target_rate`: 99.7304
- `key_interior_target_rate_mean`: 78.2746
- `random_interior_target_rate_mean`: 99.7370
- `key_normalized_target_rate_recovery`: `{'0.0000': 0.0, '0.2500': 0.3479303917510581, '0.5000': 0.9250149705021272, '0.7500': 0.9964007071648965, '1.0000': 1.0}`
- `key_normalized_margin_recovery`: `{'0.0000': 0.0, '0.2500': 0.17588176830986638, '0.5000': 0.4547054496476789, '0.7500': 0.7609508596973242, '1.0000': 1.0}`

### regmean

| Lambda | Key ASR (%) | Random ASR (%) | Key margin median | Random margin median | Key target-logit median | Random target-logit median |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 34.19 | 92.37 | -0.6291 | 3.6822 | 28.7627 | 28.1966 |
| 0.25 | 49.55 | 92.53 | -0.0168 | 3.5807 | 29.0276 | 28.3765 |
| 0.50 | 73.22 | 92.53 | 0.9584 | 3.3029 | 29.8362 | 28.5652 |
| 0.75 | 87.22 | 91.74 | 1.9255 | 2.9288 | 30.0692 | 28.7480 |
| 1.00 | 90.43 | 90.43 | 2.5091 | 2.5091 | 28.9157 | 28.9157 |

- `key_auc_target_rate`: 68.0760
- `random_auc_target_rate`: 92.0502
- `key_interior_target_rate_mean`: 69.9982
- `random_interior_target_rate_mean`: 92.2672
- `key_normalized_target_rate_recovery`: `{'0.0000': 0.0, '0.2500': 0.27315248043959006, '0.5000': 0.6941065787093683, '0.7500': 0.9429372839991916, '1.0000': 1.0}`
- `key_normalized_margin_recovery`: `{'0.0000': 0.0, '0.2500': 0.19509891455039752, '0.5000': 0.5058715414551309, '0.7500': 0.8140379709829583, '1.0000': 1.0}`

## Short Interpretation

- BIND-EXACT keeps the original DTK trigger and `resblocks.11.ln_2` interface, changing only key-component `.detach()` removal.
- BIND-EXACT does not collapse: TA/TIES are near 100%, and RegMean remains about 92%.
- Therefore BAB collapse is more consistent with moving the interface to block9 than with exact projection backward semantics.
- Dose-response shows RegMean has a slower key response than TA/TIES, but it is not key-disconnected.
