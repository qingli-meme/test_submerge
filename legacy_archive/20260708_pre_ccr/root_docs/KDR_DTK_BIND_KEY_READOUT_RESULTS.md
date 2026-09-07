# KDR-DTK-BIND Key Readout Efficiency Results

## Source Files

- `analysis/kdr_dtk_bind_key_readout_efficiency/20260707_213907_kdr_key_readout_efficiency.json`
- `analysis/kdr_dtk_bind_key_readout_efficiency/20260707_213907_kdr_key_readout_efficiency_samples.csv`
- `logs/kdr_dtk_bind_key_readout_efficiency_20260707_213236.log`

## Setup

- Attack checkpoint: `./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BIND_CIFAR100_Tgt_1_L_22/finetuned.pt`
- Trigger: `./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy`
- Key layer: `model.visual.transformer.resblocks.11.ln_2`
- Methods: `local_attack, ta, ties, regmean`
- Eval batches: `30`
- Batch size: `64`

## Stage-1 Key Stability

| Metric | Value |
|---|---:|
| within-batch cosine median | 0.809560 |
| batch-to-global cosine median | 0.995586 |
| batch-to-global cosine q10 | 0.994994 |
| shift norm median | 14.089871 |

## Main Readout Table

| Method | Group | N | Target rate (%) | |a| med | g_key med | g_spec med | r med | r_spec med | M_full med |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local_attack | all | 1901 | 100.00 | 16.8966 | 12.2303 | 13.4776 | 0.7257 | 0.8001 | 8.1706 |
| local_attack | success | 1901 | 100.00 | 16.8966 | 12.2303 | 13.4776 | 0.7257 | 0.8001 | 8.1706 |
| local_attack | failure | 0 | nan | nan | nan | nan | nan | nan | nan |
| ta | all | 1901 | 99.89 | 24.1457 | 6.6527 | 7.9812 | 0.2768 | 0.3327 | 4.4371 |
| ta | success | 1899 | 100.00 | 24.1484 | 6.6527 | 7.9821 | 0.2769 | 0.3328 | 4.4400 |
| ta | failure | 2 | 0.00 | 20.7088 | 4.9824 | 6.0405 | 0.2411 | 0.2909 | -0.6046 |
| ties | all | 1901 | 99.63 | 27.7905 | 6.3128 | 7.6751 | 0.2260 | 0.2759 | 4.9942 |
| ties | success | 1894 | 100.00 | 27.8076 | 6.3176 | 7.6779 | 0.2261 | 0.2760 | 5.0012 |
| ties | failure | 7 | 0.00 | 18.5841 | 2.9963 | 3.9194 | 0.1613 | 0.2062 | -1.0924 |
| regmean | all | 1901 | 90.43 | 23.5752 | 3.0850 | 4.1441 | 0.1302 | 0.1789 | 2.5091 |
| regmean | success | 1719 | 100.00 | 23.9832 | 3.1554 | 4.2700 | 0.1316 | 0.1806 | 2.6776 |
| regmean | failure | 182 | 0.00 | 18.6926 | 2.0733 | 2.5395 | 0.1051 | 0.1463 | -1.1321 |

## Success vs Failure Ratios

| Method | Metric | Success median | Failure median | Failure - Success | Failure / Success |
|---|---|---:|---:|---:|---:|
| local_attack | `key_amplitude_abs` | 16.8966 | nan | nan | nan |
| local_attack | `causal_margin_gain` | 12.2303 | nan | nan | nan |
| local_attack | `specific_margin_gain` | 13.4776 | nan | nan | nan |
| local_attack | `readout_efficiency` | 0.7257 | nan | nan | nan |
| local_attack | `specific_readout_efficiency` | 0.8001 | nan | nan | nan |
| ta | `key_amplitude_abs` | 24.1484 | 20.7088 | -3.4396 | 0.8576 |
| ta | `causal_margin_gain` | 6.6527 | 4.9824 | -1.6704 | 0.7489 |
| ta | `specific_margin_gain` | 7.9821 | 6.0405 | -1.9416 | 0.7568 |
| ta | `readout_efficiency` | 0.2769 | 0.2411 | -0.0358 | 0.8708 |
| ta | `specific_readout_efficiency` | 0.3328 | 0.2909 | -0.0419 | 0.8741 |
| ties | `key_amplitude_abs` | 27.8076 | 18.5841 | -9.2235 | 0.6683 |
| ties | `causal_margin_gain` | 6.3176 | 2.9963 | -3.3212 | 0.4743 |
| ties | `specific_margin_gain` | 7.6779 | 3.9194 | -3.7585 | 0.5105 |
| ties | `readout_efficiency` | 0.2261 | 0.1613 | -0.0648 | 0.7134 |
| ties | `specific_readout_efficiency` | 0.2760 | 0.2062 | -0.0698 | 0.7471 |
| regmean | `key_amplitude_abs` | 23.9832 | 18.6926 | -5.2906 | 0.7794 |
| regmean | `causal_margin_gain` | 3.1554 | 2.0733 | -1.0820 | 0.6571 |
| regmean | `specific_margin_gain` | 4.2700 | 2.5395 | -1.7304 | 0.5947 |
| regmean | `readout_efficiency` | 0.1316 | 0.1051 | -0.0265 | 0.7987 |
| regmean | `specific_readout_efficiency` | 0.1806 | 0.1463 | -0.0344 | 0.8097 |

## Key Findings

- RegMean failure is mixed, not a pure amplitude-only failure and not a pure readout-only failure.
- RegMean failure/success key amplitude ratio: `0.7794`.
- RegMean failure/success causal gain ratio: `0.6571`.
- RegMean failure/success specific gain ratio: `0.5947`.
- RegMean failure/success readout efficiency ratio: `0.7987`.
- RegMean failure/success specific readout efficiency ratio: `0.8097`.
- Across all samples, RegMean specific readout efficiency is much lower than TA/TIES, so unit key signal is under-read by RegMean.

## Interpretation

RegMean compresses both the emitted/effective key amplitude and the target-margin readout per unit key. The stronger signal is readout compression: all-sample `r_spec` is `0.1789` for RegMean versus `0.3327` for TA and `0.2759` for TIES.
