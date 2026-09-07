# KDR-DTK-BIND Sender-Dose Audit Results

## Source Files

- `analysis/kdr_dtk_bind_sender_dose_audit/20260707_221737_kdr_sender_dose_audit.json`
- `analysis/kdr_dtk_bind_sender_dose_audit/20260707_221737_kdr_sender_dose_samples.csv`
- `logs/kdr_dtk_bind_sender_dose_audit_20260707_221057.log`

## Setup

- Attack checkpoint: `./checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BIND_CIFAR100_Tgt_1_L_22/finetuned.pt`
- Trigger: `./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy`
- Key layer: `model.visual.transformer.resblocks.11.ln_2`
- Methods: `local_attack, ta, ties, regmean`
- Eval batches: `30`
- Batch size: `64`

## Stage-1 Native Sender Key

| Metric | Value |
|---|---:|
| within-batch cosine median | 0.809560 |
| batch-to-global cosine median | 0.995586 |
| batch-to-global cosine q10 | 0.994994 |
| shift norm median | 14.089871 |

## Main Sender-Dose Table

| Method | N | Full ASR (%) | Sender-dose ASR (%) | |a0| med | |am| med | |am|/|a0| med | M_full med | M_sender med | M_sender - M_full med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_attack | 1901 | 100.00 | 98.05 | 11.3478 | 16.8966 | 1.4615 | 8.1706 | 4.0530 | -4.0816 |
| ta | 1901 | 99.89 | 67.60 | 11.3478 | 24.1457 | 2.1060 | 4.4371 | 0.5420 | -3.9141 |
| ties | 1901 | 99.63 | 78.64 | 11.3478 | 27.7905 | 2.4243 | 4.9942 | 0.9067 | -4.0134 |
| regmean | 1901 | 90.43 | 71.59 | 11.3478 | 23.5752 | 2.0356 | 2.5091 | 0.8946 | -1.5713 |

## Success / Failure Split Under Full Current Dose

| Method | Group | N | Full ASR (%) | Sender-dose ASR (%) | |a0| med | |am| med | |am|/|a0| med | M_full med | M_sender med |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local_attack | success | 1901 | 100.00 | 98.05 | 11.3478 | 16.8966 | 1.4615 | 8.1706 | 4.0530 |
| local_attack | failure | 0 | nan | nan | nan | nan | nan | nan | nan |
| ta | success | 1899 | 100.00 | 67.67 | 11.3488 | 24.1484 | 2.1060 | 4.4400 | 0.5421 |
| ta | failure | 2 | 0.00 | 0.00 | 8.5888 | 20.7088 | 2.6091 | -0.6046 | -4.1917 |
| ties | success | 1894 | 100.00 | 78.93 | 11.3534 | 27.8076 | 2.4241 | 5.0012 | 0.9116 |
| ties | failure | 7 | 0.00 | 0.00 | 8.2898 | 18.5841 | 2.4525 | -1.0924 | -4.0837 |
| regmean | success | 1719 | 100.00 | 79.06 | 11.5078 | 23.9832 | 2.0446 | 2.6776 | 1.0909 |
| regmean | failure | 182 | 0.00 | 1.10 | 9.7374 | 18.6926 | 1.9067 | -1.1321 | -2.3869 |

## Key Comparisons

| Method | Current/native dose ratio | Sender-dose ASR drop vs full (pp) | Sender margin drop |
|---|---:|---:|---:|
| local_attack | 1.4615 | 1.95 | -4.0816 |
| ta | 2.1060 | 32.30 | -3.9141 |
| ties | 2.4243 | 20.99 | -4.0134 |
| regmean | 2.0356 | 18.83 | -1.5713 |

## Interpretation

- Current BIND does not merely use the Stage-1 native sender dose. Current key coordinate is amplified relative to frozen sender dose in every receiver.
- Median current/native dose ratio: local `1.46x`, TA `2.11x`, TIES `2.42x`, RegMean `2.04x`.
- Replacing current dose with Stage-1 native sender dose keeps local high (`98.05%`) but drops merged models: TA `67.60%`, TIES `78.64%`, RegMean `71.59%`.
- This supports sender-receiver dose mismatch: merged attacks rely on a self-amplified current key coordinate, not just the original Stage-1 native dose.
- RegMean is not uniquely worse under sender-dose than TA/TIES here; TA drops the most. The bigger result is that native sender dose is insufficient for all merged contexts.
