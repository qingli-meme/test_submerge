# KDR-DTK-SCB Experiment Results

Date: 2026-07-08  
Workspace: `/data0/BadMerging`  
Method: `KDR_DTK_SCB`  
Base trigger: `trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy`  
Checkpoint: `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_SCB_CIFAR100_Tgt_1_L_22/finetuned.pt`

## What Ran

Completed:

- Static checks for SCB training / diagnostics / eval entrypoints.
- SCB 5-epoch training.
- Key causality audit.
- Sender-dose calibration audit.
- ASR evaluation for TA, TIES, RegMean.
- AdaMerging initial ASR before adaptive coefficient optimization.

Stopped by request:

- AdaMerging 500-epoch coefficient optimization. Final AdaMerging ASR is not available.

## Source Files

- Training summary: `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_SCB_CIFAR100_Tgt_1_L_22/20260707_225850_dtk_scb_summary.json`
- Key causality: `analysis/kdr_dtk_scb_key_causality/20260708_000014_kdr_key_causality.json`
- Sender calibration: `analysis/kdr_dtk_scb_sender_calibration/20260708_000814_kdr_dtk_scb_sender_calibration.json`
- TA/TIES ASR: `results/kdr_dtk_scb/20260708_000831_KDR_DTK_SCB_CIFAR100_tgt1_L22.json`
- RegMean ASR log: `logs/kdr_dtk_scb_eval_regmean_20260708_000823.log`
- AdaMerging partial log: `logs/kdr_dtk_scb_eval_adamerging_20260708_000823.log`

## Training Summary

Stage-1 key stability:

| metric | value |
|---|---:|
| within-batch key cosine median | 0.8096 |
| batch-prototype to global cosine median | 0.9956 |
| trigger shift norm median | 14.0899 |

Final training metrics:

| metric | value |
|---|---:|
| elapsed | 3252 sec |
| full triggered target rate | 99.05% |
| sender-calibrated target rate | 99.76% |
| key-removed target rate | 3.91% |
| full margin mean | 7.9857 |
| sender margin mean | 5.1267 |
| key-removed margin mean | -2.2817 |
| current key coeff abs mean | 18.6113 |
| sender key coeff abs mean | 11.3680 |
| current / sender dose ratio median | 1.6481 |
| residual target gain mean | 6.1849 |
| bind loss mean | 0.0192 |

Interpretation: training objective itself was satisfied. The sender-calibrated branch is target-effective, and the key-removed branch is mostly suppressed on training batches.

## ASR Results

| merge method | ASR | count | note |
|---|---:|---:|---|
| TA | 99.92% | 9892 / 9900 | completed |
| TIES | 99.86% | 9886 / 9900 | completed |
| RegMean | 93.72% | 9278 / 9900 | completed |
| AdaMerging initial | 99.92% | 9892 / 9900 | before lambda optimization |
| AdaMerging final | N/A | N/A | stopped during 500-epoch optimization |

RegMean improved only slightly over the prior BIND-EXACT result reported earlier at 92.25%. SCB did not produce a large RegMean breakthrough.

## Key Causality Audit

Key emission:

| method | key cosine median | key energy ratio median | \|a\| median |
|---|---:|---:|---:|
| local_attack | 0.2616 | 0.0685 | 14.3694 |
| TA | 0.4504 | 0.2029 | 23.4811 |
| TIES | 0.6139 | 0.3769 | 28.5636 |
| RegMean | 0.6336 | 0.4015 | 24.0942 |

Key necessity:

| method | ASR | ASR after key removal | drop | random-control drop | specificity |
|---|---:|---:|---:|---:|---:|
| local_attack | 98.53% | 0.79% | 97.74 | -0.32 | 98.05 |
| TA | 99.95% | 4.31% | 95.63 | -0.05 | 95.69 |
| TIES | 99.84% | 24.41% | 75.43 | -0.05 | 75.49 |
| RegMean | 93.37% | 54.18% | 39.19 | -1.42 | 40.61 |

Margin medians:

| method | triggered margin | key-removed margin | random-control margin |
|---|---:|---:|---:|
| local_attack | 10.7068 | -4.1109 | 10.3101 |
| TA | 5.2620 | -1.6481 | 5.8691 |
| TIES | 5.7629 | -0.7110 | 6.3497 |
| RegMean | 2.9060 | 0.1537 | 3.7822 |

Interpretation: SCB creates strong key necessity for local and TA, acceptable but weaker necessity for TIES, and still weak necessity for RegMean. Under RegMean, removing the key leaves ASR at 54.18%, so the receiver still has substantial key-independent target evidence.

## Sender Calibration Audit

This audit replaces the current model key dose with the frozen Stage-1 sender dose on the same samples.

| method | full ASR | sender-dose ASR | off ASR | \|a0\| median | \|am\| median | \|am\|/\|a0\| median |
|---|---:|---:|---:|---:|---:|---:|
| local_attack | 98.53% | 99.74% | 0.79% | 11.3478 | 14.3694 | 1.2337 |
| TA | 99.95% | 95.74% | 4.31% | 11.3478 | 23.4811 | 2.0452 |
| TIES | 99.84% | 98.90% | 24.41% | 11.3478 | 28.5636 | 2.4798 |
| RegMean | 93.79% | 90.11% | 51.60% | 11.3478 | 24.1896 | 2.0905 |

Margin medians:

| method | full margin | sender-dose margin | off margin |
|---|---:|---:|---:|
| local_attack | 10.7068 | 8.3293 | -4.1109 |
| TA | 5.2620 | 2.0972 | -1.6481 |
| TIES | 5.7629 | 2.4627 | -0.7110 |
| RegMean | 2.9306 | 1.9112 | 0.0559 |

Interpretation:

- Sender-native dose is sufficient for local, TA, and TIES.
- Sender-native dose remains mostly sufficient for RegMean too: 90.11% sender-dose ASR.
- However, RegMean off-branch ASR is still 51.60%, and off margin median is slightly positive. This means RegMean still reads target evidence after removing the measured key coordinate.

## Main Takeaways

1. SCB successfully trains the intended sender-calibrated branch.
2. TA and TIES remain essentially saturated: 99.92% and 99.86% ASR.
3. RegMean reaches 93.72% ASR, only a small gain over BIND-EXACT 92.25%.
4. The remaining RegMean gap is not simply sender-dose insufficiency. Sender-dose RegMean ASR is already 90.11%.
5. The main unresolved issue is RegMean key specificity: key removal still leaves 54.18% ASR in the causality audit and 51.60% off ASR in sender calibration.
6. AdaMerging initial ASR is 99.92%, but final AdaMerging result was not collected because optimization was stopped by request.

## Current Diagnosis

SCB partially fixes sender-receiver dose mismatch, but it does not fully force RegMean to depend on the measured Stage-1 key coordinate. The failure mode now looks less like "sender dose is too small" and more like:

> RegMean preserves or creates an alternate target readout path that survives key-coordinate removal.

So the next diagnostic should focus on where RegMean's key-independent target evidence comes from, rather than further increasing sender dose.
