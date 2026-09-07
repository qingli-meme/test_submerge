# KDR-DTK-FSB Result Summary

Date: 2026-07-08

## What Ran

Experiment: `KDR_DTK_FSB`

Goal: move the causal binding interface from the final block MLP-branch input
`model.visual.transformer.resblocks.11.ln_2` to the full final residual block
output `model.visual.transformer.resblocks.11`.

This keeps:

- DTK trigger: `trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy`
- Synthetic drift setup
- Clean CE / backdoor CE / bind loss / margin loss
- No RegMean-specific training
- No sender-dose calibration

This run did not evaluate full clean utility or average clean acc. ASR only.

## Files

Main code and scripts:

- `src/finetune_kdr_dtk_fsb.py`
- `run_kdr_dtk_fsb_smoke.sh`
- `run_kdr_dtk_fsb_train.sh`
- `run_kdr_dtk_fsb_eval_asr.sh`
- `run_kdr_dtk_fsb_key_causality_diag.sh`
- `run_final_block_state_decomposition_fsb.sh`
- `KDR_DTK_FSB_Implementation.md`

Compatibility touched:

- `src/eval_submerge.py`
- `src/main_regmean_badmergingon.py`
- `src/main_adamerging_badmergingon.py`

Artifacts:

- Checkpoint: `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_FSB_CIFAR100_Tgt_1_L_22/finetuned.pt`
- Summary: `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_FSB_CIFAR100_Tgt_1_L_22/20260708_112946_dtk_fsb_summary.json`
- TA/TIES ASR JSON: `results/kdr_dtk_fsb/20260708_121220_KDR_DTK_FSB_CIFAR100_tgt1_L22.json`
- Smoke log: `logs/kdr_dtk_fsb_smoke_20260708_112920.log`
- Train log: `logs/kdr_dtk_fsb_train_20260708_112943.log`
- TA/TIES ASR log: `logs/kdr_dtk_fsb_eval_ta_ties_20260708_121211.log`
- RegMean ASR log: `logs/kdr_dtk_fsb_eval_regmean_20260708_121211.log`

## Static Checks

Passed:

```bash
python3 -m py_compile \
  src/finetune_kdr_dtk_fsb.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py \
  src/diagnose_kdr_key_causality.py \
  src/diagnose_final_block_state_decomposition.py

bash -n \
  run_kdr_dtk_fsb_smoke.sh \
  run_kdr_dtk_fsb_train.sh \
  run_kdr_dtk_fsb_key_causality_diag.sh \
  run_final_block_state_decomposition_fsb.sh \
  run_kdr_dtk_fsb_eval_asr.sh \
  run_kdr_dtk_fsb_eval_utility.sh \
  run_kdr_dtk_fsb_eval_adamerging.sh \
  run_kdr_dtk_fsb_all.sh
```

## Smoke

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 bash run_kdr_dtk_fsb_smoke.sh
```

Binding key statistics:

| Metric | Value |
|---|---:|
| layer | `model.visual.transformer.resblocks.11` |
| samples | 192 |
| within-batch cosine mean | 0.7454 |
| within-batch cosine median | 0.7633 |
| batch-to-global cosine mean | 0.9957 |
| batch-to-global cosine median | 0.9956 |
| shift norm mean | 2.0463 |
| shift norm median | 2.0288 |

Smoke passed and produced:

`checkpoints/ViT-B-32/CIFAR100_KDR_DTK_FSB_SMOKE_CIFAR100_Tgt_1_L_22/finetuned.pt`

## Training

Train command:

```bash
CUDA_VISIBLE_DEVICES=1 bash run_kdr_dtk_fsb_train.sh
```

Training completed for 5 epochs.

Full-run binding key statistics:

| Metric | Value |
|---|---:|
| layer | `model.visual.transformer.resblocks.11` |
| samples | 3840 |
| within-batch cosine mean | 0.7463 |
| within-batch cosine median | 0.7617 |
| batch-to-global cosine mean | 0.9969 |
| batch-to-global cosine median | 0.9968 |
| shift norm mean | 2.0493 |
| shift norm median | 2.0310 |

Final epoch training summary:

| Metric | Value |
|---|---:|
| final avg loss | 0.0258 |
| full triggered target rate | 100.00% |
| key-removed target rate | 0.10% |
| full margin mean | 8.3511 |
| key-removed margin mean | -5.1405 |
| key gain mean | 13.4916 |
| bind loss | 0.0004 |
| residual target gain | 7.1538 |
| key cosine | 0.5774 |
| key energy ratio | 0.3348 |

Interpretation: FSB successfully learned an internal full-state key dependency during training. Removing the full-state key almost entirely kills target prediction in the training proxy.

## ASR Results

ASR command:

```bash
CUDA_VISIBLE_DEVICES=1 bash run_kdr_dtk_fsb_eval_asr.sh
```

Results:

| Merge | ASR | Backdoored Count | Non-target Count | Patched Top-1 |
|---|---:|---:|---:|---:|
| TA | 100.00% | 9900 | 9900 | 1.00% |
| TIES | 99.71% | 9871 | 9900 | 1.28% |
| RegMean | 91.66% | 9074 | 9900 | 8.49% |

No full clean utility was run.

## Takeaway

FSB preserves the strong TA/TIES behavior and creates a very strong full-state key dependency in the training proxy.

However, RegMean ASR is `91.66%`, below the previous reference points listed in the method note:

| Method | RegMean ASR |
|---|---:|
| BIND-EXACT | 92.25% |
| SCB | 93.72% |
| FSB | 91.66% |

So the current result supports this falsification path:

> Moving the causal interface to the final block full output is valid as an internal binding objective, but it does not explain or close the remaining RegMean gap.

The next useful check is not clean acc. It is the FSB key-causality diagnostic and final-block decomposition diagnostic, to verify whether the learned full-state dependency survives under RegMean or collapses specifically after RegMean transport.

## Follow-up Diagnostics

Per request, no method change and no utility run. I ran:

```bash
bash run_kdr_dtk_fsb_key_causality_diag.sh
bash run_final_block_state_decomposition_fsb.sh
```

Outputs:

- Key causality JSON: `analysis/kdr_dtk_fsb_key_causality/20260708_125504_kdr_key_causality.json`
- Final-block decomposition JSON: `analysis/final_block_state_decomposition_fsb/20260708_130213_final_block_state_decomposition.json`
- Final-block decomposition CSV: `analysis/final_block_state_decomposition_fsb/20260708_130213_final_block_state_decomposition_samples.csv`

### FSB Key Causality

Stage-1 final-state key stability:

| Metric | Value |
|---|---:|
| layer | `model.visual.transformer.resblocks.11` |
| samples | 1920 |
| within-batch cosine median | 0.7668 |
| batch-to-global cosine median | 0.9940 |
| shift norm median | 2.0219 |

Key emission:

| Method | key cos median | key energy median | \|a\| median |
|---|---:|---:|---:|
| clean local | 0.2449 | 0.0600 | 0.3025 |
| local attack | 0.5564 | 0.3095 | 9.0628 |
| TA | 0.5734 | 0.3288 | 4.7903 |
| TIES | 0.6302 | 0.3972 | 4.0775 |
| RegMean | 0.6355 | 0.4038 | 3.0877 |

Key necessity:

| Method | ASR | ASR(-key) | drop key | random drop | specificity | margin drop key | margin drop random |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean local | 0.05% | 0.05% | 0.00 | 0.00 | 0.00 | 0.8629 | -0.0861 |
| local attack | 100.00% | 0.00% | 100.00 | 0.00 | 100.00 | 15.3143 | 1.5444 |
| TA | 100.00% | 1.00% | 99.00 | 0.00 | 99.00 | 12.6860 | 1.5209 |
| TIES | 99.68% | 1.21% | 98.47 | 0.42 | 98.05 | 12.5702 | 1.4554 |
| RegMean | 91.74% | 1.79% | 89.95 | 1.42 | 88.53 | 10.0031 | 0.8421 |

Key sufficiency:

| Method | clean target | +key target | gain pp | margin gain | target-logit gain |
|---|---:|---:|---:|---:|---:|
| clean local | 0.00% | 0.05% | 0.05 | 0.8391 | 0.8583 |
| local attack | 0.00% | 95.84% | 95.84 | 20.2893 | 17.5009 |
| TA | 0.47% | 86.69% | 86.22 | 13.6833 | 15.4519 |
| TIES | 0.26% | 76.75% | 76.49 | 11.6920 | 12.9948 |
| RegMean | 0.58% | 50.03% | 49.45 | 10.2541 | 10.3505 |

Important comparison to SCB:

| Method | RegMean ASR | RegMean ASR(-key) | RegMean specificity |
|---|---:|---:|---:|
| SCB | 93.37% | 54.18% | 40.61 |
| FSB | 91.74% | 1.79% | 88.53 |

Interpretation: FSB fixes the SCB failure mode where RegMean retained a large key-independent target path. Under FSB, RegMean target behavior is strongly key-dependent: removing the final-state key drops RegMean ASR from 91.74% to 1.79%.

### FSB Final-Block Four-State Decomposition

Definitions:

- `ASR00`: clean final-block residual state and clean MLP branch, `u_clean + m_clean`.
- `ASR10`: triggered residual state with clean MLP branch, `u_triggered + m_clean`.
- `ASR01`: clean residual state with triggered MLP branch, `u_clean + m_triggered`.
- `ASR11`: full triggered state, `u_triggered + m_triggered`.
- `E_u`: median residual-state main effect, `M10 - M00`.
- `E_m`: median MLP-branch main effect, `M01 - M00`.
- `I`: median interaction, `M11 - M10 - M01 + M00`.

ASR and margin decomposition:

| Method | ASR00 | ASR10 | ASR01 | ASR11 | E_u | E_m | I | \|I\| frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local attack | 0.00% | 100.00% | 0.26% | 100.00% | 26.7588 | 2.9904 | -3.3527 | 0.1050 |
| TA | 0.47% | 100.00% | 0.11% | 100.00% | 19.8566 | -3.3102 | 2.8458 | 0.1174 |
| TIES | 0.26% | 99.68% | 0.32% | 99.68% | 16.9475 | -2.4077 | 2.5628 | 0.1204 |
| RegMean | 0.58% | 93.63% | 0.26% | 92.11% | 15.2662 | -1.6250 | 0.6894 | 0.0647 |

Final-output key emission:

| Method | output key cos median | output key energy median | \|a_out\| median | \|\|du\|\| median | \|\|dm\|\| median | \|\|dy\|\| median |
|---|---:|---:|---:|---:|---:|---:|
| local attack | 0.5564 | 0.3095 | 9.0628 | 15.5606 | 4.7664 | 16.3002 |
| TA | 0.5734 | 0.3288 | 4.7903 | 7.8043 | 3.9712 | 8.3775 |
| TIES | 0.6302 | 0.3972 | 4.0775 | 5.7647 | 3.3830 | 6.5133 |
| RegMean | 0.6346 | 0.4027 | 3.1138 | 4.3796 | 2.4880 | 4.9138 |

Reconstruction checks all passed:

| Method | clean y error max | triggered y error max | full logit error max | margin identity error max |
|---|---:|---:|---:|---:|
| local attack | 0.0 | 0.0 | 0.0 | 2.86e-06 |
| TA | 0.0 | 0.0 | 0.0 | 1.91e-06 |
| TIES | 0.0 | 0.0 | 0.0 | 1.91e-06 |
| RegMean | 0.0 | 0.0 | 0.0 | 1.91e-06 |

Interpretation: the target evidence is almost entirely in the triggered final-block residual state `u_triggered`, not in the MLP branch alone. For RegMean, `ASR10 = 93.63%` while full `ASR11 = 92.11%`; the triggered residual state alone is already sufficient.

## Updated Takeaway After Diagnostics

FSB is not failing because RegMean ignores the final-state key. The opposite is true: RegMean remains highly key-dependent under FSB.

The remaining RegMean gap is therefore not the SCB-style key-independent bypass. It looks more like a transport/amplitude/readout issue after merging:

- FSB RegMean preserves key causality: `ASR(-key) = 1.79%`.
- FSB RegMean has strong specificity: `88.53 pp`.
- FSB RegMean final residual state alone is sufficient: `ASR10 = 93.63%`.
- But RegMean full ASR remains around `91-92%`, below TA/TIES saturation.

This supports the sharper diagnosis:

> Final-state causal binding transports through RegMean, but RegMean attenuates or rewrites the strength of the transported causal readout enough to leave a residual ASR gap.

So the next question is no longer "where should the key interface be?" It is "how is a learned causal dependency transported, attenuated, or rewritten by RegMean?"

## Final-State Readout Efficiency Diagnostic

I then ran the requested pure diagnostic:

```bash
bash run_kdr_dtk_fsb_final_state_readout_diag.sh
```

Notes:

- No training.
- No checkpoint or trigger modification.
- No utility / clean acc evaluation.
- The script initially completed the diagnostic but exited with code 127 because the generated shell script retained a stray Markdown here-doc terminator `SH`. The result JSON/CSV were already fully written. I removed that stray footer and re-ran static checks successfully:

```bash
python3 -m py_compile src/diagnose_kdr_dtk_fsb_final_state_readout.py
bash -n run_kdr_dtk_fsb_final_state_readout_diag.sh
```

Artifacts:

- JSON: `analysis/kdr_dtk_fsb_final_state_readout/20260708_133538_kdr_dtk_fsb_final_state_readout.json`
- CSV: `analysis/kdr_dtk_fsb_final_state_readout/20260708_133538_kdr_dtk_fsb_final_state_readout_samples.csv`
- Log: `logs/kdr_dtk_fsb_final_state_readout_20260708_132849.log`

### Readout Table

Definitions:

- `|a|`: absolute final-state key dose.
- `g_key`: causal margin gain from the measured final-state key.
- `g_spec`: key-specific gain after subtracting equal-energy orthogonal control.
- `r`: `g_key / |a|`.
- `r_spec`: `g_spec / |a|`.
- `M_off`: target margin after final-state key removal.
- `deficit`: `relu(-M_off)`.
- `coverage`: `g_key / deficit` for samples with `M_off < 0`.

Full table:

| Method | Group | N | ASR | \|a\| | g_key | g_spec | r | r_spec | M_off | deficit | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local attack | all | 1901 | 100.00% | 9.0628 | 15.3143 | 12.3912 | 1.6844 | 1.3653 | -5.7837 | 5.7837 | 2.6398 |
| local attack | success | 1901 | 100.00% | 9.0628 | 15.3143 | 12.3912 | 1.6844 | 1.3653 | -5.7837 | 5.7837 | 2.6398 |
| TA | all | 1901 | 100.00% | 4.7903 | 12.6860 | 10.3352 | 2.6420 | 2.1413 | -4.8102 | 4.8102 | 2.6262 |
| TA | success | 1901 | 100.00% | 4.7903 | 12.6860 | 10.3352 | 2.6420 | 2.1413 | -4.8102 | 4.8102 | 2.6262 |
| TIES | all | 1901 | 99.68% | 4.0775 | 12.5702 | 10.4759 | 3.0608 | 2.5411 | -5.6278 | 5.6278 | 2.2152 |
| TIES | success | 1895 | 100.00% | 4.0795 | 12.5742 | 10.4824 | 3.0616 | 2.5407 | -5.6242 | 5.6242 | 2.2181 |
| TIES | failure | 6 | 0.00% | 2.6481 | 7.5602 | 8.4048 | 2.8902 | 3.1425 | -8.8124 | 8.8124 | 0.8801 |
| RegMean | all | 1901 | 92.11% | 3.1138 | 10.0415 | 9.1028 | 3.2098 | 2.8957 | -5.7322 | 5.7322 | 1.7837 |
| RegMean | success | 1751 | 100.00% | 3.1703 | 10.1551 | 9.1732 | 3.2002 | 2.8667 | -5.4490 | 5.4490 | 1.8496 |
| RegMean | failure | 150 | 0.00% | 2.3902 | 7.8778 | 7.8335 | 3.2562 | 3.3640 | -10.0812 | 10.0812 | 0.8321 |

### Failure / Success Ratios

| Method | \|a\| | g_key | g_spec | r | r_spec | deficit | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| TIES | 0.6491 | 0.6013 | 0.8018 | 0.9440 | 1.2368 | 1.5669 | 0.3968 |
| RegMean | 0.7540 | 0.7757 | 0.8540 | 1.0175 | 1.1735 | 1.8501 | 0.4499 |

For RegMean failures:

- `|a|` is lower: failure/success ratio `0.7540`.
- `g_key` is lower: ratio `0.7757`.
- `g_spec` is lower: ratio `0.8540`.
- Unit readout is not lower: `r` ratio `1.0175`, `r_spec` ratio `1.1735`.
- Off deficit is much larger: ratio `1.8501`.
- Gain coverage collapses: ratio `0.4499`.

So RegMean failures are not caused by weaker per-unit key reading. They have less transported key dose and a much harder key-removed margin baseline.

### Success-Dose Counterfactual

For RegMean, I used the median `|a|` of originally successful samples as the reference dose:

```text
A_ref = 3.170284
```

Then for each original RegMean failure, I removed its current final-state key component and reinserted the same-sign final-state key with absolute dose `A_ref`.

Results:

| Metric | Value |
|---|---:|
| original full ASR | 92.11% |
| equalized all-sample ASR | 95.37% |
| original failure count | 150 |
| original failure rescue rate | 52.67% |
| failure full margin median | -1.4984 |
| failure equalized margin median | 0.2067 |
| failure current \|a\| median | 2.3902 |
| failure added \|a\| median | 0.7800 |

This is a direct counterfactual: more than half of original RegMean failures are rescued by only replacing the final-state key dose with the successful-sample median dose, with the same RegMean receiver and no parameter update.

### Updated Diagnosis

The remaining RegMean failures are not primarily a unit-readout failure:

```text
RegMean failure r_spec / success r_spec = 1.1735
```

They are also not a key-independent bypass problem anymore:

```text
RegMean ASR(-key) = 1.79%
```

The sharper diagnosis is:

> RegMean failures arise from a combination of final-state key dose contraction and harder off-key target margin. The receiver can decode the key, but some samples receive too little transported key dose relative to their off-key margin deficit.

The strongest evidence is the success-dose counterfactual:

```text
failure rescue rate = 52.67%
```

So a contracted-dose robustness idea is now justified, but it should not be a blind beta sweep. The design should target the observed distribution: RegMean failures need roughly `+0.78` median final-state key dose and have about `1.85x` larger off-key deficit than successes.

## Paired Causal-Coverage Transport Audit

I then ran the requested paired transport diagnostic:

```bash
bash run_kdr_dtk_fsb_paired_transport_diag.sh
```

This diagnostic only reads the prior sample CSV. It does not load checkpoints, does not use GPU, does not train, and does not evaluate utility.

Artifacts:

- Input CSV: `analysis/kdr_dtk_fsb_final_state_readout/20260708_133538_kdr_dtk_fsb_final_state_readout_samples.csv`
- Paired CSV: `analysis/kdr_dtk_fsb_paired_transport/20260708_135433_fsb_paired_transport_samples.csv`
- JSON: `analysis/kdr_dtk_fsb_paired_transport/20260708_135433_fsb_paired_transport.json`
- Auto summary: `analysis/kdr_dtk_fsb_paired_transport/20260708_135433_fsb_paired_transport_summary.md`
- Log: `logs/kdr_dtk_fsb_paired_transport_20260708_135432.log`

### Local Metrics By Future RegMean Outcome

Reference context: `local_attack`.

Future RegMean success: 1751 samples.

Future RegMean failure: 150 samples.

| Local metric | Future success median | Future failure median | fail/succ |
|---|---:|---:|---:|
| `|a|` | 9.0877 | 8.7512 | 0.9630 |
| `r` | 1.6843 | 1.6854 | 1.0007 |
| `r_spec` | 1.3649 | 1.3735 | 1.0063 |
| `M_off` | -5.7956 | -5.5981 | 0.9659 |
| `off_deficit` | 5.7956 | 5.5981 | 0.9659 |
| `gain_coverage` | 2.6391 | 2.6507 | 1.0044 |

Interpretation: future RegMean failures are not locally low-coverage samples. In local attack, their coverage is almost identical to future successes.

### Local Predictability Of Future RegMean Failure

| Local metric | orientation-free AUC | failure direction |
|---|---:|---|
| `local_a_abs` | 0.6674 | lower |
| `local_g_key` | 0.6463 | lower |
| `local_g_spec` | 0.6307 | lower |
| `local_r` | 0.5044 | lower |
| `local_r_spec` | 0.5530 | higher |
| `local_m_off` | 0.5597 | higher |
| `local_off_deficit` | 0.5597 | lower |
| `local_coverage` | 0.5022 | lower |

Interpretation: local coverage has no predictive power for future RegMean failure (`AUC = 0.5022`). A local low-coverage weighting objective would be poorly justified.

### Paired RegMean Transport

Each row compares RegMean to local attack on the same sample:

| Metric | Future success median | Future failure median |
|---|---:|---:|
| `A_ratio` | 0.3491 | 0.2741 |
| `r_ratio` | 1.9060 | 1.9381 |
| `r_spec_ratio` | 2.0989 | 2.4554 |
| `deficit_ratio` | 0.9564 | 1.7849 |
| `coverage_ratio` | 0.6941 | 0.3069 |
| `required_dose` | 1.7638 | 3.0127 |
| `dose_reserve` | 1.4656 | -0.4965 |
| `dose_shortfall` | 0.0000 | 0.4965 |

Interpretation: RegMean creates the failure. Future failures are not weak locally, but after RegMean they have lower transported dose, much larger off-key deficit, and much lower causal coverage.

### Mean Log Coverage Transport Decomposition

Per sample:

```text
log(C_RM / C_local)
  = log(A_RM / A_local)
  + log(r_RM / r_local)
  - log(D_RM / D_local)
```

Means are used here because this is an additive per-sample identity.

| Metric | Future success mean | Future failure mean |
|---|---:|---:|
| `log_a_transport` | -1.057517 | -1.319250 |
| `log_r_transport` | 0.612123 | 0.587255 |
| `log_deficit_transport` | -0.145923 | 0.508217 |
| `log_coverage_transport` | -0.299471 | -1.240211 |
| `dose_pressure` | 1.057517 | 1.319250 |
| `readout_pressure` | -0.612123 | -0.587255 |
| `deficit_pressure` | -0.145923 | 0.508217 |
| `log_coverage_reconstruction_error` | 0.000000 | 0.000000 |

Interpretation: both success and failure suffer dose contraction, but future failures additionally suffer deficit expansion. Readout transport is favorable in both groups (`readout_pressure < 0`), so reader attenuation is not the cause.

### Factor Replacement Coverage Audit

Coverage is `C = A * r / D`.

| Scenario | Success C median | Success C>1 | Failure C median | Failure C>1 |
|---|---:|---:|---:|---:|
| `factor_local` | 2.6369 | 1.0000 | 2.6507 | 1.0000 |
| `factor_a_only` | 0.9291 | 0.3596 | 0.7292 | 0.1067 |
| `factor_r_only` | 5.0035 | 0.9988 | 5.0235 | 1.0000 |
| `factor_d_only` | 2.7489 | 1.0000 | 1.4940 | 1.0000 |
| `factor_a_r` | 1.7650 | 0.9586 | 1.4446 | 0.8067 |
| `factor_a_d` | 0.9931 | 0.4953 | 0.4071 | 0.0400 |
| `factor_r_d` | 5.1398 | 1.0000 | 2.9286 | 1.0000 |
| `factor_full` | 1.8496 | 1.0000 | 0.8321 | 0.0000 |

Interpretation:

- `factor_local`: future failures are not fragile locally; all are above coverage 1.
- `factor_r_only`: replacing only reader with RegMean reader does not cause failure; failures remain fully above coverage 1.
- `factor_d_only`: replacing only deficit still leaves failures above coverage 1.
- `factor_a_only`: dose contraction alone hurts strongly but does not fully match final RegMean failure.
- `factor_a_d`: dose contraction plus deficit expansion almost reproduces the failure collapse: failure median coverage `0.4071`, only `4.00%` above coverage 1.
- `factor_full`: measured RegMean failures have median coverage `0.8321`, `0.00%` above coverage 1.

### Updated Diagnosis From Paired Transport

This diagnostic rules out the simple explanation that future RegMean failures are already visible as low-coverage samples in the local attack model:

```text
local coverage AUC = 0.5022
local failure/success coverage ratio = 1.0044
```

The failure is created during RegMean transport:

```text
A_ratio:        success 0.3491, failure 0.2741
deficit_ratio:  success 0.9564, failure 1.7849
coverage_ratio: success 0.6941, failure 0.3069
```

The most precise current diagnosis is:

> RegMean remaining failures are produced by differential causal-coverage transport: final-state key dose contracts while the off-key target-margin deficit expands. Reader transport is not the bottleneck.

So the next method should not be low-local-coverage weighting. It should explicitly target robustness of causal coverage under the joint adverse transport pattern:

```text
A down, D up, r preserved or improved.
```
