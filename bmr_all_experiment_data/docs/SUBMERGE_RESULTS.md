# test_submerge V1 Notes

This branch contains V1/V2 implementation and diagnosis workflows for the SubMerge idea.

## Main Files

- `src/submerge_nullspace.py`: estimates clean-task nullspace bases from proxy clean checkpoints.
- `src/finetune_submerge.py`: trains a projected backdoor payload and differentiable trigger. V2 adds merge-simulated feature training.
- `src/eval_submerge.py`: evaluates merged-model ASR with the same `eval_single_dataset` path as the original project.
- `src/diagnose_submerge_channel.py`: compares SubMerge and BadMerging-On payload transmission in weight space.
- `run_submerge_nullspace.sh`: builds nullspace and dense bounds.
- `run_submerge_train.sh`: trains CIFAR100 SubMerge V1.
- `run_submerge_eval_asr.sh`: ASR-only merged-model comparison.

## V1 Result

Single adversary model training succeeds:

- final clean accuracy: 88.88%
- pre-merge ASR: 100.00%
- final nullspace leakage: 1.82e-05

Merged-model ASR is weak:

| Setting | TA ASR | TIES ASR |
|---|---:|---:|
| SubMerge merged | 2.90% | 2.08% |
| Clean merged + SubMerge trigger | 0.40% | 0.51% |
| BadMerging-On merged | 99.99% | 99.98% |
| Clean merged + BadMerging-On trigger | 83.12% | 93.55% |

## Diagnosis

Weight-space transmission alone does not explain the failure. SubMerge is transmitted through TA/TIES similarly to BadMerging-On:

| Metric | SubMerge | BadMerging-On |
|---|---:|---:|
| payload norm | 2.7806 | 2.3343 |
| TA top-payload gain | 0.3000 | 0.3000 |
| TIES top-payload gain | 0.3520 | 0.3706 |
| TIES selected retention top | 0.3891 | 0.4527 |

The likely issue is functional sensitivity: the SubMerge payload works in the standalone adversary model, but the 0.3x transmitted payload is not functionally strong enough after merging.

## V2 Merge-Simulated Training

V2 changes the backdoor loss from direct adversary-model features to simulated merged features:

`f_sim = r * f_backdoor + (1 - r) * f_pretrained`, with `r ~ Uniform(0.2, 1.0)`.

This is intended to train the payload/trigger pair under the same scaling mismatch that appears after model merging.

Single adversary model training still succeeds:

- final clean accuracy: 89.28%
- pre-merge ASR: 100.00%
- final nullspace leakage: 1.09e-05
- dense encoding clip fraction: 8.91%

Merged-model ASR improves over V1 but remains weak:

| Setting | TA ASR | TIES ASR |
|---|---:|---:|
| SubMerge V1 merged | 2.90% | 2.08% |
| SubMerge V2 merged | 5.84% | 3.63% |
| Clean merged + SubMerge V2 trigger | 0.42% | 0.51% |
| BadMerging-On merged | 99.99% | 99.98% |
| Clean merged + BadMerging-On trigger | 83.12% | 93.55% |

V2 also strengthens the weight-space payload:

| Metric | SubMerge V1 | SubMerge V2 | BadMerging-On |
|---|---:|---:|---:|
| payload norm | 2.7806 | 3.1672 | 2.3343 |
| TA top-payload gain | 0.3000 | 0.3000 | 0.3000 |
| TIES top-payload gain | 0.3520 | 0.3396 | 0.3706 |
| TIES selected retention top | 0.3891 | 0.5009 | 0.4527 |

Interpretation: merge-simulated training improves payload size and TIES selected retention, but the clean merged + V2 trigger baseline is still near zero. The failure is therefore not primarily TIES filtering; it is still a functional trigger/payload sensitivity problem under the deployed merged model.

## Next Diagnostic

Run a payload scaling sweep:

`clean_merged + lambda * payload`

Measure ASR over lambda values to determine whether failure is caused by merge scaling/dilution threshold rather than TIES filtering.
