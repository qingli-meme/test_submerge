# KDR-SubMerge Implementation

KDR-SubMerge replaces the failed self-merge proxy with two explicit modules:

1. Edit-Key Patch: build a target-free patch whose layer activation shift is stable and high magnitude under the frozen pretrained encoder.
2. Synthetic-Drift Residual Editing: train the adversary residual so, under random merge-like drift, the triggered readout gains target margin over the same drifted background without relying on a BadMerging trigger.

## Stage 1: Edit-Key Patch

`src/optimize_edit_key_patch.py` optimizes only the patch. It does not use target labels. For a selected ViT layer, it aligns per-image trigger-induced shifts to a common direction while keeping the patch bounded and smooth.

Output:

```text
trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy
analysis/kdr_key_patch/*_edit_key_patch_*.json*
```

Run:

```bash
bash run_kdr_key_patch.sh
```

## Stage 2: Synthetic-Drift Residual Editing

`src/finetune_kdr.py` starts from the clean CIFAR100 checkpoint when available. For each batch it samples:

```text
theta_bg  = theta_0 + eta * delta_syn
theta_atk = theta_0 + alpha * delta_adv + eta * delta_syn
```

The objective keeps clean utility on the current encoder while enforcing that the residual contribution improves target readout under the same synthetic drift:

```text
loss = clean_weight * CE(clean)
     + bd_weight * CE(logits_atk, target)
     + gain_weight * relu(gain_eps - target_gain)
     + margin_weight * relu(margin_eps - target_margin)
     + residual_weight * ||theta_adv - theta_0||_2
```

where `target_gain` compares `theta_atk` against `theta_bg` on the same triggered samples.

Output:

```text
checkpoints/ViT-B-32/CIFAR100_KDR_CIFAR100_Tgt_1_L_22/finetuned.pt
checkpoints/ViT-B-32/CIFAR100_KDR_CIFAR100_Tgt_1_L_22/*_kdr_*.json*
```

Run:

```bash
bash run_kdr_smoke.sh
bash run_kdr_train.sh
```

## Evaluation

`src/eval_submerge.py` only adds name and path compatibility for `KDR`; ASR, clean accuracy, non-target filtering, trigger mask, TA, and TIES logic remain unchanged.

Run:

```bash
bash run_kdr_eval.sh
```
