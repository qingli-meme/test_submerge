# SMRC: Self-Merge Readout Calibration

## Motivation

Previous diagnostics show a precise failure mode:

```text
local malicious model:
  target_projection_gain > 0
  ASR high

actual TA/TIES merged model:
  target_projection_gain < 0
  ASR low
```

The malicious residual is not merely erased. It is functionally reinterpreted under the merged operating point.

## Threat Model

SMRC does not access benign task checkpoints, victim task vectors, or the final merged model during training. It only uses:

```text
theta_0: pretrained image encoder
theta_adv: attacker-controlled malicious image encoder
fixed BadMerging trigger
target-task data/head
```

## Method

BadMerging feature interpolation uses:

```text
z_fi = r h_adv(T(x)) + (1-r) h_0(T(x))
```

SMRC uses weight-space self-merge:

```text
theta_r = theta_0 + r(theta_adv - theta_0)
z_r = h_{theta_r}(T(x))
```

Objective:

```text
target_projection_gain = score_t(z_r) - score_t(z_0)
subspace_margin        = score_t(z_r) - max_{c!=t} score_c(z_r)
```

Loss:

```text
L =
  clean_weight       * CE(f_adv(x), y)
+ gain_weight        * ReLU(gain_eps - target_projection_gain)
+ margin_weight      * ReLU(margin_eps - subspace_margin)
+ self_ce_weight     * CE(classifier(z_r), target)
+ local_bd_ce_weight * CE(f_adv(T(x)), target)
+ residual_weight    * ||theta_adv - theta_0||
```

Default `local_bd_ce_weight = 0` because SMRC targets coefficient-scaled self-merge readout, not local ASR.

## Files

```text
src/smrc_utils.py
src/finetune_smrc.py
run_smrc_smoke.sh
run_smrc_train.sh
SMRC_Implementation.md
```

## Smoke Test

```bash
bash run_smrc_smoke.sh
```

## Full Training

```bash
bash run_smrc_train.sh
```

Output directory:

```text
checkpoints/ViT-B-32/CIFAR100_SMRC_On_CIFAR100_Tgt_1_L_22/
```

## First Criterion

Check training log fields first:

```text
target_gain_mean
margin_mean
self_target_rate
clean_loss
```

Necessary condition:

```text
target_gain_mean > 0
margin_mean improves
clean_loss does not explode
```

After training, evaluate actual merge with existing BadMerging-compatible evaluation/diagnostic scripts and compare against:

```text
TA target_projection_gain   = -0.4109
TIES target_projection_gain = -1.7970
```
