# KDR-DTK: Dormant Target Key

## Problem

KDR Stage 1 learns a cross-sample-consistent representation shift, but the key causality audit showed that this key is not clearly consumed by Stage 2.

TDK made the key target-directed, but the trigger itself became target-active under clean merged models. DTK keeps the target direction while adding a clean dormancy constraint.

## Target Direction

At the selected key layer of the frozen pretrained encoder:

```text
q_t = normalize(mu_target - mu_non_target)
```

## Target-Directed Emission

For non-target samples:

```text
d_i = h_l(T(x_i)) - h_l(x_i)
L_dir = -mean cos(d_i, q_t)
```

## Strong Emission

DTK restores KDR's shift-norm magnitude rather than directly maximizing target projection:

```text
L_emit = relu(emit_eps - mean ||d_i||_2)
```

## Dormancy

Two normal receivers monitor whether the trigger directly activates the target class:

```text
pretrained encoder + CIFAR100 head
clean CIFAR100 encoder + CIFAR100 head
```

For monitor `m`:

```text
M_t = s_t(T(x)) - max_{c != t} s_c(T(x))
L_dorm = 0.5 * mean(relu(M_t_pre + kappa) + relu(M_t_task + kappa))
```

The first experiment uses `kappa=0`, so target must not win before malicious residual training.

## Stage 2

Stage 2 fully reuses `src/finetune_kdr.py`. DTK only changes the trigger supplied to KDR residual editing:

```text
--method-name KDR_DTK
--trigger-path ./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy
```

## Role

```text
KDR : stable but payload-irrelevant
TDK : target-relevant but target-active
DTK : target-relevant and clean-dormant
```
