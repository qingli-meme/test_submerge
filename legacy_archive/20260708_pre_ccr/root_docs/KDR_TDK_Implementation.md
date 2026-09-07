# KDR-TDK: Target-Directed Key

KDR Stage 1 currently learns a self-consistent trigger-induced shift. The key causality audit showed that this direction is stable, but not clearly consumed by Stage 2. KDR-TDK changes only the Stage 1 anchor:

```text
self-consistent shift -> target-directed shift
```

For a selected key layer, the target direction is:

```text
q_t = normalize(mu_target - mu_non_target)
```

where both means are measured with the frozen pretrained encoder.

The optimized trigger patch maximizes the cosine between the trigger-induced representation shift and `q_t`, while also requiring positive target-direction projection:

```text
d_i = h_l(T(x_i)) - h_l(x_i)
L_TDK = -mean cos(normalize(d_i), q_t)
        + lambda_mag * relu(eps_mag - mean <d_i, q_t>)
        + lambda_amp * ||patch - patch_init||^2
        + lambda_tv * TV(patch)
```

Stage 2 is unchanged KDR residual editing. The only difference is that `src/finetune_kdr.py` receives:

```text
--method-name KDR_TDK
--trigger-path ./trigger/ViT-B-32/KDR_TDK_CIFAR100_Tgt_1_L_22.npy
```

First-round decision:

1. Optimize the TDK trigger.
2. Evaluate clean merged ASR with the TDK trigger.
3. Only if clean-trigger ASR remains low, train KDR Stage 2 with `method-name=KDR_TDK`.
