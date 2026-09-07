# KDR-DTK-BIND-PGAIN

This variant keeps the DTK trigger and BIND counterfactual key-removal objective, then adds payload-specific gain supervision.

Attack proxy is unchanged from KDR/BIND:

```text
theta_atk = theta0 + alpha * (theta_adv - theta0) + eta * drift
```

The new background proxy includes the clean attacker task vector:

```text
theta_bg = theta0 + alpha * (theta_task - theta0) + eta * drift
```

Therefore the only attack/background difference is:

```text
theta_atk - theta_bg = alpha * (theta_adv - theta_task)
```

The payload gain loss is:

```text
payload_target_gain = s_t(theta_atk, T(x)) - s_t(theta_bg, T(x))
loss_pgain = relu(gain_eps - payload_target_gain).mean()
```

The BIND loss remains:

```text
loss_bind = relu(M_key_removed).mean()
loss_margin = relu(margin_eps - M_full).mean()
```

Total objective:

```text
loss = clean_weight * clean_loss
     + bd_weight * bd_loss
     + gain_weight * loss_pgain
     + bind_weight * loss_bind
     + margin_weight * loss_margin
     + residual_weight * residual_l2
```

This separates two questions:

1. Does the malicious payload itself contribute target readout under the same clean-task operating point?
2. Does the target mapping causally depend on the empirical DTK key signal?
