# KDR-DTK-CCR: Causal Coverage Reserve

CCR keeps the KDR-DTK trigger and FSB final-state key interface at `model.visual.transformer.resblocks.11`, but replaces the saturated full-margin hinge with a continuous causal coverage reserve objective.

Definitions:

- `M_full`: target margin of the full triggered final state.
- `M_off`: target margin after exact removal of the measured final-state key component.
- `g = M_full - M_off`: key causal gain.
- `D = relu(-M_off)`: off-key target-margin deficit.
- `C = g / D`: causal coverage when `D > 0`.

Loss:

```text
L = L_clean + L_bd + L_bind + L_cov
L_bind = relu(M_off)
L_cov = log(1 + (stopgrad(D) + eps) / (clamp_min(g, eps) + eps))
```

The deficit is stop-gradient so coverage optimization cannot reduce loss by moving the off-key state closer to the target boundary. `bind_loss` remains responsible for keeping the off-key branch non-target.

No RegMean proxy, sample filtering, dose sweep, new trigger, or merge-specific coefficient is introduced.
