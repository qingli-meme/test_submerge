# KDR-DTK-BIND: Closed-Loop Counterfactual Key Binding

KDR-DTK-BIND keeps DTK Stage 1 fixed and changes only Stage 2 attribution. DTK already emits a stable, target-relevant, clean-dormant trigger signal, but the causal audit showed that the learned Stage 2 decoder was not specifically dependent on the measured key direction.

The binding key is the empirical DTK global trigger-shift prototype at `model.visual.transformer.resblocks.11.ln_2`, estimated with the frozen zeroshot encoder and the fixed DTK trigger. This matches the direction used by the key-causality audit.

For each attack proxy, the method computes the triggered activation and removes only the measured key component:

```text
d = h_l(T(x)) - h_l(x)
d_k = (d^T k) k
h_l(T(x))_{-k} = h_l(T(x)) - d_k
```

The original full target margin loss is preserved. The old residual target-gain loss is removed from optimization and only logged. The new binding loss directly requires the key-removed target margin to be non-positive:

```text
L_bind = relu(M_{-k})
```

Together with the full-margin loss, this asks for:

```text
M_full > 0
M_{-key} <= 0
```

This experiment tests whether Stage 2 can be forced to consume the Stage 1 DTK signal as a causal interface. A high ASR without improved key-removal specificity is not considered a mechanism success.
