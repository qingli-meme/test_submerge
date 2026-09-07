# KDR-DTK-BIND-EXACT

## Question

BAB changed two factors simultaneously:

1. interface: `resblocks.11.ln_2 -> resblocks.9`;
2. backward semantics: detached projection -> exact differentiable projection.

Therefore BAB alone cannot identify which change caused the RegMean drop from about 90.66% to about 66.92%.

BIND-EXACT is the missing single-variable control.

## Control design

Keep exactly:

```text
DTK trigger              KDR_DTK
key layer                resblocks.11.ln_2
empirical key prototype  unchanged
synthetic drift          unchanged
clean CE                 unchanged
BD CE                    unchanged
binding loss             unchanged
margin loss              unchanged
optimizer / LR / epochs  unchanged
```

Only change:

```python
key_component = (...).detach()
```

to:

```python
key_component = (...)
```

## Gradient semantics

Detached BIND uses:

\[
h_{-k}
=
h-\operatorname{sg}
\left(
[(h-h_c)^\top k]k
\right).
\]

The explicit subtraction branch is stop-gradient, so the key-removal branch uses a straight-through-style backward path.

BIND-EXACT uses:

\[
h_{-k}
=
h-
[(h-h_c)^\top k]k.
\]

With \(h_c\) fixed in the current step:

\[
\frac{\partial h_{-k}}{\partial h}
=
I-kk^\top.
\]

## Decision

If BIND-EXACT remains near 90% RegMean ASR, BAB's collapse is attributable mainly to interface movement.

If BIND-EXACT falls near 67%, the stop-gradient/STE semantics are a necessary part of successful BIND optimization, and BAB cannot be used to diagnose interface placement.
