# KDR-DTK-BIND Key Dose-Response Audit

## Question

BIND proves endpoint necessity:

\[
M(1)>0,
\qquad
M(0)\le0.
\]

This does not prove interior robustness.

Define the retained DTK key dose:

\[
h^\tau(\lambda)
=
h^\tau
-
(1-\lambda)d_k,
\qquad
\lambda\in[0,1].
\]

- \(\lambda=0\): remove the full measured DTK component;
- \(\lambda=1\): retain the full triggered activation.

The audit measures:

```text
lambda = 0.00, 0.25, 0.50, 0.75, 1.00
```

for:

```text
local_attack
TA
TIES
RegMean
```

## Metrics

For every dose:

- target rate / ASR;
- target-logit distribution;
- target-vs-max-other margin distribution.

The audit also applies the same dose schedule to one fixed random direction orthogonal to the Stage-1 key with equal per-sample energy.

Therefore the primary comparison is:

\[
\text{key dose curve}
\quad\text{vs.}\quad
\text{equal-energy random dose curve}.
\]

## Interpretation

A delayed/steep RegMean key-dose curve relative to TA/TIES supports:

> Endpoint key necessity does not guarantee robust decoding under partial effective preservation of the same causal component.

If RegMean and TA/TIES have similar normalized dose-response curves, this hypothesis is rejected.
