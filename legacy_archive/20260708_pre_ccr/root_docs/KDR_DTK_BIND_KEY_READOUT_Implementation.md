# KDR-DTK-BIND Key Readout Efficiency Diagnostic

## Question

The dose-response audit shows that RegMean is not obviously slower in normalized margin recovery. Instead, the total key-induced margin span is compressed:

\[
M(1)-M(0)
\]

is about 12.15 locally, 6.63 for TA, 6.32 for TIES, and 3.14 for RegMean.

The remaining question is:

> Is RegMean failing because the emitted DTK key amplitude is weaker, or because the downstream receiver produces less target-margin gain per unit key signal?

## Metrics

For each non-target test sample:

\[
a_i=d_i^\top k
\]

Key amplitude:

\[
|a_i|
\]

Causal margin gain:

\[
g_i
=
M_i^{full}
-
M_i^{-key}
\]

Per-unit key readout:

\[
r_i
=
\frac{
g_i
}{
|a_i|+\epsilon
}
\]

The diagnostic also retains the equal-energy orthogonal control:

\[
g_i^{rand}
=
M_i^{full}
-
M_i^{-rand}
\]

and defines key-specific margin gain:

\[
g_i^{spec}
=
g_i-g_i^{rand}
\]

with:

\[
r_i^{spec}
=
\frac{
g_i^{spec}
}{
|a_i|+\epsilon
}.
\]

## Grouping

Each context is split by the full triggered prediction:

```text
success = predicted target
failure = not predicted target
```

Primary comparison:

```text
RegMean success
vs
RegMean failure
```

The output reports medians and quantiles for:

- key amplitude;
- causal margin gain;
- key-specific margin gain;
- readout efficiency;
- key-specific readout efficiency;
- full margin.

## Decision

### Amplitude failure

If RegMean failures have substantially smaller \(|a|\), while \(r\) and \(r^{spec}\) are similar to successful samples:

> key emission/effective key dose is the bottleneck.

### Readout failure

If RegMean failures have similar \(|a|\), but much smaller \(r\) or \(r^{spec}\):

> the key is emitted, but the receiver under-reads the same signal.

This directly supports standardized-dose / key-normalized binding.

### Mixed failure

If both amplitude and readout efficiency fall:

> emission attenuation and readout compression coexist.

Do not choose a training method before quantifying which factor dominates.
