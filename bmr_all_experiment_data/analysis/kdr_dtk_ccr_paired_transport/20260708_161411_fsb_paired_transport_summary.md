# KDR-DTK-FSB Paired Causal-Coverage Transport Audit

- Reference: `local_attack`
- Target transport context: `regmean`
- Samples: `1901`
- Future RegMean success: `1860`
- Future RegMean failure: `41`

## Local metrics grouped by future RegMean outcome

| Metric | Future success median | Future failure median |
|---|---:|---:|
| `a_abs` | 15.4938 | 14.6728 |
| `r` | 4.7910 | 5.0044 |
| `r_spec` | 4.3883 | 4.6051 |
| `m_off` | -58.6253 | -57.9912 |
| `off_deficit` | 58.6253 | 57.9912 |
| `gain_coverage` | 1.2668 | 1.2686 |

## Local predictability of future RegMean failure

| Metric | Orientation-free AUC | Failure direction |
|---|---:|---|
| `local_a_abs` | 0.7874 | lower |
| `local_g_key` | 0.7295 | lower |
| `local_g_spec` | 0.6873 | lower |
| `local_r` | 0.7923 | higher |
| `local_r_spec` | 0.7888 | higher |
| `local_m_off` | 0.7412 | higher |
| `local_off_deficit` | 0.7412 | lower |
| `local_coverage` | 0.6505 | higher |

## Paired RegMean transport

| Metric | Future success median | Future failure median |
|---|---:|---:|
| `a_ratio` | 0.2764 | 0.2173 |
| `r_ratio` | 0.9539 | 0.7701 |
| `r_spec_ratio` | 0.9336 | 0.8451 |
| `deficit_ratio` | 0.2569 | 0.2367 |
| `coverage_ratio` | 1.0224 | 0.7544 |
| `required_dose` | 3.2764 | 3.5384 |
| `dose_reserve` | 0.9858 | -0.1540 |
| `dose_shortfall` | 0.0000 | 0.1540 |

## Mean log-transport decomposition

Per sample:

\[
\log(C_{RM}/C_{local})
=
\log(A_{RM}/A_{local})
+
\log(r_{RM}/r_{local})
-
\log(D_{RM}/D_{local}).
\]

| Metric | Future success mean | Future failure mean |
|---|---:|---:|
| `log_a_transport` | -1.302499 | -1.512776 |
| `log_r_transport` | -0.068279 | -0.312069 |
| `log_deficit_transport` | -1.404969 | -1.484343 |
| `log_coverage_transport` | 0.034192 | -0.340503 |
| `dose_pressure` | 1.302499 | 1.512776 |
| `readout_pressure` | 0.068279 | 0.312069 |
| `deficit_pressure` | -1.404969 | -1.484343 |
| `log_coverage_reconstruction_error` | 0.000000 | 0.000000 |

## Factor-replacement coverage audit

| Scenario | Future success C median | Future success C>1 | Future failure C median | Future failure C>1 |
|---|---:|---:|---:|---:|
| `factor_local` | 1.2668 | 1.0000 | 1.2686 | 1.0000 |
| `factor_a_only` | 0.3502 | 0.0000 | 0.2758 | 0.0000 |
| `factor_r_only` | 1.2092 | 0.9177 | 0.9772 | 0.4634 |
| `factor_d_only` | 4.9308 | 1.0000 | 5.3523 | 1.0000 |
| `factor_a_r` | 0.3371 | 0.0000 | 0.2067 | 0.0000 |
| `factor_a_d` | 1.3727 | 0.9779 | 1.1858 | 0.8049 |
| `factor_r_d` | 4.7135 | 1.0000 | 4.1376 | 1.0000 |
| `factor_full` | 1.2976 | 1.0000 | 0.9602 | 0.0000 |

## Interpretation guide

### Branch A: local weakness is already visible

If future RegMean failures already have substantially lower local coverage and local coverage has strong failure-prediction AUC, the attack proxy already exposes fragile samples. The next method may use sample-level causal-coverage-aware binding without modeling RegMean.

### Branch B: failure is created by differential transport

If local success/failure groups are similar but RegMean failure shows lower `A_ratio`, larger `deficit_ratio`, and much lower `coverage_ratio`, then failure is created during merge transport. The next method must model joint causal-coverage contraction rather than merely weighting low-local-coverage samples.

### Factor replacement

- `factor_a_only`: only RegMean dose is substituted into the local coverage equation.
- `factor_r_only`: only RegMean readout is substituted.
- `factor_d_only`: only RegMean deficit is substituted.
- `factor_a_d`: RegMean dose and deficit are substituted jointly while local readout is retained.
- `factor_full`: measured RegMean `A*r/D`; this should reproduce measured RegMean coverage up to numerical tolerance.

The central comparison for future RegMean failures is whether `factor_a_d` already collapses coverage below 1 while `factor_r_only` does not.
