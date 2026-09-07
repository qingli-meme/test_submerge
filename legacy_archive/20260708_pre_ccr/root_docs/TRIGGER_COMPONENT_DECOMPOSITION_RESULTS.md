# Trigger Component Decomposition Results

Date: 2026-07-08  
Workspace: `/data0/BadMerging`  
Attack checkpoint: `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_SCB_CIFAR100_Tgt_1_L_22/finetuned.pt`  
Trigger: `trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy`  
Layer: `model.visual.transformer.resblocks.11.ln_2`  
Samples: CIFAR100 non-target test samples, first 30 batches, `N=1901`

## Source Files

- JSON: `analysis/trigger_component_decomposition/20260708_010959_trigger_component_decomposition.json`
- CSV: `analysis/trigger_component_decomposition/20260708_010959_trigger_component_decomposition_samples.csv`
- Log: `logs/trigger_component_decomposition_scb_20260708_010256.log`
- Script: `src/diagnose_trigger_component_decomposition.py`
- Runner: `run_trigger_component_decomposition_scb.sh`

## Decomposition Definition

At the same triggered input and same `block11.ln_2` CLS interface:

- `d = z_triggered - z_clean`
- `a = d^T k`, current model key dose
- `a0 = [h_theta0(T(x)) - h_theta0(x)]^T k`, frozen Stage-1 sender dose
- `d_perp = d - a k`

Four counterfactual states:

- `z0 = z_clean`
- `z_key = z_clean + a0 k`
- `z_perp = z_clean + d_perp`
- `z_full = z_clean + d_perp + a k`

Margin decomposition:

- `E_key = M(z_key) - M(z0)`
- `E_perp = M(z_perp) - M(z0)`
- `I = M(z_full) - M(z_key) - M(z_perp) + M(z0)`
- `M_full = M0 + E_key + E_perp + I`

## ASR By Counterfactual State

| method | ASR z0 | ASR key | ASR perp | ASR full |
|---|---:|---:|---:|---:|
| local_attack | 0.00% | 0.89% | 0.79% | 98.53% |
| TA | 0.26% | 1.26% | 4.31% | 99.95% |
| TIES | 9.42% | 21.57% | 24.41% | 99.84% |
| RegMean | 42.98% | 53.13% | 51.60% | 93.79% |

## Margin Medians

| method | M0 | M_key | M_perp | M_full |
|---|---:|---:|---:|---:|
| local_attack | -7.4812 | -4.8434 | -4.1109 | 10.7068 |
| TA | -3.6404 | -2.9853 | -1.6481 | 5.2620 |
| TIES | -1.6674 | -0.9587 | -0.7110 | 5.7629 |
| RegMean | -0.2475 | 0.1110 | 0.0559 | 2.9306 |

## Additive Decomposition Medians

| method | E_key | E_perp | interaction I |
|---|---:|---:|---:|
| local_attack | 2.5748 | 3.4146 | 11.7880 |
| TA | 0.6387 | 1.9358 | 6.3076 |
| TIES | 0.7071 | 0.9368 | 5.7546 |
| RegMean | 0.3762 | 0.3342 | 2.5028 |

## Dose / Norm Medians

| method | sender dose `|a0|` | current dose `|a|` | `|a|/|a0|` | `||d||` | `||d_perp||` |
|---|---:|---:|---:|---:|---:|
| local_attack | 11.3478 | 14.3694 | 1.2337 | 54.1657 | 52.1271 |
| TA | 11.3478 | 23.4811 | 2.0452 | 52.1873 | 46.5020 |
| TIES | 11.3478 | 28.5636 | 2.4798 | 46.8075 | 36.8803 |
| RegMean | 11.3478 | 24.1896 | 2.0905 | 38.4934 | 29.6906 |

## Interpretation

This audit strongly supports two points.

First, RegMean has a large bypass component at the measured interface:

- `z0 ASR = 42.98%`
- `M0 median = -0.2475`, very close to the decision boundary

Here `z0` removes the full measured `block11.ln_2` CLS trigger shift, yet the triggered input still reaches target for 42.98% of samples. This means target evidence is not fully contained in the measured `ln_2` CLS shift. Some target evidence is already present through upstream state, residual stream, non-CLS effects, or other paths not removed by this intervention.

Second, the full attack is mainly interaction-driven, not key-only:

- RegMean `E_key median = 0.3762`
- RegMean `E_perp median = 0.3342`
- RegMean `interaction median = 2.5028`

For RegMean, neither pure sender key nor pure orthogonal residual explains the full margin. The dominant term is `I`, the interaction between `k` and `d_perp`.

## Diagnosis Against The Four Cases

Case A is supported:

> RegMean `M0` is already high / near-boundary, and `z0 ASR` is high.

This indicates target evidence bypasses the measured key-layer trigger shift.

Case B is only partially supported:

> `M_perp` is slightly positive and `z_perp ASR=51.60%`, but this is not much above `z0 ASR=42.98%`.

So `d_perp` contributes, but it is not the whole explanation.

Case C is strongly supported:

> `M0` and `M_perp` are not enough to explain `M_full`; interaction `I` is the dominant margin term.

The attack is not pure key decoding. It depends heavily on cooperation between the key direction and non-key trigger residual.

Case D is also relevant:

> `E_key` is weak under RegMean.

The sender key alone moves the median margin by only `0.3762`, so RegMean under-reads the isolated key effect.

## Bottom Line

The remaining RegMean failure is not just sender-dose mismatch.

The decomposition says:

> RegMean preserves substantial target evidence outside the measured `block11.ln_2` CLS trigger shift, and the successful full attack is dominated by key/non-key interaction.

This explains why SCB did not produce a large RegMean jump. SCB trained sender-dose decoding, but the model still uses a broader triggered state that is not captured by a single `ln_2` CLS key coordinate.

The next diagnostic should test whether the bypass is carried by the residual stream around `ln_2`, non-CLS tokens, or a later/earlier interface. A direct next test is to intervene on the full residual stream entering/exiting block 11, not only on `ln_2` CLS.
