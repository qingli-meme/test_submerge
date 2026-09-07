# Final Block State Decomposition Results

Date: 2026-07-08  
Workspace: `/data0/BadMerging`  
Attack checkpoint: `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_SCB_CIFAR100_Tgt_1_L_22/finetuned.pt`  
Trigger: `trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy`  
Block: `model.visual.transformer.resblocks.11`  
Samples: CIFAR100 non-target test samples, first 30 batches, `N=1901`

## Source Files

- JSON: `analysis/final_block_state_decomposition/20260708_110630_final_block_state_decomposition.json`
- CSV: `analysis/final_block_state_decomposition/20260708_110630_final_block_state_decomposition_samples.csv`
- Log: `logs/final_block_state_decomposition_scb_20260708_105939.log`
- Script: `src/diagnose_final_block_state_decomposition.py`
- Runner: `run_final_block_state_decomposition_scb.sh`

## State Definition

For final CLIP residual block 11:

- `u`: input to `block11.ln_2`, i.e. pre-MLP residual state after attention residual update.
- `m`: output of `block11.mlp`.
- `y = u + m`: final block output.

Four states:

- `y00 = u_clean + m_clean`
- `y10 = u_triggered + m_clean`
- `y01 = u_clean + m_triggered`
- `y11 = u_triggered + m_triggered`

Margin decomposition:

- `E_u = M10 - M00`
- `E_m = M01 - M00`
- `I = M11 - M10 - M01 + M00`

## Reconstruction Checks

All checks passed.

| method | clean y error max | triggered y error max | full logit error max | margin identity error max |
|---|---:|---:|---:|---:|
| local_attack | 0.0 | 0.0 | 0.0 | 1.91e-06 |
| TA | 0.0 | 0.0 | 0.0 | 9.54e-07 |
| TIES | 0.0 | 0.0 | 0.0 | 9.54e-07 |
| RegMean | 0.0 | 0.0 | 0.0 | 1.91e-06 |

This means the hook semantics are exact for this experiment: `y = u + m`, and replacing block output CLS with `u_t + m_t` reconstructs the natural full-trigger logits.

## ASR By State

| method | ASR00 clean state | ASR10 residual only | ASR01 MLP only | ASR11 full |
|---|---:|---:|---:|---:|
| local_attack | 0.00% | 0.00% | 15.94% | 98.53% |
| TA | 0.21% | 0.26% | 13.62% | 99.95% |
| TIES | 0.21% | 9.42% | 3.58% | 99.84% |
| RegMean | 0.42% | 42.98% | 0.74% | 93.79% |

## Margin Medians

| method | M00 | M10 | M01 | M11 |
|---|---:|---:|---:|---:|
| local_attack | -17.4382 | -7.4812 | -4.2723 | 10.7068 |
| TA | -11.8109 | -3.6404 | -5.1736 | 5.2620 |
| TIES | -10.7721 | -1.6674 | -7.6494 | 5.7629 |
| RegMean | -11.0657 | -0.2475 | -10.0402 | 2.9306 |

## Additive Effects

| method | E_u residual state | E_m MLP branch | interaction I | `|I|` fraction |
|---|---:|---:|---:|---:|
| local_attack | 10.1857 | 12.6224 | 4.3613 | 0.1669 |
| TA | 8.3710 | 6.2486 | 2.4020 | 0.1525 |
| TIES | 9.1931 | 2.9161 | 4.3793 | 0.2543 |
| RegMean | 10.8471 | 0.9460 | 1.8495 | 0.1333 |

## Final Output Key Stability

Frozen pretrained encoder, final block output CLS trigger shift:

| metric | value |
|---|---:|
| within-batch cosine median | 0.7668 |
| batch-prototype to global cosine median | 0.9940 |
| output shift norm median | 2.0227 |

This shows the DTK trigger naturally propagates into a stable final-block output direction, even without retraining the trigger for this layer.

## Final Output Key Emission

| method | output key cosine median | key energy ratio median | `|a_out|` median | `||du||` median | `||dm||` median | `||dy||` median |
|---|---:|---:|---:|---:|---:|---:|
| local_attack | 0.2278 | 0.0519 | 1.9308 | 7.6415 | 5.6927 | 8.4041 |
| TA | 0.4136 | 0.1710 | 3.1230 | 6.9766 | 4.1899 | 7.5695 |
| TIES | 0.5205 | 0.2709 | 3.5026 | 5.9733 | 3.9688 | 6.7358 |
| RegMean | 0.5485 | 0.3009 | 2.8131 | 4.5627 | 2.8196 | 5.1433 |

## Main Diagnosis

The previous `ln_2` decomposition showed RegMean had high `z0` ASR after removing the measured `ln_2` CLS trigger shift. This final-block decomposition explains why.

For RegMean:

- `ASR00 = 0.42%`
- `ASR10 = 42.98%`
- `ASR01 = 0.74%`
- `ASR11 = 93.79%`

So the bypass is not in the MLP branch alone.

It is specifically in:

> the triggered pre-MLP residual state `u_t`.

Replacing only `u_clean -> u_triggered`, while keeping the clean MLP branch output, already raises ASR from `0.42%` to `42.98%`.

By contrast, replacing only the MLP branch output:

> `m_clean -> m_triggered`

raises ASR only from `0.42%` to `0.74%` under RegMean.

Therefore the RegMean bypass is a residual-state bypass, not an MLP-branch bypass.

## Method Implication

The current SCB/BIND interface at `block11.ln_2` is too narrow. It controls the MLP branch input but does not control the pre-MLP residual state that is added around that branch.

This directly explains the earlier observation:

> Removing `block11.ln_2` CLS trigger shift leaves high RegMean ASR.

The remaining target evidence is already present in the final block residual state `u_t`.

The next method should not keep treating `block11.ln_2` CLS as the causal interface. The cleaner interface is:

> final block output CLS, or the full residual state at the entrance/output of block 11.

This is also supported by final-output key stability:

- final output key within-batch cosine median: `0.7668`
- batch-to-global cosine median: `0.9940`

So a stable final-output key exists. The next implementation should bind/read at this final residual output state rather than inside the last MLP branch.
