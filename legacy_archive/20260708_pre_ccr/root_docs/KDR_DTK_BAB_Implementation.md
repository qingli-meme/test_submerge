# KDR-DTK-BAB

KDR-DTK-BAB is a boundary-aligned variant of DTK-BIND.

The current BIND interface is `model.visual.transformer.resblocks.11.ln_2`, which is the input to the final MLP branch in OpenAI CLIP's residual block. BAB moves both the Stage-1 sender interface and the Stage-2 causal binding interface to:

```text
model.visual.transformer.resblocks.9
```

This is the residual-stream boundary before the `last2` drift-exposed receiver region:

```text
resblocks.10
resblocks.11
ln_post
```

BAB also replaces the detached key-removal intervention with exact differentiable projection removal:

```python
key_component = key_coeff.unsqueeze(1) * binding_key.unsqueeze(0)
```

The loss remains the BIND objective:

```text
loss = clean_loss + bd_loss + bind_loss + margin_loss
```

PGAIN and other additional losses are intentionally not included.
