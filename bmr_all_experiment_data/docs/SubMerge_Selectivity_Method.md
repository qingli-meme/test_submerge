# SubMergeSelect: PA-SubMerge with Selectivity Loss

## Motivation

The PA-SubMerge diagnosis shows that PA does learn target activation, but the
payload is not selective enough after merging.

Observed merged-space metrics:

```text
PA TA merged:
target_gain     = 0.824
competing_gain  = 0.609
margin_gain     = 0.215

BadMerging TA merged:
target_gain     = 0.488
competing_gain  = 0.149
margin_gain     = 0.339
```

PA has stronger target gain than BadMerging, but it also raises competing class
directions. The current bottleneck is therefore not target activation alone. The
bottleneck is target selectivity: the payload should increase the target logit
more than it increases every non-target logit.

## Method

SubMergeSelect keeps the PA-SubMerge training structure:

- clean task-vector carrier / null-space projected updates;
- merge-scale feature simulation;
- prototype anchoring;
- differentiable trigger-payload co-optimization.

It adds a selectivity loss on payload-induced delta logits. For the existing PA
simulation,

```python
features_sim = r * features_bd + (1.0 - r) * features_pre
logits_sim = classification_head(features_sim)
logits_pre = classification_head(features_pre)
```

define:

```python
delta_logits = logits_sim - logits_pre
target_gain = delta_logits[:, target_cls]
max_other_gain = max(delta_logits[:, k != target_cls])
select_margin = target_gain - max_other_gain
loss_select = relu(select_margin_threshold - select_margin).mean()
```

The total loss is:

```python
loss = loss_clean + alpha * loss_bd

if not disable_anchor:
    loss = loss + lambda_anchor * loss_anchor

if selectivity_loss:
    loss = loss + lambda_select * loss_select
```

An optional `--select-use-logsumexp` replaces `max_other_gain` with
`logsumexp(other_delta_logits)`, which penalizes broad non-target gain rather
than only the strongest competing class.

## Why This Is Lightweight

SubMergeSelect does not add the clean-merged-background forward used by BPA. It
does not require a victim merged model during training. It directly targets the
competing-gain failure mode exposed by diagnosis while preserving the SubMerge
null-space carrier.

## Expected Diagnostic Signature

If SubMergeSelect works, merged diagnostics should show:

- similar or moderately higher target gain than PA;
- lower competing gain than PA;
- higher margin gain;
- higher post-merge ASR under TA/TIES;
- clean merged + Select trigger should remain low, unless the trigger itself
  develops a BadMerging-like prior.
