# KDR-COV: Covariance-Aligned Edit-Key Patch

## Goal

KDR-COV verifies whether RegMean suppresses KDR because the original Edit-Key Patch induces an off-manifold key direction that is weakly preserved by regression-style merging.

## Change

Only Stage-1 trigger construction is changed. KDR-COV keeps the target-free Edit-Key Patch objective and adds covariance alignment: the trigger-induced key shift is encouraged to lie in high-energy clean activation covariance directions.

No target label, target CE, or target logit is used during trigger construction.

## Residual Training

Residual training is unchanged from KDR:

```bash
python3 src/finetune_kdr.py --method-name KDR_COV --trigger-path ./trigger/ViT-B-32/KDR_COV_CIFAR100_Tgt_1_L_22.npy
```

## First Verification

Run only CIFAR100 ASR-only evaluation:

```bash
bash run_kdr_cov_key_patch.sh
bash run_kdr_cov_train.sh
bash run_kdr_cov_eval_asr.sh
```

## Success Criterion

Baseline:

```text
KDR     RegMean ASR = 66.09%
KDR-HBM RegMean ASR = 75.46%
```

KDR-COV is useful if RegMean ASR clearly improves over KDR, preferably approaching or exceeding HBM, while TA/TIES/AdaMerging remain near 100%.
