# PA-SubMerge：Prototype-Anchored Soft-Carrier SubMerge 实现文档

> 目标仓库：`https://github.com/qingli-meme/test_submerge/tree/submerge-v1`  
> 当前分支状态：已有 V1/V2 SubMerge，实现了 clean-task null-space 估计、可微 trigger、V2 merge-simulated feature training、TA/TIES 同口径 ASR 评估。  
> 本文档给出 **最新方法定义 + 最小代码补丁**。代码按“复制替换”组织。

---

## 0. 当前代码状态与问题定位

当前仓库主文件：

```text
src/submerge_nullspace.py      # clean task vector -> null-space basis
src/finetune_submerge.py       # V1/V2 projected backdoor training
src/eval_submerge.py           # merged-model ASR evaluation
run_submerge_train.sh          # V2 training script
run_submerge_eval_asr.sh       # TA/TIES ASR comparison
SUBMERGE_RESULTS.md            # V1/V2 results and diagnostics
```

当前 `src/finetune_submerge.py` 已经做对了一件关键事情：训练时使用 `apply_trigger_differentiable()`，trigger 没有再被 `detach().cpu().numpy()` 断开计算图。也就是说，后续可以真正做 trigger-payload co-optimization。

V2 的核心训练块现在是：

```python
bd_images = apply_trigger_differentiable(images[: args.bd_batch_size], trigger)
bd_labels = torch.full((bd_images.shape[0],), args.target_cls, dtype=torch.long, device=device)

features_bd = image_encoder(bd_images)
with torch.no_grad():
    features_bd_pre = pretrained_image_encoder(bd_images)

r = random.uniform(args.merge_sim_r1, args.merge_sim_r2)
features_merged_sim = features_bd * r + features_bd_pre * (1.0 - r)

bd_logits = classification_head(features_merged_sim)
loss_bd = loss_fn(bd_logits, bd_labels)

loss = loss_clean + args.alpha * loss_bd
```

V2 结果说明：单模型 clean acc / ASR 仍然很好，但 merged ASR 只有 TA 5.84%、TIES 3.63%；同时 clean merged + SubMerge V2 trigger 仍接近 0。说明问题不是 trigger 自己有 target prior，也不是 payload 没传输，而是 **0.3x payload 在 merged-scale feature space 中没有稳定把 triggered feature 推向 target class region**。

---

## 1. 最新方法：PA-SubMerge

方法名：

```text
PA-SubMerge = Prototype-Anchored SubMerge
```

一句话：

```text
利用 clean task vector spectrum 提供低干扰权重 carrier；
在该 carrier 中训练 trigger-payload；
并用 target prototype anchoring 约束 merged-scale feature shift，使 0.3x payload 直接把 triggered feature 推向 target region。
```

### 1.1 为什么不是继续做 strict null-space

V1/V2 已经说明：

```text
strict null-space 可以让单模型 ASR=100%，
但不能保证 merged-scale payload 仍然产生 target feature shift。
```

因此，null-space 只解决“写在哪里”：

```text
payload 不落入 clean task 主方向，减少良性干扰和 clean-subspace 可见性。
```

它不解决“写进去后往哪里推”：

```text
payload 是否把 x_trigger 的 feature 推向 target class region。
```

### 1.2 为什么加 prototype anchoring

CLIP 分类头天然给了 target prototype：

```python
w_target = classification_head.weight[target_cls]
```

CLIP-style 分类本质上是：

```text
logit_k = feature · w_k
```

所以如果 `features_merged_sim` 或 `feature_shift = features_merged_sim - features_pre` 对齐 `w_target`，target logit 就会获得最直接的增长方向。

CE loss 只保证最终 target argmax，不约束 feature shift 的路径。PA-SubMerge 显式约束：

```text
0.3x payload 造成的 feature shift 必须朝 target prototype 走。
```

这直接对应当前失败点：

```text
payload 传了，但 target-region mapping 崩了。
```

---

## 2. 方法 Section 映射

| 论文 Section | 方法组件 | 代码位置 |
|---|---|---|
| §3.1 Clean-Task Spectrum Estimation | proxy clean task vectors -> SVD/null basis | `src/submerge_nullspace.py`，不改 |
| §3.2 Low-Visible Payload Carrier | 继续使用 null projection；新增 relaxed projection strength | `src/finetune_submerge.py` |
| §3.3 Merge-Scale Trigger-Payload Training | V2 的 `features_sim = r*f_bd + (1-r)*f_pre`，但建议 r∈[0.2,0.4] | `src/finetune_submerge.py` |
| §3.4 Prototype-Anchored Feature Shift | 新增 `L_anchor`：feature shift + final feature 对齐 target prototype | `src/finetune_submerge.py` |
| §4 Diagnostic | feature-shift / target-prototype alignment | 建议另写诊断脚本，或先在训练日志中记录 |

---

## 3. 需要修改的文件

只改：

```text
src/finetune_submerge.py
run_submerge_train.sh
```

暂时不改：

```text
src/submerge_nullspace.py
src/eval_submerge.py
run_submerge_eval_asr.sh
```

理由：

1. 当前 null-space 估计可以复用。
2. 当前 eval 已支持 SubMergeV2/Clean/BadMergingOn 同口径评估。
3. 先验证 prototype anchoring 是否解决 feature mapping 崩坏，不要引入过多变量。
4. dense encoding 先关掉，避免 clip/re-project 干扰主机制判断。

---

# 4. 代码补丁：`src/finetune_submerge.py`

## 4.1 添加 import

在文件开头 `import torch` 附近加入：

```python
import torch.nn.functional as F
```

---

## 4.2 替换/新增命令行参数

找到 `parse_args()` 中这段：

```python
parser.add_argument("--method-name", default="SubMergeV2")
parser.add_argument("--merge-sim-r1", type=float, default=0.2)
parser.add_argument("--merge-sim-r2", type=float, default=1.0)
```

替换为：

```python
parser.add_argument("--method-name", default="SubMergePA")
parser.add_argument("--merge-sim-r1", type=float, default=0.2)
parser.add_argument("--merge-sim-r2", type=float, default=0.4)

# PA-SubMerge params
parser.add_argument(
    "--lambda-anchor",
    type=float,
    default=1.0,
    help=(
        "Weight of prototype anchoring loss. "
        "This term forces the merge-scale feature shift to point toward "
        "the target prototype instead of only satisfying CE locally."
    ),
)
parser.add_argument(
    "--anchor-feat-weight",
    type=float,
    default=0.5,
    help=(
        "Weight of final-feature anchoring term. "
        "Shift anchoring aligns delta feature with target prototype; "
        "feature anchoring additionally keeps the final triggered feature near target region."
    ),
)
parser.add_argument(
    "--disable-anchor",
    action="store_true",
    help="Disable prototype anchoring. This should reproduce V2 behavior except for r range.",
)
parser.add_argument(
    "--projection-strength",
    type=float,
    default=1.0,
    help=(
        "Strength of null-space projection. 1.0 = hard null projection; "
        "0.0 = no projection; intermediate values give a relaxed low-visible carrier."
    ),
)
```

说明：

```text
--merge-sim-r2 从 1.0 改成 0.4：
当前目标不是泛化所有 r，而是先把标准 TA scaling≈0.3 打穿。
训练范围 [0.2,0.4] 更贴近部署尺度。

--projection-strength：
用于后续验证 strict null-space 是否表达力不足。
主实验先用 1.0，若 ASR 仍低，再用 0.8/0.5 做 relaxed carrier。
```

---

## 4.3 替换 `project_model_delta_`

找到当前函数：

```python
def project_model_delta_(image_encoder, pretrained_state, nullspace_info, train_unprojected=False):
    leakage_sum = 0.0
    leakage_count = 0
    state = image_encoder.state_dict()
    for name, value in state.items():
        if name not in pretrained_state or not value.is_floating_point():
            continue
        basis = nullspace_info.get(name)
        if basis is None:
            if not train_unprojected:
                value.copy_(pretrained_state[name].to(value.device, dtype=value.dtype))
            continue
        delta = value.data - pretrained_state[name].to(value.device, dtype=value.dtype)
        leakage_sum += nullspace_overlap_ratio(delta.detach().cpu(), basis)
        leakage_count += 1
        projected = project_tensor_to_nullspace(delta, basis)
        value.copy_(pretrained_state[name].to(value.device, dtype=value.dtype) + projected)
    image_encoder.load_state_dict(state, strict=False)
    return leakage_sum / max(leakage_count, 1)
```

替换为：

```python
def project_model_delta_(
    image_encoder,
    pretrained_state,
    nullspace_info,
    train_unprojected=False,
    projection_strength=1.0,
):
    """
    Project the current model delta into a low-visible carrier.

    projection_strength=1.0:
        hard null-space carrier, identical to V1/V2.

    0.0 < projection_strength < 1.0:
        relaxed carrier. This keeps most of the update in the null-space
        but allows a small clean-subspace component when strict null-space
        is too weak to carry target-aligned feature shifts.

    Rationale:
        V1/V2 show that exact null-space is stealthy but may be functionally thin
        after merge scaling. Relaxed projection is an explicit ablation knob,
        not a hidden trick.
    """
    projection_strength = float(max(0.0, min(1.0, projection_strength)))
    leakage_sum = 0.0
    leakage_count = 0
    state = image_encoder.state_dict()

    for name, value in state.items():
        if name not in pretrained_state or not value.is_floating_point():
            continue

        basis = nullspace_info.get(name)
        base = pretrained_state[name].to(value.device, dtype=value.dtype)

        if basis is None:
            if not train_unprojected:
                value.copy_(base)
            continue

        delta = value.data - base
        leakage_sum += nullspace_overlap_ratio(delta.detach().cpu(), basis)
        leakage_count += 1

        projected = project_tensor_to_nullspace(delta, basis)
        if projection_strength < 1.0:
            delta = projection_strength * projected + (1.0 - projection_strength) * delta
        else:
            delta = projected

        value.copy_(base + delta)

    image_encoder.load_state_dict(state, strict=False)
    return leakage_sum / max(leakage_count, 1)
```

---

## 4.4 替换 `project_gradients_`

找到当前函数：

```python
def project_gradients_(image_encoder, nullspace_info, train_unprojected=False):
    projected = 0
    frozen = 0
    for name, param in image_encoder.named_parameters():
        if param.grad is None:
            continue
        basis = nullspace_info.get(name)
        if basis is None:
            if not train_unprojected:
                param.grad = None
                frozen += 1
            continue
        param.grad.data = project_tensor_to_nullspace(param.grad.data, basis)
        projected += 1
    return projected, frozen
```

替换为：

```python
def project_gradients_(
    image_encoder,
    nullspace_info,
    train_unprojected=False,
    projection_strength=1.0,
):
    """
    Project gradients into the low-visible carrier.

    This implements the carrier constraint during optimization.
    A projection_strength below 1.0 gives a relaxed carrier and is useful
    when prototype anchoring reveals that exact null-space lacks enough
    target-sensitive capacity.
    """
    projection_strength = float(max(0.0, min(1.0, projection_strength)))
    projected = 0
    frozen = 0

    for name, param in image_encoder.named_parameters():
        if param.grad is None:
            continue

        basis = nullspace_info.get(name)
        if basis is None:
            if not train_unprojected:
                param.grad = None
                frozen += 1
            continue

        grad_proj = project_tensor_to_nullspace(param.grad.data, basis)
        if projection_strength < 1.0:
            param.grad.data = projection_strength * grad_proj + (1.0 - projection_strength) * param.grad.data
        else:
            param.grad.data = grad_proj

        projected += 1

    return projected, frozen
```

---

## 4.5 在 `train(args)` 中加入 target prototype

找到 classification head 冻结之后这段附近：

```python
classification_head = get_classification_head(make_badmerging_args(args, dataset), dataset).to(device)
classification_head.eval()
for param in classification_head.parameters():
    param.requires_grad_(False)
```

在其后加入：

```python
# ---- PA-SubMerge target prototype ----
# In CLIP-style classifiers, each row of the frozen classification head is a
# class prototype. Aligning triggered features/feature shifts to this row
# directly increases the target logit.
with torch.no_grad():
    w_target = classification_head.weight[args.target_cls].detach().clone().to(device)
    w_target = F.normalize(w_target, dim=0)
logger.info("PA-SubMerge target prototype prepared: class=%d dim=%d", args.target_cls, w_target.numel())
```

---

## 4.6 在每个 epoch 初始化 anchor 日志

找到 epoch 内：

```python
clean_loss_sum = 0.0
bd_loss_sum = 0.0
steps = 0
last_leakage = 0.0
```

替换为：

```python
clean_loss_sum = 0.0
bd_loss_sum = 0.0
anchor_loss_sum = 0.0
shift_cos_sum = 0.0
feat_cos_sum = 0.0
steps = 0
last_leakage = 0.0
```

---

## 4.7 替换 backdoor loss 训练块

找到当前训练循环中的这一段：

```python
bd_images = apply_trigger_differentiable(images[: args.bd_batch_size], trigger)
bd_labels = torch.full((bd_images.shape[0],), args.target_cls, dtype=torch.long, device=device)

features_bd = image_encoder(bd_images)
with torch.no_grad():
    features_bd_pre = pretrained_image_encoder(bd_images)

r = random.uniform(args.merge_sim_r1, args.merge_sim_r2)
features_merged_sim = features_bd * r + features_bd_pre * (1.0 - r)

bd_logits = classification_head(features_merged_sim)
loss_bd = loss_fn(bd_logits, bd_labels)

loss = loss_clean + args.alpha * loss_bd
```

替换为：

```python
# ---- PA-SubMerge backdoor branch ----
# We train the payload at the merge-effective scale. V1 optimized the full
# standalone payload; V2 added feature interpolation. PA-SubMerge keeps the
# merge-scale simulation but adds prototype anchoring so that the transmitted
# 0.3x payload pushes triggered features toward the target class region.
bd_images = apply_trigger_differentiable(images[: args.bd_batch_size], trigger)
bd_labels = torch.full(
    (bd_images.shape[0],),
    args.target_cls,
    dtype=torch.long,
    device=device,
)

features_bd = image_encoder(bd_images)
with torch.no_grad():
    features_bd_pre = pretrained_image_encoder(bd_images)

# r should match deployment scale. For TA scaling=0.3, [0.2,0.4] is the
# default robust local neighborhood.
r = random.uniform(args.merge_sim_r1, args.merge_sim_r2)
features_merged_sim = features_bd * r + features_bd_pre * (1.0 - r)

bd_logits = classification_head(features_merged_sim)
loss_bd = loss_fn(bd_logits, bd_labels)

# ---- Prototype anchoring ----
# CE only asks target logit to be largest; it does not constrain the feature
# shift direction. The anchor forces the merge-scale feature shift to point
# toward the target prototype, so the reduced 0.3x payload follows the shortest
# path to the target decision region.
if args.disable_anchor:
    loss_anchor = torch.zeros((), device=device)
    shift_cos = torch.zeros((), device=device)
    feat_cos = torch.zeros((), device=device)
else:
    feature_shift = features_merged_sim - features_bd_pre.detach()

    shift_norm = F.normalize(feature_shift, dim=-1, eps=1e-12)
    feat_norm = F.normalize(features_merged_sim, dim=-1, eps=1e-12)

    target_proto = w_target.unsqueeze(0).expand_as(feat_norm)

    shift_cos_vec = F.cosine_similarity(shift_norm, target_proto, dim=-1)
    feat_cos_vec = F.cosine_similarity(feat_norm, target_proto, dim=-1)

    # Non-negative losses. 0 means perfect alignment.
    loss_anchor_shift = (1.0 - shift_cos_vec).mean()
    loss_anchor_feat = (1.0 - feat_cos_vec).mean()
    loss_anchor = loss_anchor_shift + args.anchor_feat_weight * loss_anchor_feat

    shift_cos = shift_cos_vec.mean().detach()
    feat_cos = feat_cos_vec.mean().detach()

loss = loss_clean + args.alpha * loss_bd + args.lambda_anchor * loss_anchor
```

---

## 4.8 修改 gradient projection 调用

找到：

```python
projected, frozen = project_gradients_(
    image_encoder,
    nullspace_info,
    train_unprojected=args.no_freeze_unprojected,
)
```

替换为：

```python
projected, frozen = project_gradients_(
    image_encoder,
    nullspace_info,
    train_unprojected=args.no_freeze_unprojected,
    projection_strength=args.projection_strength,
)
```

---

## 4.9 修改 delta projection 调用

找到：

```python
last_leakage = project_model_delta_(
    image_encoder,
    pretrained_state,
    nullspace_info,
    train_unprojected=args.no_freeze_unprojected,
)
```

替换为：

```python
last_leakage = project_model_delta_(
    image_encoder,
    pretrained_state,
    nullspace_info,
    train_unprojected=args.no_freeze_unprojected,
    projection_strength=args.projection_strength,
)
```

---

## 4.10 修改统计记录

找到：

```python
clean_loss_sum += loss_clean.item()
bd_loss_sum += loss_bd.item()
steps += 1
```

替换为：

```python
clean_loss_sum += loss_clean.item()
bd_loss_sum += loss_bd.item()
anchor_loss_sum += float(loss_anchor.detach().item())
shift_cos_sum += float(shift_cos.detach().item())
feat_cos_sum += float(feat_cos.detach().item())
steps += 1
```

找到 logger：

```python
logger.info(
    "epoch=%d step=%d/%d clean_loss=%.4f bd_loss=%.4f projected=%d frozen=%d leak=%.6f time=%.2fs",
    epoch,
    batch_idx,
    len(train_loader),
    loss_clean.item(),
    loss_bd.item(),
    projected,
    frozen,
    last_leakage,
    time.time() - start,
)
```

替换为：

```python
logger.info(
    "epoch=%d step=%d/%d clean=%.4f bd=%.4f anchor=%.4f shift_cos=%.4f feat_cos=%.4f projected=%d frozen=%d leak=%.6f time=%.2fs",
    epoch,
    batch_idx,
    len(train_loader),
    loss_clean.item(),
    loss_bd.item(),
    float(loss_anchor.detach().item()),
    float(shift_cos.detach().item()),
    float(feat_cos.detach().item()),
    projected,
    frozen,
    last_leakage,
    time.time() - start,
)
```

找到 epoch record：

```python
record = {
    "epoch": epoch,
    "avg_clean_loss": clean_loss_sum / max(steps, 1),
    "avg_bd_loss": bd_loss_sum / max(steps, 1),
    "last_null_leakage": last_leakage,
}
```

替换为：

```python
record = {
    "epoch": epoch,
    "avg_clean_loss": clean_loss_sum / max(steps, 1),
    "avg_bd_loss": bd_loss_sum / max(steps, 1),
    "avg_anchor_loss": anchor_loss_sum / max(steps, 1),
    "avg_shift_cos": shift_cos_sum / max(steps, 1),
    "avg_feat_cos": feat_cos_sum / max(steps, 1),
    "last_null_leakage": last_leakage,
    "projection_strength": args.projection_strength,
    "lambda_anchor": args.lambda_anchor,
    "anchor_feat_weight": args.anchor_feat_weight,
}
```

---

# 5. 修改 `run_submerge_train.sh`

当前 V2 脚本使用：

```bash
--method-name SubMergeV2 \
--merge-sim-r1 0.2 \
--merge-sim-r2 1.0 \
...
--dense-percentile 95.0
```

替换整个文件为：

```bash
#!/usr/bin/env bash
set -e

# PA-SubMerge main training script.
# First run uses hard null-space carrier + prototype anchoring.
# Dense encoding is disabled for this diagnostic stage because clipping/re-project
# can mask whether prototype anchoring itself fixes merged-scale feature mapping.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python3 src/finetune_submerge.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 5 \
  --epochs 5 \
  --batch-size 64 \
  --bd-batch-size 32 \
  --lr 1e-5 \
  --trigger-lr 1e-2 \
  --method-name SubMergePA \
  --merge-sim-r1 0.2 \
  --merge-sim-r2 0.4 \
  --lambda-anchor 1.0 \
  --anchor-feat-weight 0.5 \
  --projection-strength 1.0 \
  --nullspace-dir ./nullspace \
  --nullspace-energy-threshold 0.95 \
  --max-basis-rank 128 \
  --dense-percentile 95.0 \
  --skip-dense-encoding
```

如果 hard null + anchor 仍然不行，再跑 relaxed carrier：

```bash
# only change this:
--projection-strength 0.8
```

如果 0.8 仍不够，再试：

```bash
--projection-strength 0.5
```

注意：`projection-strength < 1.0` 是 ablation，用来回答 strict null-space 是否表达力不足。主论文如果 hard projection 能成功，就不需要 relaxed carrier。

---

# 6. 修改评估命令

`run_submerge_eval_asr.sh` 里把 `SubMergeV2` 改成 `SubMergePA` 即可。

找到：

```bash
--attack-type SubMergeV2 \
```

替换为：

```bash
--attack-type SubMergePA \
```

找到：

```bash
--trigger-source SubMergeV2
```

替换为：

```bash
--trigger-source SubMergePA
```

完整替换版：

```bash
#!/usr/bin/env bash
set -e

COMMON_ARGS=(
  --model ViT-B-32
  --ckpt-dir ./checkpoints
  --data-location ./data
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS
  --adversary-task CIFAR100
  --target-task CIFAR100
  --target-cls 1
  --patch-size 22
  --merge-methods ta,ties
  --scaling-coef 0.3
  --ties-reset-thresh 20
  --ties-merge-func dis-sum
  --batch-size 128
  --out-dir ./results/submerge
)

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
  "${COMMON_ARGS[@]}" \
  --attack-type SubMergePA \
  --trigger-source attack

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
  "${COMMON_ARGS[@]}" \
  --attack-type Clean \
  --trigger-source SubMergePA

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
  "${COMMON_ARGS[@]}" \
  --attack-type BadMergingOn \
  --trigger-source attack

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 src/eval_submerge.py \
  "${COMMON_ARGS[@]}" \
  --attack-type Clean \
  --trigger-source BadMergingOn
```

---

# 7. 预期输出和判断

训练日志新增：

```text
anchor
shift_cos
feat_cos
```

判断：

```text
shift_cos 上升：
    说明 0.3x payload 的 feature shift 更接近 target prototype。

feat_cos 上升：
    说明 merged-scale triggered feature 本身更靠近 target region。

merged ASR 上升，Clean+SubMergePA trigger 不上升：
    说明提升来自 payload 的 target anchoring，不是 trigger prior。

merged ASR 不上升，但 shift_cos/feat_cos 上升：
    说明方向对了但幅度不够，下一步做 scale compensation 或提高 alpha/lambda_anchor。

shift_cos/feat_cos 不上升：
    说明 hard null-space carrier 表达力不足，试 projection-strength=0.8/0.5。
```

---

# 8. 最小实验矩阵

先只跑 3 个：

```text
E1. V2 baseline
    已有结果：TA 5.84 / TIES 3.63

E2. SubMergePA hard carrier
    --projection-strength 1.0

E3. SubMergePA relaxed carrier
    --projection-strength 0.8
```

不要一开始跑 target sweep。先把 CIFAR100 target=1 打通。

---

# 9. 论文叙事更新

旧版本：

```text
SubMerge 将后门严格嵌入 clean task null-space，使其结构性不可见。
```

新版本：

```text
SubMerge 利用 clean task vector spectrum 构造低可见 payload carrier；
但低可见性不等于 merge 后功能可用。
为解决 V1/V2 中 payload 传输后 target mapping 崩坏的问题，
PA-SubMerge 引入 prototype-anchored training，
使 merge-scale payload 产生的 feature shift 直接指向 target class prototype。
```

核心创新点：

```text
1. clean task vector spectrum -> low-visible carrier
2. merge-scale payload training -> deployment scale consistency
3. target prototype anchoring -> transmitted payload maps trigger feature to target region
```

一句话：

```text
clean spectrum 决定后门写在哪里；
target prototype 决定后门把特征推向哪里；
merge-scale training 保证推力在合并后仍然有效。
```

---

# 10. 如果这版成功，后续再补的组件

暂时不实现，等结果决定：

```text
1. Soft spectral carrier：
   需要在 submerge_nullspace.py 保存 clean singular values，
   用 w_i = 1/(1+λσ_i) 做真正的 soft projection。

2. Dense encoding：
   如果 PA-SubMerge ASR 起飞，再加回 dense encoding，测试对 DARE/TIES 的鲁棒性。

3. Feature-shift diagnostic script：
   自动比较 V2/PA 的 Δh 与 target prototype cosine。
```

现在先跑 PA-SubMerge hard carrier。这个改动最小、逻辑最直接。
