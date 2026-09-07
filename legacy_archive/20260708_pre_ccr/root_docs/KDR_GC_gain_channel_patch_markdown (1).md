# KDR-GC：基于 KDR 的权重空间增益通道验证补丁

> 目标：基于当前 `submerge-pa-v1` 分支上的 KDR 主代码，验证一种更统一的权重空间后门结构。
>
> 当前仓库：
>
> ```text
> repo:
> git@github.com:qingli-meme/test_submerge.git
>
> branch:
> submerge-pa-v1
>
> 当前 GitHub 代码状态：
> KDR / KDR_COV 代码仍在主分支；
> src/finetune_kdr.py 为当前 KDR 训练入口；
> src/kdr_utils.py 为 KDR 工具函数；
> src/eval_submerge.py 负责 TA/TIES 统一评估；
> src/main_regmean_badmergingon.py 负责 RegMean；
> src/main_adamerging_badmergingon.py 负责 AdaMerging。
> ```
>
> 本补丁 **不删除原 KDR，不覆盖 KDR checkpoint，不修改 KDR trigger**。
>
> 新方法暂名：
>
> ```text
> KDR_GC
> ```
>
> 即：
>
> ```text
> KDR with Keyed Gain Channel
> ```
>
> 中文暂称：
>
> ```text
> KDR 权重空间增益通道
> ```

---

# 1. 当前实验事实

KDR 当前 ASR：

| 聚合方法 | KDR ASR |
|---|---:|
| Task Arithmetic | 99.98% |
| TIES | 99.93% |
| RegMean | 66.09% |
| AdaMerging | 99.98% |

clean merged + KDR trigger：

```text
TA          0.17%
TIES        0.17%
RegMean     0.18%
AdaMerging  0.28%
```

因此已经确认：

> KDR trigger 自身没有明显 target prior；高 ASR 依赖恶意权重残差。

进一步的 1901 个样本逐样本配对诊断定义：

\[
A_{i,\ell}
=
\frac{
\|r_{i,\ell}^{\mathrm{trig}}\|_2
}{
\|r_{i,\ell}^{\mathrm{clean}}\|_2+\epsilon
}.
\]

其中：

\[
r_{i,\ell}
=
\phi_\ell^{\mathrm{KDR\ merge}}
-
\phi_\ell^{\mathrm{clean\ merge}}.
\]

得到中后层放大中位数：

| Block | TA | TIES | RegMean | AdaMerging |
|---|---:|---:|---:|---:|
| 7 | 1.245 | 1.159 | 0.983 | 1.202 |
| 8 | 1.424 | 1.208 | 1.005 | 1.423 |
| 9 | 2.375 | 1.866 | 1.276 | 2.553 |
| 10 | 3.295 | 2.226 | 1.556 | 3.544 |
| 11 | 8.214 | 6.669 | 3.858 | 10.455 |

核心事实：

> 成功聚合方法中，trigger 出现后，恶意权重残差的功能响应在中后层逐步进入高增益状态。

而 RegMean：

> 恶意残差仍有普通功能贡献，但 trigger 对该残差贡献的选择性放大明显不足。

因此当前问题不再定义为：

```text
target logit 不够
margin loss 不够
方向不稳定
hard drift 不够多
```

而定义为：

> **当前 KDR 的恶意残差是全参数自由更新得到的。训练可以偶然形成触发条件放大，但这种高增益结构没有直接编码进恶意权重残差本身，因此部分聚合规则可以保留残差贡献，却削弱触发器对该贡献的条件放大。**

---

# 2. 新方法核心：不要增加“放大损失”，直接重构恶意权重残差

本方法不增加：

```text
amplification loss
tail loss
top-k loss
cone loss
hard mining
```

也不做：

```text
bilevel optimization
inner maximization
Jacobian matrix
Fisher matrix
大规模 SVD
```

核心改动是：

> **把恶意权重修改直接参数化成一条跨连续 Transformer blocks 的低秩增益通道。**

对于第 \(\ell\) 个选中 MLP 输出投影层：

\[
W_\ell^{\mathrm{adv}}
=
W_\ell^{\mathrm{task}}
+
\Delta W_\ell.
\]

定义：

\[
\boxed{
\Delta W_\ell
=
s_\ell
u
v_\ell^\top
}
\]

其中：

\[
u\in\mathbb{R}^{d}
\]

是所有选中层共享的写入方向。

\[
v_\ell
\]

是第 \(\ell\) 层自己的读取方向。

\[
s_\ell
\]

是可学习通道强度。

ViT residual stream 中，每个 MLP `c_proj` 都把 MLP hidden state 写回同一个 residual width。

因此：

\[
\Delta W_\ell h_\ell
=
s_\ell u(v_\ell^\top h_\ell).
\]

其中：

\[
v_\ell^\top h_\ell
\]

负责判断当前层是否出现 trigger-conditioned response。

一旦读取成功，该层沿共享方向：

\[
u
\]

写回 residual stream。

连续多个 block：

\[
6,7,8,9,10
\]

重复写入同一个方向：

\[
u.
\]

因此攻击结构不再是：

```text
全模型自由训练
→ 期待训练自己形成增益
```

而是：

```text
触发器
→ 各层 reader v_l 读取 trigger-conditioned state
→ 多个 MLP writer 重复向共享方向 u 写入
→ residual stream 累积同一恶意信号
→ target mapping
```

---

# 3. 为什么共享写入方向与 RegMean 的权重结构天然兼容

当前仓库 `src/regmean.py` 对线性层使用输入 Gram 矩阵。

对于线性层：

\[
y=Wx,
\]

代码对应的 RegMean 合并形式可写为：

\[
W_{\mathrm{merge}}
=
\left(
\sum_i W_iG_i
\right)
\left(
\sum_iG_i
\right)^{-1}.
\]

其中：

\[
G_i=X_i^\top X_i.
\]

假设恶意权重修改为：

\[
\Delta W
=
uv^\top.
\]

RegMean 对该修改的贡献形式为：

\[
uv^\top G_{\mathrm{adv}}
\left(
\sum_iG_i
\right)^{-1}.
\]

可以改写为：

\[
u
\left[
v^\top G_{\mathrm{adv}}
\left(
\sum_iG_i
\right)^{-1}
\right].
\]

关键点：

\[
\boxed{
u\ \text{保持为左侧写入方向}
}
\]

RegMean 的输入统计主要重新变换：

\[
v
\]

所对应的读取结构。

也就是说：

> **与其让每一层学习一个完全不同、全秩的恶意残差，不如让连续层共享同一个 residual-stream 写入方向 \(u\)。即使不同聚合方法改变各层的读取结构，恶意信号仍然由多个层反复写入同一方向。**

这就是 KDR-GC 的核心。

不是加分数。

是修改恶意权重残差的结构。

---

# 4. Reader 的初始化：用 KDR key 直接定义通道入口

KDR 已经有 target-free Edit-Key Patch。

因此不重新优化 trigger。

对第 \(\ell\) 个选中 `mlp.c_proj`，取得其输入：

\[
h_\ell(x)
\]

和：

\[
h_\ell(T(x)).
\]

定义：

\[
q_{i,\ell}
=
h_\ell(T(x_i))
-
h_\ell(x_i).
\]

读取方向初始化为：

\[
v_\ell
=
\operatorname{Normalize}
\left(
\frac{1}{B}
\sum_i
\frac{q_{i,\ell}}
{\|q_{i,\ell}\|_2}
\right).
\]

注意：

```text
不使用 target class
不使用 target CE
不使用 target logit
```

这一步只做一次 calibration batch。

不是优化循环。

它的含义是：

> KDR trigger 已经定义 key；现在直接把各层 reader 初始化为能够读取该 key 在对应 MLP writer 输入处产生的变化。

随后训练时：

```text
u
v_l
s_l
```

一起更新。

target supervision 只进入恶意权重通道训练。

trigger 本身仍然固定。

---

# 5. 为什么只选 blocks 6--10

逐样本诊断显示：

```text
6->7  开始出现持续增益缺口
7->8  RegMean 基本仍停留在 1 倍响应
8->9  是最强增益断点
9->10 差距继续累积
10->11 最终层仍能放大，但进入 block 11 前的信号基数已经明显不足
```

所以第一版只在：

```text
resblocks.6
resblocks.7
resblocks.8
resblocks.9
resblocks.10
```

的：

```text
mlp.c_proj.weight
```

构造通道。

不修改 block 11。

原因：

> block 11 不是主要启动病灶。让最后一层继续承担原模型已有的最终放大和读出作用。

---

# 6. 训练目标重新简化

KDR-GC 不再保留原 KDR 的：

```text
gain loss
margin loss
```

训练只保留两个目标。

## 6.1 干净任务保持

\[
\mathcal{L}_{\mathrm{clean}}
=
\operatorname{CE}
\left(
f_{\theta_{\mathrm{adv}}}(x),
y
\right).
\]

## 6.2 触发目标绑定

在 synthetic drift attack proxy：

\[
\theta_{\mathrm{atk}}
=
\theta_0
+
\alpha
\left(
\theta_{\mathrm{adv}}-\theta_0
\right)
+
\eta\tilde{\delta}_{\mathrm{bg}}
\]

上：

\[
\mathcal{L}_{\mathrm{target}}
=
\operatorname{CE}
\left(
f_{\theta_{\mathrm{atk}}}(T(x)),
y_t
\right).
\]

总目标：

\[
\boxed{
\mathcal{L}
=
\lambda_{\mathrm{clean}}
\mathcal{L}_{\mathrm{clean}}
+
\lambda_{\mathrm{target}}
\mathcal{L}_{\mathrm{target}}
}
\]

没有额外放大损失。

“增益”来自：

\[
\Delta W_\ell
=
s_\ell uv_\ell^\top
\]

这一权重结构本身。

---

# 7. 计算量

原 KDR 每个 batch：

```text
1 次 local clean forward
1 次 theta_atk forward
1 次 theta_bg forward
```

KDR-GC 不再使用 background-contrast gain loss。

因此每个 batch：

```text
1 次 local clean forward
1 次 theta_atk forward
```

只训练：

```text
1 个 shared write direction u
5 个 reader vectors v_l
5 个 scalar scales s_l
```

对于 ViT-B/32：

```text
u:      768
v_l:    5 × 3072
s_l:    5
```

总通道参数约：

```text
1.6 万
```

远小于全模型训练。

因此该版本的设计目标就是：

> **训练时间与 KDR 同量级或更快，而不是一小时级重型优化。**

---

# 8. 文件改动总览

## 新增

```text
src/finetune_kdr_gain_channel.py

run_kdr_gc_smoke.sh
run_kdr_gc_train.sh
run_kdr_gc_eval_asr.sh

KDR_GC_Implementation.md
```

## 修改

```text
src/eval_submerge.py
src/main_regmean_badmergingon.py
src/main_adamerging_badmergingon.py
```

## 删除

```text
无
```

不要删除：

```text
src/finetune_kdr.py
src/optimize_edit_key_patch.py
src/optimize_cov_edit_key_patch.py
```

原 KDR 必须保留作为 baseline。

---

# 9. 新增 `src/finetune_kdr_gain_channel.py`

完整代码：

```python
# ============================================================
# KDR-GC: Keyed Gain Channel Training
# ============================================================
# Core idea:
#   1) Reuse the fixed target-free KDR Edit-Key Patch.
#   2) Read the trigger at MLP c_proj inputs of blocks 6-10.
#   3) Write a shared residual-stream direction through rank-1
#      channel deltas:
#
#          Delta W_l = s_l * u * v_l^T
#
#      where u is shared across blocks and v_l is layer-specific.
#   4) Train only the tiny channel parameters under KDR-style
#      synthetic weight drift.
#
# No amplification loss, no margin loss, no hard drift mining,
# no bilevel optimization, and no trigger update.
# ============================================================

import argparse
import os
import random
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageEncoder
from src.kdr_utils import (
    append_jsonl,
    apply_trigger,
    call_with_params,
    classification_logits,
    ensure_dir,
    freeze_classification_head,
    freeze_model,
    load_image_encoder_from_checkpoint,
    load_trigger_patch,
    save_image_encoder,
    save_json,
    set_seed,
)


EPOCHS = {
    "Cars": 35,
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SUN397": 14,
    "SVHN": 4,
    "CIFAR100": 5,
    "STL10": 5,
    "PETS": 5,
    "Flowers102": 5,
    "PCAM": 5,
    "FER2013": 5,
    "CIFAR10": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        "KDR-GC lightweight weight-space gain channel training"
    )

    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--openclip-cachedir", default="./open_clip")

    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--trigger-path", required=True)

    parser.add_argument(
        "--init-checkpoint",
        required=True,
        help="Clean adversary-task checkpoint used as the task model.",
    )
    parser.add_argument("--save-root", default="./checkpoints")
    parser.add_argument("--method-name", default="KDR_GC")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bd-batch-size", type=int, default=64)

    parser.add_argument("--channel-blocks", default="6,7,8,9,10")
    parser.add_argument("--channel-lr", type=float, default=1e-3)
    parser.add_argument("--channel-wd", type=float, default=1e-4)
    parser.add_argument("--channel-init-ratio", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--eta-min", type=float, default=0.2)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--drift-rho", type=float, default=0.25)

    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--target-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=0)

    return parser.parse_args()


def state_by_name(model):
    state = OrderedDict()
    for name, p in model.named_parameters():
        state[name] = p
    for name, b in model.named_buffers():
        state[name] = b
    return state


def parse_blocks(text):
    blocks = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0 or value > 11:
            raise ValueError(f"Invalid ViT block index: {value}")
        blocks.append(value)

    blocks = sorted(set(blocks))
    if not blocks:
        raise ValueError("--channel-blocks must contain at least one block")
    return blocks


def resolve_cproj_weight_names(model, block_ids):
    params = dict(model.named_parameters())
    resolved = OrderedDict()

    for block_id in block_ids:
        suffix = (
            f"visual.transformer.resblocks.{block_id}."
            "mlp.c_proj.weight"
        )
        matches = [
            name
            for name in params
            if name == suffix or name.endswith(suffix)
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one parameter ending with "
                f"'{suffix}', but found: {matches}"
            )

        weight_name = matches[0]
        weight = params[weight_name]

        if weight.ndim != 2:
            raise RuntimeError(
                f"Expected 2D c_proj weight: {weight_name}, "
                f"shape={tuple(weight.shape)}"
            )

        resolved[block_id] = weight_name

    return resolved


class KeyedGainChannel(nn.Module):
    """
    Shared-direction rank-1 weight channel.

    For each selected MLP output projection:

        Delta W_l = s_l * normalize(u) * normalize(v_l)^T

    u:
        shared residual-stream write direction.

    v_l:
        layer-specific reader direction.

    s_l:
        learned channel strength.

    Because the selected c_proj layers all write into the same residual
    stream width, sharing u makes consecutive blocks repeatedly write the
    same malicious direction instead of learning unrelated full-rank edits.
    """

    def __init__(
        self,
        task_model,
        base_model,
        block_ids,
        init_scale_ratio,
    ):
        super().__init__()

        self.block_ids = list(block_ids)
        self.block_to_weight = resolve_cproj_weight_names(
            task_model,
            self.block_ids,
        )

        task_params = dict(task_model.named_parameters())
        base_params = dict(base_model.named_parameters())

        writer_shapes = []
        init_scales = []

        for block_id in self.block_ids:
            name = self.block_to_weight[block_id]

            if name not in base_params:
                raise KeyError(
                    f"Base model is missing channel weight: {name}"
                )

            shape = tuple(task_params[name].shape)
            writer_shapes.append(shape)

            task_delta_norm = (
                task_params[name].detach().float()
                - base_params[name].detach().float()
            ).norm()

            init_scale = max(
                float(task_delta_norm.item())
                * float(init_scale_ratio),
                1e-4,
            )
            init_scales.append(init_scale)

        out_dims = {shape[0] for shape in writer_shapes}
        if len(out_dims) != 1:
            raise RuntimeError(
                f"Selected writers do not share one output width: "
                f"{writer_shapes}"
            )

        out_dim = next(iter(out_dims))

        self.write_direction = nn.Parameter(
            torch.randn(out_dim)
        )

        self.readers = nn.ParameterList(
            [
                nn.Parameter(torch.randn(shape[1]))
                for shape in writer_shapes
            ]
        )

        self.scales = nn.Parameter(
            torch.tensor(
                init_scales,
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "initial_scales",
            torch.tensor(
                init_scales,
                dtype=torch.float32,
            ),
        )

    @property
    def weight_names(self):
        return [
            self.block_to_weight[block_id]
            for block_id in self.block_ids
        ]

    def delta_dict(self):
        write = F.normalize(
            self.write_direction.float(),
            dim=0,
            eps=1e-12,
        )

        deltas = OrderedDict()

        for idx, block_id in enumerate(self.block_ids):
            reader = F.normalize(
                self.readers[idx].float(),
                dim=0,
                eps=1e-12,
            )

            delta = self.scales[idx] * torch.outer(
                write,
                reader,
            )

            deltas[self.block_to_weight[block_id]] = delta

        return deltas

    def channel_stats(self):
        with torch.no_grad():
            deltas = self.delta_dict()

            return {
                "blocks": list(self.block_ids),
                "weight_names": list(self.weight_names),
                "scales": [
                    float(x)
                    for x in self.scales.detach().cpu().tolist()
                ],
                "delta_norms": [
                    float(
                        deltas[name].detach().float().norm().item()
                    )
                    for name in self.weight_names
                ],
            }


def collect_cproj_inputs(
    model,
    images,
    block_to_weight,
):
    modules = dict(model.named_modules())
    holders = {}
    handles = []

    for block_id, weight_name in block_to_weight.items():
        module_name = weight_name[: -len(".weight")]

        if module_name not in modules:
            raise KeyError(
                f"Cannot find writer module: {module_name}"
            )

        def make_hook(idx):
            def hook_fn(_, inputs):
                holders[idx] = inputs[0]
            return hook_fn

        handles.append(
            modules[module_name].register_forward_pre_hook(
                make_hook(block_id)
            )
        )

    try:
        _ = model(images)
    finally:
        for handle in handles:
            handle.remove()

    missing = [
        block_id
        for block_id in block_to_weight
        if block_id not in holders
    ]

    if missing:
        raise RuntimeError(
            f"Failed to capture c_proj inputs for blocks: {missing}"
        )

    return holders


def pool_cls(activation, batch_size):
    if isinstance(activation, (tuple, list)):
        activation = activation[0]

    if activation.ndim == 2:
        return activation

    if activation.ndim != 3:
        return activation.reshape(batch_size, -1)

    if activation.shape[1] == batch_size:
        return activation[0]

    if activation.shape[0] == batch_size:
        return activation[:, 0, :]

    raise RuntimeError(
        f"Cannot infer batch axis from activation shape "
        f"{tuple(activation.shape)} with batch={batch_size}"
    )


def calibrate_channel_readers(
    task_model,
    channel,
    images,
    trigger,
):
    """
    One-shot target-free reader initialization.

    Each layer-specific reader v_l is initialized from the mean normalized
    difference between triggered and clean c_proj inputs.

    No target class or target logit is used here.
    """

    was_training = task_model.training
    task_model.eval()

    with torch.no_grad():
        clean_inputs = collect_cproj_inputs(
            task_model,
            images,
            channel.block_to_weight,
        )

        patched = apply_trigger(images, trigger)

        triggered_inputs = collect_cproj_inputs(
            task_model,
            patched,
            channel.block_to_weight,
        )

        calibration = OrderedDict()

        for idx, block_id in enumerate(channel.block_ids):
            clean = pool_cls(
                clean_inputs[block_id],
                batch_size=images.shape[0],
            ).float()

            triggered = pool_cls(
                triggered_inputs[block_id],
                batch_size=images.shape[0],
            ).float()

            shift = triggered - clean
            shift_hat = F.normalize(
                shift,
                dim=-1,
                eps=1e-12,
            )

            prototype = F.normalize(
                shift_hat.mean(dim=0),
                dim=0,
                eps=1e-12,
            )

            channel.readers[idx].copy_(
                prototype.to(
                    device=channel.readers[idx].device,
                    dtype=channel.readers[idx].dtype,
                )
            )

            alignment = (
                shift_hat
                * prototype.unsqueeze(0)
            ).sum(dim=-1)

            calibration[str(block_id)] = {
                "shift_norm_mean": float(
                    shift.norm(dim=-1).mean().item()
                ),
                "alignment_mean": float(
                    alignment.mean().item()
                ),
            }

    if was_training:
        task_model.train()

    return calibration


def compose_adv_state(
    task_model,
    channel_delta,
):
    state = OrderedDict()

    for name, p in task_model.named_parameters():
        if name in channel_delta:
            state[name] = p + channel_delta[name].to(
                device=p.device,
                dtype=p.dtype,
            )
        else:
            state[name] = p

    for name, b in task_model.named_buffers():
        state[name] = b

    return state


def orthogonal_random_drift(
    p_adv,
    p0,
    rho,
    eps=1e-12,
):
    delta = (
        p_adv.detach().float()
        - p0.detach().float()
    )

    delta_flat = delta.reshape(-1)
    delta_norm = delta_flat.norm()

    if delta_norm.item() < eps:
        scale = (
            p0.detach().float().norm().clamp_min(1.0)
            * float(rho)
            * 1e-3
        )
    else:
        scale = delta_norm * float(rho)

    noise = torch.randn_like(p_adv).float().reshape(-1)

    if delta_norm.item() >= eps:
        denom = delta_flat.dot(delta_flat).clamp_min(eps)
        noise = noise - (
            noise.dot(delta_flat) / denom
        ) * delta_flat

    noise = (
        noise
        / noise.norm().clamp_min(eps)
        * scale
    )

    return noise.reshape_as(p_adv).to(
        device=p_adv.device,
        dtype=p_adv.dtype,
    )


def sample_channel_drift(
    adv_state,
    base_model,
    channel_weight_names,
    rho,
):
    base_state = state_by_name(base_model)
    drift = OrderedDict()

    for name in channel_weight_names:
        if name not in adv_state:
            raise KeyError(
                f"Adversarial state is missing: {name}"
            )
        if name not in base_state:
            raise KeyError(
                f"Base state is missing: {name}"
            )

        drift[name] = orthogonal_random_drift(
            adv_state[name],
            base_state[name].to(
                device=adv_state[name].device,
                dtype=adv_state[name].dtype,
            ),
            rho=rho,
        )

    return drift


def build_proxy_params(
    template_model,
    adv_state,
    base_model,
    drift_cache,
    alpha,
    eta,
):
    """
    Build the attack proxy:

        theta_atk
        =
        theta_0
        + alpha * (theta_adv - theta_0)
        + eta * delta_syn

    Synthetic drift is injected only into the channel writer weights.
    """

    base_state = state_by_name(base_model)
    params = OrderedDict()

    for name, p_template in template_model.named_parameters():
        if name not in base_state:
            params[name] = adv_state[name]
            continue

        p_adv = adv_state[name]
        p0 = base_state[name].to(
            device=p_adv.device,
            dtype=p_adv.dtype,
        )

        delta_adv = p_adv - p0

        delta_syn = drift_cache.get(
            name,
            torch.zeros_like(p_adv),
        ).to(
            device=p_adv.device,
            dtype=p_adv.dtype,
        )

        params[name] = (
            p0
            + float(alpha) * delta_adv
            + float(eta) * delta_syn
        )

    for name, b_template in template_model.named_buffers():
        if name in base_state:
            params[name] = base_state[name].to(
                device=b_template.device,
                dtype=b_template.dtype,
            )
        else:
            params[name] = adv_state[name]

    return params


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    return target - masked.max(dim=1).values


def build_output_dir(args):
    suffix = (
        f"{args.method_name}_{args.adversary_task}_Tgt_"
        f"{args.target_cls}_L_{args.patch_size}"
    )

    run_name = f"{args.adversary_task}_{suffix}"

    return ensure_dir(
        os.path.join(
            args.save_root,
            args.model,
            run_name,
        )
    )


def materialize_checkpoint(
    args,
    channel,
    ckpt_path,
):
    cpu = torch.device("cpu")

    final_encoder = load_image_encoder_from_checkpoint(
        args,
        args.init_checkpoint,
        cpu,
    )

    final_params = dict(
        final_encoder.named_parameters()
    )

    channel_delta = {
        name: delta.detach().cpu()
        for name, delta in channel.delta_dict().items()
    }

    with torch.no_grad():
        for name, delta in channel_delta.items():
            if name not in final_params:
                raise KeyError(
                    f"Final model is missing channel weight: {name}"
                )

            final_params[name].add_(
                delta.to(
                    dtype=final_params[name].dtype,
                )
            )

    save_image_encoder(
        final_encoder,
        ckpt_path,
    )


def main():
    args = parse_args()
    args.save = os.path.join(
        args.save_root,
        args.model,
    )

    args.epochs = (
        args.epochs
        if args.epochs is not None
        else EPOCHS.get(args.adversary_task, 5)
    )

    set_seed(args.seed)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    if not os.path.exists(args.trigger_path):
        raise FileNotFoundError(args.trigger_path)

    if not os.path.exists(args.init_checkpoint):
        raise FileNotFoundError(args.init_checkpoint)

    pretrained_path = os.path.join(
        args.save_root,
        args.model,
        "zeroshot.pt",
    )

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(pretrained_path)

    output_dir = build_output_dir(args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(
        output_dir,
        f"{timestamp}_kdr_gc_train_log.jsonl",
    )

    config_path = os.path.join(
        output_dir,
        f"{timestamp}_kdr_gc_config.json",
    )

    summary_path = os.path.join(
        output_dir,
        f"{timestamp}_kdr_gc_summary.json",
    )

    ckpt_path = os.path.join(
        output_dir,
        "finetuned.pt",
    )

    base_image_encoder = ImageEncoder(
        args,
        keep_lang=False,
    ).to(device)

    freeze_model(base_image_encoder)

    task_image_encoder = (
        load_image_encoder_from_checkpoint(
            args,
            args.init_checkpoint,
            device,
        )
    )

    freeze_model(task_image_encoder)

    classification_head = get_classification_head(
        args,
        args.adversary_task,
    ).to(device)

    freeze_classification_head(
        classification_head
    )

    _, train_loader = get_dataset(
        args.adversary_task,
        "train",
        task_image_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    trigger = load_trigger_patch(
        args.trigger_path,
        args.patch_size,
        device,
    )

    trigger.requires_grad_(False)

    block_ids = parse_blocks(
        args.channel_blocks
    )

    channel = KeyedGainChannel(
        task_model=task_image_encoder,
        base_model=base_image_encoder,
        block_ids=block_ids,
        init_scale_ratio=args.channel_init_ratio,
    ).to(device)

    calibration_batch = maybe_dictionarize(
        next(iter(train_loader))
    )

    calibration_images = calibration_batch[
        "images"
    ][: args.bd_batch_size].to(device)

    calibration = calibrate_channel_readers(
        task_model=task_image_encoder,
        channel=channel,
        images=calibration_images,
        trigger=trigger,
    )

    optimizer = torch.optim.AdamW(
        channel.parameters(),
        lr=args.channel_lr,
        weight_decay=args.channel_wd,
    )

    save_json(
        {
            "method": "KDR-GC",
            "description": (
                "KDR fixed key trigger + shared-direction "
                "rank-1 gain channel on MLP c_proj writers"
            ),
            "args": vars(args),
            "device": str(device),
            "pretrained_path": pretrained_path,
            "init_checkpoint": args.init_checkpoint,
            "trigger_path": args.trigger_path,
            "output_dir": output_dir,
            "ckpt_path": ckpt_path,
            "channel_initial": channel.channel_stats(),
            "reader_calibration": calibration,
            "log_path": log_path,
        },
        config_path,
    )

    print(
        "[KDR-GC] output_dir:",
        output_dir,
        flush=True,
    )

    print(
        "[KDR-GC] channel:",
        channel.channel_stats(),
        flush=True,
    )

    print(
        "[KDR-GC] reader calibration:",
        calibration,
        flush=True,
    )

    global_step = 0
    start = time.time()
    last_epoch_records = []

    for epoch in range(args.epochs):
        epoch_records = []

        for batch_idx, batch in enumerate(train_loader):
            if (
                args.max_train_batches
                and batch_idx >= args.max_train_batches
            ):
                break

            batch = maybe_dictionarize(batch)

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            bd_images = apply_trigger(
                images[: args.bd_batch_size],
                trigger,
            )

            target_labels = torch.full(
                (bd_images.shape[0],),
                args.target_cls,
                dtype=torch.long,
                device=device,
            )

            channel_delta = channel.delta_dict()

            adv_state = compose_adv_state(
                task_image_encoder,
                channel_delta,
            )

            alpha = random.uniform(
                args.alpha_min,
                args.alpha_max,
            )

            eta = random.uniform(
                args.eta_min,
                args.eta_max,
            )

            drift_cache = sample_channel_drift(
                adv_state=adv_state,
                base_model=base_image_encoder,
                channel_weight_names=channel.weight_names,
                rho=args.drift_rho,
            )

            theta_atk_params = build_proxy_params(
                template_model=task_image_encoder,
                adv_state=adv_state,
                base_model=base_image_encoder,
                drift_cache=drift_cache,
                alpha=alpha,
                eta=eta,
            )

            clean_features = call_with_params(
                task_image_encoder,
                adv_state,
                images,
            )

            clean_logits = classification_logits(
                classification_head,
                clean_features,
            )

            clean_loss = F.cross_entropy(
                clean_logits,
                labels,
            )

            z_atk = call_with_params(
                task_image_encoder,
                theta_atk_params,
                bd_images,
            )

            logits_atk = classification_logits(
                classification_head,
                z_atk,
            )

            target_loss = F.cross_entropy(
                logits_atk,
                target_labels,
            )

            loss = (
                args.clean_weight * clean_loss
                + args.target_weight * target_loss
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            if (
                args.grad_clip
                and args.grad_clip > 0
            ):
                torch.nn.utils.clip_grad_norm_(
                    channel.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            with torch.no_grad():
                pred = logits_atk.argmax(dim=1)

                margin = target_margin(
                    logits_atk,
                    args.target_cls,
                )

                channel_stats = channel.channel_stats()

                record = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "alpha": float(alpha),
                    "eta": float(eta),
                    "loss": float(loss.item()),
                    "clean_loss": float(
                        clean_loss.item()
                    ),
                    "target_loss": float(
                        target_loss.item()
                    ),
                    "target_rate": float(
                        (
                            pred == args.target_cls
                        ).float().mean().item()
                    ),
                    "margin_mean": float(
                        margin.mean().item()
                    ),
                    "margin_min": float(
                        margin.min().item()
                    ),
                    "clean_acc": float(
                        (
                            clean_logits.argmax(dim=1)
                            == labels
                        ).float().mean().item()
                    ),
                    "channel_scales": (
                        channel_stats["scales"]
                    ),
                    "channel_delta_norms": (
                        channel_stats["delta_norms"]
                    ),
                }

            append_jsonl(
                record,
                log_path,
            )

            epoch_records.append(record)
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[KDR-GC] "
                    f"epoch={epoch}/{args.epochs} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss={record['loss']:.4f} "
                    f"clean={record['clean_loss']:.4f} "
                    f"target={record['target_loss']:.4f} "
                    f"rate={record['target_rate']:.3f} "
                    f"margin={record['margin_mean']:.4f} "
                    f"clean_acc={record['clean_acc']:.3f}",
                    flush=True,
                )

        last_epoch_records = epoch_records

        if epoch_records:
            epoch_summary = {
                "epoch": epoch,
                "steps": len(epoch_records),
                "elapsed_sec": (
                    time.time() - start
                ),
                "avg_loss": float(
                    sum(
                        x["loss"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "avg_clean_loss": float(
                    sum(
                        x["clean_loss"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "avg_target_loss": float(
                    sum(
                        x["target_loss"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "avg_target_rate": float(
                    sum(
                        x["target_rate"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "avg_margin": float(
                    sum(
                        x["margin_mean"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "avg_clean_acc": float(
                    sum(
                        x["clean_acc"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "channel": channel.channel_stats(),
            }

            append_jsonl(
                {"epoch_summary": epoch_summary},
                log_path,
            )

            print(
                "[KDR-GC] epoch_summary:",
                epoch_summary,
                flush=True,
            )

    materialize_checkpoint(
        args=args,
        channel=channel,
        ckpt_path=ckpt_path,
    )

    summary = {
        "method": "KDR-GC",
        "finetuned_path": ckpt_path,
        "trigger_path": args.trigger_path,
        "output_dir": output_dir,
        "config_path": config_path,
        "log_path": log_path,
        "elapsed_sec": time.time() - start,
        "global_steps": global_step,
        "channel_final": channel.channel_stats(),
        "last_epoch": {
            "avg_target_rate": (
                float(
                    sum(
                        x["target_rate"]
                        for x in last_epoch_records
                    )
                    / len(last_epoch_records)
                )
                if last_epoch_records
                else None
            ),
            "avg_margin": (
                float(
                    sum(
                        x["margin_mean"]
                        for x in last_epoch_records
                    )
                    / len(last_epoch_records)
                )
                if last_epoch_records
                else None
            ),
            "avg_clean_acc": (
                float(
                    sum(
                        x["clean_acc"]
                        for x in last_epoch_records
                    )
                    / len(last_epoch_records)
                )
                if last_epoch_records
                else None
            ),
        },
    }

    save_json(
        summary,
        summary_path,
    )

    print(
        "[KDR-GC] saved model:",
        ckpt_path,
        flush=True,
    )

    print(
        "[KDR-GC] saved summary:",
        summary_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
```

---

# 10. 新增 `run_kdr_gc_smoke.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/kdr_utils.py src/finetune_kdr_gain_channel.py

test -f ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy
test -f ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt
test -f ./checkpoints/ViT-B-32/zeroshot.pt

python3 src/finetune_kdr_gain_channel.py \
  --model ViT-B-32 \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy \
  --init-checkpoint ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  --save-root ./checkpoints \
  --method-name KDR_GC \
  --epochs 1 \
  --batch-size 64 \
  --bd-batch-size 32 \
  --channel-blocks 6,7,8,9,10 \
  --channel-lr 1e-3 \
  --channel-wd 1e-4 \
  --channel-init-ratio 0.01 \
  --grad-clip 1.0 \
  --alpha-min 0.2 \
  --alpha-max 1.0 \
  --eta-min 0.2 \
  --eta-max 1.0 \
  --drift-rho 0.25 \
  --clean-weight 1.0 \
  --target-weight 1.0 \
  --max-train-batches 5 \
  --seed 2026

echo "[KDR-GC-Smoke] done"
```

---

# 11. 新增 `run_kdr_gc_train.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/kdr_utils.py src/finetune_kdr_gain_channel.py

test -f ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy
test -f ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt
test -f ./checkpoints/ViT-B-32/zeroshot.pt

python3 src/finetune_kdr_gain_channel.py \
  --model ViT-B-32 \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy \
  --init-checkpoint ./checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  --save-root ./checkpoints \
  --method-name KDR_GC \
  --epochs 5 \
  --batch-size 128 \
  --bd-batch-size 64 \
  --channel-blocks 6,7,8,9,10 \
  --channel-lr 1e-3 \
  --channel-wd 1e-4 \
  --channel-init-ratio 0.01 \
  --grad-clip 1.0 \
  --alpha-min 0.2 \
  --alpha-max 1.0 \
  --eta-min 0.2 \
  --eta-max 1.0 \
  --drift-rho 0.25 \
  --clean-weight 1.0 \
  --target-weight 1.0 \
  --seed 2026

echo "[KDR-GC-Train] done"
```

---

# 12. 修改 `src/eval_submerge.py`

当前 GitHub 版本已经支持：

```text
KDR
KDR_COV
```

现在加入：

```text
KDR_GC
```

## 12.1 修改 `--attack-type`

找到：

```python
        "KDR",
        "KDR_COV",
        "BadMergingOn",
```

替换为：

```python
        "KDR",
        "KDR_COV",
        "KDR_GC",
        "BadMergingOn",
```

---

## 12.2 修改 `--trigger-source`

找到：

```python
        "KDR",
        "KDR_COV",
        "BadMergingOn",
```

替换为：

```python
        "KDR",
        "KDR_COV",
        "KDR_GC",
        "BadMergingOn",
```

---

## 12.3 checkpoint path 支持 KDR_GC

在 `checkpoint_path(args, dataset)` 中找到：

```python
        "KDR",
        "KDR_COV",
    ) and dataset == args.adversary_task:
```

替换为：

```python
        "KDR",
        "KDR_COV",
        "KDR_GC",
    ) and dataset == args.adversary_task:
```

这样：

```text
--attack-type KDR_GC
```

会读取：

```text
checkpoints/ViT-B-32/
CIFAR100_KDR_GC_CIFAR100_Tgt_1_L_22/
finetuned.pt
```

---

## 12.4 KDR_GC 复用原 KDR trigger

找到：

```python
def trigger_path(args):
    root = os.path.join("./trigger", args.model)

    source = args.attack_type if args.trigger_source == "attack" else args.trigger_source
```

替换为：

```python
def trigger_path(args):
    root = os.path.join("./trigger", args.model)

    source = (
        args.attack_type
        if args.trigger_source == "attack"
        else args.trigger_source
    )

    if source == "KDR_GC":
        source = "KDR"
```

后面的 trigger tuple 不需要加入 `KDR_GC`。

因为经过映射：

```text
KDR_GC -> KDR
```

最终读取：

```text
trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy
```

不要生成：

```text
KDR_GC_CIFAR100_Tgt_1_L_22.npy
```

KDR-GC 不改 trigger。

---

# 13. 修改 `src/main_regmean_badmergingon.py`

## 13.1 Trigger 路径

当前代码：

```python
elif attack_type in ('KDR', 'KDR_COV'):

    trigger_path = os.path.join(
        args.trigger_dir,
        f'{attack_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy'
    )

    trigger = np.load(trigger_path)
    trigger = torch.from_numpy(trigger)
```

替换为：

```python
elif attack_type in ('KDR', 'KDR_COV', 'KDR_GC'):

    trigger_type = (
        'KDR'
        if attack_type == 'KDR_GC'
        else attack_type
    )

    trigger_path = os.path.join(
        args.trigger_dir,
        f'{trigger_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy'
    )

    trigger = np.load(trigger_path)
    trigger = torch.from_numpy(trigger)
```

---

## 13.2 RegMean experiment name

当前：

```python
if attack_type in ('KDR', 'KDR_COV'):

    exp_name = f'{attack_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}'
```

替换为：

```python
if attack_type in ('KDR', 'KDR_COV', 'KDR_GC'):

    exp_name = (
        f'{attack_type}_{adversary_task}_'
        f'Tgt_{target_cls}_L_{patch_size}'
    )
```

这样 RegMean 会读取：

```text
CIFAR100_KDR_GC_CIFAR100_Tgt_1_L_22/finetuned.pt
```

---

# 14. 修改 `src/main_adamerging_badmergingon.py`

## 14.1 Trigger 路径

当前：

```python
elif attack_type in ('KDR', 'KDR_COV'):

    trigger_path = os.path.join(
        args.trigger_dir,
        f'{attack_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy'
    )

    trigger = np.load(trigger_path)
    trigger = torch.from_numpy(trigger)
```

替换为：

```python
elif attack_type in ('KDR', 'KDR_COV', 'KDR_GC'):

    trigger_type = (
        'KDR'
        if attack_type == 'KDR_GC'
        else attack_type
    )

    trigger_path = os.path.join(
        args.trigger_dir,
        f'{trigger_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy'
    )

    trigger = np.load(trigger_path)
    trigger = torch.from_numpy(trigger)
```

---

## 14.2 恶意 checkpoint 路径

当前：

```python
if attack_type in ('KDR', 'KDR_COV'):

    ckpt_name = os.path.join(
        args.save,
        dataset_name
        + f'_{attack_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}',
        'finetuned.pt'
    )
```

替换为：

```python
if attack_type in ('KDR', 'KDR_COV', 'KDR_GC'):

    ckpt_name = os.path.join(
        args.save,
        dataset_name
        + f'_{attack_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}',
        'finetuned.pt'
    )
```

---

## 14.3 AdaMerging lambda 保存名

当前：

```python
elif attack_type in ('KDR', 'KDR_COV'):

    torch.save(
        adamerging_mtl_model.lambdas_raw,
        os.path.join(
            adamerging_dir,
            f"{attack_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}_Epoch_{ep}.pt"
        )
    )
```

替换为：

```python
elif attack_type in ('KDR', 'KDR_COV', 'KDR_GC'):

    torch.save(
        adamerging_mtl_model.lambdas_raw,
        os.path.join(
            adamerging_dir,
            f"{attack_type}_{adversary_task}_Tgt_{target_cls}_L_{patch_size}_Epoch_{ep}.pt"
        )
    )
```

---

# 15. 新增 `run_kdr_gc_eval_asr.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs results/kdr_gc

python3 -m py_compile \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py

test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_GC_CIFAR100_Tgt_1_L_22/finetuned.pt

test -f \
  ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy

echo "[KDR-GC ASR] Task Arithmetic / TIES"

python3 src/eval_submerge.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --attack-type KDR_GC \
  --trigger-source KDR_GC \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --batch-size 128 \
  --out-dir ./results/kdr_gc

echo "[KDR-GC ASR] RegMean"

python3 src/main_regmean_badmergingon.py \
  --attack-type KDR_GC \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 5 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --batch-size 128

echo "[KDR-GC ASR] AdaMerging"

python3 src/main_adamerging_badmergingon.py \
  --attack-type KDR_GC \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --alpha 5 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --batch-size 128 \
  --adamerging-epochs "${ADAMERGING_EPOCHS:-500}"

echo "[KDR-GC ASR] done"
```

---

# 16. 新增 `KDR_GC_Implementation.md`

```markdown
# KDR-GC: Keyed Gain Channel

## 核心问题

KDR 已证明 target-free Edit-Key Patch 可以作为非对抗 key。

但逐样本诊断显示，成功攻击并不只依赖“识别 key”。

真正稳定的攻击表现为：

\[
\frac{
\|r_\ell^{\mathrm{trig}}\|
}{
\|r_\ell^{\mathrm{clean}}\|
}
\]

在中后层逐步增长。

RegMean 没有擦除恶意权重残差，却明显削弱这种触发条件放大。

## 核心方法

KDR-GC 不增加 amplification loss。

直接把恶意权重更新写成：

\[
\Delta W_\ell
=
s_\ell u v_\ell^\top.
\]

连续 MLP output projections 共享：

\[
u.
\]

每层有自己的 reader：

\[
v_\ell.
\]

因此：

\[
\Delta W_\ell h_\ell
=
s_\ell u(v_\ell^\top h_\ell).
\]

reader 判断 trigger-conditioned state。

writer 把读取结果重复写入同一个 residual-stream direction。

## Reader calibration

使用固定 KDR trigger。

对 clean / triggered input：

\[
q_{i,\ell}
=
h_\ell(T(x_i))
-
h_\ell(x_i).
\]

初始化：

\[
v_\ell
=
\operatorname{Normalize}
\left(
\frac{1}{B}
\sum_i
\frac{q_{i,\ell}}
{\|q_{i,\ell}\|}
\right).
\]

不使用 target supervision。

## Weight-space structure

当前默认：

```text
blocks = 6,7,8,9,10
writers = mlp.c_proj.weight
rank = 1
shared write direction = u
```

## Training objective

只保留：

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{clean}}
+
\mathcal{L}_{\mathrm{target}}.
\]

不使用：

```text
gain loss
margin loss
amplification loss
hard drift mining
cone loss
```

## Synthetic drift

保留 KDR 的未知背景思想。

但 synthetic drift 只施加在 channel writer weights。

训练 attack proxy：

\[
\theta_{\mathrm{atk}}
=
\theta_0
+
\alpha(\theta_{\mathrm{adv}}-\theta_0)
+
\eta\tilde{\delta}_{\mathrm{bg}}.
\]

## 第一轮实验

```bash
bash run_kdr_gc_smoke.sh
bash run_kdr_gc_train.sh
bash run_kdr_gc_eval_asr.sh
```

只看：

```text
TA
TIES
RegMean
AdaMerging
```

的 ASR。

KDR baseline：

```text
TA          99.98%
TIES        99.93%
RegMean     66.09%
AdaMerging  99.98%
```

第一判断：

```text
RegMean >= 85%
且 TA/TIES/AdaMerging 保持接近 100%
```

则继续 utility 和机制诊断。

如果：

```text
RegMean <= 70%
```

说明 rank-1 shared write channel 没解决条件增益问题。

不要调 amplification loss。

先重新分析 channel reader 是否在 merge 后被破坏。
```

---

# 17. 静态检查

执行：

```bash
python3 -m py_compile \
  src/kdr_utils.py \
  src/finetune_kdr_gain_channel.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py

bash -n run_kdr_gc_smoke.sh
bash -n run_kdr_gc_train.sh
bash -n run_kdr_gc_eval_asr.sh
```

---

# 18. 运行顺序

先：

```bash
bash run_kdr_gc_smoke.sh
```

smoke 重点看：

```text
reader calibration 不报错
blocks 6--10 均找到 mlp.c_proj.weight
target loss 能下降
target rate 有上升趋势
channel scales / delta norms 非 NaN
```

如果 smoke 正常：

```bash
bash run_kdr_gc_train.sh
```

训练完成后确认：

```bash
test -f \
  ./checkpoints/ViT-B-32/CIFAR100_KDR_GC_CIFAR100_Tgt_1_L_22/finetuned.pt
```

然后：

```bash
bash run_kdr_gc_eval_asr.sh
```

---

# 19. 第一轮实验只看 ASR

结果表：

| Method | TA | TIES | RegMean | AdaMerging |
|---|---:|---:|---:|---:|
| KDR | 99.98 | 99.93 | 66.09 | 99.98 |
| KDR-GC |  |  |  |  |

第一轮不要跑：

```text
utility
新的 amplification diagnostic
多 target
多 dataset
rank sweep
block sweep
learning-rate sweep
init-ratio sweep
```

先验证：

> **把恶意残差直接结构化为共享写入方向的低秩增益通道，能否修复 RegMean 下触发条件放大不足。**

---

# 20. 结果判据

## 强成功

```text
RegMean ASR >= 90%
TA/TIES/AdaMerging 接近 100%
```

说明 weight-space gain channel 基本成立。

## 有效

```text
RegMean ASR >= 85%
```

且其他方法不掉。

说明结构有明确价值，下一步做 utility 和增益诊断。

## 弱信号

```text
75% <= RegMean ASR < 85%
```

说明结构方向有帮助，但 reader 或 channel placement 仍可能不稳。

不要立即 sweep。

先诊断 merge 前后：

```text
shared write direction u 是否保留
reader v_l 是否被旋转
```

## 失败

```text
RegMean ASR <= 70%
```

则不继续给 KDR-GC 加 loss。

直接检查：

> RegMean 是否确实保留 shared write direction，但破坏 layer-specific reader。

---

# 21. Git 提交建议

只提交 KDR-GC 相关文件：

```bash
git add \
  src/finetune_kdr_gain_channel.py \
  run_kdr_gc_smoke.sh \
  run_kdr_gc_train.sh \
  run_kdr_gc_eval_asr.sh \
  KDR_GC_Implementation.md \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py

git status --short
```

不要使用：

```bash
git add .
```

确认没有误提交：

```text
MRC_patch_markdown.md
src/vis/*.png
历史 deleted docs
```

然后：

```bash
git commit -m "Add KDR keyed gain channel experiment"
git push origin submerge-pa-v1
```

---

# 22. 本次方法的一句话定义

\[
\boxed{
\text{KDR-GC 将恶意权重残差直接参数化为跨连续 Transformer 层共享写入方向的低秩通道，使固定的 target-free key 在各层被读取后反复向同一 residual-stream direction 写入，从权重结构上形成触发条件增益，而不是通过额外放大损失逼出该现象。}
}
\]
