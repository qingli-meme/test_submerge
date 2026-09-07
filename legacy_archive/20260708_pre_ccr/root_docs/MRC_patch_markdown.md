# MRC Patch for `qingli-meme/test_submerge`

目标：在当前 `test_submerge` 仓库上新增一套 **MRC（Merge-Readout Calibration）** 最小实现，用于验证一个更精确的失败模式：

> 恶意残差不是没有 survive，也不是 trigger key 被 merge drift 打散，而是本地 target-positive 的残差在 merged operating point 下发生了 **functional sign flip**。  
> 因此训练目标应从 local CE / local mapping 改成：在 sampled merged contexts 中，让恶意残差相对 clean merge 产生正的 target-subspace projection gain，并提升 target-vs-nontarget subspace margin。

本 patch **不改 BadMerging 的 trigger 设置，不改 BadMerging 的攻击设置，不改原始评估脚本**。  
只新增文件；不覆盖你当前未提交的诊断脚本。

---

## 0. 当前仓库状态与操作原则

你当前本地信息：

```bash
repo: https://github.com/qingli-meme/test_submerge
branch: submerge-pa-v1
last committed commit: 10c5924 Add PA relaxed projection experiments

uncommitted files include:
  src/diagnose_badmerging_stability.py
  src/diagnose_badmerging_subspace_distance.py
  some abandoned-method-related files
  several historical docs in deleted state
```

执行前先做：

```bash
cd /home/zlz422/BadMerging

git branch --show-current
git status --short
```

要求：

1. **不要 `git reset --hard`。**
2. **不要 `git checkout -- .`。**
3. **不要恢复或删除那些 historical deleted docs。**
4. **不要覆盖下面两个诊断脚本：**

```bash
src/diagnose_badmerging_stability.py
src/diagnose_badmerging_subspace_distance.py
```

5. 本 patch 只新增这些文件：

```bash
src/mrc_utils.py
src/finetune_mrc.py
src/eval_mrc_readout.py
run_mrc_smoke.sh
run_mrc_train.sh
run_mrc_eval.sh
MRC_Implementation.md
```

---

## 1. 方法定位

### 1.1 已有诊断结论

你已经跑出的关键现象：

```text
FOPATwoPhase_local:
  subspace_margin = 0.0456
  ASR = 100%

FOPATwoPhase_ta:
  subspace_margin = -0.1982
  ASR = 27.93%
  hard_ASR = 27.79%

FOPATwoPhase_ties:
  subspace_margin = -0.1713
  ASR = 34.96%
  hard_ASR = 34.58%

clean_ta + same trigger:
  ASR = 0.20%

clean_ties + same trigger:
  ASR = 0.59%

target_projection_gain:
  local - clean_local = +0.7854
  TA    - clean_ta    = -0.4109
  TIES  - clean_ties  = -1.7970
```

结论：

```text
local model 里 trigger 确实把表示推向 target subspace；
merged model 里同一 residual 不但没有继续增强 target projection，
反而把 target projection 拉低。
```

所以失败模式是：

```text
merge-induced functional sign flip
```

而不是：

```text
trigger key collapse
target subspace collapse
residual disappearance
```

### 1.2 MRC 的最小目标

MRC 不优化 local ASR。  
MRC 直接优化 merged context 中的 residual readout：

```text
clean merged model:
  θ_clean = Merge(θ0 + δ_clean_1, ..., θ0 + δ_clean_k)

attack merged model:
  θ_attack = Merge(θ0 + δ_adv, θ0 + δ_clean_1, ..., θ0 + δ_clean_k)

triggered features:
  z_clean  = h(θ_clean,  T(x))
  z_attack = h(θ_attack, T(x))

target projection gain:
  gain_t = score_t(z_attack) - score_t(z_clean)

target-vs-nontarget margin:
  margin_t = score_t(z_attack) - max_{c≠t} score_c(z_attack)
```

训练目标：

```text
gain_t > eps_gain
margin_t > eps_margin
clean utility preserved
residual budget controlled
```

这套目标直接对应当前诊断结果：  
**把 TA/TIES 下的 negative target_projection_gain 拉回 positive。**

---

## 2. 新增文件 1：`src/mrc_utils.py`

创建文件：

```bash
cat > src/mrc_utils.py <<'PY'
# ============================================================
# MRC Utilities
# ============================================================
# Method role:
#   Shared utilities for Merge-Readout Calibration (MRC).
#
# Design rationale:
#   MRC must train an uploadable malicious residual under merged
#   operating points. Therefore, the code needs differentiable
#   model merging. We use torch functional_call to evaluate a model
#   under merged parameters without copying weights into the module.
#
# Key diagnostics preserved in outputs:
#   - target_projection_gain
#   - subspace_margin
#   - clean CE
#   - residual norm
#
# This file does not modify any existing BadMerging/SubMerge source file.
# ============================================================

import json
import os
import random
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.func import functional_call as _functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call as _functional_call

# ---------------------------------------------------------------------
# Compatibility: the original BadMerging code often serializes complete
# model objects, not only plain state_dicts. PyTorch 2.6 changed the
# default weights_only behavior. Keep this local helper explicit.
# ---------------------------------------------------------------------

def torch_load(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


class EvalArgs:
    """Minimal args object expected by BadMerging eval/head utilities."""
    pass


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible diagnostics/training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    """Create a directory and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Mapping, path: str) -> None:
    """Save a JSON object with stable formatting."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def append_jsonl(obj: Mapping, path: str) -> None:
    """Append one JSON record for training logs."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def make_eval_args(args, dataset: str, device: Optional[str] = None):
    """Build the minimal object required by get_classification_head/eval."""
    evargs = EvalArgs()
    evargs.data_location = args.data_location
    evargs.batch_size = args.batch_size
    evargs.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    evargs.model = args.model
    evargs.cache_dir = getattr(args, "cache_dir", "")
    evargs.openclip_cachedir = getattr(args, "openclip_cachedir", "./open_clip")
    evargs.save = os.path.join(args.ckpt_dir, args.model)
    evargs.eval_datasets = [dataset]
    return evargs


def checkpoint_to_state_dict(path: str, map_location="cpu") -> OrderedDict:
    """
    Load a checkpoint as a plain state_dict.

    BadMerging checkpoints are usually serialized ImageEncoder objects.
    Some local scripts may save a dict with 'state_dict'. This helper
    handles both.
    """
    obj = torch_load(path, map_location=map_location)
    if hasattr(obj, "state_dict"):
        state = obj.state_dict()
    elif isinstance(obj, Mapping) and "state_dict" in obj:
        state = obj["state_dict"]
    elif isinstance(obj, Mapping):
        state = obj
    else:
        raise TypeError(f"Unsupported checkpoint format: {path}")

    out = OrderedDict()
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().clone()
    return out


def load_image_encoder(args, checkpoint_path: str, device: torch.device):
    """
    Load ImageEncoder from an existing checkpoint.

    This is intentionally imported lazily so that mrc_utils remains
    lightweight and unit-testable without importing the whole repo.
    """
    from src.modeling import ImageEncoder

    encoder = ImageEncoder(args, keep_lang=False).to(device)
    state = checkpoint_to_state_dict(checkpoint_path, map_location=device)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        print(f"[MRC] load_image_encoder missing keys: {len(missing)}")
    if unexpected:
        print(f"[MRC] load_image_encoder unexpected keys: {len(unexpected)}")
    return encoder


def save_image_encoder(encoder, path: str) -> None:
    """
    Save an ImageEncoder in the same style as the original BadMerging code.
    """
    ensure_dir(os.path.dirname(path))
    if hasattr(encoder, "save"):
        encoder.save(path)
    else:
        torch.save(encoder, path)


def clean_checkpoint_paths(
    ckpt_dir: str,
    model: str,
    clean_tasks: Sequence[str],
) -> List[str]:
    """Return checkpoint paths for clean task-specific models."""
    paths = []
    for task in clean_tasks:
        path = os.path.join(ckpt_dir, model, task, "finetuned.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing clean checkpoint for task={task}: {path}"
            )
        paths.append(path)
    return paths


def load_clean_states(
    ckpt_dir: str,
    model: str,
    clean_tasks: Sequence[str],
    map_location="cpu",
) -> Dict[str, OrderedDict]:
    """Load clean task checkpoints as state_dicts."""
    states = {}
    for task, path in zip(clean_tasks, clean_checkpoint_paths(ckpt_dir, model, clean_tasks)):
        states[task] = checkpoint_to_state_dict(path, map_location=map_location)
    return states


def load_trigger_patch(
    trigger_path: str,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load a trigger patch from .npy.

    Supports both patch-shaped arrays [3, L, L] and full-image arrays
    [3, 224, 224]. Full-image triggers are cropped from bottom-right.
    """
    arr = np.load(trigger_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected trigger array [C,H,W], got shape={arr.shape}")
    if arr.shape[0] != 3:
        raise ValueError(f"Expected 3-channel trigger, got shape={arr.shape}")

    if arr.shape[-2:] == (224, 224):
        arr = arr[:, -patch_size:, -patch_size:]
    elif arr.shape[-2:] != (patch_size, patch_size):
        raise ValueError(
            f"Trigger shape {arr.shape} is neither full 224 nor patch_size={patch_size}"
        )

    trigger = torch.tensor(arr, dtype=torch.float32, device=device)
    return trigger


def apply_trigger(images: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    """Apply a fixed bottom-right trigger patch to normalized images."""
    patched = images.clone()
    h, w = trigger.shape[-2:]
    patched[:, :, -h:, -w:] = trigger.unsqueeze(0).to(
        device=images.device, dtype=images.dtype
    )
    return patched


def trigger_to_backdoor_info(trigger: torch.Tensor, target_cls: int):
    """
    Build backdoor_info compatible with src.eval.eval_single_dataset.

    The original BadMerging utility expects a full applied patch and mask.
    """
    from src.utils import corner_mask_generation

    applied_patch, mask, _, _ = corner_mask_generation(
        trigger.detach().cpu().numpy(),
        image_size=(3, 224, 224),
    )
    return {
        "mask": torch.from_numpy(mask).float(),
        "applied_patch": torch.from_numpy(applied_patch).float(),
        "target_cls": int(target_cls),
    }


def get_head_weights(classification_head, feature_dim: Optional[int] = None) -> torch.Tensor:
    """
    Extract class weights from BadMerging ClassificationHead.

    The repo has used both 'weight' and 'weights' naming in different forks.
    This helper accepts either.
    """
    candidates = []
    for name in ["weight", "weights"]:
        if hasattr(classification_head, name):
            val = getattr(classification_head, name)
            if torch.is_tensor(val):
                candidates.append(val)
    if hasattr(classification_head, "linear") and hasattr(classification_head.linear, "weight"):
        candidates.append(classification_head.linear.weight)

    if not candidates:
        raise AttributeError("Cannot find classification head weights.")

    w = candidates[0].detach()
    if feature_dim is not None:
        if w.ndim != 2:
            raise ValueError(f"Expected 2D head weights, got {w.shape}")
        if w.shape[1] != feature_dim and w.shape[0] == feature_dim:
            w = w.t()
        if w.shape[1] != feature_dim:
            raise ValueError(
                f"Head weight shape {w.shape} incompatible with feature_dim={feature_dim}"
            )
    return w.float()


def normalized_class_scores(features: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine-style class scores.

    BadMerging's ClassificationHead normalizes features. MRC uses the same
    normalized geometry when measuring target-subspace readout.
    """
    z = F.normalize(features.float(), dim=-1)
    w = F.normalize(class_weights.to(z.device, dtype=z.dtype), dim=-1)
    return z @ w.t()


def gain_and_margin(
    z_attack: torch.Tensor,
    z_clean: torch.Tensor,
    class_weights: torch.Tensor,
    target_cls: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute MRC readout metrics.

    target_projection_gain:
        score_t(z_attack) - score_t(z_clean)

    subspace_margin:
        score_t(z_attack) - max_{c != t} score_c(z_attack)
    """
    scores_attack = normalized_class_scores(z_attack, class_weights)
    scores_clean = normalized_class_scores(z_clean, class_weights)

    target_attack = scores_attack[:, target_cls]
    target_clean = scores_clean[:, target_cls]
    target_gain = target_attack - target_clean.detach()

    masked = scores_attack.clone()
    masked[:, target_cls] = -1e9
    max_non_target = masked.max(dim=1).values
    margin = target_attack - max_non_target

    pred = scores_attack.argmax(dim=1)

    return {
        "target_attack": target_attack,
        "target_clean": target_clean.detach(),
        "target_gain": target_gain,
        "max_non_target": max_non_target,
        "margin": margin,
        "pred": pred,
    }


def _floating_state_value(state: Mapping[str, torch.Tensor], name: str, like: torch.Tensor) -> torch.Tensor:
    """Fetch a floating tensor from a state dict and move it to the target tensor device/dtype."""
    if name not in state:
        raise KeyError(f"Missing parameter in checkpoint state: {name}")
    return state[name].to(device=like.device, dtype=like.dtype)


def _task_delta(
    state: Mapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
    name: str,
    like: torch.Tensor,
) -> torch.Tensor:
    """Return θ_task - θ_base for one parameter name."""
    return _floating_state_value(state, name, like) - _floating_state_value(base_state, name, like)


def _ties_merge_delta(
    deltas: Sequence[torch.Tensor],
    density: float,
) -> torch.Tensor:
    """
    TIES-like delta merge for one parameter tensor.

    This is a compact implementation for MRC training/eval. It follows
    the TIES intuition:
      1. trim small-magnitude entries per task vector;
      2. elect an aggregate sign;
      3. average sign-consistent entries.

    Non-differentiability of topk/sign is acceptable here because MRC only
    needs gradients through retained malicious entries. This is used as a
    training proxy, not as a replacement for the official TIES evaluator.
    """
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0,1], got {density}")

    flats = [d.reshape(-1) for d in deltas]
    stacked = torch.stack(flats, dim=0)
    n, numel = stacked.shape
    k = max(1, int(round(density * numel)))

    masks = torch.zeros_like(stacked, dtype=torch.bool)
    abs_stacked = stacked.abs()
    for i in range(n):
        idx = torch.topk(abs_stacked[i], k=k, largest=True).indices
        masks[i, idx] = True

    trimmed = torch.where(masks, stacked, torch.zeros_like(stacked))
    elected = torch.sign(trimmed.sum(dim=0, keepdim=True))
    sign_match = torch.sign(trimmed) == elected
    aligned = torch.where(sign_match, trimmed, torch.zeros_like(trimmed))

    denom = (aligned != 0).sum(dim=0).clamp_min(1).to(aligned.dtype)
    merged = aligned.sum(dim=0) / denom
    return merged.reshape_as(deltas[0])


def build_functional_params(
    model: torch.nn.Module,
    base_state: Mapping[str, torch.Tensor],
    clean_states: Sequence[Mapping[str, torch.Tensor]],
    operator: str,
    scaling: float,
    ties_density: float,
    include_adv: bool,
) -> Dict[str, torch.Tensor]:
    """
    Build a differentiable merged-parameter dict for functional_call.

    include_adv=True:
        θ_merge = Merge(θ_adv, θ_clean_1, ..., θ_clean_k)

    include_adv=False:
        θ_merge = Merge(θ_clean_1, ..., θ_clean_k)

    The malicious model parameters are the current trainable parameters of
    `model`, so gradients flow back into the uploadable residual.
    """
    operator = operator.lower()
    params: Dict[str, torch.Tensor] = {}

    for name, p in model.named_parameters():
        if name not in base_state:
            params[name] = p
            continue

        base = _floating_state_value(base_state, name, p)
        deltas: List[torch.Tensor] = []

        if include_adv:
            deltas.append(p - base)

        for clean_state in clean_states:
            if name in clean_state:
                deltas.append(_task_delta(clean_state, base_state, name, p))

        if not deltas:
            params[name] = base
            continue

        if operator in ["ta", "task_arithmetic", "task-arithmetic"]:
            merged_delta = sum(deltas)
        elif operator in ["ties", "ties_merging", "ties-merging"]:
            merged_delta = _ties_merge_delta(deltas, density=ties_density)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        params[name] = base + float(scaling) * merged_delta

    # Buffers are copied from base. They do not require gradients.
    for name, b in model.named_buffers():
        if name in base_state and torch.is_tensor(base_state[name]):
            params[name] = base_state[name].to(device=b.device, dtype=b.dtype)
        else:
            params[name] = b

    return params


def call_with_params(
    model: torch.nn.Module,
    params: Mapping[str, torch.Tensor],
    images: torch.Tensor,
) -> torch.Tensor:
    """Forward ImageEncoder under a merged parameter dict."""
    return _functional_call(model, params, (images,))


@torch.no_grad()
def build_merged_state_dict(
    attack_state: Mapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
    clean_states: Sequence[Mapping[str, torch.Tensor]],
    operator: str,
    scaling: float,
    ties_density: float,
    include_adv: bool = True,
) -> OrderedDict:
    """
    Build an explicit merged state_dict for evaluation with eval_single_dataset.

    This mirrors build_functional_params but does not require gradients.
    """
    operator = operator.lower()
    out = OrderedDict()

    for name, base_v in base_state.items():
        if not torch.is_tensor(base_v):
            continue

        if base_v.dtype in [torch.int64, torch.int32, torch.uint8, torch.bool]:
            out[name] = base_v.detach().clone()
            continue

        base = base_v.detach().clone().float()
        deltas: List[torch.Tensor] = []

        if include_adv:
            if name in attack_state and torch.is_tensor(attack_state[name]):
                deltas.append(attack_state[name].detach().float() - base)

        for clean_state in clean_states:
            if name in clean_state and torch.is_tensor(clean_state[name]):
                deltas.append(clean_state[name].detach().float() - base)

        if not deltas:
            out[name] = base_v.detach().clone()
            continue

        if operator in ["ta", "task_arithmetic", "task-arithmetic"]:
            merged_delta = sum(deltas)
        elif operator in ["ties", "ties_merging", "ties-merging"]:
            merged_delta = _ties_merge_delta(deltas, density=ties_density)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        merged = base + float(scaling) * merged_delta
        out[name] = merged.to(dtype=base_v.dtype)

    return out


def split_csv(value: str) -> List[str]:
    """Split comma-separated CLI strings."""
    return [x.strip() for x in value.split(",") if x.strip()]


def select_clean_contexts(
    clean_states_by_task: Mapping[str, Mapping[str, torch.Tensor]],
    context_size: int,
) -> Tuple[List[str], List[Mapping[str, torch.Tensor]]]:
    """Randomly sample clean task states for one merged context."""
    tasks = list(clean_states_by_task.keys())
    if context_size <= 0 or context_size >= len(tasks):
        selected = tasks
    else:
        selected = random.sample(tasks, context_size)
    return selected, [clean_states_by_task[t] for t in selected]


def residual_l2_norm(
    model: torch.nn.Module,
    base_state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Compute ||θ_adv - θ0||_2 over floating parameters."""
    total = None
    for name, p in model.named_parameters():
        if name not in base_state:
            continue
        base = _floating_state_value(base_state, name, p)
        val = (p - base).float().pow(2).sum()
        total = val if total is None else total + val
    if total is None:
        return torch.zeros([], device=next(model.parameters()).device)
    return torch.sqrt(total + 1e-12)
PY
```

---

## 3. 新增文件 2：`src/finetune_mrc.py`

创建文件：

```bash
cat > src/finetune_mrc.py <<'PY'
# ============================================================
# Section 3: Merge-Readout Calibration Training
# ============================================================
# Method role:
#   Train one uploadable malicious model under the original BadMerging
#   one-malicious-model setting, but replace local trigger-target CE
#   with merged-context readout calibration.
#
# Core objective:
#   For sampled clean merge contexts, optimize the malicious residual so
#   that the attack merged model increases target-subspace projection over
#   the clean merged baseline:
#
#       gain_t = score_t(h_attack(T(x))) - score_t(h_clean(T(x)))
#
#   and has positive target-vs-nontarget margin:
#
#       margin_t = score_t(h_attack(T(x))) - max_{c != t} score_c(h_attack(T(x)))
#
# Why this is necessary:
#   Diagnostics showed that local target-positive residuals can become
#   target-negative under TA/TIES merged operating points. MRC therefore
#   calibrates the residual's functional sign under merge, rather than
#   optimizing local ASR.
# ============================================================

import argparse
import os
import random
import time
from typing import Dict

import torch
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head

from src.mrc_utils import (
    append_jsonl,
    apply_trigger,
    build_functional_params,
    call_with_params,
    clean_checkpoint_paths,
    ensure_dir,
    gain_and_margin,
    get_head_weights,
    load_clean_states,
    load_image_encoder,
    load_trigger_patch,
    make_eval_args,
    residual_l2_norm,
    save_image_encoder,
    save_json,
    select_clean_contexts,
    set_seed,
    split_csv,
    torch_load,
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
    "STL10": 5,
    "CIFAR100": 5,
    "Flowers": 251,
    "PETS": 77,
    "ImageNet100": 3,
}


def parse_args():
    p = argparse.ArgumentParser("MRC: Merge-Readout Calibration")

    # ---- Repository / dataset ----
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--openclip-cachedir", default="./open_clip")
    p.add_argument("--adversary-task", default="CIFAR100")
    p.add_argument("--target-cls", type=int, default=1)

    # ---- Attack assets ----
    p.add_argument("--patch-size", type=int, default=22)
    p.add_argument("--trigger-path", required=True)
    p.add_argument("--init-checkpoint", default=None,
                   help="Initial malicious checkpoint. If omitted, use zeroshot.pt.")
    p.add_argument("--method-name", default="MRC")

    # ---- Clean merge context bank ----
    p.add_argument(
        "--clean-tasks",
        default="Cars,SUN397,PETS,EuroSAT,GTSRB,SVHN",
        help="Comma-separated clean checkpoints used as merge drift bank.",
    )
    p.add_argument(
        "--context-size",
        type=int,
        default=0,
        help="Number of clean task vectors sampled per step. 0 = use all clean tasks.",
    )
    p.add_argument(
        "--operator-bank",
        default="ta",
        help="Comma-separated operators used during training: ta,ties. Default ta for speed.",
    )
    p.add_argument("--ta-scaling", type=float, default=0.3)
    p.add_argument("--ties-density", type=float, default=0.2)

    # ---- Optimization ----
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bd-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # ---- MRC loss weights ----
    p.add_argument("--gain-weight", type=float, default=1.0)
    p.add_argument("--margin-weight", type=float, default=1.0)
    p.add_argument("--merged-ce-weight", type=float, default=0.25)
    p.add_argument("--clean-weight", type=float, default=1.0)
    p.add_argument("--local-clean-weight", type=float, default=0.25)
    p.add_argument("--residual-weight", type=float, default=0.0)

    # ---- MRC thresholds ----
    p.add_argument(
        "--gain-eps",
        type=float,
        default=0.05,
        help="Minimum positive target projection gain required under merged context.",
    )
    p.add_argument(
        "--margin-eps",
        type=float,
        default=0.02,
        help="Minimum target-vs-nontarget margin under merged context.",
    )

    # ---- Runtime / output ----
    p.add_argument("--save-root", default="./checkpoints")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="For smoke test. 0 means full epoch.")
    p.add_argument("--save-every-epoch", action="store_true")
    return p.parse_args()


def build_output_dir(args) -> str:
    run_name = (
        f"{args.adversary_task}_{args.method_name}_"
        f"{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    )
    out = os.path.join(args.save_root, args.model, run_name)
    ensure_dir(out)
    return out


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.epochs = args.epochs if args.epochs is not None else EPOCHS.get(args.adversary_task, 5)

    clean_tasks = split_csv(args.clean_tasks)
    operator_bank = [op.lower() for op in split_csv(args.operator_bank)]
    for op in operator_bank:
        if op not in ["ta", "ties", "task_arithmetic", "task-arithmetic"]:
            raise ValueError(f"Unsupported operator in --operator-bank: {op}")

    output_dir = build_output_dir(args)
    log_path = os.path.join(output_dir, "train_log.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    config_path = os.path.join(output_dir, "config.json")

    zeroshot_path = os.path.join(args.ckpt_dir, args.model, "zeroshot.pt")
    init_checkpoint = args.init_checkpoint or zeroshot_path
    if not os.path.exists(init_checkpoint):
        raise FileNotFoundError(f"Missing init checkpoint: {init_checkpoint}")
    if not os.path.exists(args.trigger_path):
        raise FileNotFoundError(f"Missing trigger: {args.trigger_path}")

    # Validate clean checkpoint paths early.
    clean_checkpoint_paths(args.ckpt_dir, args.model, clean_tasks)

    # Load trainable malicious encoder.
    image_encoder = load_image_encoder(args, init_checkpoint, device=device)
    image_encoder.train()

    # Load immutable states.
    base_state = torch_load(zeroshot_path, map_location="cpu").state_dict()
    base_state = {k: v.detach().cpu() for k, v in base_state.items() if torch.is_tensor(v)}
    clean_states = load_clean_states(
        args.ckpt_dir, args.model, clean_tasks, map_location="cpu"
    )

    # Dataset/head.
    evargs = make_eval_args(args, args.adversary_task, device=str(device))
    classification_head = get_classification_head(evargs, args.adversary_task).to(device)
    classification_head.eval()
    for p in classification_head.parameters():
        p.requires_grad_(False)

    preprocess = image_encoder.train_preprocess
    train_dataset, train_loader = get_dataset(
        args.adversary_task,
        "train",
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    trigger = load_trigger_patch(args.trigger_path, args.patch_size, device=device)
    trigger.requires_grad_(False)

    # Determine feature dimension for robust head weight extraction.
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_batch = maybe_dictionarize(sample_batch)
        sample_images = sample_batch["images"][:2].to(device)
        sample_feat = image_encoder(sample_images)
    class_weights = get_head_weights(classification_head, feature_dim=sample_feat.shape[-1]).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in image_encoder.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.wd,
    )

    save_json(
        {
            "method": "MRC",
            "rationale": (
                "Train uploadable residual by enforcing positive target-subspace "
                "projection gain under sampled merged contexts."
            ),
            "args": vars(args),
            "output_dir": output_dir,
            "zeroshot_path": zeroshot_path,
            "init_checkpoint": init_checkpoint,
            "clean_tasks": clean_tasks,
            "operator_bank": operator_bank,
        },
        config_path,
    )

    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        image_encoder.train()
        agg: Dict[str, float] = {
            "loss_total": 0.0,
            "loss_gain": 0.0,
            "loss_margin": 0.0,
            "loss_merged_ce": 0.0,
            "loss_clean": 0.0,
            "loss_local_clean": 0.0,
            "target_gain": 0.0,
            "subspace_margin": 0.0,
            "merged_trigger_acc": 0.0,
        }
        n_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            bd_images = apply_trigger(images[: args.bd_batch_size], trigger)
            target_labels = torch.full(
                (bd_images.shape[0],),
                int(args.target_cls),
                dtype=torch.long,
                device=device,
            )

            selected_tasks, selected_clean_states = select_clean_contexts(
                clean_states, args.context_size
            )
            operator = random.choice(operator_bank)

            attack_params = build_functional_params(
                model=image_encoder,
                base_state=base_state,
                clean_states=selected_clean_states,
                operator=operator,
                scaling=args.ta_scaling,
                ties_density=args.ties_density,
                include_adv=True,
            )
            clean_params = build_functional_params(
                model=image_encoder,
                base_state=base_state,
                clean_states=selected_clean_states,
                operator=operator,
                scaling=args.ta_scaling,
                ties_density=args.ties_density,
                include_adv=False,
            )

            z_attack = call_with_params(image_encoder, attack_params, bd_images)
            with torch.no_grad():
                z_clean = call_with_params(image_encoder, clean_params, bd_images)

            metrics = gain_and_margin(
                z_attack=z_attack,
                z_clean=z_clean,
                class_weights=class_weights,
                target_cls=args.target_cls,
            )

            loss_gain = F.relu(args.gain_eps - metrics["target_gain"]).mean()
            loss_margin = F.relu(args.margin_eps - metrics["margin"]).mean()

            logits_attack = classification_head(z_attack)
            loss_merged_ce = F.cross_entropy(logits_attack, target_labels)

            z_attack_clean = call_with_params(image_encoder, attack_params, images)
            logits_clean = classification_head(z_attack_clean)
            loss_clean = F.cross_entropy(logits_clean, labels)

            local_logits = classification_head(image_encoder(images))
            loss_local_clean = F.cross_entropy(local_logits, labels)

            res_norm = residual_l2_norm(image_encoder, base_state)

            loss = (
                args.gain_weight * loss_gain
                + args.margin_weight * loss_margin
                + args.merged_ce_weight * loss_merged_ce
                + args.clean_weight * loss_clean
                + args.local_clean_weight * loss_local_clean
                + args.residual_weight * res_norm
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred = metrics["pred"]
                trigger_acc = (pred == args.target_cls).float().mean().item()

            record = {
                "epoch": epoch,
                "batch": batch_idx,
                "global_step": global_step,
                "operator": operator,
                "selected_tasks": selected_tasks,
                "loss_total": float(loss.item()),
                "loss_gain": float(loss_gain.item()),
                "loss_margin": float(loss_margin.item()),
                "loss_merged_ce": float(loss_merged_ce.item()),
                "loss_clean": float(loss_clean.item()),
                "loss_local_clean": float(loss_local_clean.item()),
                "residual_l2": float(res_norm.detach().item()),
                "target_gain_mean": float(metrics["target_gain"].mean().detach().item()),
                "target_gain_min": float(metrics["target_gain"].min().detach().item()),
                "subspace_margin_mean": float(metrics["margin"].mean().detach().item()),
                "subspace_margin_min": float(metrics["margin"].min().detach().item()),
                "merged_trigger_acc_batch": float(trigger_acc),
            }
            append_jsonl(record, log_path)

            for k in agg:
                if k == "target_gain":
                    agg[k] += record["target_gain_mean"]
                elif k == "subspace_margin":
                    agg[k] += record["subspace_margin_mean"]
                elif k == "merged_trigger_acc":
                    agg[k] += record["merged_trigger_acc_batch"]
                else:
                    agg[k] += record[k]
            n_steps += 1
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[MRC] "
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                    f"op={operator} tasks={selected_tasks} "
                    f"loss={record['loss_total']:.4f} "
                    f"gain={record['target_gain_mean']:.4f} "
                    f"margin={record['subspace_margin_mean']:.4f} "
                    f"trig_acc={trigger_acc:.4f}"
                )

        epoch_record = {f"epoch_avg_{k}": v / max(n_steps, 1) for k, v in agg.items()}
        epoch_record.update({"epoch": epoch, "steps": n_steps, "elapsed_sec": time.time() - start_time})
        append_jsonl({"epoch_summary": epoch_record}, log_path)
        print(f"[MRC] epoch summary: {epoch_record}")

        if args.save_every_epoch:
            ep_path = os.path.join(output_dir, f"finetuned_epoch_{epoch}.pt")
            save_image_encoder(image_encoder, ep_path)

    ft_path = os.path.join(output_dir, "finetuned.pt")
    save_image_encoder(image_encoder, ft_path)

    summary = {
        "method": "MRC",
        "finetuned_path": ft_path,
        "trigger_path": args.trigger_path,
        "output_dir": output_dir,
        "final_epoch": args.epochs - 1,
        "global_steps": global_step,
        "train_log": log_path,
        "config": config_path,
    }
    save_json(summary, summary_path)
    print(f"[MRC] saved model: {ft_path}")
    print(f"[MRC] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
PY
```

---

## 4. 新增文件 3：`src/eval_mrc_readout.py`

创建文件：

```bash
cat > src/eval_mrc_readout.py <<'PY'
# ============================================================
# Section 4: MRC Evaluation and Readout Diagnostics
# ============================================================
# Method role:
#   Evaluate an MRC-trained malicious checkpoint under TA/TIES merged
#   contexts and report both conventional attack metrics and readout
#   metrics.
#
# Metrics:
#   - clean accuracy
#   - ASR
#   - target_projection_gain
#   - subspace_margin
#
# Why this is necessary:
#   MRC is motivated by functional sign flip. Therefore, ASR alone is
#   insufficient. The evaluation must show whether target_projection_gain
#   changed from negative to positive under merged contexts.
# ============================================================

import argparse
import os
from collections import OrderedDict
from typing import Dict, List

import torch

import sys
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.modeling import ImageEncoder

from src.mrc_utils import (
    apply_trigger,
    build_merged_state_dict,
    call_with_params,
    build_functional_params,
    checkpoint_to_state_dict,
    clean_checkpoint_paths,
    gain_and_margin,
    get_head_weights,
    load_clean_states,
    load_image_encoder,
    load_trigger_patch,
    make_eval_args,
    save_json,
    select_clean_contexts,
    set_seed,
    split_csv,
    torch_load,
    trigger_to_backdoor_info,
)


def parse_args():
    p = argparse.ArgumentParser("Evaluate MRC merged readout")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--openclip-cachedir", default="./open_clip")
    p.add_argument("--adversary-task", default="CIFAR100")
    p.add_argument("--target-cls", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=22)
    p.add_argument("--attack-checkpoint", required=True)
    p.add_argument("--trigger-path", required=True)
    p.add_argument("--clean-tasks", default="Cars,SUN397,PETS,EuroSAT,GTSRB,SVHN")
    p.add_argument("--operators", default="ta,ties")
    p.add_argument("--ta-scaling", type=float, default=0.3)
    p.add_argument("--ties-density", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-readout-batches", type=int, default=20)
    p.add_argument("--output-dir", default="./analysis/mrc_eval")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


@torch.no_grad()
def load_eval_encoder(args, state_dict: OrderedDict, device: torch.device):
    enc = ImageEncoder(args, keep_lang=False).to(device)
    missing, unexpected = enc.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[MRC-Eval] merged encoder missing keys: {len(missing)}")
    if unexpected:
        print(f"[MRC-Eval] merged encoder unexpected keys: {len(unexpected)}")
    enc.eval()
    return enc


@torch.no_grad()
def conventional_eval(args, encoder, trigger, dataset, device):
    evargs = make_eval_args(args, dataset, device=str(device))
    clean = eval_single_dataset(encoder, dataset, evargs, backdoor_info=None)
    bd_info = trigger_to_backdoor_info(trigger, args.target_cls)
    bd = eval_single_dataset(encoder, dataset, evargs, backdoor_info=bd_info)

    return {
        "clean_acc": 100.0 * float(clean["top1"]),
        "asr": 100.0 * float(bd["backdoored_acc"]),
        "backdoored_cnt": int(bd.get("backdoored_cnt", 0)),
        "non_target_cnt": int(bd.get("non_target_cnt", 0)),
    }


@torch.no_grad()
def readout_eval(args, attack_model, base_state, clean_states_for_context, class_weights, trigger, device):
    """
    Compute target_projection_gain and subspace_margin on test batches.

    Uses functional_call for both:
      attack merged model = Merge(attack, clean context)
      clean merged model  = Merge(clean context)
    """
    preprocess = attack_model.val_preprocess
    _, loader = get_dataset(
        args.adversary_task,
        "test",
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    operator_records: Dict[str, Dict[str, float]] = {}
    for operator in split_csv(args.operators):
        attack_params = build_functional_params(
            model=attack_model,
            base_state=base_state,
            clean_states=clean_states_for_context,
            operator=operator,
            scaling=args.ta_scaling,
            ties_density=args.ties_density,
            include_adv=True,
        )
        clean_params = build_functional_params(
            model=attack_model,
            base_state=base_state,
            clean_states=clean_states_for_context,
            operator=operator,
            scaling=args.ta_scaling,
            ties_density=args.ties_density,
            include_adv=False,
        )

        vals = {
            "target_gain_sum": 0.0,
            "target_gain_min": 1e9,
            "subspace_margin_sum": 0.0,
            "subspace_margin_min": 1e9,
            "trigger_pred_target_sum": 0.0,
            "n": 0,
        }

        for bidx, batch in enumerate(loader):
            if args.max_readout_batches and bidx >= args.max_readout_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            bd_images = apply_trigger(images, trigger)

            z_attack = call_with_params(attack_model, attack_params, bd_images)
            z_clean = call_with_params(attack_model, clean_params, bd_images)

            m = gain_and_margin(z_attack, z_clean, class_weights, args.target_cls)
            n = images.shape[0]

            vals["target_gain_sum"] += float(m["target_gain"].sum().item())
            vals["target_gain_min"] = min(vals["target_gain_min"], float(m["target_gain"].min().item()))
            vals["subspace_margin_sum"] += float(m["margin"].sum().item())
            vals["subspace_margin_min"] = min(vals["subspace_margin_min"], float(m["margin"].min().item()))
            vals["trigger_pred_target_sum"] += float((m["pred"] == args.target_cls).float().sum().item())
            vals["n"] += n

        n = max(vals["n"], 1)
        operator_records[operator] = {
            "target_projection_gain_mean": vals["target_gain_sum"] / n,
            "target_projection_gain_min": vals["target_gain_min"],
            "subspace_margin_mean": vals["subspace_margin_sum"] / n,
            "subspace_margin_min": vals["subspace_margin_min"],
            "readout_target_rate": vals["trigger_pred_target_sum"] / n,
            "num_readout_samples": vals["n"],
        }

    return operator_records


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clean_tasks = split_csv(args.clean_tasks)
    operators = split_csv(args.operators)

    clean_checkpoint_paths(args.ckpt_dir, args.model, clean_tasks)

    zeroshot_path = os.path.join(args.ckpt_dir, args.model, "zeroshot.pt")
    base_obj = torch_load(zeroshot_path, map_location="cpu")
    base_state = base_obj.state_dict()
    base_state = {k: v.detach().cpu() for k, v in base_state.items() if torch.is_tensor(v)}

    attack_state = checkpoint_to_state_dict(args.attack_checkpoint, map_location="cpu")
    clean_states_by_task = load_clean_states(args.ckpt_dir, args.model, clean_tasks, map_location="cpu")
    clean_states_for_context = [clean_states_by_task[t] for t in clean_tasks]

    attack_model = load_image_encoder(args, args.attack_checkpoint, device=device)
    attack_model.eval()

    trigger = load_trigger_patch(args.trigger_path, args.patch_size, device=device)

    evargs = make_eval_args(args, args.adversary_task, device=str(device))
    head = get_classification_head(evargs, args.adversary_task).to(device)
    head.eval()

    # Infer feature dimension.
    preprocess = attack_model.val_preprocess
    _, tmp_loader = get_dataset(
        args.adversary_task,
        "test",
        preprocess,
        location=args.data_location,
        batch_size=2,
    )
    tmp_batch = maybe_dictionarize(next(iter(tmp_loader)))
    tmp_feat = attack_model(tmp_batch["images"].to(device))
    class_weights = get_head_weights(head, feature_dim=tmp_feat.shape[-1]).to(device)

    results = {
        "config": vars(args),
        "attack_checkpoint": args.attack_checkpoint,
        "trigger_path": args.trigger_path,
        "operators": {},
        "readout": {},
    }

    for operator in operators:
        print(f"[MRC-Eval] conventional eval operator={operator}")

        merged_state = build_merged_state_dict(
            attack_state=attack_state,
            base_state=base_state,
            clean_states=clean_states_for_context,
            operator=operator,
            scaling=args.ta_scaling,
            ties_density=args.ties_density,
            include_adv=True,
        )
        merged_encoder = load_eval_encoder(args, merged_state, device)
        results["operators"][operator] = conventional_eval(
            args, merged_encoder, trigger, args.adversary_task, device
        )

    print("[MRC-Eval] readout diagnostics")
    results["readout"] = readout_eval(
        args=args,
        attack_model=attack_model,
        base_state=base_state,
        clean_states_for_context=clean_states_for_context,
        class_weights=class_weights,
        trigger=trigger,
        device=device,
    )

    out_path = os.path.join(args.output_dir, "mrc_eval_results.json")
    save_json(results, out_path)
    print(f"[MRC-Eval] saved: {out_path}")
    print(results)


if __name__ == "__main__":
    main()
PY
```

---

## 5. 新增文件 4：`run_mrc_smoke.sh`

创建 smoke test 脚本：

```bash
cat > run_mrc_smoke.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/mrc_utils.py src/finetune_mrc.py src/eval_mrc_readout.py

python3 src/finetune_mrc.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/On_CIFAR100_Tgt_1_L_22.npy \
  --init-checkpoint ./checkpoints/ViT-B-32/CIFAR100_On_CIFAR100_Tgt_1_L_22/finetuned.pt \
  --method-name MRC_Smoke \
  --clean-tasks Cars,SUN397,PETS,EuroSAT,GTSRB,SVHN \
  --operator-bank ta \
  --context-size 3 \
  --ta-scaling 0.3 \
  --epochs 1 \
  --max-train-batches 5 \
  --batch-size 64 \
  --bd-batch-size 32 \
  --lr 1e-6 \
  --gain-weight 1.0 \
  --margin-weight 1.0 \
  --merged-ce-weight 0.25 \
  --clean-weight 1.0 \
  --local-clean-weight 0.25 \
  --gain-eps 0.05 \
  --margin-eps 0.02 \
  --save-root ./checkpoints \
  --seed 2026

echo "[MRC-Smoke] done"
SH

chmod +x run_mrc_smoke.sh
```

---

## 6. 新增文件 5：`run_mrc_train.sh`

创建完整训练脚本：

```bash
cat > run_mrc_train.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/mrc_utils.py src/finetune_mrc.py src/eval_mrc_readout.py

python3 src/finetune_mrc.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/On_CIFAR100_Tgt_1_L_22.npy \
  --init-checkpoint ./checkpoints/ViT-B-32/CIFAR100_On_CIFAR100_Tgt_1_L_22/finetuned.pt \
  --method-name MRC_On_CIFAR100 \
  --clean-tasks Cars,SUN397,PETS,EuroSAT,GTSRB,SVHN \
  --operator-bank ta \
  --context-size 3 \
  --ta-scaling 0.3 \
  --ties-density 0.2 \
  --epochs 3 \
  --batch-size 128 \
  --bd-batch-size 64 \
  --lr 1e-6 \
  --wd 0.05 \
  --grad-clip 1.0 \
  --gain-weight 1.0 \
  --margin-weight 1.0 \
  --merged-ce-weight 0.25 \
  --clean-weight 1.0 \
  --local-clean-weight 0.25 \
  --residual-weight 0.0 \
  --gain-eps 0.05 \
  --margin-eps 0.02 \
  --save-root ./checkpoints \
  --save-every-epoch \
  --seed 2026

echo "[MRC-Train] done"
SH

chmod +x run_mrc_train.sh
```

说明：

- 默认训练只用 `ta`，因为 TIES-style differentiable proxy 显存和时间开销更大。
- 如果 TA 成功把 target gain 拉正，再跑：

```bash
sed -i 's/--operator-bank ta/--operator-bank ta,ties/g' run_mrc_train.sh
```

或者直接命令行改为：

```bash
--operator-bank ta,ties
```

---

## 7. 新增文件 6：`run_mrc_eval.sh`

创建评估脚本：

```bash
cat > run_mrc_eval.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ATTACK_CKPT=${1:-"./checkpoints/ViT-B-32/CIFAR100_MRC_On_CIFAR100_CIFAR100_Tgt_1_L_22/finetuned.pt"}

python3 -m py_compile src/mrc_utils.py src/eval_mrc_readout.py

python3 src/eval_mrc_readout.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --attack-checkpoint "${ATTACK_CKPT}" \
  --trigger-path ./trigger/ViT-B-32/On_CIFAR100_Tgt_1_L_22.npy \
  --clean-tasks Cars,SUN397,PETS,EuroSAT,GTSRB,SVHN \
  --operators ta,ties \
  --ta-scaling 0.3 \
  --ties-density 0.2 \
  --batch-size 128 \
  --max-readout-batches 20 \
  --output-dir ./analysis/mrc_eval \
  --seed 2026

echo "[MRC-Eval] done"
SH

chmod +x run_mrc_eval.sh
```

---

## 8. 新增文件 7：`MRC_Implementation.md`

创建说明文档：

```bash
cat > MRC_Implementation.md <<'MD'
# MRC: Merge-Readout Calibration

## Core diagnosis

Previous diagnostics indicate:

- local malicious model has high ASR;
- trigger keys are stable across merged contexts;
- target prototype/subspace is stable across merged contexts;
- nevertheless, the malicious residual's target projection gain becomes negative under TA/TIES merged contexts.

Therefore, the failure is not residual disappearance, trigger-key collapse, or target-subspace collapse.  
The failure is merge-induced functional sign flip.

## Method

MRC trains the uploadable malicious residual under sampled merged operating points.

For each batch:

1. sample clean task vectors;
2. construct clean merged model;
3. construct attack merged model;
4. apply the fixed BadMerging trigger;
5. compute target projection gain:
   `score_t(z_attack) - score_t(z_clean)`;
6. compute target-vs-nontarget margin:
   `score_t(z_attack) - max_{c != t} score_c(z_attack)`;
7. optimize:
   - positive target gain;
   - positive subspace margin;
   - clean utility preservation.

## Files

```text
src/mrc_utils.py
src/finetune_mrc.py
src/eval_mrc_readout.py
run_mrc_smoke.sh
run_mrc_train.sh
run_mrc_eval.sh
```

## Smoke test

```bash
bash run_mrc_smoke.sh
```

## Full train

```bash
bash run_mrc_train.sh
```

## Eval

```bash
bash run_mrc_eval.sh
```

or:

```bash
bash run_mrc_eval.sh ./checkpoints/ViT-B-32/CIFAR100_MRC_On_CIFAR100_CIFAR100_Tgt_1_L_22/finetuned.pt
```

## Expected diagnostic direction

The first criterion is not raw ASR.  
The first criterion is whether the previously negative readout metrics become positive:

```text
target_projection_gain:
  TA:   negative -> positive
  TIES: negative -> less negative / positive

subspace_margin:
  TA:   negative -> positive
  TIES: negative -> less negative / positive
```

If projection gain becomes positive but ASR is still weak, increase `--merged-ce-weight` moderately.

If clean accuracy collapses, reduce:

```text
--merged-ce-weight
--gain-weight
--margin-weight
```

and increase:

```text
--clean-weight
--local-clean-weight
```

## Recommended ablation grid

```bash
# gain/margin threshold
--gain-eps 0.00 / 0.05 / 0.10
--margin-eps 0.00 / 0.02 / 0.05

# loss weights
--merged-ce-weight 0.0 / 0.25 / 0.5
--clean-weight 1.0 / 2.0

# operator bank
--operator-bank ta
--operator-bank ta,ties

# context size
--context-size 1 / 3 / 0
```

## Failure diagnosis

### Case 1: target gain remains negative

The residual is still functionally inverted.  
Increase `--gain-weight`; reduce `--merged-ce-weight`; try smaller `--lr`.

### Case 2: gain positive but margin negative

The residual increases target projection but also increases a stronger non-target direction.  
Increase `--margin-weight`.

### Case 3: gain and margin positive but ASR low

The subspace objective is working but the final classifier boundary is not crossed.  
Increase `--merged-ce-weight` from 0.25 to 0.5.

### Case 4: clean accuracy collapses

The residual is too invasive.  
Increase `--clean-weight`, reduce `--lr`, reduce `--merged-ce-weight`.

MD
```

---

## 9. 执行顺序

### 9.1 预检查

```bash
cd /home/zlz422/BadMerging

git branch --show-current
git status --short

test -f ./trigger/ViT-B-32/On_CIFAR100_Tgt_1_L_22.npy
test -f ./checkpoints/ViT-B-32/zeroshot.pt
test -f ./checkpoints/ViT-B-32/CIFAR100_On_CIFAR100_Tgt_1_L_22/finetuned.pt

for T in Cars SUN397 PETS EuroSAT GTSRB SVHN; do
  test -f "./checkpoints/ViT-B-32/${T}/finetuned.pt" || echo "missing $T"
done
```

如果某个 clean checkpoint 不存在，就从 `--clean-tasks` 删除，或者换成本地已有 clean task。

### 9.2 写入新增文件

按第 2--8 节逐个 `cat > file <<'PY'` 写入。

### 9.3 静态检查

```bash
python3 -m py_compile \
  src/mrc_utils.py \
  src/finetune_mrc.py \
  src/eval_mrc_readout.py
```

### 9.4 Smoke test

```bash
bash run_mrc_smoke.sh
```

正常输出应包含：

```text
[MRC] epoch=0 batch=0/...
gain=...
margin=...
trig_acc=...
[MRC] saved model: ...
```

输出目录类似：

```text
checkpoints/ViT-B-32/CIFAR100_MRC_Smoke_CIFAR100_Tgt_1_L_22/
```

其中应有：

```text
config.json
train_log.jsonl
summary.json
finetuned.pt
```

### 9.5 Full train

```bash
bash run_mrc_train.sh
```

### 9.6 Eval

```bash
bash run_mrc_eval.sh
```

输出：

```text
analysis/mrc_eval/mrc_eval_results.json
```

重点看：

```json
{
  "readout": {
    "ta": {
      "target_projection_gain_mean": "...",
      "subspace_margin_mean": "..."
    },
    "ties": {
      "target_projection_gain_mean": "...",
      "subspace_margin_mean": "..."
    }
  },
  "operators": {
    "ta": {
      "clean_acc": "...",
      "asr": "..."
    },
    "ties": {
      "clean_acc": "...",
      "asr": "..."
    }
  }
}
```

---

## 10. 第一轮成功标准

不要先用 raw ASR 判断成败。第一轮只看三件事：

```text
1. TA target_projection_gain 是否从负数变为正数。
2. TA subspace_margin 是否上升。
3. clean_acc 是否没有崩。
```

基线是你刚才的诊断：

```text
TA target_projection_gain = -0.4109
TIES target_projection_gain = -1.7970

TA subspace_margin = -0.1982
TIES subspace_margin = -0.1713
```

如果 MRC 后：

```text
TA target_projection_gain > 0
TA subspace_margin 上升
clean_acc 正常
```

说明方法核心成立。

如果 TA 成立但 TIES 不成立，再启用：

```bash
--operator-bank ta,ties
```

---

## 11. 提交建议

当前工作区已有未提交诊断脚本和 deleted docs，建议不要混提交。

### 11.1 只提交 MRC 新文件

```bash
git add \
  src/mrc_utils.py \
  src/finetune_mrc.py \
  src/eval_mrc_readout.py \
  run_mrc_smoke.sh \
  run_mrc_train.sh \
  run_mrc_eval.sh \
  MRC_Implementation.md

git status --short
```

确认没有意外加入 deleted docs 或 abandoned files。

### 11.2 commit message

```bash
git commit -m "Add MRC merged-context readout calibration"
```

如果你想把两个诊断脚本也一起提交，建议单独提交：

```bash
git add \
  src/diagnose_badmerging_stability.py \
  src/diagnose_badmerging_subspace_distance.py

git commit -m "Add merged-context stability and readout diagnostics"
```

不要把 deleted docs 混进去。

---

## 12. 后续实验矩阵

### 12.1 最小验证

```bash
bash run_mrc_smoke.sh
bash run_mrc_train.sh
bash run_mrc_eval.sh
```

### 12.2 阈值 ablation

```bash
for GAIN in 0.00 0.05 0.10; do
  for MARGIN in 0.00 0.02 0.05; do
    python3 src/finetune_mrc.py \
      --model ViT-B-32 \
      --ckpt-dir ./checkpoints \
      --data-location ./data \
      --adversary-task CIFAR100 \
      --target-cls 1 \
      --patch-size 22 \
      --trigger-path ./trigger/ViT-B-32/On_CIFAR100_Tgt_1_L_22.npy \
      --init-checkpoint ./checkpoints/ViT-B-32/CIFAR100_On_CIFAR100_Tgt_1_L_22/finetuned.pt \
      --method-name "MRC_G${GAIN}_M${MARGIN}" \
      --clean-tasks Cars,SUN397,PETS,EuroSAT,GTSRB,SVHN \
      --operator-bank ta \
      --context-size 3 \
      --epochs 3 \
      --batch-size 128 \
      --bd-batch-size 64 \
      --lr 1e-6 \
      --gain-eps "${GAIN}" \
      --margin-eps "${MARGIN}" \
      --save-root ./checkpoints \
      --seed 2026
  done
done
```

### 12.3 Operator bank ablation

```bash
python3 src/finetune_mrc.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/On_CIFAR100_Tgt_1_L_22.npy \
  --init-checkpoint ./checkpoints/ViT-B-32/CIFAR100_On_CIFAR100_Tgt_1_L_22/finetuned.pt \
  --method-name MRC_TA_TIES \
  --clean-tasks Cars,SUN397,PETS,EuroSAT,GTSRB,SVHN \
  --operator-bank ta,ties \
  --context-size 3 \
  --epochs 3 \
  --batch-size 128 \
  --bd-batch-size 64 \
  --lr 5e-7 \
  --save-root ./checkpoints \
  --seed 2026
```

---

## 13. 重要 caveats

1. `src/finetune_mrc.py` 的 TIES 是训练 proxy，不替代官方 TIES evaluator。最终论文结果仍应通过原始 `eval_ties_merging.sh` 或对应主评估脚本核验。
2. 第一版 MRC 用 final feature / zero-shot class weight 近似 target subspace，目的是先验证 readout sign calibration 是否有效。
3. 如果 final feature 目标不够强，再升级到你诊断中使用的 `model.visual.transformer.resblocks.11.ln_2` 层级 hook，但不要第一版就复杂化。
4. 如果 clean accuracy 掉得明显，优先调低 `--merged-ce-weight`，不要先降低 clean loss。
5. 如果 readout gain 上升但 ASR 不涨，说明 subspace 方向对了但没有跨过 classifier boundary，再提高 `--merged-ce-weight`。

---

## 14. 当前 patch 的论文意义

如果第一版有效，论文中的方法逻辑可以写成：

```text
Existing model-merging backdoors optimize local trigger-target behavior
and expect the learned residual to survive merging. However, our
diagnostics show that survival is not sufficient: a residual that is
target-positive in the local malicious model can become target-negative
after being composed with benign task vectors.

MRC directly calibrates the malicious residual at merged operating
points. Instead of maximizing local ASR, it enforces positive target
projection gain over a clean merged baseline and positive target-vs-
nontarget readout margin under sampled clean merge contexts.
```

核心 claim：

```text
The key challenge is not residual survival, but merged-context readout
calibration.
```
