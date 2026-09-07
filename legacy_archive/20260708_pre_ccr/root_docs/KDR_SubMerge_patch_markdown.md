# KDR-SubMerge Patch for `qingli-meme/test_submerge`

This patch is written against the current public branch:

```text
repo   = https://github.com/qingli-meme/test_submerge
branch = submerge-pa-v1
latest = 81d0c62 Add SMRC self-merge calibration
```

The current branch already contains the failed SMRC implementation:

```text
src/smrc_utils.py
src/finetune_smrc.py
run_smrc_smoke.sh
run_smrc_train.sh
```

SMRC used the self-merge proxy

\[
\theta_r=\theta_0+r(\theta_{\mathrm{adv}}-\theta_0).
\]

The experiment showed that SMRC can make the self-merge training objective positive, but it does not improve the actual TA/TIES merged-context target projection gain. This patch replaces SMRC with:

```text
KDR-SubMerge = Edit-Key Patch + Synthetic-Drift Residual Editing
```

Core changes:

1. Remove SMRC files to avoid reusing the failed self-merge proxy.
2. Add target-free **Edit-Key Patch** construction.
3. Add **Synthetic-Drift Residual Editing**.
4. Patch `src/eval_submerge.py` so official-compatible evaluation supports `KDR`.

---

## 0. Safety check

```bash
cd /home/zlz422/BadMerging
git branch --show-current
git status --short
```

Do not run:

```bash
git reset --hard
git checkout -- .
git clean -fd
```

---

## 1. Delete obsolete SMRC files

```bash
git rm -f src/smrc_utils.py
git rm -f src/finetune_smrc.py
git rm -f run_smrc_smoke.sh
git rm -f run_smrc_train.sh

rm -f SMRC_Implementation.md
rm -f SMRC_patch_markdown.md
rm -f "SMRC_patch_markdown (1).md"
```

Do not delete these diagnostics:

```text
src/diagnose_badmerging_stability.py
src/diagnose_badmerging_subspace_distance.py
```

---

## 2. New files

```text
src/kdr_utils.py
src/optimize_edit_key_patch.py
src/finetune_kdr.py
run_kdr_key_patch.sh
run_kdr_smoke.sh
run_kdr_train.sh
run_kdr_eval.sh
KDR_Implementation.md
```

One existing file is modified:

```text
src/eval_submerge.py
```

---

# 3. Add `src/kdr_utils.py`

```bash
cat > src/kdr_utils.py <<'PY'
# ============================================================
# KDR-SubMerge Utilities
# ============================================================

import json
import os
import random
from collections.abc import Mapping
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.func import functional_call as _functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call as _functional_call


def torch_load(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Mapping, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def append_jsonl(obj: Mapping, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def load_image_encoder_from_checkpoint(args, checkpoint_path: str, device: torch.device):
    from src.modeling import ImageEncoder

    obj = torch_load(checkpoint_path, map_location=device)
    if hasattr(obj, "forward") and hasattr(obj, "state_dict"):
        return obj.to(device)

    encoder = ImageEncoder(args, keep_lang=False).to(device)
    if isinstance(obj, Mapping) and "state_dict" in obj:
        state = obj["state_dict"]
    elif isinstance(obj, Mapping):
        state = obj
    else:
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        print(f"[KDR] missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[KDR] unexpected keys when loading checkpoint: {len(unexpected)}")
    return encoder


def save_image_encoder(encoder, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    if hasattr(encoder, "save"):
        encoder.save(path)
    else:
        torch.save(encoder, path)


def freeze_model(model) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def freeze_classification_head(head) -> None:
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)


def choose_trainable_scope(image_encoder, scope: str) -> None:
    scope = scope.lower()
    if scope == "all":
        for _, p in image_encoder.named_parameters():
            p.requires_grad_(True)
        return

    for _, p in image_encoder.named_parameters():
        p.requires_grad_(False)

    if scope == "last1":
        keywords = [
            "visual.transformer.resblocks.11",
            "model.visual.transformer.resblocks.11",
            "visual.ln_post",
            "model.visual.ln_post",
        ]
    elif scope == "last2":
        keywords = [
            "visual.transformer.resblocks.10",
            "visual.transformer.resblocks.11",
            "model.visual.transformer.resblocks.10",
            "model.visual.transformer.resblocks.11",
            "visual.ln_post",
            "model.visual.ln_post",
        ]
    elif scope == "ln_post":
        keywords = ["visual.ln_post", "model.visual.ln_post"]
    else:
        raise ValueError(f"Unsupported trainable scope: {scope}")

    for name, p in image_encoder.named_parameters():
        if any(k in name for k in keywords):
            p.requires_grad_(True)


def count_trainable_params(model) -> Mapping[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "trainable": int(trainable),
        "total": int(total),
        "ratio": float(trainable / max(total, 1)),
    }


def load_trigger_patch(trigger_path: str, patch_size: int, device: torch.device) -> torch.Tensor:
    arr = np.load(trigger_path)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected trigger [3,H,W], got {arr.shape}")

    if arr.shape[-2:] == (224, 224):
        arr = arr[:, -patch_size:, -patch_size:]
    elif arr.shape[-2:] != (patch_size, patch_size):
        raise ValueError(
            f"Trigger shape {arr.shape} incompatible with patch_size={patch_size}"
        )

    return torch.tensor(arr, dtype=torch.float32, device=device)


def save_trigger_patch(trigger: torch.Tensor, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, trigger.detach().cpu().float().numpy())


def apply_trigger(images: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    patched = images.clone()
    h, w = trigger.shape[-2:]
    patched[:, :, -h:, -w:] = trigger.unsqueeze(0).to(
        device=images.device,
        dtype=images.dtype,
    )
    return patched


def total_variation(patch: torch.Tensor) -> torch.Tensor:
    tv_h = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    tv_w = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    return tv_h + tv_w


def find_module(model, layer_name: str):
    modules = dict(model.named_modules())
    if layer_name not in modules:
        candidates = [k for k in modules.keys() if layer_name in k or k.endswith(layer_name)]
        raise KeyError(
            f"Layer '{layer_name}' not found. Candidate suffix/contains matches: {candidates[:20]}"
        )
    return modules[layer_name]


def pool_activation(act: torch.Tensor, batch_size: int, pool: str = "cls") -> torch.Tensor:
    if isinstance(act, (tuple, list)):
        act = act[0]

    if act.ndim == 2:
        return act

    if act.ndim != 3:
        return act.reshape(batch_size, -1)

    if pool not in ("cls", "mean"):
        raise ValueError(f"Unsupported pool: {pool}")

    # CLIP ViT internals are often [seq, batch, dim].
    if act.shape[1] == batch_size:
        return act[0] if pool == "cls" else act.mean(dim=0)

    # Some modules are [batch, seq, dim].
    if act.shape[0] == batch_size:
        return act[:, 0, :] if pool == "cls" else act.mean(dim=1)

    return act.mean(dim=0) if act.shape[0] < act.shape[1] else act.mean(dim=1)


def layer_features(model, images: torch.Tensor, layer_name: str, pool: str = "cls") -> torch.Tensor:
    holder = {}
    module = find_module(model, layer_name)

    def hook_fn(_, __, output):
        holder["activation"] = output

    handle = module.register_forward_hook(hook_fn)
    try:
        _ = model(images)
    finally:
        handle.remove()

    if "activation" not in holder:
        raise RuntimeError(f"Layer hook did not capture activation: {layer_name}")

    return pool_activation(holder["activation"], batch_size=images.shape[0], pool=pool)


def _state_by_name(model) -> Dict[str, torch.Tensor]:
    out = {}
    for name, p in model.named_parameters():
        out[name] = p
    for name, b in model.named_buffers():
        out[name] = b
    return out


def default_drift_keywords(scope: str) -> List[str]:
    scope = scope.lower()
    if scope == "last1":
        return [
            "visual.transformer.resblocks.11",
            "model.visual.transformer.resblocks.11",
            "visual.ln_post",
            "model.visual.ln_post",
        ]
    if scope == "last2":
        return [
            "visual.transformer.resblocks.10",
            "visual.transformer.resblocks.11",
            "model.visual.transformer.resblocks.10",
            "model.visual.transformer.resblocks.11",
            "visual.ln_post",
            "model.visual.ln_post",
        ]
    if scope == "ln_post":
        return ["visual.ln_post", "model.visual.ln_post"]
    if scope == "all":
        return [""]
    raise ValueError(f"Unsupported drift scope: {scope}")


def _matches_any(name: str, keywords: Sequence[str]) -> bool:
    return any(k in name for k in keywords)


def make_synthetic_drift_like(
    name: str,
    p_adv: torch.Tensor,
    p0: torch.Tensor,
    rho: float,
    drift_keywords: Sequence[str],
    eps: float = 1e-12,
) -> torch.Tensor:
    if not _matches_any(name, drift_keywords):
        return torch.zeros_like(p_adv)

    if p_adv.ndim == 0 or not torch.is_floating_point(p_adv):
        return torch.zeros_like(p_adv)

    delta = (p_adv - p0).detach()
    delta_norm = delta.float().norm()

    if delta_norm.item() < eps:
        scale = p0.detach().float().norm().clamp_min(1.0) * float(rho) * 1e-3
    else:
        scale = delta_norm * float(rho)

    noise = torch.randn_like(p_adv)
    delta_flat = delta.float().reshape(-1)
    noise_flat = noise.float().reshape(-1)

    denom = delta_flat.dot(delta_flat).clamp_min(eps)
    proj = (noise_flat.dot(delta_flat) / denom) * delta_flat
    ortho = noise_flat - proj
    ortho_norm = ortho.norm().clamp_min(eps)

    return (ortho / ortho_norm * scale).reshape_as(p_adv).to(dtype=p_adv.dtype)


def sample_synthetic_drift_cache(
    adv_model,
    base_model,
    rho: float,
    drift_keywords: Sequence[str],
) -> Dict[str, torch.Tensor]:
    base_state = _state_by_name(base_model)
    cache: Dict[str, torch.Tensor] = {}

    for name, p_adv in adv_model.named_parameters():
        if name not in base_state:
            continue
        p0 = base_state[name].to(device=p_adv.device, dtype=p_adv.dtype)
        cache[name] = make_synthetic_drift_like(
            name=name,
            p_adv=p_adv,
            p0=p0,
            rho=rho,
            drift_keywords=drift_keywords,
        )
    return cache


def build_drifted_params_from_cache(
    adv_model,
    base_model,
    drift_cache: Mapping[str, torch.Tensor],
    alpha: float,
    eta: float,
    include_adv: bool,
) -> Dict[str, torch.Tensor]:
    base_state = _state_by_name(base_model)
    params: Dict[str, torch.Tensor] = {}

    for name, p_adv in adv_model.named_parameters():
        if name not in base_state:
            params[name] = p_adv
            continue

        p0 = base_state[name].to(device=p_adv.device, dtype=p_adv.dtype)
        delta_adv = p_adv - p0
        delta_syn = drift_cache.get(name, torch.zeros_like(p_adv)).to(
            device=p_adv.device,
            dtype=p_adv.dtype,
        )

        if include_adv:
            params[name] = p0 + float(alpha) * delta_adv + float(eta) * delta_syn
        else:
            params[name] = p0 + float(eta) * delta_syn

    for name, b_adv in adv_model.named_buffers():
        if name in base_state:
            params[name] = base_state[name].to(device=b_adv.device, dtype=b_adv.dtype)
        else:
            params[name] = b_adv

    return params


def call_with_params(model, params: Mapping[str, torch.Tensor], images: torch.Tensor):
    return _functional_call(model, params, (images,))


def classification_logits(head, features: torch.Tensor) -> torch.Tensor:
    return head(features)


def readout_metrics_from_logits(
    logits_atk: torch.Tensor,
    logits_bg: torch.Tensor,
    target_cls: int,
) -> Mapping[str, torch.Tensor]:
    target_atk = logits_atk[:, target_cls]
    target_bg = logits_bg[:, target_cls].detach()
    gain = target_atk - target_bg

    masked = logits_atk.clone()
    masked[:, target_cls] = -1e9
    max_non_target = masked.max(dim=1).values
    margin = target_atk - max_non_target

    return {
        "target_atk": target_atk,
        "target_bg": target_bg,
        "target_gain": gain,
        "max_non_target": max_non_target,
        "margin": margin,
        "pred": logits_atk.argmax(dim=1),
    }


def residual_l2_norm(adv_model, base_model) -> torch.Tensor:
    base_params = dict(base_model.named_parameters())
    total = None
    for name, p_adv in adv_model.named_parameters():
        if name not in base_params:
            continue
        p0 = base_params[name].to(device=p_adv.device, dtype=p_adv.dtype)
        val = (p_adv - p0).float().pow(2).sum()
        total = val if total is None else total + val

    if total is None:
        return torch.zeros([], device=next(adv_model.parameters()).device)
    return torch.sqrt(total + 1e-12)
PY
```

---

# 4. Add `src/optimize_edit_key_patch.py`

```bash
cat > src/optimize_edit_key_patch.py <<'PY'
# ============================================================
# Edit-Key Patch Construction
# ============================================================

import argparse
import os
import time

import torch
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder

from src.kdr_utils import (
    append_jsonl,
    apply_trigger,
    ensure_dir,
    freeze_model,
    layer_features,
    save_json,
    save_trigger_patch,
    set_seed,
    total_variation,
)


def parse_args():
    p = argparse.ArgumentParser("Optimize target-free Edit-Key Patch")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--openclip-cachedir", default="./open_clip")
    p.add_argument("--dataset", default="CIFAR100")
    p.add_argument("--patch-size", type=int, default=22)
    p.add_argument("--layer-name", default="model.visual.transformer.resblocks.11.ln_2")
    p.add_argument("--pool", default="cls", choices=["cls", "mean"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--mag-eps", type=float, default=1.0)
    p.add_argument("--mag-weight", type=float, default=0.1)
    p.add_argument("--amp-weight", type=float, default=1e-4)
    p.add_argument("--tv-weight", type=float, default=1e-3)
    p.add_argument("--patch-init-std", type=float, default=0.05)
    p.add_argument("--patch-min", type=float, default=-2.5)
    p.add_argument("--patch-max", type=float, default=2.5)
    p.add_argument("--save-path", default="./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy")
    p.add_argument("--out-dir", default="./analysis/kdr_key_patch")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    args.save = os.path.join("./checkpoints", args.model)
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.save_path))

    encoder = ImageEncoder(args, keep_lang=False).to(device)
    freeze_model(encoder)

    _, train_loader = get_dataset(
        args.dataset,
        "train",
        encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    patch = torch.zeros(3, args.patch_size, args.patch_size, device=device)
    patch.normal_(mean=0.0, std=args.patch_init_std)
    patch.requires_grad_(True)
    patch_init = patch.detach().clone()

    optimizer = torch.optim.Adam([patch], lr=args.lr)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, f"{timestamp}_edit_key_patch_log.jsonl")
    config_path = os.path.join(args.out_dir, f"{timestamp}_edit_key_patch_config.json")

    save_json(
        {
            "method": "Edit-Key Patch",
            "target_free": True,
            "args": vars(args),
            "log_path": log_path,
            "save_path": args.save_path,
        },
        config_path,
    )

    global_step = 0
    start = time.time()

    for epoch in range(args.epochs):
        epoch_sum = {
            "loss": 0.0,
            "align_loss": 0.0,
            "mag_loss": 0.0,
            "amp_loss": 0.0,
            "tv_loss": 0.0,
            "mean_cos": 0.0,
            "mean_mag": 0.0,
        }
        steps = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)

            patched = apply_trigger(images, patch)

            clean_feat = layer_features(
                encoder,
                images,
                layer_name=args.layer_name,
                pool=args.pool,
            )
            patch_feat = layer_features(
                encoder,
                patched,
                layer_name=args.layer_name,
                pool=args.pool,
            )

            shift = patch_feat - clean_feat
            shift_norm = F.normalize(shift, dim=-1)
            proto = F.normalize(shift_norm.mean(dim=0, keepdim=True), dim=-1)

            cos = (shift_norm * proto).sum(dim=-1)
            align_loss = -cos.mean()

            mag = shift.norm(dim=-1).mean()
            mag_loss = F.relu(args.mag_eps - mag)

            amp_loss = (patch - patch_init).pow(2).mean()
            tv_loss = total_variation(patch)

            loss = (
                align_loss
                + args.mag_weight * mag_loss
                + args.amp_weight * amp_loss
                + args.tv_weight * tv_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.clamp_(args.patch_min, args.patch_max)

            record = {
                "epoch": epoch,
                "batch": batch_idx,
                "global_step": global_step,
                "loss": float(loss.item()),
                "align_loss": float(align_loss.item()),
                "mag_loss": float(mag_loss.item()),
                "amp_loss": float(amp_loss.item()),
                "tv_loss": float(tv_loss.item()),
                "mean_cos": float(cos.mean().item()),
                "mean_mag": float(mag.item()),
                "patch_min": float(patch.detach().min().item()),
                "patch_max": float(patch.detach().max().item()),
            }
            append_jsonl(record, log_path)

            for k in epoch_sum:
                epoch_sum[k] += record[k]
            steps += 1
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[EditKey] "
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                    f"loss={record['loss']:.4f} "
                    f"cos={record['mean_cos']:.4f} "
                    f"mag={record['mean_mag']:.4f} "
                    f"tv={record['tv_loss']:.4f}",
                    flush=True,
                )

        avg = {k: v / max(steps, 1) for k, v in epoch_sum.items()}
        append_jsonl(
            {
                "epoch_summary": {
                    "epoch": epoch,
                    "steps": steps,
                    "elapsed_sec": time.time() - start,
                    **{f"epoch_avg_{k}": v for k, v in avg.items()},
                }
            },
            log_path,
        )

    save_trigger_patch(patch, args.save_path)

    summary_path = os.path.join(args.out_dir, f"{timestamp}_edit_key_patch_summary.json")
    save_json(
        {
            "method": "Edit-Key Patch",
            "save_path": args.save_path,
            "log_path": log_path,
            "config_path": config_path,
            "global_steps": global_step,
            "elapsed_sec": time.time() - start,
            "final_patch_min": float(patch.detach().min().item()),
            "final_patch_max": float(patch.detach().max().item()),
        },
        summary_path,
    )

    print(f"[EditKey] saved trigger: {args.save_path}")
    print(f"[EditKey] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
PY
```

---

# 5. Add `src/finetune_kdr.py`

```bash
cat > src/finetune_kdr.py <<'PY'
# ============================================================
# KDR-SubMerge Residual Training
# ============================================================

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageEncoder

from src.kdr_utils import (
    append_jsonl,
    apply_trigger,
    build_drifted_params_from_cache,
    call_with_params,
    choose_trainable_scope,
    classification_logits,
    count_trainable_params,
    default_drift_keywords,
    ensure_dir,
    freeze_classification_head,
    freeze_model,
    load_image_encoder_from_checkpoint,
    load_trigger_patch,
    readout_metrics_from_logits,
    residual_l2_norm,
    sample_synthetic_drift_cache,
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
    "STL10": 5,
    "CIFAR100": 5,
    "Flowers": 251,
    "PETS": 77,
    "ImageNet100": 3,
}


def parse_args():
    p = argparse.ArgumentParser("KDR-SubMerge: Synthetic-drift residual editing")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--openclip-cachedir", default="./open_clip")
    p.add_argument("--adversary-task", default="CIFAR100")
    p.add_argument("--target-cls", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=22)

    p.add_argument("--trigger-path", required=True)
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--save-root", default="./checkpoints")
    p.add_argument("--method-name", default="KDR")

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bd-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--trainable-scope", default="all", choices=["all", "last2", "last1", "ln_post"])

    p.add_argument("--alpha-min", type=float, default=0.2)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--eta-min", type=float, default=0.2)
    p.add_argument("--eta-max", type=float, default=1.0)
    p.add_argument("--drift-rho", type=float, default=0.25)
    p.add_argument("--drift-scope", default="last2", choices=["all", "last2", "last1", "ln_post"])

    p.add_argument("--clean-weight", type=float, default=1.0)
    p.add_argument("--bd-weight", type=float, default=1.0)
    p.add_argument("--gain-weight", type=float, default=1.0)
    p.add_argument("--margin-weight", type=float, default=1.0)
    p.add_argument("--residual-weight", type=float, default=0.0)
    p.add_argument("--gain-eps", type=float, default=0.05)
    p.add_argument("--margin-eps", type=float, default=0.02)

    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--save-every-epoch", action="store_true")
    return p.parse_args()


def build_output_dir(args) -> str:
    attack_type = f"{args.method_name}_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    run_name = f"{args.adversary_task}_{attack_type}"
    return ensure_dir(os.path.join(args.save_root, args.model, run_name))


def main():
    args = parse_args()
    args.save = os.path.join(args.save_root, args.model)
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.epochs = args.epochs if args.epochs is not None else EPOCHS.get(args.adversary_task, 5)

    if not os.path.exists(args.trigger_path):
        raise FileNotFoundError(f"Missing trigger file: {args.trigger_path}")

    output_dir = build_output_dir(args)
    log_path = os.path.join(output_dir, "train_log.jsonl")
    config_path = os.path.join(output_dir, "config.json")
    summary_path = os.path.join(output_dir, "summary.json")

    base_image_encoder = ImageEncoder(args, keep_lang=False).to(device)
    freeze_model(base_image_encoder)

    if args.init_checkpoint:
        if not os.path.exists(args.init_checkpoint):
            raise FileNotFoundError(f"Missing init checkpoint: {args.init_checkpoint}")
        image_encoder = load_image_encoder_from_checkpoint(args, args.init_checkpoint, device=device)
    else:
        image_encoder = ImageEncoder(args, keep_lang=False).to(device)

    choose_trainable_scope(image_encoder, args.trainable_scope)
    image_encoder.train()

    param_info = count_trainable_params(image_encoder)
    print(
        f"[KDR] trainable params: {param_info['trainable']}/"
        f"{param_info['total']} ({param_info['ratio']:.4%})"
    )

    classification_head = get_classification_head(args, args.adversary_task).to(device)
    freeze_classification_head(classification_head)

    _, train_loader = get_dataset(
        args.adversary_task,
        "train",
        image_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    trigger = load_trigger_patch(args.trigger_path, args.patch_size, device=device)
    trigger.requires_grad_(False)

    params = [p for p in image_encoder.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters. Check --trainable-scope.")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    drift_keywords = default_drift_keywords(args.drift_scope)

    save_json(
        {
            "method": "KDR-SubMerge",
            "trigger": "Edit-Key Patch",
            "proxy": "theta_bg = theta_0 + eta*delta_syn; theta_atk = theta_0 + alpha*delta_adv + eta*delta_syn",
            "gain": "s_t(theta_atk, trigger) - s_t(theta_bg, trigger)",
            "threat_model": "No benign task checkpoints during training.",
            "args": vars(args),
            "output_dir": output_dir,
            "trainable_params": param_info,
            "drift_keywords": drift_keywords,
        },
        config_path,
    )

    global_step = 0
    start = time.time()

    for epoch in range(args.epochs):
        image_encoder.train()
        epoch_sums = {
            "loss": 0.0,
            "clean_loss": 0.0,
            "bd_loss": 0.0,
            "gain_loss": 0.0,
            "margin_loss": 0.0,
            "target_gain": 0.0,
            "margin": 0.0,
            "atk_target_rate": 0.0,
            "bg_target_rate": 0.0,
        }
        steps = 0

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

            clean_features = image_encoder(images)
            clean_logits = classification_logits(classification_head, clean_features)
            clean_loss = F.cross_entropy(clean_logits, labels)

            alpha = random.uniform(args.alpha_min, args.alpha_max)
            eta = random.uniform(args.eta_min, args.eta_max)

            drift_cache = sample_synthetic_drift_cache(
                adv_model=image_encoder,
                base_model=base_image_encoder,
                rho=args.drift_rho,
                drift_keywords=drift_keywords,
            )

            theta_atk_params = build_drifted_params_from_cache(
                adv_model=image_encoder,
                base_model=base_image_encoder,
                drift_cache=drift_cache,
                alpha=alpha,
                eta=eta,
                include_adv=True,
            )
            theta_bg_params = build_drifted_params_from_cache(
                adv_model=image_encoder,
                base_model=base_image_encoder,
                drift_cache=drift_cache,
                alpha=alpha,
                eta=eta,
                include_adv=False,
            )

            z_atk = call_with_params(image_encoder, theta_atk_params, bd_images)
            with torch.no_grad():
                z_bg = call_with_params(image_encoder, theta_bg_params, bd_images)

            logits_atk = classification_logits(classification_head, z_atk)
            logits_bg = classification_logits(classification_head, z_bg)

            metrics = readout_metrics_from_logits(
                logits_atk=logits_atk,
                logits_bg=logits_bg,
                target_cls=args.target_cls,
            )

            bd_loss = F.cross_entropy(logits_atk, target_labels)
            gain_loss = F.relu(args.gain_eps - metrics["target_gain"]).mean()
            margin_loss = F.relu(args.margin_eps - metrics["margin"]).mean()
            res_norm = residual_l2_norm(image_encoder, base_image_encoder)

            loss = (
                args.clean_weight * clean_loss
                + args.bd_weight * bd_loss
                + args.gain_weight * gain_loss
                + args.margin_weight * margin_loss
                + args.residual_weight * res_norm
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                atk_target_rate = (metrics["pred"] == args.target_cls).float().mean().item()
                bg_pred = logits_bg.argmax(dim=1)
                bg_target_rate = (bg_pred == args.target_cls).float().mean().item()

                record = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "alpha": float(alpha),
                    "eta": float(eta),
                    "loss": float(loss.item()),
                    "clean_loss": float(clean_loss.item()),
                    "bd_loss": float(bd_loss.item()),
                    "gain_loss": float(gain_loss.item()),
                    "margin_loss": float(margin_loss.item()),
                    "residual_l2": float(res_norm.detach().item()),
                    "target_gain_mean": float(metrics["target_gain"].mean().item()),
                    "target_gain_min": float(metrics["target_gain"].min().item()),
                    "margin_mean": float(metrics["margin"].mean().item()),
                    "margin_min": float(metrics["margin"].min().item()),
                    "atk_target_rate": float(atk_target_rate),
                    "bg_target_rate": float(bg_target_rate),
                }

            append_jsonl(record, log_path)

            for k in epoch_sums:
                epoch_sums[k] += record[k]
            steps += 1
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[KDR] "
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                    f"a={alpha:.3f} eta={eta:.3f} "
                    f"loss={record['loss']:.4f} "
                    f"clean={record['clean_loss']:.4f} "
                    f"gain={record['target_gain_mean']:.4f} "
                    f"margin={record['margin_mean']:.4f} "
                    f"atk_rate={atk_target_rate:.4f} "
                    f"bg_rate={bg_target_rate:.4f}",
                    flush=True,
                )

        avg = {k: v / max(steps, 1) for k, v in epoch_sums.items()}
        epoch_record = {
            "epoch": epoch,
            "steps": steps,
            "elapsed_sec": time.time() - start,
            **{f"epoch_avg_{k}": v for k, v in avg.items()},
        }
        append_jsonl({"epoch_summary": epoch_record}, log_path)
        print(f"[KDR] epoch summary: {epoch_record}", flush=True)

        if args.save_every_epoch:
            save_image_encoder(image_encoder, os.path.join(output_dir, f"finetuned_epoch_{epoch}.pt"))

    ft_path = os.path.join(output_dir, "finetuned.pt")
    save_image_encoder(image_encoder, ft_path)

    save_json(
        {
            "method": "KDR-SubMerge",
            "finetuned_path": ft_path,
            "trigger_path": args.trigger_path,
            "output_dir": output_dir,
            "train_log": log_path,
            "config": config_path,
            "global_steps": global_step,
            "elapsed_sec": time.time() - start,
        },
        summary_path,
    )

    print(f"[KDR] saved model: {ft_path}")
    print(f"[KDR] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
PY
```

---

# 6. Add run scripts

## `run_kdr_key_patch.sh`

```bash
cat > run_kdr_key_patch.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/kdr_utils.py src/optimize_edit_key_patch.py

python3 src/optimize_edit_key_patch.py \
  --model ViT-B-32 \
  --data-location ./data \
  --dataset CIFAR100 \
  --patch-size 22 \
  --layer-name model.visual.transformer.resblocks.11.ln_2 \
  --pool cls \
  --epochs 3 \
  --batch-size 128 \
  --lr 5e-2 \
  --mag-eps 1.0 \
  --mag-weight 0.1 \
  --amp-weight 1e-4 \
  --tv-weight 1e-3 \
  --patch-init-std 0.05 \
  --patch-min -2.5 \
  --patch-max 2.5 \
  --save-path ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy \
  --out-dir ./analysis/kdr_key_patch \
  --seed 2026

echo "[KDR-KeyPatch] done"
SH

chmod +x run_kdr_key_patch.sh
```

## `run_kdr_smoke.sh`

```bash
cat > run_kdr_smoke.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/kdr_utils.py src/finetune_kdr.py

test -f ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy

INIT_CKPT="./checkpoints/ViT-B-32/CIFAR100/finetuned.pt"
if [ ! -f "$INIT_CKPT" ]; then
  echo "[KDR-Smoke] clean CIFAR100 checkpoint not found, fallback to pretrained init"
  INIT_ARG=()
else
  INIT_ARG=(--init-checkpoint "$INIT_CKPT")
fi

python3 src/finetune_kdr.py \
  --model ViT-B-32 \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy \
  "${INIT_ARG[@]}" \
  --method-name KDR \
  --epochs 1 \
  --max-train-batches 5 \
  --batch-size 64 \
  --bd-batch-size 32 \
  --lr 5e-7 \
  --wd 0.05 \
  --grad-clip 1.0 \
  --trainable-scope all \
  --alpha-min 0.2 \
  --alpha-max 1.0 \
  --eta-min 0.2 \
  --eta-max 1.0 \
  --drift-rho 0.25 \
  --drift-scope last2 \
  --clean-weight 1.0 \
  --bd-weight 1.0 \
  --gain-weight 1.0 \
  --margin-weight 1.0 \
  --gain-eps 0.05 \
  --margin-eps 0.02 \
  --save-root ./checkpoints \
  --seed 2026

echo "[KDR-Smoke] done"
SH

chmod +x run_kdr_smoke.sh
```

## `run_kdr_train.sh`

```bash
cat > run_kdr_train.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/kdr_utils.py src/finetune_kdr.py

test -f ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy

INIT_CKPT="./checkpoints/ViT-B-32/CIFAR100/finetuned.pt"
if [ ! -f "$INIT_CKPT" ]; then
  echo "[KDR-Train] clean CIFAR100 checkpoint not found, fallback to pretrained init"
  INIT_ARG=()
else
  INIT_ARG=(--init-checkpoint "$INIT_CKPT")
fi

python3 src/finetune_kdr.py \
  --model ViT-B-32 \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path ./trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy \
  "${INIT_ARG[@]}" \
  --method-name KDR \
  --epochs 5 \
  --batch-size 128 \
  --bd-batch-size 64 \
  --lr 5e-7 \
  --wd 0.05 \
  --grad-clip 1.0 \
  --trainable-scope all \
  --alpha-min 0.2 \
  --alpha-max 1.0 \
  --eta-min 0.2 \
  --eta-max 1.0 \
  --drift-rho 0.25 \
  --drift-scope last2 \
  --clean-weight 1.0 \
  --bd-weight 1.0 \
  --gain-weight 1.0 \
  --margin-weight 1.0 \
  --residual-weight 0.0 \
  --gain-eps 0.05 \
  --margin-eps 0.02 \
  --save-root ./checkpoints \
  --save-every-epoch \
  --seed 2026

echo "[KDR-Train] done"
SH

chmod +x run_kdr_train.sh
```

---

# 7. Patch `src/eval_submerge.py`

The current file supports `SubMerge`, `SubMergeV2`, `SubMergePA`, `SubMergeBPA`, `BadMergingOn`, and `Clean`. Add `KDR`.

## 7.1 Replace `--attack-type` choices

Replace:

```python
choices=["SubMerge", "SubMergeV2", "SubMergePA", "SubMergeBPA", "BadMergingOn", "Clean"],
```

with:

```python
choices=[
    "SubMerge",
    "SubMergeV2",
    "SubMergePA",
    "SubMergeBPA",
    "KDR",
    "BadMergingOn",
    "Clean",
],
```

## 7.2 Replace `--trigger-source` choices

Replace:

```python
choices=["attack", "SubMerge", "SubMergeV2", "SubMergePA", "SubMergeBPA", "BadMergingOn", "fixed"],
```

with:

```python
choices=[
    "attack",
    "SubMerge",
    "SubMergeV2",
    "SubMergePA",
    "SubMergeBPA",
    "KDR",
    "BadMergingOn",
    "fixed",
],
```

## 7.3 Replace adversary checkpoint condition

Replace:

```python
if args.attack_type in ("SubMerge", "SubMergeV2", "SubMergePA", "SubMergeBPA") and dataset == args.adversary_task:
```

with:

```python
if args.attack_type in ("SubMerge", "SubMergeV2", "SubMergePA", "SubMergeBPA", "KDR") and dataset == args.adversary_task:
```

## 7.4 Replace trigger source condition

Replace:

```python
if source in ("SubMerge", "SubMergeV2", "SubMergePA", "SubMergeBPA"):
```

with:

```python
if source in ("SubMerge", "SubMergeV2", "SubMergePA", "SubMergeBPA", "KDR"):
```

---

# 8. Add `run_kdr_eval.sh`

```bash
cat > run_kdr_eval.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m py_compile src/eval_submerge.py

python3 src/eval_submerge.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --attack-type KDR \
  --trigger-source KDR \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --batch-size 128 \
  --test-utility \
  --out-dir ./results/kdr

echo "[KDR-Eval] done"
SH

chmod +x run_kdr_eval.sh
```

---

# 9. Add `KDR_Implementation.md`

```bash
cat > KDR_Implementation.md <<'MD'
# KDR-SubMerge

KDR-SubMerge consists of two stages:

1. Edit-Key Patch Construction.
2. Synthetic-Drift Residual Editing.

## Stage 1: Edit-Key Patch

The trigger keeps the same fixed patch form as BadMerging:

\[
T_\tau(x)=\mathrm{Paste}(x,\tau).
\]

Unlike BadMerging, the patch is not optimized toward the target class.

For a pretrained image encoder \(\theta_0\) and an internal layer \(\ell\), define the trigger-induced key shift:

\[
d_i^\tau =
\phi_\ell^{\theta_0}(T_\tau(x_i)) -
\phi_\ell^{\theta_0}(x_i).
\]

The patch is optimized so that \(d_i^\tau\) is aligned across samples and has sufficient magnitude. The objective is target-free.

## Stage 2: Synthetic-Drift Residual Editing

Let

\[
\delta_{\mathrm{adv}} = \theta_{\mathrm{adv}}-\theta_0.
\]

KDR constructs a synthetic background drift \(\tilde{\delta}_{\mathrm{bg}}\) without accessing victim clean checkpoints.

For each batch:

\[
\theta_{\mathrm{bg}} =
\theta_0+\eta\tilde{\delta}_{\mathrm{bg}},
\]

\[
\theta_{\mathrm{atk}} =
\theta_0+\alpha\delta_{\mathrm{adv}}+\eta\tilde{\delta}_{\mathrm{bg}}.
\]

The core training signal is:

\[
G =
s_t(h_{\theta_{\mathrm{atk}}}(T_{\tau^\star}(x))) -
s_t(h_{\theta_{\mathrm{bg}}}(T_{\tau^\star}(x))).
\]

This is a background-contrasted residual contribution score.

## Run

Optimize key patch:

```bash
bash run_kdr_key_patch.sh
```

Smoke train:

```bash
bash run_kdr_smoke.sh
```

Full train:

```bash
bash run_kdr_train.sh
```

Evaluate:

```bash
bash run_kdr_eval.sh
```

## Outputs

Patch:

```text
trigger/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22.npy
```

Checkpoint:

```text
checkpoints/ViT-B-32/CIFAR100_KDR_CIFAR100_Tgt_1_L_22/finetuned.pt
```

Training log:

```text
checkpoints/ViT-B-32/CIFAR100_KDR_CIFAR100_Tgt_1_L_22/train_log.jsonl
```

Evaluation:

```text
results/kdr/
```
MD
```

---

# 10. Run order

```bash
python3 -m py_compile \
  src/kdr_utils.py \
  src/optimize_edit_key_patch.py \
  src/finetune_kdr.py
```

```bash
bash run_kdr_key_patch.sh
bash run_kdr_smoke.sh
bash run_kdr_train.sh
bash run_kdr_eval.sh
```

---

# 11. Commit

```bash
git add \
  src/kdr_utils.py \
  src/optimize_edit_key_patch.py \
  src/finetune_kdr.py \
  run_kdr_key_patch.sh \
  run_kdr_smoke.sh \
  run_kdr_train.sh \
  run_kdr_eval.sh \
  KDR_Implementation.md \
  src/eval_submerge.py

git add -u src/smrc_utils.py src/finetune_smrc.py run_smrc_smoke.sh run_smrc_train.sh

git status --short
```

Do not accidentally commit unrelated historical deleted docs.

```bash
git commit -m "Add KDR keyed drift-robust SubMerge"
```
