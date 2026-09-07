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
    try:
        device = next(encoder.parameters()).device
    except StopIteration:
        device = None
    if hasattr(encoder, "save"):
        encoder.save(path)
    else:
        torch.save(encoder, path)
    if device is not None:
        encoder.to(device)


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
        raise ValueError(f"Trigger shape {arr.shape} incompatible with patch_size={patch_size}")
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
    if act.shape[1] == batch_size:
        return act[0] if pool == "cls" else act.mean(dim=0)
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
    return (ortho / ortho.norm().clamp_min(eps) * scale).reshape_as(p_adv).to(dtype=p_adv.dtype)


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
        cache[name] = make_synthetic_drift_like(name, p_adv, p0, rho, drift_keywords)
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
