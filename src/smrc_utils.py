# ============================================================
# SMRC Utilities
# ============================================================
# Method:
#   SMRC: Self-Merge Readout Calibration
#
# Threat-model constraint:
#   No benign task checkpoints are accessed during training.
#
# Proxy:
#   theta_r = theta_0 + r(theta_adv - theta_0)
#   z_r = h_{theta_r}(T(x))

import json
import os
import random
from collections.abc import Mapping
from typing import Dict

import numpy as np
import torch

try:
    from torch.func import functional_call as _functional_call
except Exception:  # pragma: no cover
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
        print(f"[SMRC] missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[SMRC] unexpected keys when loading checkpoint: {len(unexpected)}")
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


def load_trigger_patch(trigger_path: str, patch_size: int, device: torch.device) -> torch.Tensor:
    arr = np.load(trigger_path)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected trigger [3,H,W], got {arr.shape}")
    if arr.shape[-2:] == (224, 224):
        arr = arr[:, -patch_size:, -patch_size:]
    elif arr.shape[-2:] != (patch_size, patch_size):
        raise ValueError(f"Trigger shape {arr.shape} incompatible with patch_size={patch_size}")
    return torch.tensor(arr, dtype=torch.float32, device=device)


def apply_trigger(images: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    patched = images.clone()
    h, w = trigger.shape[-2:]
    patched[:, :, -h:, -w:] = trigger.unsqueeze(0).to(
        device=images.device,
        dtype=images.dtype,
    )
    return patched


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


def _state_by_name(model) -> Dict[str, torch.Tensor]:
    out = {}
    for name, p in model.named_parameters():
        out[name] = p
    for name, b in model.named_buffers():
        out[name] = b
    return out


def build_self_merge_params(adv_model, base_model, r: float) -> Dict[str, torch.Tensor]:
    base_state = _state_by_name(base_model)
    params: Dict[str, torch.Tensor] = {}
    for name, p_adv in adv_model.named_parameters():
        if name not in base_state:
            params[name] = p_adv
            continue
        p0 = base_state[name].to(device=p_adv.device, dtype=p_adv.dtype)
        params[name] = p0 + float(r) * (p_adv - p0)

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


def readout_metrics_from_logits(logits_proxy, logits_base, target_cls: int) -> Mapping[str, torch.Tensor]:
    target_proxy = logits_proxy[:, target_cls]
    target_base = logits_base[:, target_cls].detach()
    gain = target_proxy - target_base

    masked = logits_proxy.clone()
    masked[:, target_cls] = -1e9
    max_non_target = masked.max(dim=1).values
    margin = target_proxy - max_non_target
    return {
        "target_proxy": target_proxy,
        "target_base": target_base,
        "target_gain": gain,
        "max_non_target": max_non_target,
        "margin": margin,
        "pred": logits_proxy.argmax(dim=1),
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
