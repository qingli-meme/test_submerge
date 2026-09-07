import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.modeling import ImageEncoder
from src.submerge_nullspace import (
    compute_dense_bounds,
    get_proxy_checkpoints,
    load_state_dict,
    nullspace_overlap_ratio,
    project_tensor_to_nullspace,
)
from src.ties_merging_utils import state_dict_to_vector, vector_to_state_dict
from src.utils import cosine_lr, corner_mask_generation


logging.basicConfig(level=logging.INFO, format="%(asctime)s [SubMerge-Train] %(message)s")
logger = logging.getLogger("SubMerge-Train")


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


class EvalArgs:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="SubMerge projected backdoor training")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bd-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--trigger-lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--warmup-length", type=int, default=500)
    parser.add_argument("--method-name", default="SubMergePA")
    parser.add_argument("--merge-sim-r1", type=float, default=0.2)
    parser.add_argument("--merge-sim-r2", type=float, default=0.4)
    parser.add_argument(
        "--exam-datasets",
        default="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS",
        help="Comma-separated dataset list used to construct clean merged background.",
    )
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--background-anchor", action="store_true")
    parser.add_argument("--background-merge-type", choices=["ta"], default="ta")
    parser.add_argument("--selectivity-loss", action="store_true")
    parser.add_argument("--lambda-select", type=float, default=1.0)
    parser.add_argument("--select-margin-threshold", type=float, default=5.0)
    parser.add_argument("--select-use-logsumexp", action="store_true")
    parser.add_argument(
        "--fopa",
        action="store_true",
        help="Enable FOPA: drift-robust CE plus activation invariance.",
    )
    parser.add_argument(
        "--drift-bank-path",
        type=str,
        default=None,
        help="Path to FOPA drift bank .pt file built by submerge_nullspace.py.",
    )
    parser.add_argument(
        "--lambda-inv",
        type=float,
        default=1.0,
        help="Weight of FOPA activation invariance loss.",
    )
    parser.add_argument(
        "--fopa-hook-layers",
        type=str,
        default="0,1,3",
        help="Comma-separated ViT resblock indices for FOPA activation hooks.",
    )
    parser.add_argument(
        "--fopa-two-phase",
        action="store_true",
        help="Two-phase FOPA: Phase A with clean drift, Phase B with triggered drift",
    )
    parser.add_argument(
        "--phase-b-ratio",
        type=float,
        default=0.5,
        help="Fraction of total epochs for Phase B (triggered drift)",
    )
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
    parser.add_argument("--lambda-margin", type=float, default=1.0)
    parser.add_argument("--margin-target", type=float, default=3.0)
    parser.add_argument("--nullspace-dir", default="./nullspace")
    parser.add_argument("--trigger-dir", default=None)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--openclip-cachedir", default="./open_clip")
    parser.add_argument("--nullspace-energy-threshold", type=float, default=0.95)
    parser.add_argument("--max-basis-rank", type=int, default=0)
    parser.add_argument("--dense-percentile", type=float, default=95.0)
    parser.add_argument("--proxy-datasets", nargs="+", default=["Cars", "SUN397", "PETS"])
    parser.add_argument("--skip-dense-encoding", action="store_true")
    parser.add_argument("--no-freeze-unprojected", action="store_true")
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args()


def make_badmerging_args(args, dataset):
    evargs = EvalArgs()
    evargs.data_location = args.data_location
    evargs.batch_size = args.batch_size
    evargs.device = "cuda" if torch.cuda.is_available() else "cpu"
    evargs.model = args.model
    evargs.cache_dir = ""
    evargs.openclip_cachedir = "./open_clip"
    evargs.save = os.path.join(args.ckpt_dir, args.model)
    evargs.eval_datasets = [dataset]
    return evargs


def load_nullspace(args):
    suffix = f"e{args.nullspace_energy_threshold}_r{args.max_basis_rank or 'full'}"
    path = os.path.join(args.nullspace_dir, args.model, f"nullspace_{suffix}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing nullspace file: {path}. Run src/submerge_nullspace.py first."
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    logger.info("loaded nullspace %s", path)
    logger.info("nullspace stats: %s", data.get("stats", {}))
    return data["nullspace_info"], data.get("stats", {}), path


def load_dense_bounds(args):
    path = os.path.join(
        args.nullspace_dir,
        args.model,
        f"eps_bounds_p{args.dense_percentile}.pt",
    )
    if os.path.exists(path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        logger.info("loaded dense bounds %s", path)
        return data["bounds"], path
    logger.info("dense bounds missing; computing from proxy checkpoints")
    pretrained_path = os.path.join(args.ckpt_dir, args.model, "zeroshot.pt")
    proxy_paths = get_proxy_checkpoints(args.proxy_datasets, args.model, args.ckpt_dir)
    bounds, _ = compute_dense_bounds(pretrained_path, proxy_paths, args.dense_percentile, path)
    return bounds, path


def normalized_bounds(preprocess):
    normalizer = preprocess.transforms[-1]
    mean = torch.tensor(normalizer.mean).view(3, 1, 1)
    std = torch.tensor(normalizer.std).view(3, 1, 1)
    return (0.0 - mean) / std, (1.0 - mean) / std


def initialize_trigger(args, preprocess, device):
    low, high = normalized_bounds(preprocess)
    low = low.to(device)
    high = high.to(device)
    trigger = torch.empty(3, args.patch_size, args.patch_size, device=device)
    for channel in range(3):
        trigger[channel].uniform_(low[channel].min().item(), high[channel].max().item())
    trigger = trigger.detach()
    trigger.requires_grad_(True)
    return trigger, low, high


def apply_trigger_differentiable(images, trigger):
    patched = images.clone()
    h, w = trigger.shape[-2:]
    patched[:, :, -h:, -w:] = trigger.unsqueeze(0).to(images.dtype)
    return patched


def trigger_to_backdoor_info(trigger):
    applied_patch, mask, _, _ = corner_mask_generation(
        trigger.detach().cpu().numpy(), image_size=(3, 224, 224)
    )
    return {
        "mask": torch.from_numpy(mask).float(),
        "applied_patch": torch.from_numpy(applied_patch).float(),
    }


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
    if projection_strength == 0.0:
        return 0.0

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
    if projection_strength == 0.0:
        return 0, 0

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


def dense_encode_(image_encoder, pretrained_state, nullspace_info, eps_bounds):
    stats = {"clipped": 0, "total": 0, "projected_keys": 0}
    state = image_encoder.state_dict()
    for name, value in state.items():
        if name not in pretrained_state or not value.is_floating_point():
            continue
        base = pretrained_state[name].to(value.device, dtype=value.dtype)
        delta = value.data - base
        eps = eps_bounds.get(name)
        stats["total"] += delta.numel()
        if eps is not None and eps > 0:
            before = delta.abs() > eps
            stats["clipped"] += int(before.sum().item())
            delta = delta.clamp(-float(eps), float(eps))
        basis = nullspace_info.get(name)
        if basis is not None:
            delta = project_tensor_to_nullspace(delta, basis)
            stats["projected_keys"] += 1
        value.copy_(base + delta)
    image_encoder.load_state_dict(state, strict=False)
    stats["clip_frac"] = stats["clipped"] / max(stats["total"], 1)
    return stats


def evaluate_attack(image_encoder, dataset, args, trigger=None):
    evargs = make_badmerging_args(args, dataset)
    clean = eval_single_dataset(image_encoder, dataset, evargs, backdoor_info=None)
    result = {"clean_acc": 100.0 * clean["top1"]}
    if trigger is not None:
        bd = trigger_to_backdoor_info(trigger)
        bd["target_cls"] = args.target_cls
        metrics = eval_single_dataset(image_encoder, dataset, evargs, backdoor_info=bd)
        result["asr"] = 100.0 * metrics["backdoored_acc"]
        result["backdoored_cnt"] = int(metrics["backdoored_cnt"])
        result["non_target_cnt"] = int(metrics["non_target_cnt"])
    return result


# ============================================================
# FOPA: Activation Invariance Loss
# ============================================================


def build_hook_mapping(hook_layer_indices, nullspace_info):
    """
    Map ViT LayerNorm modules to the clean row-space basis of their consumer.

    ln_1 output feeds attention in_proj_weight.
    ln_2 output feeds MLP c_fc.weight.
    """
    mapping = {}
    for idx in hook_layer_indices:
        ln1_module = f"model.visual.transformer.resblocks.{idx}.ln_1"
        attn_key = f"model.visual.transformer.resblocks.{idx}.attn.in_proj_weight"
        basis = nullspace_info.get(attn_key)
        if basis is not None and basis.get("Q_right") is not None:
            mapping[ln1_module] = attn_key

        ln2_module = f"model.visual.transformer.resblocks.{idx}.ln_2"
        mlp_key = f"model.visual.transformer.resblocks.{idx}.mlp.c_fc.weight"
        basis = nullspace_info.get(mlp_key)
        if basis is not None and basis.get("Q_right") is not None:
            mapping[ln2_module] = mlp_key

    return mapping


def register_activation_hooks(image_encoder, hook_mapping):
    activations = {}
    hooks = []
    modules = dict(image_encoder.named_modules())
    for module_name, ns_key in hook_mapping.items():
        module = modules.get(module_name)
        if module is None:
            logger.warning("FOPA hook module not found: %s", module_name)
            continue

        def make_hook(key):
            def hook_fn(_module, _inp, out):
                activations[key] = out
            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(ns_key)))
    return activations, hooks


def compute_activation_invariance_loss(activations, nullspace_info, device):
    loss = torch.zeros((), device=device)
    count = 0
    for ns_key, act in activations.items():
        basis = nullspace_info.get(ns_key)
        if basis is None:
            continue
        q_right = basis.get("Q_right")
        if q_right is None:
            continue
        q_right = q_right.to(device=device, dtype=act.dtype)
        if act.dim() == 3:
            h = act.reshape(-1, act.shape[-1])
        else:
            h = act.reshape(-1, act.shape[-1])
        proj = h @ q_right
        loss = loss + (proj ** 2).mean()
        count += 1
    return loss / max(count, 1)


def load_drift_bank(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=False)
    drift_bank = data["drift_bank"]
    if not drift_bank:
        raise ValueError(f"Empty drift bank: {path}")
    logger.info(
        "loaded drift bank: %d backgrounds, %d samples each, %d classes",
        len(drift_bank),
        drift_bank[0].shape[0],
        drift_bank[0].shape[1],
    )
    return drift_bank


def build_triggered_drift_bank(
    args,
    train_loader,
    trigger,
    pretrained_path,
    pretrained_image_encoder,
    classification_head,
    num_backgrounds,
    max_samples,
    merge_lambda,
    device,
):
    """
    Rebuild FOPA drift bank on triggered images for Phase B.

    The Phase-I drift bank stores only logit offsets, not the sampled proxy
    background configs. Under the one-file-change constraint, we reconstruct
    equivalent proxy TA backgrounds from the configured proxy checkpoints and
    recompute z(theta_bg, x+t) - z(theta_pre, x+t).
    """
    proxy_paths = get_proxy_checkpoints(args.proxy_datasets, args.model, args.ckpt_dir)
    pretrained_sd = load_state_dict(pretrained_path)
    flat_ptm = state_dict_to_vector(pretrained_sd, [])

    proxy_tvs = []
    for path in proxy_paths:
        proxy_sd = load_state_dict(path)
        proxy_tvs.append(state_dict_to_vector(proxy_sd, []) - flat_ptm)
    if not proxy_tvs:
        raise ValueError("No proxy checkpoints available for triggered drift bank")
    proxy_stack = torch.vstack(proxy_tvs)

    sample_batches = []
    pre_logits = []
    sample_count = 0
    logger.info(
        "collecting triggered samples for Phase B drift bank: max_samples=%d",
        max_samples,
    )
    with torch.no_grad():
        for batch in train_loader:
            if sample_count >= max_samples:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"]
            remaining = max_samples - sample_count
            if images.shape[0] > remaining:
                images = images[:remaining]
            bd_images = apply_trigger_differentiable(images.to(device), trigger)
            logits = classification_head(pretrained_image_encoder(bd_images))
            sample_batches.append(images.cpu())
            pre_logits.append(logits.detach().cpu())
            sample_count += images.shape[0]

    if not sample_batches:
        raise RuntimeError("No samples collected for triggered drift bank")
    pre_logits = torch.cat(pre_logits, dim=0)

    triggered_drift_bank = []
    num_proxies = proxy_stack.shape[0]
    rng = random.Random(args.seed + 9109)
    model_args = copy.copy(args)

    for bg_idx in range(num_backgrounds):
        num_selected = rng.randint(1, min(5, num_proxies))
        if num_proxies >= 3:
            num_selected = rng.randint(3, min(5, num_proxies))
        selected = rng.sample(range(num_proxies), num_selected)
        logger.info(
            "Phase B triggered drift bg %d/%d: using proxies %s",
            bg_idx + 1,
            num_backgrounds,
            selected,
        )

        bg_vector = flat_ptm + merge_lambda * proxy_stack[selected].sum(dim=0)
        bg_state = vector_to_state_dict(bg_vector, pretrained_sd, [])
        bg_encoder = ImageEncoder(model_args, keep_lang=False).to(device)
        bg_encoder.load_state_dict(bg_state, strict=False)
        bg_encoder.eval()
        for param in bg_encoder.parameters():
            param.requires_grad_(False)

        bg_logits = []
        with torch.no_grad():
            for images in sample_batches:
                bd_images = apply_trigger_differentiable(images.to(device), trigger)
                logits = classification_head(bg_encoder(bd_images))
                bg_logits.append(logits.detach().cpu())
        bg_logits = torch.cat(bg_logits, dim=0)

        drift = bg_logits - pre_logits
        triggered_drift_bank.append(drift)
        logger.info(
            "  triggered drift norm mean=%.4f std=%.4f",
            drift.norm(dim=1).mean().item(),
            drift.norm(dim=1).std().item(),
        )

        del bg_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return triggered_drift_bank


def load_clean_ta_merged_encoder(args, pretrained_path, device):
    if args.background_merge_type != "ta":
        raise ValueError(f"Unsupported background merge type: {args.background_merge_type}")

    datasets = [x.strip() for x in args.exam_datasets.split(",") if x.strip()]
    root = os.path.join(args.ckpt_dir, args.model)
    ptm_sd = load_state_dict(pretrained_path)
    flat_ptm = state_dict_to_vector(ptm_sd, [])

    clean_tvs = []
    clean_paths = {}
    for ds in datasets:
        path = os.path.join(root, ds, "finetuned.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        clean_paths[ds] = path
        clean_sd = load_state_dict(path)
        clean_tvs.append(state_dict_to_vector(clean_sd, []) - flat_ptm)

    clean_stack = torch.vstack(clean_tvs)
    clean_merged = flat_ptm + args.scaling_coef * clean_stack.sum(dim=0)
    clean_state = vector_to_state_dict(clean_merged, ptm_sd, [])

    clean_encoder = ImageEncoder(args, keep_lang=False).to(device)
    clean_encoder.load_state_dict(clean_state, strict=False)
    clean_encoder.eval()
    for param in clean_encoder.parameters():
        param.requires_grad_(False)

    logger.info(
        "loaded clean TA background encoder: datasets=%s scaling=%.3f paths=%s",
        datasets,
        args.scaling_coef,
        clean_paths,
    )
    return clean_encoder, clean_paths


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = args.adversary_task
    use_background_anchor = args.background_anchor or args.method_name == "SubMergeBPA"
    use_selectivity_loss = args.selectivity_loss or args.method_name in (
        "SubMergeSelect",
        "SubMergeSelectM8",
    )
    use_fopa = args.fopa or args.method_name.startswith("FOPA")
    args.epochs = args.epochs if args.epochs is not None else EPOCHS.get(dataset, 5)
    args.save_dir = args.save_dir or os.path.join(args.ckpt_dir, args.model)
    args.save = args.save_dir
    run_name = f"{dataset}_{args.method_name}_{dataset}_Tgt_{args.target_cls}_L_{args.patch_size}"
    output_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    nullspace_info, nullspace_stats, nullspace_path = load_nullspace(args)
    eps_bounds, eps_path = load_dense_bounds(args)

    pretrained_path = os.path.join(args.ckpt_dir, args.model, "zeroshot.pt")
    image_encoder = ImageEncoder(args, keep_lang=False).to(device)
    image_encoder.load_state_dict(load_state_dict(pretrained_path), strict=False)
    pretrained_image_encoder = ImageEncoder(args, keep_lang=False).to(device)
    pretrained_image_encoder.load_state_dict(load_state_dict(pretrained_path), strict=False)
    pretrained_image_encoder.eval()
    for param in pretrained_image_encoder.parameters():
        param.requires_grad_(False)

    clean_merged_image_encoder = None
    clean_background_paths = None
    if use_background_anchor:
        clean_merged_image_encoder, clean_background_paths = load_clean_ta_merged_encoder(
            args,
            pretrained_path,
            device,
        )

    drift_bank = None
    hook_mapping = None
    resolved_drift_bank_path = args.drift_bank_path
    if use_fopa:
        if resolved_drift_bank_path is None:
            resolved_drift_bank_path = os.path.join(
                args.nullspace_dir,
                args.model,
                "drift_bank_bg8_s2000_lam0.3.pt",
            )
        if not os.path.exists(resolved_drift_bank_path):
            raise FileNotFoundError(
                f"Drift bank not found: {resolved_drift_bank_path}. "
                "Run run_fopa_nullspace.sh first."
            )
        drift_bank = load_drift_bank(resolved_drift_bank_path, device="cpu")
        hook_indices = [
            int(x.strip()) for x in args.fopa_hook_layers.split(",") if x.strip()
        ]
        hook_mapping = build_hook_mapping(hook_indices, nullspace_info)
        logger.info("FOPA hook mapping: %s", hook_mapping)
        if not hook_mapping:
            logger.warning("No FOPA hook layers matched nullspace keys; L_invariance will be 0.")
    phase_b_start = int(args.epochs * (1.0 - args.phase_b_ratio))
    phase_b_start = max(0, min(args.epochs, phase_b_start))
    phase_b_started = False

    classification_head = get_classification_head(make_badmerging_args(args, dataset), dataset).to(device)
    classification_head.eval()
    for param in classification_head.parameters():
        param.requires_grad_(False)

    # ---- PA-SubMerge target prototype ----
    # In CLIP-style classifiers, each row of the frozen classification head is a
    # class prototype. Aligning triggered features/feature shifts to this row
    # directly increases the target logit.
    with torch.no_grad():
        w_target = classification_head.weight[args.target_cls].detach().clone().to(device)
        w_target = F.normalize(w_target, dim=0)
    logger.info(
        "PA-SubMerge target prototype prepared: class=%d dim=%d",
        args.target_cls,
        w_target.numel(),
    )

    pretrained_state = OrderedDict(
        (k, v.detach().clone().to(device)) for k, v in image_encoder.state_dict().items()
        if torch.is_tensor(v)
    )
    preprocess = image_encoder.train_preprocess
    train_dataset, train_loader = get_dataset(
        dataset,
        "train",
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    logger.info("training dataset %s size=%d batches=%d", dataset, len(train_dataset), len(train_loader))

    trigger, trig_low, trig_high = initialize_trigger(args, preprocess, device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    model_params = [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": model_params, "lr": args.lr},
            {"params": [trigger], "lr": args.trigger_lr, "weight_decay": 0.0},
        ],
        lr=args.lr,
        weight_decay=args.wd,
    )
    scheduler = cosine_lr(optimizer, [args.lr, args.trigger_lr], args.warmup_length, args.epochs * len(train_loader))

    config = vars(args).copy()
    config.update({
        "nullspace_path": nullspace_path,
        "eps_bounds_path": eps_path,
        "pretrained_path": pretrained_path,
        "nullspace_stats": nullspace_stats,
        "output_dir": output_dir,
        "use_background_anchor": use_background_anchor,
        "use_selectivity_loss": use_selectivity_loss,
        "use_fopa": use_fopa,
        "lambda_inv": args.lambda_inv,
        "fopa_hook_layers": args.fopa_hook_layers,
        "fopa_two_phase": args.fopa_two_phase,
        "phase_b_ratio": args.phase_b_ratio,
        "phase_b_start": phase_b_start,
        "drift_bank_path": resolved_drift_bank_path,
        "clean_background_paths": clean_background_paths,
    })
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    if args.test_only:
        return output_dir

    log_path = os.path.join(output_dir, "train_log.jsonl")
    with open(log_path, "w") as log_f:
        for epoch in range(args.epochs):
            if (
                use_fopa
                and args.fopa_two_phase
                and not phase_b_started
                and epoch == phase_b_start
            ):
                logger.info(
                    "=== Entering FOPA Phase B: freezing trigger, rebuilding triggered drift bank ==="
                )
                trigger.requires_grad_(False)
                model_params = [p for p in image_encoder.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(
                    [{"params": model_params, "lr": args.lr}],
                    lr=args.lr,
                    weight_decay=args.wd,
                )
                scheduler = cosine_lr(
                    optimizer,
                    [args.lr],
                    args.warmup_length,
                    args.epochs * len(train_loader),
                )

                image_encoder.eval()
                drift_bank = build_triggered_drift_bank(
                    args=args,
                    train_loader=train_loader,
                    trigger=trigger,
                    pretrained_path=pretrained_path,
                    pretrained_image_encoder=pretrained_image_encoder,
                    classification_head=classification_head,
                    num_backgrounds=len(drift_bank),
                    max_samples=drift_bank[0].shape[0],
                    merge_lambda=args.scaling_coef,
                    device=device,
                )
                phase_b_started = True
                image_encoder.train()
                logger.info(
                    "Phase B triggered drift bank ready, %d backgrounds",
                    len(drift_bank),
                )
            image_encoder.train()
            clean_loss_sum = 0.0
            bd_loss_sum = 0.0
            anchor_loss_sum = 0.0
            shift_cos_sum = 0.0
            feat_cos_sum = 0.0
            margin_loss_sum = 0.0
            margin_sum = 0.0
            margin_positive_sum = 0.0
            target_logit_sum = 0.0
            max_other_logit_sum = 0.0
            select_loss_sum = 0.0
            select_margin_sum = 0.0
            select_target_gain_sum = 0.0
            select_competing_gain_sum = 0.0
            select_success_sum = 0.0
            fopa_loss_sum = 0.0
            inv_loss_sum = 0.0
            drift_norm_sum = 0.0
            steps = 0
            last_leakage = 0.0
            for batch_idx, batch in enumerate(train_loader):
                start = time.time()
                step = epoch * len(train_loader) + batch_idx
                scheduler(step)
                optimizer.zero_grad(set_to_none=True)
                batch = maybe_dictionarize(batch)
                images = batch["images"].to(device)
                labels = batch["labels"].to(device)

                clean_logits = classification_head(image_encoder(images))
                loss_clean = loss_fn(clean_logits, labels)

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

                fopa_activations = {}
                fopa_hooks = []
                if use_fopa and hook_mapping:
                    fopa_activations, fopa_hooks = register_activation_hooks(
                        image_encoder,
                        hook_mapping,
                    )
                features_bd = image_encoder(bd_images)
                for fh in fopa_hooks:
                    fh.remove()
                with torch.no_grad():
                    features_bd_pre = pretrained_image_encoder(bd_images)

                # r should match deployment scale. For TA scaling=0.3, [0.2,0.4] is the
                # default robust local neighborhood.
                r = random.uniform(args.merge_sim_r1, args.merge_sim_r2)
                if use_background_anchor:
                    with torch.no_grad():
                        features_bg = clean_merged_image_encoder(bd_images)
                    payload_shift = features_bd - features_bd_pre
                    features_merged_sim = features_bg + r * payload_shift
                else:
                    features_bg = features_bd_pre
                    features_merged_sim = features_bd * r + features_bd_pre * (1.0 - r)

                bd_logits = classification_head(features_merged_sim)
                logits_pre = classification_head(features_bd_pre)
                loss_bd = loss_fn(bd_logits, bd_labels)
                if use_fopa and drift_bank is not None:
                    bg_idx = random.randint(0, len(drift_bank) - 1)
                    db = drift_bank[bg_idx]
                    n_db = db.shape[0]
                    n_bd = bd_logits.shape[0]
                    sample_idx = torch.randint(0, n_db, (n_bd,))
                    d_bg = db[sample_idx].to(device=device, dtype=bd_logits.dtype)
                    z_fopa = bd_logits + d_bg
                    loss_fopa = loss_fn(z_fopa, bd_labels)
                    drift_norm = d_bg.detach().float().norm(dim=1).mean()
                else:
                    loss_fopa = torch.zeros((), device=device)
                    drift_norm = torch.zeros((), device=device)
                if use_fopa and fopa_activations:
                    loss_inv = compute_activation_invariance_loss(
                        fopa_activations,
                        nullspace_info,
                        device,
                    )
                else:
                    loss_inv = torch.zeros((), device=device)
                target_logit = bd_logits[:, args.target_cls]
                other_logits = bd_logits.clone()
                other_logits[:, args.target_cls] = -float("inf")
                max_other_logit = other_logits.max(dim=1).values
                margin = target_logit - max_other_logit
                if use_background_anchor:
                    loss_margin = F.relu(args.margin_target - margin).mean()
                else:
                    loss_margin = torch.zeros((), device=device)

                if use_selectivity_loss:
                    delta_logits = bd_logits - logits_pre
                    select_target_gain = delta_logits[:, args.target_cls]
                    other_delta_logits = delta_logits.clone()
                    other_delta_logits[:, args.target_cls] = -1e9
                    if args.select_use_logsumexp:
                        select_competing_gain = torch.logsumexp(other_delta_logits, dim=1)
                    else:
                        select_competing_gain = other_delta_logits.max(dim=1).values
                    select_margin = select_target_gain - select_competing_gain
                    loss_select = F.relu(
                        args.select_margin_threshold - select_margin
                    ).mean()
                    select_success_ratio = (
                        select_margin > args.select_margin_threshold
                    ).float().mean()
                else:
                    select_target_gain = torch.zeros_like(target_logit)
                    select_competing_gain = torch.zeros_like(target_logit)
                    select_margin = torch.zeros_like(target_logit)
                    loss_select = torch.zeros((), device=device)
                    select_success_ratio = torch.zeros((), device=device)

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
                    feature_shift = features_merged_sim - features_bg.detach()

                    shift_norm = F.normalize(feature_shift, dim=-1, eps=1e-12)
                    feat_norm = F.normalize(features_merged_sim, dim=-1, eps=1e-12)

                    target_proto = w_target.unsqueeze(0).expand_as(feat_norm)

                    shift_cos_vec = F.cosine_similarity(shift_norm, target_proto, dim=-1)
                    feat_cos_vec = F.cosine_similarity(feat_norm, target_proto, dim=-1)

                    if use_background_anchor:
                        loss_anchor_shift = -shift_cos_vec.mean()
                        loss_anchor_feat = -feat_cos_vec.mean()
                    else:
                        # Non-negative losses. 0 means perfect alignment.
                        loss_anchor_shift = (1.0 - shift_cos_vec).mean()
                        loss_anchor_feat = (1.0 - feat_cos_vec).mean()
                    loss_anchor = loss_anchor_shift + args.anchor_feat_weight * loss_anchor_feat

                    shift_cos = shift_cos_vec.mean().detach()
                    feat_cos = feat_cos_vec.mean().detach()

                if use_background_anchor:
                    loss = (
                        loss_clean
                        + args.alpha * loss_bd
                        + args.lambda_anchor * loss_anchor
                        + args.lambda_margin * loss_margin
                    )
                else:
                    if use_fopa:
                        loss = loss_clean + args.alpha * loss_fopa
                    else:
                        loss = loss_clean + args.alpha * loss_bd
                    if not args.disable_anchor:
                        loss = loss + args.lambda_anchor * loss_anchor
                    if use_selectivity_loss:
                        loss = loss + args.lambda_select * loss_select
                    if use_fopa:
                        loss = loss + args.lambda_inv * loss_inv
                loss.backward()
                projected, frozen = project_gradients_(
                    image_encoder,
                    nullspace_info,
                    train_unprojected=args.no_freeze_unprojected,
                    projection_strength=args.projection_strength,
                )
                torch.nn.utils.clip_grad_norm_(model_params, 1.0)
                optimizer.step()
                with torch.no_grad():
                    for c in range(3):
                        trigger.data[c].clamp_(trig_low[c].min().item(), trig_high[c].max().item())
                    last_leakage = project_model_delta_(
                        image_encoder,
                        pretrained_state,
                        nullspace_info,
                        train_unprojected=args.no_freeze_unprojected,
                        projection_strength=args.projection_strength,
                    )

                clean_loss_sum += loss_clean.item()
                bd_loss_sum += loss_bd.item()
                anchor_loss_sum += float(loss_anchor.detach().item())
                shift_cos_sum += float(shift_cos.detach().item())
                feat_cos_sum += float(feat_cos.detach().item())
                margin_loss_sum += float(loss_margin.detach().item())
                margin_sum += float(margin.detach().mean().item())
                margin_positive_sum += float((margin.detach() > 0).float().mean().item())
                target_logit_sum += float(target_logit.detach().mean().item())
                max_other_logit_sum += float(max_other_logit.detach().mean().item())
                select_loss_sum += float(loss_select.detach().item())
                select_margin_sum += float(select_margin.detach().mean().item())
                select_target_gain_sum += float(select_target_gain.detach().mean().item())
                select_competing_gain_sum += float(select_competing_gain.detach().mean().item())
                select_success_sum += float(select_success_ratio.detach().item())
                fopa_loss_sum += float(loss_fopa.detach().item()) if use_fopa else 0.0
                inv_loss_sum += float(loss_inv.detach().item()) if use_fopa else 0.0
                drift_norm_sum += float(drift_norm.detach().item()) if use_fopa else 0.0
                steps += 1
                if step % 20 == 0:
                    logger.info(
                        "epoch=%d step=%d/%d clean=%.4f bd=%.4f fopa=%.4f inv=%.4f drift=%.4f anchor=%.4f select_loss=%.4f select_margin=%.4f target_gain=%.4f competing_gain=%.4f select_success=%.4f margin_loss=%.4f margin=%.4f margin_pos=%.4f target_logit=%.4f max_other=%.4f shift_cos=%.4f feat_cos=%.4f projected=%d frozen=%d leak=%.6f time=%.2fs",
                        epoch,
                        batch_idx,
                        len(train_loader),
                        loss_clean.item(),
                        loss_bd.item(),
                        float(loss_fopa.detach().item()),
                        float(loss_inv.detach().item()),
                        float(drift_norm.detach().item()),
                        float(loss_anchor.detach().item()),
                        float(loss_select.detach().item()),
                        float(select_margin.detach().mean().item()),
                        float(select_target_gain.detach().mean().item()),
                        float(select_competing_gain.detach().mean().item()),
                        float(select_success_ratio.detach().item()),
                        float(loss_margin.detach().item()),
                        float(margin.detach().mean().item()),
                        float((margin.detach() > 0).float().mean().item()),
                        float(target_logit.detach().mean().item()),
                        float(max_other_logit.detach().mean().item()),
                        float(shift_cos.detach().item()),
                        float(feat_cos.detach().item()),
                        projected,
                        frozen,
                        last_leakage,
                        time.time() - start,
                    )

            record = {
                "epoch": epoch,
                "avg_clean_loss": clean_loss_sum / max(steps, 1),
                "avg_bd_loss": bd_loss_sum / max(steps, 1),
                "avg_anchor_loss": anchor_loss_sum / max(steps, 1),
                "avg_shift_cos": shift_cos_sum / max(steps, 1),
                "avg_feat_cos": feat_cos_sum / max(steps, 1),
                "avg_margin_loss": margin_loss_sum / max(steps, 1),
                "avg_margin_mean": margin_sum / max(steps, 1),
                "avg_margin_positive_ratio": margin_positive_sum / max(steps, 1),
                "avg_target_logit_mean": target_logit_sum / max(steps, 1),
                "avg_max_other_logit_mean": max_other_logit_sum / max(steps, 1),
                "avg_select_loss": select_loss_sum / max(steps, 1),
                "avg_select_margin": select_margin_sum / max(steps, 1),
                "avg_select_target_gain": select_target_gain_sum / max(steps, 1),
                "avg_select_competing_gain": select_competing_gain_sum / max(steps, 1),
                "avg_select_success_ratio": select_success_sum / max(steps, 1),
                "avg_fopa_loss": fopa_loss_sum / max(steps, 1),
                "avg_inv_loss": inv_loss_sum / max(steps, 1),
                "avg_drift_norm": drift_norm_sum / max(steps, 1),
                "last_null_leakage": last_leakage,
                "projection_strength": args.projection_strength,
                "lambda_anchor": args.lambda_anchor,
                "anchor_feat_weight": args.anchor_feat_weight,
                "lambda_margin": args.lambda_margin,
                "margin_target": args.margin_target,
                "use_background_anchor": use_background_anchor,
                "use_selectivity_loss": use_selectivity_loss,
                "use_fopa": use_fopa,
                "lambda_inv": args.lambda_inv if use_fopa else 0.0,
                "lambda_select": args.lambda_select,
                "select_margin_threshold": args.select_margin_threshold,
                "select_use_logsumexp": args.select_use_logsumexp,
            }
            if args.eval_every_epoch:
                image_encoder.eval()
                record.update(evaluate_attack(image_encoder, dataset, args, trigger))
            log_f.write(json.dumps(record, default=str) + "\n")
            log_f.flush()
            logger.info("epoch summary %s", record)

    ft_path = os.path.join(output_dir, "finetuned.pt")
    trigger_dir = args.trigger_dir or os.path.join("./trigger", args.model)
    trigger_path = os.path.join(
        trigger_dir,
        f"{args.method_name}_{dataset}_Tgt_{args.target_cls}_L_{args.patch_size}.npy",
    )
    os.makedirs(os.path.dirname(trigger_path), exist_ok=True)

    image_encoder.eval()
    image_encoder.save(ft_path)
    np.save(trigger_path, trigger.detach().cpu().numpy())
    logger.info("saved pre-eval model to %s", ft_path)
    logger.info("saved trigger to %s", trigger_path)

    pre_dense_metrics = evaluate_attack(image_encoder, dataset, args, trigger)
    dense_stats = None
    if not args.skip_dense_encoding:
        logger.info("applying dense encoding")
        dense_stats = dense_encode_(image_encoder, pretrained_state, nullspace_info, eps_bounds)
        logger.info("dense stats %s", dense_stats)
        image_encoder.save(ft_path)
        logger.info("saved dense-encoded model to %s", ft_path)
    final_metrics = evaluate_attack(image_encoder, dataset, args, trigger)

    summary = {
        "pre_dense_metrics": pre_dense_metrics,
        "final_metrics": final_metrics,
        "dense_stats": dense_stats,
        "finetuned_path": ft_path,
        "trigger_path": trigger_path,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("saved model to %s", ft_path)
    logger.info("saved trigger to %s", trigger_path)
    return output_dir


if __name__ == "__main__":
    train(parse_args())
