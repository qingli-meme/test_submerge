# ============================================================
# KDR-DTK Exact-Projection Counterfactual Key Binding
# ============================================================

import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import tqdm

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
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
    find_module,
    freeze_classification_head,
    freeze_model,
    load_image_encoder_from_checkpoint,
    load_trigger_patch,
    pool_activation,
    residual_l2_norm,
    sample_synthetic_drift_cache,
    save_image_encoder,
    save_json,
    set_seed,
)
from src.modeling import ImageEncoder


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
        "KDR-DTK exact-projection counterfactual key binding"
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--openclip-cachedir", default="./open_clip")
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--trigger-path", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--save-root", default="./checkpoints")
    parser.add_argument("--method-name", default="KDR_DTK_BIND_EXACT")
    parser.add_argument(
        "--key-layer",
        default="model.visual.transformer.resblocks.11.ln_2",
    )
    parser.add_argument("--key-pool", default="cls", choices=["cls"])
    parser.add_argument("--key-prototype-batches", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bd-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--trainable-scope",
        default="all",
        choices=["all", "last2", "last1", "ln_post"],
    )
    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--eta-min", type=float, default=0.2)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--drift-rho", type=float, default=0.25)
    parser.add_argument(
        "--drift-scope",
        default="last2",
        choices=["all", "last2", "last1", "ln_post"],
    )
    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--bd-weight", type=float, default=1.0)
    parser.add_argument("--bind-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=0.0)
    parser.add_argument("--margin-eps", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def build_output_dir(args):
    attack_type = (
        f"{args.method_name}_{args.adversary_task}_Tgt_"
        f"{args.target_cls}_L_{args.patch_size}"
    )
    return ensure_dir(os.path.join(args.save_root, args.model, f"{args.adversary_task}_{attack_type}"))


def mean_record(records, key):
    vals = [record[key] for record in records if key in record]
    return float(sum(vals) / max(len(vals), 1))


def resolve_layer_name(model, requested):
    modules = dict(model.named_modules())
    if requested in modules:
        return requested

    suffix_matches = [
        name for name in modules if name.endswith(requested) or requested.endswith(name)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    tail = ".".join(requested.split(".")[-6:])
    tail_matches = [name for name in modules if name.endswith(tail)]
    if len(tail_matches) != 1:
        raise RuntimeError(
            f"Cannot uniquely resolve key layer '{requested}'. "
            f"suffix_matches={suffix_matches[:20]}, tail_matches={tail_matches[:20]}"
        )
    return tail_matches[0]


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    return target - masked.max(dim=1).values


def capture_regular_layer_activation(model, images, resolved_layer_name, pool):
    holder = {}
    module = find_module(model, resolved_layer_name)

    def hook_fn(_, __, output):
        holder["activation"] = output

    handle = module.register_forward_hook(hook_fn)
    try:
        features = model(images)
    finally:
        handle.remove()

    if "activation" not in holder:
        raise RuntimeError(f"Hook did not capture key layer: {resolved_layer_name}")

    activation = pool_activation(
        holder["activation"],
        batch_size=images.shape[0],
        pool=pool,
    )
    return features, activation


def capture_functional_layer_activation(model, params, images, resolved_layer_name, pool):
    holder = {}
    module = find_module(model, resolved_layer_name)

    def hook_fn(_, __, output):
        holder["activation"] = output

    handle = module.register_forward_hook(hook_fn)
    try:
        features = call_with_params(model, params, images)
    finally:
        handle.remove()

    if "activation" not in holder:
        raise RuntimeError(f"Functional hook did not capture key layer: {resolved_layer_name}")

    activation = pool_activation(
        holder["activation"],
        batch_size=images.shape[0],
        pool=pool,
    )
    return features, activation


def _edit_tensor_cls(output, delta, batch_size):
    edited = output.clone()
    delta = delta.to(device=edited.device, dtype=edited.dtype)

    if edited.ndim == 2:
        if edited.shape[0] != batch_size:
            raise RuntimeError(
                f"2D activation shape mismatch: shape={tuple(edited.shape)}, batch_size={batch_size}"
            )
        return edited + delta

    if edited.ndim != 3:
        raise RuntimeError(f"Expected 2D/3D key-layer output, got {tuple(edited.shape)}")

    if edited.shape[1] == batch_size:
        edited[0, :, :] = edited[0, :, :] + delta
        return edited

    if edited.shape[0] == batch_size:
        edited[:, 0, :] = edited[:, 0, :] + delta
        return edited

    raise RuntimeError(
        f"Cannot identify batch axis for activation {tuple(edited.shape)} with batch_size={batch_size}"
    )


def edit_layer_output(output, delta, batch_size):
    if torch.is_tensor(output):
        return _edit_tensor_cls(output, delta=delta, batch_size=batch_size)

    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            raise RuntimeError("Unsupported tuple key-layer output")
        return (
            _edit_tensor_cls(output[0], delta=delta, batch_size=batch_size),
            *output[1:],
        )

    if isinstance(output, list):
        if not output or not torch.is_tensor(output[0]):
            raise RuntimeError("Unsupported list key-layer output")
        edited = list(output)
        edited[0] = _edit_tensor_cls(edited[0], delta=delta, batch_size=batch_size)
        return edited

    raise RuntimeError(f"Unsupported key-layer output type: {type(output)}")


def functional_forward_with_layer_delta(model, params, images, resolved_layer_name, delta):
    module = find_module(model, resolved_layer_name)
    batch_size = images.shape[0]

    def hook_fn(_, __, output):
        return edit_layer_output(output, delta=delta, batch_size=batch_size)

    handle = module.register_forward_hook(hook_fn)
    try:
        features = call_with_params(model, params, images)
    finally:
        handle.remove()
    return features


@torch.no_grad()
def estimate_empirical_stage1_key(args, base_image_encoder, train_loader, trigger, device):
    resolved_layer = resolve_layer_name(base_image_encoder, args.key_layer)
    normalized_shift_sum = None
    batch_prototypes = []
    within_batch_cosines = []
    shift_norms = []
    total_samples = 0

    base_image_encoder.eval()
    for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc="estimate-dtk-binding-key")):
        if batch_idx >= args.key_prototype_batches:
            break

        batch = maybe_dictionarize(batch)
        images = batch["images"].to(device)
        patched = apply_trigger(images, trigger)

        _, clean_act = capture_regular_layer_activation(
            base_image_encoder,
            images,
            resolved_layer,
            args.key_pool,
        )
        _, patched_act = capture_regular_layer_activation(
            base_image_encoder,
            patched,
            resolved_layer,
            args.key_pool,
        )

        shift = patched_act - clean_act
        shift_hat = F.normalize(shift, dim=-1, eps=1e-12)
        batch_proto = F.normalize(shift_hat.mean(dim=0), dim=0, eps=1e-12)
        within_batch_cosines.append(torch.matmul(shift_hat, batch_proto).detach().float().cpu())
        shift_norms.append(shift.norm(dim=-1).detach().float().cpu())
        batch_prototypes.append(batch_proto.detach().float().cpu())

        batch_sum = shift_hat.sum(dim=0)
        normalized_shift_sum = batch_sum if normalized_shift_sum is None else normalized_shift_sum + batch_sum
        total_samples += int(shift_hat.shape[0])

    if normalized_shift_sum is None or total_samples <= 0:
        raise RuntimeError("No samples used to estimate DTK binding key")

    key = F.normalize(normalized_shift_sum, dim=0, eps=1e-12)
    batch_proto_tensor = torch.stack(batch_prototypes, dim=0)
    batch_to_global = torch.matmul(batch_proto_tensor.to(key.device), key)
    within = torch.cat(within_batch_cosines, dim=0)
    norms = torch.cat(shift_norms, dim=0)

    stats = {
        "resolved_layer": resolved_layer,
        "num_samples": total_samples,
        "num_batches": len(batch_prototypes),
        "within_batch_cosine_mean": float(within.mean().item()),
        "within_batch_cosine_median": float(within.median().item()),
        "batch_to_global_cosine_mean": float(batch_to_global.mean().item()),
        "batch_to_global_cosine_median": float(batch_to_global.median().item()),
        "batch_to_global_cosine_q10": float(torch.quantile(batch_to_global, 0.1).item()),
        "shift_norm_mean": float(norms.mean().item()),
        "shift_norm_median": float(norms.median().item()),
    }
    return key.detach(), stats


def main():
    args = parse_args()
    args.save = os.path.join(args.save_root, args.model)
    set_seed(args.seed)

    if not os.path.exists(args.trigger_path):
        raise FileNotFoundError(args.trigger_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.epochs = args.epochs if args.epochs is not None else EPOCHS.get(args.adversary_task, 5)

    output_dir = build_output_dir(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{timestamp}_dtk_bind_exact_train_log.jsonl")
    config_path = os.path.join(output_dir, f"{timestamp}_dtk_bind_exact_config.json")
    summary_path = os.path.join(output_dir, f"{timestamp}_dtk_bind_exact_summary.json")
    ckpt_path = os.path.join(output_dir, "finetuned.pt")
    key_path = os.path.join(output_dir, "binding_key.pt")

    pretrained_path = os.path.join(args.save_root, args.model, "zeroshot.pt")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(pretrained_path)

    base_image_encoder = load_image_encoder_from_checkpoint(args, pretrained_path, device)
    freeze_model(base_image_encoder)
    base_image_encoder.eval()

    if args.init_checkpoint:
        image_encoder = load_image_encoder_from_checkpoint(args, args.init_checkpoint, device)
    else:
        image_encoder = ImageEncoder(args, keep_lang=False).to(device)

    choose_trainable_scope(image_encoder, args.trainable_scope)
    image_encoder.train()
    trainable_stats = count_trainable_params(image_encoder)

    classification_head = get_classification_head(args, args.adversary_task).to(device)
    freeze_classification_head(classification_head)

    _, train_loader = get_dataset(
        args.adversary_task,
        "train",
        image_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    trigger = load_trigger_patch(args.trigger_path, args.patch_size, device)
    trigger.requires_grad_(False)

    binding_key, binding_key_stats = estimate_empirical_stage1_key(
        args,
        base_image_encoder,
        train_loader,
        trigger,
        device,
    )
    binding_key = binding_key.to(device)
    binding_key.requires_grad_(False)
    resolved_key_layer = binding_key_stats["resolved_layer"]

    torch.save(
        {
            "key": binding_key.detach().cpu(),
            "stats": binding_key_stats,
            "trigger_path": args.trigger_path,
            "key_layer": resolved_key_layer,
            "prototype_batches": args.key_prototype_batches,
        },
        key_path,
    )

    print("[DTK-BIND-EXACT] binding key:", binding_key_stats, flush=True)

    drift_keywords = default_drift_keywords(args.drift_scope)
    trainable_params = [parameter for parameter in image_encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

    save_json(
        {
            "method": "KDR-DTK Exact-Projection Counterfactual Key Binding",
            "stage1": "reuse fixed KDR_DTK trigger",
            "stage2_change": "replace residual target-gain loss with key-removal binding loss",
            "binding_definition": (
                "full attack margin must be positive; after removing the empirical DTK key "
                "component at the fixed Stage-1 key layer, target margin must be non-positive"
            ),
            "key_component_gradient": (
                "exact differentiable projection removal; gradient through h_minus_key is "
                "I - kk^T at the fixed empirical key for the current step"
            ),
            "args": vars(args),
            "device": str(device),
            "pretrained_path": pretrained_path,
            "init_checkpoint": args.init_checkpoint or pretrained_path,
            "trigger_path": args.trigger_path,
            "binding_key_path": key_path,
            "binding_key_stats": binding_key_stats,
            "output_dir": output_dir,
            "ckpt_path": ckpt_path,
            "trainable_stats": trainable_stats,
            "drift_keywords": drift_keywords,
            "log_path": log_path,
        },
        config_path,
    )

    print("[DTK-BIND-EXACT] output_dir:", output_dir, flush=True)
    print("[DTK-BIND-EXACT] trainable:", trainable_stats, flush=True)
    print("[DTK-BIND-EXACT] trigger:", args.trigger_path, flush=True)

    global_step = 0
    start = time.time()
    last_epoch_records = []

    for epoch in range(args.epochs):
        epoch_records = []

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            bd_clean_images = images[: args.bd_batch_size]
            bd_images = apply_trigger(bd_clean_images, trigger)
            target_labels = torch.full(
                (bd_images.shape[0],),
                args.target_cls,
                dtype=torch.long,
                device=device,
            )

            alpha = random.uniform(args.alpha_min, args.alpha_max)
            eta = random.uniform(args.eta_min, args.eta_max)
            drift_cache = sample_synthetic_drift_cache(
                image_encoder,
                base_image_encoder,
                rho=args.drift_rho,
                drift_keywords=drift_keywords,
            )
            theta_atk_params = build_drifted_params_from_cache(
                image_encoder,
                base_image_encoder,
                drift_cache,
                alpha=alpha,
                eta=eta,
                include_adv=True,
            )
            theta_bg_params = build_drifted_params_from_cache(
                image_encoder,
                base_image_encoder,
                drift_cache,
                alpha=alpha,
                eta=eta,
                include_adv=False,
            )

            clean_logits = classification_logits(classification_head, image_encoder(images))
            clean_loss = F.cross_entropy(clean_logits, labels)

            with torch.no_grad():
                _, clean_proxy_act = capture_functional_layer_activation(
                    image_encoder,
                    theta_atk_params,
                    bd_clean_images,
                    resolved_key_layer,
                    args.key_pool,
                )

            z_atk, triggered_act = capture_functional_layer_activation(
                image_encoder,
                theta_atk_params,
                bd_images,
                resolved_key_layer,
                args.key_pool,
            )
            logits_atk = classification_logits(classification_head, z_atk)

            shift = triggered_act - clean_proxy_act.detach()
            shift_norm = shift.norm(dim=-1).clamp_min(1e-12)
            key_coeff = torch.matmul(shift, binding_key)
            key_cosine = key_coeff / shift_norm
            key_energy_ratio = key_coeff.square() / shift_norm.square()
            key_component = key_coeff.unsqueeze(1) * binding_key.unsqueeze(0)

            z_key_removed = functional_forward_with_layer_delta(
                image_encoder,
                theta_atk_params,
                bd_images,
                resolved_key_layer,
                -key_component,
            )
            logits_key_removed = classification_logits(classification_head, z_key_removed)

            full_margin = target_margin(logits_atk, args.target_cls)
            key_removed_margin = target_margin(logits_key_removed, args.target_cls)
            key_gain = full_margin - key_removed_margin

            bd_loss = F.cross_entropy(logits_atk, target_labels)
            bind_loss = F.relu(key_removed_margin).mean()
            margin_loss = F.relu(args.margin_eps - full_margin).mean()
            residual_norm = residual_l2_norm(image_encoder, base_image_encoder)

            with torch.no_grad():
                z_bg = call_with_params(image_encoder, theta_bg_params, bd_images)
                logits_bg = classification_logits(classification_head, z_bg)

            residual_target_gain = (
                logits_atk[:, args.target_cls] - logits_bg[:, args.target_cls].detach()
            )

            loss = (
                args.clean_weight * clean_loss
                + args.bd_weight * bd_loss
                + args.bind_weight * bind_loss
                + args.margin_weight * margin_loss
                + args.residual_weight * residual_norm
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                atk_pred = logits_atk.argmax(dim=1)
                removed_pred = logits_key_removed.argmax(dim=1)
                bg_pred = logits_bg.argmax(dim=1)
                record = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "alpha": float(alpha),
                    "eta": float(eta),
                    "loss": float(loss.item()),
                    "clean_loss": float(clean_loss.item()),
                    "bd_loss": float(bd_loss.item()),
                    "bind_loss": float(bind_loss.item()),
                    "margin_loss": float(margin_loss.item()),
                    "residual_l2": float(residual_norm.item()),
                    "key_gain_mean": float(key_gain.mean().item()),
                    "key_gain_min": float(key_gain.min().item()),
                    "full_margin_mean": float(full_margin.mean().item()),
                    "full_margin_min": float(full_margin.min().item()),
                    "key_removed_margin_mean": float(key_removed_margin.mean().item()),
                    "key_removed_margin_max": float(key_removed_margin.max().item()),
                    "key_cosine_mean": float(key_cosine.mean().item()),
                    "key_energy_ratio_mean": float(key_energy_ratio.mean().item()),
                    "key_coeff_abs_mean": float(key_coeff.abs().mean().item()),
                    "residual_target_gain_mean": float(residual_target_gain.mean().item()),
                    "atk_target_rate": float((atk_pred == args.target_cls).float().mean().item()),
                    "key_removed_target_rate": float(
                        (removed_pred == args.target_cls).float().mean().item()
                    ),
                    "bg_target_rate": float((bg_pred == args.target_cls).float().mean().item()),
                    "clean_acc": float((clean_logits.argmax(dim=1) == labels).float().mean().item()),
                }

            append_jsonl(record, log_path)
            epoch_records.append(record)
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[DTK-BIND-EXACT] "
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{len(train_loader)} "
                    f"loss={record['loss']:.4f} clean={record['clean_loss']:.4f} "
                    f"bd={record['bd_loss']:.4f} bind={record['bind_loss']:.4f} "
                    f"key_gain={record['key_gain_mean']:.4f} "
                    f"M={record['full_margin_mean']:.4f} "
                    f"M(-k)={record['key_removed_margin_mean']:.4f} "
                    f"atk={record['atk_target_rate']:.3f} "
                    f"atk(-k)={record['key_removed_target_rate']:.3f}",
                    flush=True,
                )

        last_epoch_records = epoch_records
        epoch_summary = {
            "epoch": epoch,
            "steps": len(epoch_records),
            "elapsed_sec": time.time() - start,
        }
        for key in (
            "loss",
            "clean_loss",
            "bd_loss",
            "bind_loss",
            "margin_loss",
            "residual_l2",
            "key_gain_mean",
            "key_gain_min",
            "full_margin_mean",
            "full_margin_min",
            "key_removed_margin_mean",
            "key_removed_margin_max",
            "key_cosine_mean",
            "key_energy_ratio_mean",
            "key_coeff_abs_mean",
            "residual_target_gain_mean",
            "atk_target_rate",
            "key_removed_target_rate",
            "bg_target_rate",
            "clean_acc",
        ):
            epoch_summary[f"avg_{key}"] = mean_record(epoch_records, key)

        append_jsonl({"epoch_summary": epoch_summary}, log_path)
        print("[DTK-BIND-EXACT] epoch_summary:", epoch_summary, flush=True)

        if args.save_every_epoch:
            save_image_encoder(image_encoder, os.path.join(output_dir, f"finetuned_epoch_{epoch}.pt"))

    save_image_encoder(image_encoder, ckpt_path)
    summary = {
        "method": "KDR-DTK Exact-Projection Counterfactual Key Binding",
        "ckpt_path": ckpt_path,
        "binding_key_path": key_path,
        "binding_key_stats": binding_key_stats,
        "config_path": config_path,
        "log_path": log_path,
        "global_steps": global_step,
        "elapsed_sec": time.time() - start,
        "final_epoch": args.epochs - 1,
        "final_avg_loss": mean_record(last_epoch_records, "loss"),
        "final_avg_bind_loss": mean_record(last_epoch_records, "bind_loss"),
        "final_avg_key_gain": mean_record(last_epoch_records, "key_gain_mean"),
        "final_avg_full_margin": mean_record(last_epoch_records, "full_margin_mean"),
        "final_avg_key_removed_margin": mean_record(last_epoch_records, "key_removed_margin_mean"),
        "final_avg_atk_target_rate": mean_record(last_epoch_records, "atk_target_rate"),
        "final_avg_key_removed_target_rate": mean_record(last_epoch_records, "key_removed_target_rate"),
        "final_avg_residual_target_gain": mean_record(last_epoch_records, "residual_target_gain_mean"),
        "final_avg_key_cosine": mean_record(last_epoch_records, "key_cosine_mean"),
        "final_avg_key_energy_ratio": mean_record(last_epoch_records, "key_energy_ratio_mean"),
        "trainable_stats": trainable_stats,
    }
    save_json(summary, summary_path)
    print("[DTK-BIND-EXACT] saved checkpoint:", ckpt_path, flush=True)
    print("[DTK-BIND-EXACT] saved summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()
