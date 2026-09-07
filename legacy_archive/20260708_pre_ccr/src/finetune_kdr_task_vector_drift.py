# ============================================================
# KDR-TVD Training
# Task-Vector-Structured Drift Residual Editing
# ============================================================

import argparse
import os
import random
import sys
import time

import torch
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
    save_image_encoder,
    save_json,
    set_seed,
)
from src.task_vector_drift import (
    build_task_vector_delta_cache,
    sample_task_vector_structured_drift_cache,
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
        "KDR-TVD task-vector-structured drift residual editing"
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
    parser.add_argument("--task-reference-checkpoint", default="")
    parser.add_argument("--save-root", default="./checkpoints")
    parser.add_argument("--method-name", default="KDR_TVD")
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
        default="all",
        choices=["all", "last2", "last1", "ln_post"],
    )
    parser.add_argument("--task-drift-seed", type=int, default=32026)
    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--bd-weight", type=float, default=1.0)
    parser.add_argument("--gain-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=0.0)
    parser.add_argument("--gain-eps", type=float, default=0.05)
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
    run_name = f"{args.adversary_task}_{attack_type}"
    return ensure_dir(os.path.join(args.save_root, args.model, run_name))


def mean_record(records, key):
    vals = [r[key] for r in records if key in r]
    return float(sum(vals) / max(len(vals), 1))


def main():
    args = parse_args()
    args.save = os.path.join(args.save_root, args.model)
    set_seed(args.seed)

    if not os.path.exists(args.trigger_path):
        raise FileNotFoundError(args.trigger_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.epochs = (
        args.epochs
        if args.epochs is not None
        else EPOCHS.get(args.adversary_task, 5)
    )

    output_dir = build_output_dir(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{timestamp}_kdr_tvd_train_log.jsonl")
    config_path = os.path.join(output_dir, f"{timestamp}_kdr_tvd_config.json")
    summary_path = os.path.join(output_dir, f"{timestamp}_kdr_tvd_summary.json")
    ckpt_path = os.path.join(output_dir, "finetuned.pt")

    pretrained_path = os.path.join(args.save_root, args.model, "zeroshot.pt")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(pretrained_path)

    if args.task_reference_checkpoint:
        task_reference_path = args.task_reference_checkpoint
    else:
        task_reference_path = os.path.join(
            args.save_root,
            args.model,
            args.adversary_task,
            "finetuned.pt",
        )
    if not os.path.exists(task_reference_path):
        raise FileNotFoundError(task_reference_path)

    base_image_encoder = ImageEncoder(args, keep_lang=False).to(device)
    freeze_model(base_image_encoder)

    if args.init_checkpoint:
        image_encoder = load_image_encoder_from_checkpoint(
            args,
            args.init_checkpoint,
            device,
        )
    else:
        image_encoder = ImageEncoder(args, keep_lang=False).to(device)

    choose_trainable_scope(image_encoder, args.trainable_scope)
    image_encoder.train()
    trainable_stats = count_trainable_params(image_encoder)

    task_reference_encoder = load_image_encoder_from_checkpoint(
        args,
        task_reference_path,
        device,
    )
    freeze_model(task_reference_encoder)
    task_reference_encoder.eval()

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
    drift_keywords = default_drift_keywords(args.drift_scope)
    task_delta_cache, task_delta_stats = build_task_vector_delta_cache(
        task_reference_encoder,
        base_image_encoder,
        drift_keywords=drift_keywords,
    )

    del task_reference_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainable_params = [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

    save_json(
        {
            "method": "KDR-TVD",
            "change_from_kdr": (
                "Replace per-batch isotropic Gaussian orthogonal drift with "
                "epoch-wise task-vector-structured drift sampled from the "
                "attacker-visible clean task vector."
            ),
            "args": vars(args),
            "device": str(device),
            "pretrained_path": pretrained_path,
            "init_checkpoint": args.init_checkpoint or pretrained_path,
            "task_reference_checkpoint": task_reference_path,
            "trigger_path": args.trigger_path,
            "output_dir": output_dir,
            "ckpt_path": ckpt_path,
            "trainable_stats": trainable_stats,
            "drift_keywords": drift_keywords,
            "task_delta_stats": task_delta_stats,
            "log_path": log_path,
        },
        config_path,
    )

    print("[KDR-TVD] output_dir:", output_dir, flush=True)
    print("[KDR-TVD] trainable:", trainable_stats, flush=True)
    print("[KDR-TVD] trigger:", args.trigger_path, flush=True)
    print("[KDR-TVD] task reference:", task_reference_path, flush=True)
    print("[KDR-TVD] task delta:", task_delta_stats, flush=True)

    global_step = 0
    start = time.time()
    last_epoch_records = []
    epoch_drift_stats = []

    for epoch in range(args.epochs):
        drift_seed = args.task_drift_seed + epoch
        drift_cache, drift_stats = sample_task_vector_structured_drift_cache(
            task_delta_cache,
            rho=args.drift_rho,
            seed=drift_seed,
        )
        epoch_drift_stats.append(drift_stats)

        print(
            f"[KDR-TVD] epoch={epoch} structured_drift={drift_stats}",
            flush=True,
        )

        epoch_records = []
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            bd_images = apply_trigger(images[: args.bd_batch_size], trigger)
            target_labels = torch.full(
                (bd_images.shape[0],),
                args.target_cls,
                dtype=torch.long,
                device=device,
            )

            alpha = random.uniform(args.alpha_min, args.alpha_max)
            eta = random.uniform(args.eta_min, args.eta_max)
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

            clean_logits = classification_logits(
                classification_head,
                image_encoder(images),
            )
            clean_loss = F.cross_entropy(clean_logits, labels)

            z_atk = call_with_params(image_encoder, theta_atk_params, bd_images)
            logits_atk = classification_logits(classification_head, z_atk)
            with torch.no_grad():
                z_bg = call_with_params(image_encoder, theta_bg_params, bd_images)
                logits_bg = classification_logits(classification_head, z_bg)

            metrics = readout_metrics_from_logits(
                logits_atk,
                logits_bg,
                args.target_cls,
            )
            bd_loss = F.cross_entropy(logits_atk, target_labels)
            gain_loss = F.relu(args.gain_eps - metrics["target_gain"]).mean()
            margin_loss = F.relu(args.margin_eps - metrics["margin"]).mean()
            residual_norm = residual_l2_norm(image_encoder, base_image_encoder)

            loss = (
                args.clean_weight * clean_loss
                + args.bd_weight * bd_loss
                + args.gain_weight * gain_loss
                + args.margin_weight * margin_loss
                + args.residual_weight * residual_norm
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred = metrics["pred"]
                record = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "drift_seed": int(drift_seed),
                    "alpha": float(alpha),
                    "eta": float(eta),
                    "loss": float(loss.item()),
                    "clean_loss": float(clean_loss.item()),
                    "bd_loss": float(bd_loss.item()),
                    "gain_loss": float(gain_loss.item()),
                    "margin_loss": float(margin_loss.item()),
                    "residual_l2": float(residual_norm.item()),
                    "target_gain_mean": float(metrics["target_gain"].mean().item()),
                    "target_gain_min": float(metrics["target_gain"].min().item()),
                    "margin_mean": float(metrics["margin"].mean().item()),
                    "margin_min": float(metrics["margin"].min().item()),
                    "atk_target_rate": float((pred == args.target_cls).float().mean().item()),
                    "bg_target_rate": float(
                        (logits_bg.argmax(dim=1) == args.target_cls).float().mean().item()
                    ),
                    "clean_acc": float(
                        (clean_logits.argmax(dim=1) == labels).float().mean().item()
                    ),
                }

            append_jsonl(record, log_path)
            epoch_records.append(record)
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[KDR-TVD] "
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{len(train_loader)} "
                    f"loss={record['loss']:.4f} clean={record['clean_loss']:.4f} "
                    f"bd={record['bd_loss']:.4f} gain={record['target_gain_mean']:.4f} "
                    f"margin={record['margin_mean']:.4f} "
                    f"atk_rate={record['atk_target_rate']:.3f}",
                    flush=True,
                )

        last_epoch_records = epoch_records
        epoch_summary = {
            "epoch": epoch,
            "steps": len(epoch_records),
            "elapsed_sec": time.time() - start,
            "structured_drift": drift_stats,
        }
        for key in (
            "loss",
            "clean_loss",
            "bd_loss",
            "gain_loss",
            "margin_loss",
            "residual_l2",
            "target_gain_mean",
            "target_gain_min",
            "margin_mean",
            "margin_min",
            "atk_target_rate",
            "bg_target_rate",
            "clean_acc",
        ):
            epoch_summary[f"avg_{key}"] = mean_record(epoch_records, key)
        append_jsonl({"epoch_summary": epoch_summary}, log_path)
        print("[KDR-TVD] epoch_summary:", epoch_summary, flush=True)

        if args.save_every_epoch:
            epoch_ckpt = os.path.join(output_dir, f"finetuned_epoch_{epoch}.pt")
            save_image_encoder(image_encoder, epoch_ckpt)

        del drift_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_image_encoder(image_encoder, ckpt_path)
    summary = {
        "method": "KDR-TVD",
        "ckpt_path": ckpt_path,
        "config_path": config_path,
        "log_path": log_path,
        "global_steps": global_step,
        "elapsed_sec": time.time() - start,
        "final_epoch": args.epochs - 1,
        "final_avg_loss": mean_record(last_epoch_records, "loss"),
        "final_avg_target_gain": mean_record(last_epoch_records, "target_gain_mean"),
        "final_avg_margin": mean_record(last_epoch_records, "margin_mean"),
        "final_avg_atk_target_rate": mean_record(last_epoch_records, "atk_target_rate"),
        "trainable_stats": trainable_stats,
        "task_delta_stats": task_delta_stats,
        "epoch_drift_stats": epoch_drift_stats,
    }
    save_json(summary, summary_path)
    print("[KDR-TVD] saved checkpoint:", ckpt_path, flush=True)
    print("[KDR-TVD] saved summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()
