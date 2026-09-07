# ============================================================
# SMRC Training
# ============================================================
# Self-Merge Readout Calibration.
# Threat model: no benign task checkpoints, no victim task vectors, no
# final merged model. Training only uses theta_0, theta_adv, fixed trigger,
# and target-task data/head.

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
from src.smrc_utils import (
    append_jsonl,
    apply_trigger,
    build_self_merge_params,
    call_with_params,
    choose_trainable_scope,
    classification_logits,
    count_trainable_params,
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
    p = argparse.ArgumentParser("SMRC: Self-Merge Readout Calibration")
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
    p.add_argument("--method-name", default="SMRC")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bd-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--trainable-scope", default="all", choices=["all", "last2", "last1", "ln_post"])
    p.add_argument("--r-min", type=float, default=0.2)
    p.add_argument("--r-max", type=float, default=1.0)
    p.add_argument("--clean-weight", type=float, default=1.0)
    p.add_argument("--gain-weight", type=float, default=1.0)
    p.add_argument("--margin-weight", type=float, default=1.0)
    p.add_argument("--self-ce-weight", type=float, default=0.5)
    p.add_argument("--local-bd-ce-weight", type=float, default=0.0)
    p.add_argument("--residual-weight", type=float, default=0.0)
    p.add_argument("--gain-eps", type=float, default=0.05)
    p.add_argument("--margin-eps", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--save-every-epoch", action="store_true")
    return p.parse_args()


def build_output_dir(args) -> str:
    attack_type = f"On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    run_name = f"{args.adversary_task}_{args.method_name}_{attack_type}"
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
        f"[SMRC] trainable params: {param_info['trainable']}/"
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
    num_batches = len(train_loader)

    trigger = load_trigger_patch(args.trigger_path, args.patch_size, device=device)
    trigger.requires_grad_(False)

    params = [p for p in image_encoder.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters. Check --trainable-scope.")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    save_json(
        {
            "method": "SMRC",
            "proxy": "theta_r = theta_0 + r(theta_adv - theta_0)",
            "threat_model": "BadMerging-compatible: no benign task checkpoints during training.",
            "args": vars(args),
            "output_dir": output_dir,
            "trainable_params": param_info,
        },
        config_path,
    )

    print("[SMRC] Training start")
    print(f"[SMRC] output_dir={output_dir}")

    global_step = 0
    start = time.time()
    for epoch in range(args.epochs):
        image_encoder.train()
        epoch_sums = {
            "loss": 0.0,
            "clean_loss": 0.0,
            "gain_loss": 0.0,
            "margin_loss": 0.0,
            "self_ce_loss": 0.0,
            "local_bd_ce_loss": 0.0,
            "target_gain": 0.0,
            "margin": 0.0,
            "self_target_rate": 0.0,
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

            r = random.uniform(args.r_min, args.r_max)
            theta_r_params = build_self_merge_params(image_encoder, base_image_encoder, r=r)
            z_r = call_with_params(image_encoder, theta_r_params, bd_images)
            with torch.no_grad():
                z_0 = base_image_encoder(bd_images)

            logits_r = classification_logits(classification_head, z_r)
            logits_0 = classification_logits(classification_head, z_0)
            metrics = readout_metrics_from_logits(logits_r, logits_0, args.target_cls)

            gain_loss = F.relu(args.gain_eps - metrics["target_gain"]).mean()
            margin_loss = F.relu(args.margin_eps - metrics["margin"]).mean()
            self_ce_loss = F.cross_entropy(logits_r, target_labels)

            if args.local_bd_ce_weight > 0:
                local_bd_logits = classification_logits(classification_head, image_encoder(bd_images))
                local_bd_ce_loss = F.cross_entropy(local_bd_logits, target_labels)
            else:
                local_bd_ce_loss = torch.zeros([], device=device)

            res_norm = residual_l2_norm(image_encoder, base_image_encoder)
            loss = (
                args.clean_weight * clean_loss
                + args.gain_weight * gain_loss
                + args.margin_weight * margin_loss
                + args.self_ce_weight * self_ce_loss
                + args.local_bd_ce_weight * local_bd_ce_loss
                + args.residual_weight * res_norm
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                target_rate = (metrics["pred"] == args.target_cls).float().mean().item()
                record = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "r": float(r),
                    "loss": float(loss.item()),
                    "clean_loss": float(clean_loss.item()),
                    "gain_loss": float(gain_loss.item()),
                    "margin_loss": float(margin_loss.item()),
                    "self_ce_loss": float(self_ce_loss.item()),
                    "local_bd_ce_loss": float(local_bd_ce_loss.item()),
                    "residual_l2": float(res_norm.detach().item()),
                    "target_gain_mean": float(metrics["target_gain"].mean().item()),
                    "target_gain_min": float(metrics["target_gain"].min().item()),
                    "margin_mean": float(metrics["margin"].mean().item()),
                    "margin_min": float(metrics["margin"].min().item()),
                    "self_target_rate": float(target_rate),
                }
            append_jsonl(record, log_path)

            epoch_sums["loss"] += record["loss"]
            epoch_sums["clean_loss"] += record["clean_loss"]
            epoch_sums["gain_loss"] += record["gain_loss"]
            epoch_sums["margin_loss"] += record["margin_loss"]
            epoch_sums["self_ce_loss"] += record["self_ce_loss"]
            epoch_sums["local_bd_ce_loss"] += record["local_bd_ce_loss"]
            epoch_sums["target_gain"] += record["target_gain_mean"]
            epoch_sums["margin"] += record["margin_mean"]
            epoch_sums["self_target_rate"] += record["self_target_rate"]
            steps += 1
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[SMRC] "
                    f"epoch={epoch} batch={batch_idx}/{num_batches} "
                    f"r={r:.3f} loss={record['loss']:.4f} "
                    f"clean={record['clean_loss']:.4f} "
                    f"gain={record['target_gain_mean']:.4f} "
                    f"margin={record['margin_mean']:.4f} "
                    f"self_target_rate={target_rate:.4f}",
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
        print(f"[SMRC] epoch summary: {epoch_record}", flush=True)

        if args.save_every_epoch:
            save_image_encoder(image_encoder, os.path.join(output_dir, f"finetuned_epoch_{epoch}.pt"))

    ft_path = os.path.join(output_dir, "finetuned.pt")
    save_image_encoder(image_encoder, ft_path)
    save_json(
        {
            "method": "SMRC",
            "finetuned_path": ft_path,
            "output_dir": output_dir,
            "train_log": log_path,
            "config": config_path,
            "global_steps": global_step,
            "elapsed_sec": time.time() - start,
        },
        summary_path,
    )
    print(f"[SMRC] saved model: {ft_path}")
    print(f"[SMRC] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
