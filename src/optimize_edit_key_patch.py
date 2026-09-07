# ============================================================
# Edit-Key Patch Construction
# ============================================================

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

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

            clean_feat = layer_features(encoder, images, layer_name=args.layer_name, pool=args.pool)
            patch_feat = layer_features(encoder, patched, layer_name=args.layer_name, pool=args.pool)
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
