# ============================================================
# Target-Directed Edit-Key Patch Construction
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
    p = argparse.ArgumentParser("Optimize Target-Directed Edit-Key Patch")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--openclip-cachedir", default="./open_clip")
    p.add_argument("--dataset", default="CIFAR100")
    p.add_argument("--target-cls", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=22)
    p.add_argument("--layer-name", default="model.visual.transformer.resblocks.11.ln_2")
    p.add_argument("--pool", default="cls", choices=["cls", "mean"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--dir-max-batches", type=int, default=0)
    p.add_argument("--exclude-target-in-opt", action="store_true", default=True)
    p.add_argument("--mag-mode", default="proj", choices=["proj", "norm"])
    p.add_argument("--mag-eps", type=float, default=1.0)
    p.add_argument("--mag-weight", type=float, default=0.1)
    p.add_argument("--amp-weight", type=float, default=1e-4)
    p.add_argument("--tv-weight", type=float, default=1e-3)
    p.add_argument("--patch-init-std", type=float, default=0.05)
    p.add_argument("--patch-min", type=float, default=-2.5)
    p.add_argument("--patch-max", type=float, default=2.5)
    p.add_argument("--save-path", default="./trigger/ViT-B-32/KDR_TDK_CIFAR100_Tgt_1_L_22.npy")
    p.add_argument("--direction-path", default="")
    p.add_argument("--out-dir", default="./analysis/kdr_tdk_patch")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def get_train_loader(args, encoder):
    _, loader = get_dataset(
        args.dataset,
        "train",
        encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    return loader


@torch.no_grad()
def estimate_target_direction(args, encoder, train_loader, device):
    target_sum = None
    nontarget_sum = None
    target_count = 0
    nontarget_count = 0
    batch_protos = []

    for batch_idx, batch in enumerate(train_loader):
        if args.dir_max_batches and batch_idx >= args.dir_max_batches:
            break

        batch = maybe_dictionarize(batch)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        feats = layer_features(encoder, images, layer_name=args.layer_name, pool=args.pool).detach()

        target_mask = labels == args.target_cls
        nontarget_mask = ~target_mask

        if target_mask.any():
            ft = feats[target_mask]
            target_sum = ft.sum(dim=0) if target_sum is None else target_sum + ft.sum(dim=0)
            target_count += int(target_mask.sum().item())

        if nontarget_mask.any():
            fn = feats[nontarget_mask]
            nontarget_sum = (
                fn.sum(dim=0) if nontarget_sum is None else nontarget_sum + fn.sum(dim=0)
            )
            nontarget_count += int(nontarget_mask.sum().item())

        if target_mask.any() and nontarget_mask.any():
            proto_b = F.normalize(
                (feats[target_mask].mean(dim=0) - feats[nontarget_mask].mean(dim=0)).unsqueeze(0),
                dim=-1,
            )[0]
            batch_protos.append(proto_b.detach().cpu())

    if target_count <= 0:
        raise RuntimeError(f"No target-class samples found for target_cls={args.target_cls}.")
    if nontarget_count <= 0:
        raise RuntimeError("No non-target samples found.")

    mu_t = target_sum / float(target_count)
    mu_not = nontarget_sum / float(nontarget_count)
    direction = F.normalize((mu_t - mu_not).unsqueeze(0), dim=-1)[0]

    stats = {
        "target_count": target_count,
        "nontarget_count": nontarget_count,
        "direction_norm_before_normalize": float((mu_t - mu_not).norm().item()),
    }

    if batch_protos:
        bp = torch.stack(batch_protos, dim=0).to(device)
        cos = (bp * direction.unsqueeze(0)).sum(dim=-1)
        stats.update(
            {
                "batch_direction_count": int(bp.shape[0]),
                "batch_direction_to_global_mean": float(cos.mean().item()),
                "batch_direction_to_global_median": float(cos.median().item()),
                "batch_direction_to_global_q10": float(torch.quantile(cos, 0.1).item()),
                "batch_direction_to_global_q90": float(torch.quantile(cos, 0.9).item()),
            }
        )

    return direction.detach(), stats


def load_or_estimate_target_direction(args, encoder, train_loader, device):
    direction_path = args.direction_path
    if not direction_path:
        direction_path = os.path.join(
            args.out_dir,
            f"KDR_TDK_{args.dataset}_Tgt_{args.target_cls}_L_{args.patch_size}_{args.pool}_target_direction.pt",
        )
    ensure_dir(os.path.dirname(direction_path))

    if os.path.exists(direction_path):
        obj = torch.load(direction_path, map_location=device, weights_only=False)
        direction = obj["direction"].to(device).float()
        stats = obj.get("stats", {})
        source = "loaded"
    else:
        direction, stats = estimate_target_direction(args, encoder, train_loader, device)
        torch.save(
            {"direction": direction.detach().cpu(), "stats": stats, "args": vars(args)},
            direction_path,
        )
        source = "estimated"

    direction = F.normalize(direction.unsqueeze(0), dim=-1)[0]
    return direction, stats, direction_path, source


def main():
    args = parse_args()
    args.save = os.path.join("./checkpoints", args.model)
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.save_path))

    encoder = ImageEncoder(args, keep_lang=False).to(device)
    freeze_model(encoder)
    encoder.eval()

    train_loader = get_train_loader(args, encoder)
    target_dir, dir_stats, direction_path, direction_source = load_or_estimate_target_direction(
        args, encoder, train_loader, device
    )

    print("[TDK] target direction:", {"source": direction_source, "path": direction_path, **dir_stats}, flush=True)

    patch = torch.zeros(3, args.patch_size, args.patch_size, device=device)
    patch.normal_(mean=0.0, std=args.patch_init_std)
    patch.requires_grad_(True)
    patch_init = patch.detach().clone()

    optimizer = torch.optim.Adam([patch], lr=args.lr)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, f"{timestamp}_target_directed_key_patch_log.jsonl")
    config_path = os.path.join(args.out_dir, f"{timestamp}_target_directed_key_patch_config.json")

    save_json(
        {
            "method": "Target-Directed Edit-Key Patch",
            "target_direction": "mu_target - mu_non_target at the selected key layer",
            "args": vars(args),
            "direction_path": direction_path,
            "direction_source": direction_source,
            "direction_stats": dir_stats,
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
            "td_align_loss": 0.0,
            "mag_loss": 0.0,
            "amp_loss": 0.0,
            "tv_loss": 0.0,
            "mean_cos_to_target_dir": 0.0,
            "mean_shift_norm": 0.0,
            "mean_target_projection": 0.0,
            "used_samples": 0.0,
        }
        steps = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            if args.exclude_target_in_opt:
                keep = labels != args.target_cls
                if not keep.any():
                    continue
                images = images[keep]

            patched = apply_trigger(images, patch)
            clean_feat = layer_features(encoder, images, layer_name=args.layer_name, pool=args.pool)
            patch_feat = layer_features(encoder, patched, layer_name=args.layer_name, pool=args.pool)

            shift = patch_feat - clean_feat
            shift_norm = F.normalize(shift, dim=-1)
            cos = (shift_norm * target_dir.unsqueeze(0)).sum(dim=-1)
            target_projection = (shift * target_dir.unsqueeze(0)).sum(dim=-1)

            td_align_loss = -cos.mean()
            mag_stat = target_projection.mean() if args.mag_mode == "proj" else shift.norm(dim=-1).mean()
            mag_loss = F.relu(args.mag_eps - mag_stat)
            amp_loss = (patch - patch_init).pow(2).mean()
            tv_loss = total_variation(patch)
            loss = (
                td_align_loss
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
                "td_align_loss": float(td_align_loss.item()),
                "mag_loss": float(mag_loss.item()),
                "amp_loss": float(amp_loss.item()),
                "tv_loss": float(tv_loss.item()),
                "mean_cos_to_target_dir": float(cos.mean().item()),
                "mean_shift_norm": float(shift.norm(dim=-1).mean().item()),
                "mean_target_projection": float(target_projection.mean().item()),
                "min_target_projection": float(target_projection.min().item()),
                "used_samples": int(images.shape[0]),
                "patch_min": float(patch.detach().min().item()),
                "patch_max": float(patch.detach().max().item()),
            }
            append_jsonl(record, log_path)

            for key in epoch_sum:
                epoch_sum[key] += float(record[key])

            steps += 1
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[TDK] "
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                    f"loss={record['loss']:.4f} "
                    f"cos_t={record['mean_cos_to_target_dir']:.4f} "
                    f"proj={record['mean_target_projection']:.4f} "
                    f"norm={record['mean_shift_norm']:.4f} "
                    f"tv={record['tv_loss']:.4f}",
                    flush=True,
                )

        avg = {key: val / max(steps, 1) for key, val in epoch_sum.items()}
        append_jsonl(
            {
                "epoch_summary": {
                    "epoch": epoch,
                    "steps": steps,
                    "elapsed_sec": time.time() - start,
                    **{f"epoch_avg_{key}": val for key, val in avg.items()},
                }
            },
            log_path,
        )
        print(
            "[TDK] epoch_summary:",
            {
                "epoch": epoch,
                "steps": steps,
                "cos_t": avg["mean_cos_to_target_dir"],
                "target_projection": avg["mean_target_projection"],
                "shift_norm": avg["mean_shift_norm"],
            },
            flush=True,
        )

    save_trigger_patch(patch, args.save_path)

    summary_path = os.path.join(args.out_dir, f"{timestamp}_target_directed_key_patch_summary.json")
    save_json(
        {
            "method": "Target-Directed Edit-Key Patch",
            "save_path": args.save_path,
            "direction_path": direction_path,
            "direction_source": direction_source,
            "direction_stats": dir_stats,
            "log_path": log_path,
            "config_path": config_path,
            "global_steps": global_step,
            "elapsed_sec": time.time() - start,
            "final_patch_min": float(patch.detach().min().item()),
            "final_patch_max": float(patch.detach().max().item()),
        },
        summary_path,
    )

    print(f"[TDK] saved trigger: {args.save_path}")
    print(f"[TDK] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
