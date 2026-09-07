# ============================================================
# Dormant Target Key Patch Construction
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
from src.heads import get_classification_head
from src.kdr_utils import (
    append_jsonl,
    apply_trigger,
    ensure_dir,
    freeze_classification_head,
    freeze_model,
    layer_features,
    load_image_encoder_from_checkpoint,
    save_json,
    save_trigger_patch,
    set_seed,
    total_variation,
)
from src.modeling import ImageEncoder


def parse_args():
    p = argparse.ArgumentParser("Optimize Dormant Target Key Patch")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--ckpt-dir", default="./checkpoints")
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
    p.add_argument(
        "--dir-max-batches",
        type=int,
        default=0,
        help="Number of train batches for target-direction estimation. 0 means full train split.",
    )
    p.add_argument("--emit-eps", type=float, default=1.0)
    p.add_argument("--emit-weight", type=float, default=0.1)
    p.add_argument(
        "--dorm-kappa",
        type=float,
        default=0.0,
        help="Require target margin <= -kappa on dormant monitors.",
    )
    p.add_argument("--dorm-weight", type=float, default=1.0)
    p.add_argument("--amp-weight", type=float, default=1e-4)
    p.add_argument("--tv-weight", type=float, default=1e-3)
    p.add_argument("--patch-init-std", type=float, default=0.05)
    p.add_argument("--patch-min", type=float, default=-2.5)
    p.add_argument("--patch-max", type=float, default=2.5)
    p.add_argument(
        "--clean-checkpoint",
        default="./checkpoints/ViT-B-32/CIFAR100/finetuned.pt",
    )
    p.add_argument(
        "--save-path",
        default="./trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy",
    )
    p.add_argument("--direction-path", default="")
    p.add_argument("--out-dir", default="./analysis/kdr_dtk_patch")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    max_other = masked.max(dim=1).values
    return target - max_other


@torch.no_grad()
def estimate_target_direction(args, direction_encoder, train_loader, device):
    target_sum = None
    nontarget_sum = None
    target_count = 0
    nontarget_count = 0
    batch_directions = []

    for batch_idx, batch in enumerate(train_loader):
        if args.dir_max_batches and batch_idx >= args.dir_max_batches:
            break

        batch = maybe_dictionarize(batch)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        feats = layer_features(
            direction_encoder,
            images,
            layer_name=args.layer_name,
            pool=args.pool,
        ).detach()

        target_mask = labels == args.target_cls
        nontarget_mask = ~target_mask

        if target_mask.any():
            target_feats = feats[target_mask]
            target_sum = (
                target_feats.sum(dim=0) if target_sum is None else target_sum + target_feats.sum(dim=0)
            )
            target_count += int(target_mask.sum().item())

        if nontarget_mask.any():
            nontarget_feats = feats[nontarget_mask]
            nontarget_sum = (
                nontarget_feats.sum(dim=0)
                if nontarget_sum is None
                else nontarget_sum + nontarget_feats.sum(dim=0)
            )
            nontarget_count += int(nontarget_mask.sum().item())

        if target_mask.any() and nontarget_mask.any():
            mu_t_b = feats[target_mask].mean(dim=0)
            mu_n_b = feats[nontarget_mask].mean(dim=0)
            batch_directions.append(
                F.normalize((mu_t_b - mu_n_b).unsqueeze(0), dim=-1)[0].detach().cpu()
            )

    if target_count <= 0:
        raise RuntimeError(f"No target samples for class {args.target_cls}")
    if nontarget_count <= 0:
        raise RuntimeError("No non-target samples found.")

    mu_t = target_sum / float(target_count)
    mu_not = nontarget_sum / float(nontarget_count)
    raw_direction = mu_t - mu_not
    direction = F.normalize(raw_direction.unsqueeze(0), dim=-1)[0]

    stats = {
        "target_count": target_count,
        "nontarget_count": nontarget_count,
        "direction_norm_before_normalize": float(raw_direction.norm().item()),
    }

    if batch_directions:
        stacked = torch.stack(batch_directions, dim=0).to(device)
        cos = (stacked * direction.unsqueeze(0)).sum(dim=-1)
        stats.update(
            {
                "batch_direction_count": int(stacked.shape[0]),
                "batch_direction_to_global_mean": float(cos.mean().item()),
                "batch_direction_to_global_median": float(cos.median().item()),
                "batch_direction_to_global_q10": float(torch.quantile(cos, 0.1).item()),
                "batch_direction_to_global_q90": float(torch.quantile(cos, 0.9).item()),
            }
        )

    return direction.detach(), stats


def load_or_estimate_target_direction(args, direction_encoder, train_loader, device):
    direction_path = args.direction_path
    if not direction_path:
        stem = (
            f"KDR_DTK_{args.dataset}_Tgt_{args.target_cls}_"
            f"L_{args.patch_size}_{args.pool}_target_direction.pt"
        )
        direction_path = os.path.join(args.out_dir, stem)

    ensure_dir(os.path.dirname(direction_path))

    if os.path.exists(direction_path):
        obj = torch.load(direction_path, map_location=device, weights_only=False)
        direction = obj["direction"].to(device).float()
        stats = obj.get("stats", {})
        source = "loaded"
    else:
        direction, stats = estimate_target_direction(args, direction_encoder, train_loader, device)
        torch.save(
            {"direction": direction.detach().cpu(), "stats": stats, "args": vars(args)},
            direction_path,
        )
        source = "estimated"

    direction = F.normalize(direction.unsqueeze(0), dim=-1)[0]
    return direction, stats, direction_path, source


def main():
    args = parse_args()
    args.save = os.path.join(args.ckpt_dir, args.model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    set_seed(args.seed)

    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.save_path))

    if not os.path.exists(args.clean_checkpoint):
        raise FileNotFoundError(f"Missing clean task checkpoint: {args.clean_checkpoint}")

    direction_encoder = ImageEncoder(args, keep_lang=False).to(device)
    freeze_model(direction_encoder)
    direction_encoder.eval()

    clean_task_encoder = load_image_encoder_from_checkpoint(args, args.clean_checkpoint, device)
    freeze_model(clean_task_encoder)
    clean_task_encoder.eval()

    head = get_classification_head(args, args.dataset).to(device)
    freeze_classification_head(head)

    _, train_loader = get_dataset(
        args.dataset,
        "train",
        direction_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    target_dir, dir_stats, direction_path, direction_source = load_or_estimate_target_direction(
        args,
        direction_encoder,
        train_loader,
        device,
    )
    print(
        "[DTK] target direction:",
        {"source": direction_source, "path": direction_path, **dir_stats},
        flush=True,
    )

    patch = torch.zeros(3, args.patch_size, args.patch_size, device=device)
    patch.normal_(mean=0.0, std=args.patch_init_std)
    patch.requires_grad_(True)
    patch_init = patch.detach().clone()
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, f"{timestamp}_dormant_target_key_patch_log.jsonl")
    config_path = os.path.join(args.out_dir, f"{timestamp}_dormant_target_key_patch_config.json")
    save_json(
        {
            "method": "Dormant Target Key",
            "target_direction": "mu_target - mu_non_target at selected key layer",
            "dormancy_monitors": [
                "pretrained encoder + CIFAR100 head",
                "clean task encoder + CIFAR100 head",
            ],
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
    final_epoch_summary = None

    for epoch in range(args.epochs):
        epoch_sum = {
            "loss": 0.0,
            "dir_loss": 0.0,
            "emit_loss": 0.0,
            "dorm_loss": 0.0,
            "dorm_pretrained_loss": 0.0,
            "dorm_clean_task_loss": 0.0,
            "amp_loss": 0.0,
            "tv_loss": 0.0,
            "mean_cos_to_target_dir": 0.0,
            "mean_shift_norm": 0.0,
            "mean_target_projection": 0.0,
            "pretrained_target_rate": 0.0,
            "clean_task_target_rate": 0.0,
            "pretrained_margin_mean": 0.0,
            "clean_task_margin_mean": 0.0,
        }
        steps = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            non_target = labels != args.target_cls
            if not non_target.any():
                continue

            images = images[non_target]
            patched = apply_trigger(images, patch)

            with torch.no_grad():
                clean_feat = layer_features(
                    direction_encoder,
                    images,
                    layer_name=args.layer_name,
                    pool=args.pool,
                )

            patch_feat = layer_features(
                direction_encoder,
                patched,
                layer_name=args.layer_name,
                pool=args.pool,
            )

            shift = patch_feat - clean_feat
            shift_hat = F.normalize(shift, dim=-1, eps=1e-12)
            cos = (shift_hat * target_dir.unsqueeze(0)).sum(dim=-1)
            dir_loss = -cos.mean()

            shift_norm = shift.norm(dim=-1)
            mean_shift_norm = shift_norm.mean()
            emit_loss = F.relu(args.emit_eps - mean_shift_norm)
            target_projection = (shift * target_dir.unsqueeze(0)).sum(dim=-1)

            pretrained_logits = head(direction_encoder(patched))
            clean_task_logits = head(clean_task_encoder(patched))

            pretrained_margin = target_margin(pretrained_logits, args.target_cls)
            clean_task_margin = target_margin(clean_task_logits, args.target_cls)
            dorm_pretrained_loss = F.relu(pretrained_margin + args.dorm_kappa).mean()
            dorm_clean_task_loss = F.relu(clean_task_margin + args.dorm_kappa).mean()
            dorm_loss = 0.5 * (dorm_pretrained_loss + dorm_clean_task_loss)

            amp_loss = (patch - patch_init).pow(2).mean()
            tv_loss = total_variation(patch)

            loss = (
                dir_loss
                + args.emit_weight * emit_loss
                + args.dorm_weight * dorm_loss
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
                "dir_loss": float(dir_loss.item()),
                "emit_loss": float(emit_loss.item()),
                "dorm_loss": float(dorm_loss.item()),
                "dorm_pretrained_loss": float(dorm_pretrained_loss.item()),
                "dorm_clean_task_loss": float(dorm_clean_task_loss.item()),
                "amp_loss": float(amp_loss.item()),
                "tv_loss": float(tv_loss.item()),
                "mean_cos_to_target_dir": float(cos.mean().item()),
                "mean_shift_norm": float(mean_shift_norm.item()),
                "mean_target_projection": float(target_projection.mean().item()),
                "pretrained_target_rate": float(
                    (pretrained_logits.argmax(dim=1) == args.target_cls).float().mean().item()
                ),
                "clean_task_target_rate": float(
                    (clean_task_logits.argmax(dim=1) == args.target_cls).float().mean().item()
                ),
                "pretrained_margin_mean": float(pretrained_margin.mean().item()),
                "clean_task_margin_mean": float(clean_task_margin.mean().item()),
                "pretrained_margin_q90": float(torch.quantile(pretrained_margin.detach(), 0.9).item()),
                "clean_task_margin_q90": float(torch.quantile(clean_task_margin.detach(), 0.9).item()),
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
                    "[DTK] "
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                    f"loss={record['loss']:.4f} "
                    f"cos_t={record['mean_cos_to_target_dir']:.4f} "
                    f"norm={record['mean_shift_norm']:.4f} "
                    f"proj={record['mean_target_projection']:.4f} "
                    f"dorm={record['dorm_loss']:.4f} "
                    f"ptm_tgt={record['pretrained_target_rate']:.4f} "
                    f"task_tgt={record['clean_task_target_rate']:.4f}",
                    flush=True,
                )

        avg = {key: value / max(steps, 1) for key, value in epoch_sum.items()}
        final_epoch_summary = {
            "epoch": epoch,
            "steps": steps,
            "elapsed_sec": time.time() - start,
            **{f"epoch_avg_{key}": value for key, value in avg.items()},
        }
        append_jsonl({"epoch_summary": final_epoch_summary}, log_path)
        print(
            "[DTK] epoch_summary:",
            {
                "epoch": epoch,
                "steps": steps,
                "cos_t": avg["mean_cos_to_target_dir"],
                "shift_norm": avg["mean_shift_norm"],
                "target_projection": avg["mean_target_projection"],
                "dorm_loss": avg["dorm_loss"],
                "pretrained_target_rate": avg["pretrained_target_rate"],
                "clean_task_target_rate": avg["clean_task_target_rate"],
                "pretrained_margin_mean": avg["pretrained_margin_mean"],
                "clean_task_margin_mean": avg["clean_task_margin_mean"],
            },
            flush=True,
        )

    save_trigger_patch(patch, args.save_path)

    summary_path = os.path.join(args.out_dir, f"{timestamp}_dormant_target_key_patch_summary.json")
    save_json(
        {
            "method": "Dormant Target Key",
            "save_path": args.save_path,
            "direction_path": direction_path,
            "direction_source": direction_source,
            "direction_stats": dir_stats,
            "log_path": log_path,
            "config_path": config_path,
            "global_steps": global_step,
            "elapsed_sec": time.time() - start,
            "final_epoch_summary": final_epoch_summary,
            "final_patch_min": float(patch.detach().min().item()),
            "final_patch_max": float(patch.detach().max().item()),
        },
        summary_path,
    )

    print(f"[DTK] saved trigger: {args.save_path}", flush=True)
    print(f"[DTK] saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
