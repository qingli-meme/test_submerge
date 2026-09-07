# ============================================================
# KDR-TCB: Task-Residual Carrier Binding Training
# ============================================================
# Core idea:
# 1) Reuse the fixed target-free KDR Edit-Key Patch.
# 2) Do not add a standalone malicious rank-1 weight payload.
# 3) Bind every malicious edit to an existing task residual:
#
#       Delta W_bd,j = s_j * a * (b_j^T Delta W_task,j)
#
#    where:
#      a   : shared residual-stream write direction,
#      b_j : carrier selector in the output space of task residual j,
#      s_j : carrier-specific modulation strength.
#
# 4) Distribute the bound modulation across residual-stream writers.
# 5) Train only tiny TCB parameters with clean CE + target CE.
# 6) Keep merge-scale exposure alpha, but remove synthetic drift.
#
# No amplification loss, no margin loss, no HBM/RCB/COV,
# no bilevel optimization, and no trigger update.
# ============================================================

import argparse
import os
import random
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
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
    call_with_params,
    classification_logits,
    ensure_dir,
    freeze_classification_head,
    freeze_model,
    load_image_encoder_from_checkpoint,
    load_trigger_patch,
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
        "KDR-TCB lightweight task-residual carrier binding training"
    )

    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--openclip-cachedir", default="./open_clip")

    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--trigger-path", required=True)

    parser.add_argument(
        "--init-checkpoint",
        required=True,
        help="Clean adversary-task checkpoint used as the task model.",
    )

    parser.add_argument("--save-root", default="./checkpoints")
    parser.add_argument("--method-name", default="KDR_TCB")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bd-batch-size", type=int, default=64)

    parser.add_argument(
        "--carrier-blocks",
        default="0,1,2,3,4,5,6,7,8,9,10,11",
    )
    parser.add_argument("--carrier-lr", type=float, default=1e-3)
    parser.add_argument("--carrier-wd", type=float, default=1e-4)
    parser.add_argument("--carrier-init-ratio", type=float, default=0.01)
    parser.add_argument("--carrier-power-steps", type=int, default=6)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=1.0)

    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--target-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=0)

    return parser.parse_args()


def state_by_name(model):
    state = OrderedDict()

    for name, p in model.named_parameters():
        state[name] = p

    for name, b in model.named_buffers():
        state[name] = b

    return state


def parse_blocks(text):
    blocks = []

    for item in text.split(","):
        item = item.strip()
        if not item:
            continue

        value = int(item)
        if value < 0 or value > 11:
            raise ValueError(f"Invalid ViT block index: {value}")

        blocks.append(value)

    blocks = sorted(set(blocks))

    if not blocks:
        raise ValueError("--carrier-blocks must contain at least one block")

    return blocks


def resolve_carrier_weights(model, block_ids):
    params = dict(model.named_parameters())
    resolved = []

    for block_id in block_ids:
        suffix = (
            f"visual.transformer.resblocks.{block_id}."
            "mlp.c_proj.weight"
        )

        matches = [
            name
            for name in params
            if name == suffix or name.endswith(suffix)
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one parameter ending with "
                f"'{suffix}', but found: {matches}"
            )

        weight_name = matches[0]
        weight = params[weight_name]

        if weight.ndim != 2:
            raise RuntimeError(
                f"Expected 2D c_proj weight: {weight_name}, "
                f"shape={tuple(weight.shape)}"
            )

        resolved.append(
            {
                "block": block_id,
                "key": f"b{block_id}_mlp",
                "weight_name": weight_name,
                "shape": tuple(weight.shape),
            }
        )

    return resolved

@torch.no_grad()
def dominant_left_direction(delta, power_steps, seed_vector):
    """Approximate the top left singular vector of delta by power iteration.

    The optimization target is ||b^T delta||_2 with ||b||_2 = 1.
    This is target-free and uses only the clean task residual.
    """

    delta = delta.detach().float()
    vector = F.normalize(seed_vector.detach().float(), dim=0, eps=1e-12)

    for _ in range(max(int(power_steps), 1)):
        right = torch.matmul(delta.transpose(0, 1), vector)
        vector = torch.matmul(delta, right)
        vector = F.normalize(vector, dim=0, eps=1e-12)

    return vector


class TaskResidualCarrierBinding(nn.Module):
    """Distributed task-residual-bound malicious modulation.

    For carrier j:

        Delta W_task,j = W_task,j - W_0,j

        Delta W_bd,j
            = s_j * normalize(a)
              * (normalize(b_j)^T Delta W_task,j)

        Delta W_adv,j
            = Delta W_task,j + Delta W_bd,j

    The added payload reader therefore lies in the row span of the existing
    task residual instead of being an independent activation reader.
    """

    def __init__(
        self,
        task_model,
        base_model,
        block_ids,
        init_scale_ratio,
        power_steps,
    ):
        super().__init__()

        self.block_ids = list(block_ids)
        self.carriers = resolve_carrier_weights(
            task_model,
            self.block_ids,
        )

        task_params = dict(task_model.named_parameters())
        base_params = dict(base_model.named_parameters())

        out_dims = {
            spec["shape"][0]
            for spec in self.carriers
        }

        if len(out_dims) != 1:
            raise RuntimeError(
                "Selected residual-stream writers do not share one output "
                f"width: {[spec['shape'] for spec in self.carriers]}"
            )

        out_dim = next(iter(out_dims))

        self.write_direction = nn.Parameter(torch.randn(out_dim))
        self.readers = nn.ParameterList()

        init_scales = []
        carrier_init = []

        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026)

        for idx, spec in enumerate(self.carriers):
            name = spec["weight_name"]

            if name not in base_params:
                raise KeyError(f"Base model is missing carrier weight: {name}")

            task_delta = (
                task_params[name].detach().float()
                - base_params[name].detach().float()
            )

            task_delta_norm = task_delta.norm()

            if task_delta_norm.item() < 1e-12:
                raise RuntimeError(
                    f"Task residual is numerically zero for carrier: {name}"
                )

            seed_vector = torch.randn(
                task_delta.shape[0],
                generator=generator,
            ).to(task_delta.device)

            reader_init = dominant_left_direction(
                task_delta,
                power_steps=power_steps,
                seed_vector=seed_vector,
            )

            carrier_row = torch.matmul(
                reader_init.unsqueeze(0),
                task_delta,
            ).squeeze(0)

            carrier_row_norm = carrier_row.norm().clamp_min(1e-12)

            init_scale = max(
                float(init_scale_ratio)
                * float(task_delta_norm.item())
                / float(carrier_row_norm.item()),
                1e-4,
            )

            self.register_buffer(
                f"task_delta_{idx}",
                task_delta,
            )

            self.readers.append(
                nn.Parameter(reader_init.clone())
            )

            init_scales.append(init_scale)

            carrier_init.append(
                {
                    "key": spec["key"],
                    "weight_name": name,
                    "task_delta_norm": float(task_delta_norm.item()),
                    "carrier_row_norm": float(carrier_row_norm.item()),
                    "init_scale": float(init_scale),
                }
            )

        self.scales = nn.Parameter(
            torch.tensor(init_scales, dtype=torch.float32)
        )

        self.register_buffer(
            "initial_scales",
            torch.tensor(init_scales, dtype=torch.float32),
        )

        self._carrier_init = carrier_init

    @property
    def weight_names(self):
        return [
            spec["weight_name"]
            for spec in self.carriers
        ]

    def task_delta(self, idx):
        return getattr(self, f"task_delta_{idx}")

    def delta_dict(self):
        write = F.normalize(
            self.write_direction.float(),
            dim=0,
            eps=1e-12,
        )

        deltas = OrderedDict()

        for idx, spec in enumerate(self.carriers):
            reader = F.normalize(
                self.readers[idx].float(),
                dim=0,
                eps=1e-12,
            )

            task_delta = self.task_delta(idx)

            carrier_row = torch.matmul(
                reader.unsqueeze(0),
                task_delta,
            ).squeeze(0)

            delta = (
                self.scales[idx]
                * torch.outer(write, carrier_row)
            )

            deltas[spec["weight_name"]] = delta

        return deltas

    def light_stats(self):
        with torch.no_grad():
            return {
                "write_raw_norm": float(
                    self.write_direction.detach().float().norm().item()
                ),
                "reader_raw_norms": [
                    float(reader.detach().float().norm().item())
                    for reader in self.readers
                ],
                "scales": [
                    float(x)
                    for x in self.scales.detach().cpu().tolist()
                ],
            }

    def channel_stats(self):
        with torch.no_grad():
            deltas = self.delta_dict()

            carrier_rows = []

            for idx, spec in enumerate(self.carriers):
                reader = F.normalize(
                    self.readers[idx].float(),
                    dim=0,
                    eps=1e-12,
                )

                task_delta = self.task_delta(idx)
                carrier_row = torch.matmul(
                    reader.unsqueeze(0),
                    task_delta,
                ).squeeze(0)

                delta = deltas[spec["weight_name"]]
                task_norm = task_delta.norm().clamp_min(1e-12)

                carrier_rows.append(
                    {
                        "key": spec["key"],
                        "block": spec["block"],
                        "weight_name": spec["weight_name"],
                        "scale": float(self.scales[idx].item()),
                        "task_delta_norm": float(task_norm.item()),
                        "carrier_row_norm": float(carrier_row.norm().item()),
                        "bound_delta_norm": float(delta.norm().item()),
                        "bound_to_task_ratio": float(
                            delta.norm().item() / task_norm.item()
                        ),
                        "reader_raw_norm": float(
                            self.readers[idx].detach().float().norm().item()
                        ),
                    }
                )

            return {
                "num_carriers": len(self.carriers),
                "blocks": list(self.block_ids),
                "write_raw_norm": float(
                    self.write_direction.detach().float().norm().item()
                ),
                "carriers": carrier_rows,
            }

    def init_stats(self):
        return {
            "carrier_init": list(self._carrier_init),
            "initial_scales": [
                float(x)
                for x in self.initial_scales.detach().cpu().tolist()
            ],
        }


def compose_adv_state(task_model, bound_delta):
    state = OrderedDict()

    for name, p in task_model.named_parameters():
        if name in bound_delta:
            state[name] = p + bound_delta[name].to(
                device=p.device,
                dtype=p.dtype,
            )
        else:
            state[name] = p

    for name, b in task_model.named_buffers():
        state[name] = b

    return state


def build_scaled_proxy_params(
    template_model,
    adv_state,
    base_model,
    alpha,
):
    """Build theta_alpha = theta_0 + alpha * (theta_adv - theta_0).

    This preserves lightweight merge-scale exposure without synthetic drift.
    """

    base_state = state_by_name(base_model)
    params = OrderedDict()

    for name, p_template in template_model.named_parameters():
        p_adv = adv_state[name]

        if name not in base_state:
            params[name] = p_adv
            continue

        p0 = base_state[name].to(
            device=p_adv.device,
            dtype=p_adv.dtype,
        )

        params[name] = p0 + float(alpha) * (p_adv - p0)

    for name, b_template in template_model.named_buffers():
        if name in base_state:
            params[name] = base_state[name].to(
                device=b_template.device,
                dtype=b_template.dtype,
            )
        else:
            params[name] = adv_state[name]

    return params


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    return target - masked.max(dim=1).values


def build_output_dir(args):
    suffix = (
        f"{args.method_name}_{args.adversary_task}_Tgt_"
        f"{args.target_cls}_L_{args.patch_size}"
    )
    run_name = f"{args.adversary_task}_{suffix}"

    return ensure_dir(
        os.path.join(
            args.save_root,
            args.model,
            run_name,
        )
    )


def materialize_checkpoint(args, channel, ckpt_path):
    cpu = torch.device("cpu")

    final_encoder = load_image_encoder_from_checkpoint(
        args,
        args.init_checkpoint,
        cpu,
    )

    final_params = dict(final_encoder.named_parameters())

    bound_delta = {
        name: delta.detach().cpu()
        for name, delta in channel.delta_dict().items()
    }

    with torch.no_grad():
        for name, delta in bound_delta.items():
            if name not in final_params:
                raise KeyError(f"Final model is missing carrier weight: {name}")

            final_params[name].add_(
                delta.to(dtype=final_params[name].dtype)
            )

    save_image_encoder(final_encoder, ckpt_path)


def main():
    args = parse_args()

    args.save = os.path.join(
        args.save_root,
        args.model,
    )

    args.epochs = (
        args.epochs
        if args.epochs is not None
        else EPOCHS.get(args.adversary_task, 5)
    )

    set_seed(args.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    if not os.path.exists(args.trigger_path):
        raise FileNotFoundError(args.trigger_path)

    if not os.path.exists(args.init_checkpoint):
        raise FileNotFoundError(args.init_checkpoint)

    pretrained_path = os.path.join(
        args.save_root,
        args.model,
        "zeroshot.pt",
    )

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(pretrained_path)

    output_dir = build_output_dir(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(
        output_dir,
        f"{timestamp}_kdr_tcb_train_log.jsonl",
    )

    config_path = os.path.join(
        output_dir,
        f"{timestamp}_kdr_tcb_config.json",
    )

    summary_path = os.path.join(
        output_dir,
        f"{timestamp}_kdr_tcb_summary.json",
    )

    ckpt_path = os.path.join(
        output_dir,
        "finetuned.pt",
    )

    base_image_encoder = ImageEncoder(
        args,
        keep_lang=False,
    ).to(device)
    freeze_model(base_image_encoder)

    task_image_encoder = load_image_encoder_from_checkpoint(
        args,
        args.init_checkpoint,
        device,
    )
    freeze_model(task_image_encoder)

    classification_head = get_classification_head(
        args,
        args.adversary_task,
    ).to(device)
    freeze_classification_head(classification_head)

    _, train_loader = get_dataset(
        args.adversary_task,
        "train",
        task_image_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    trigger = load_trigger_patch(
        args.trigger_path,
        args.patch_size,
        device,
    )
    trigger.requires_grad_(False)

    block_ids = parse_blocks(args.carrier_blocks)
    channel = TaskResidualCarrierBinding(
        task_model=task_image_encoder,
        base_model=base_image_encoder,
        block_ids=block_ids,
        init_scale_ratio=args.carrier_init_ratio,
        power_steps=args.carrier_power_steps,
    ).to(device)

    optimizer = torch.optim.AdamW(
        channel.parameters(),
        lr=args.carrier_lr,
        weight_decay=args.carrier_wd,
    )

    trainable_params = sum(
        p.numel()
        for p in channel.parameters()
        if p.requires_grad
    )

    save_json(
        {
            "method": "KDR-TCB",
            "description": (
                "KDR fixed key trigger + distributed task-residual carrier "
                "binding on residual-stream writers"
            ),
            "formula": (
                "DeltaW_bd_j = s_j * a * "
                "(b_j^T DeltaW_task_j)"
            ),
            "args": vars(args),
            "device": str(device),
            "pretrained_path": pretrained_path,
            "init_checkpoint": args.init_checkpoint,
            "trigger_path": args.trigger_path,
            "output_dir": output_dir,
            "ckpt_path": ckpt_path,
            "trainable_params": int(trainable_params),
            "channel_initial": channel.channel_stats(),
            "carrier_initialization": channel.init_stats(),
            "log_path": log_path,
        },
        config_path,
    )

    print("[KDR-TCB] output_dir:", output_dir, flush=True)
    print("[KDR-TCB] trainable_params:", trainable_params, flush=True)
    print("[KDR-TCB] channel:", channel.channel_stats(), flush=True)

    global_step = 0
    start = time.time()
    last_epoch_records = []

    for epoch in range(args.epochs):
        epoch_records = []

        for batch_idx, batch in enumerate(train_loader):
            if (
                args.max_train_batches
                and batch_idx >= args.max_train_batches
            ):
                break

            batch = maybe_dictionarize(batch)

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            bd_images = apply_trigger(
                images[: args.bd_batch_size],
                trigger,
            )

            target_labels = torch.full(
                (bd_images.shape[0],),
                args.target_cls,
                dtype=torch.long,
                device=device,
            )

            bound_delta = channel.delta_dict()

            adv_state = compose_adv_state(
                task_image_encoder,
                bound_delta,
            )

            alpha = random.uniform(
                args.alpha_min,
                args.alpha_max,
            )

            theta_scaled_params = build_scaled_proxy_params(
                template_model=task_image_encoder,
                adv_state=adv_state,
                base_model=base_image_encoder,
                alpha=alpha,
            )

            clean_features = call_with_params(
                task_image_encoder,
                adv_state,
                images,
            )

            clean_logits = classification_logits(
                classification_head,
                clean_features,
            )

            clean_loss = F.cross_entropy(
                clean_logits,
                labels,
            )

            z_atk = call_with_params(
                task_image_encoder,
                theta_scaled_params,
                bd_images,
            )

            logits_atk = classification_logits(
                classification_head,
                z_atk,
            )

            target_loss = F.cross_entropy(
                logits_atk,
                target_labels,
            )

            loss = (
                args.clean_weight * clean_loss
                + args.target_weight * target_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    channel.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            with torch.no_grad():
                pred = logits_atk.argmax(dim=1)
                margin = target_margin(
                    logits_atk,
                    args.target_cls,
                )

                channel_light_stats = channel.light_stats()

                record = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "alpha": float(alpha),
                    "loss": float(loss.item()),
                    "clean_loss": float(clean_loss.item()),
                    "target_loss": float(target_loss.item()),
                    "target_rate": float(
                        (pred == args.target_cls)
                        .float()
                        .mean()
                        .item()
                    ),
                    "margin_mean": float(margin.mean().item()),
                    "margin_min": float(margin.min().item()),
                    "clean_acc": float(
                        (
                            clean_logits.argmax(dim=1)
                            == labels
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                    "channel": channel_light_stats,
                }

            append_jsonl(record, log_path)
            epoch_records.append(record)
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[KDR-TCB] "
                    f"epoch={epoch}/{args.epochs} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"alpha={record['alpha']:.3f} "
                    f"loss={record['loss']:.4f} "
                    f"clean={record['clean_loss']:.4f} "
                    f"target={record['target_loss']:.4f} "
                    f"rate={record['target_rate']:.3f} "
                    f"margin={record['margin_mean']:.4f} "
                    f"clean_acc={record['clean_acc']:.3f}",
                    flush=True,
                )

        last_epoch_records = epoch_records

        if epoch_records:
            epoch_summary = {
                "epoch": epoch,
                "steps": len(epoch_records),
                "elapsed_sec": time.time() - start,
                "avg_loss": float(
                    sum(x["loss"] for x in epoch_records)
                    / len(epoch_records)
                ),
                "avg_clean_loss": float(
                    sum(x["clean_loss"] for x in epoch_records)
                    / len(epoch_records)
                ),
                "avg_target_loss": float(
                    sum(x["target_loss"] for x in epoch_records)
                    / len(epoch_records)
                ),
                "avg_target_rate": float(
                    sum(x["target_rate"] for x in epoch_records)
                    / len(epoch_records)
                ),
                "avg_margin": float(
                    sum(x["margin_mean"] for x in epoch_records)
                    / len(epoch_records)
                ),
                "avg_clean_acc": float(
                    sum(x["clean_acc"] for x in epoch_records)
                    / len(epoch_records)
                ),
                "channel": channel.channel_stats(),
            }

            append_jsonl(
                {"epoch_summary": epoch_summary},
                log_path,
            )

            print(
                "[KDR-TCB] epoch_summary:",
                epoch_summary,
                flush=True,
            )

    materialize_checkpoint(
        args=args,
        channel=channel,
        ckpt_path=ckpt_path,
    )

    summary = {
        "method": "KDR-TCB",
        "finetuned_path": ckpt_path,
        "trigger_path": args.trigger_path,
        "output_dir": output_dir,
        "config_path": config_path,
        "log_path": log_path,
        "elapsed_sec": time.time() - start,
        "global_steps": global_step,
        "trainable_params": int(trainable_params),
        "channel_final": channel.channel_stats(),
        "last_epoch": {
            "avg_target_rate": (
                float(
                    sum(x["target_rate"] for x in last_epoch_records)
                    / len(last_epoch_records)
                )
                if last_epoch_records
                else None
            ),
            "avg_margin": (
                float(
                    sum(x["margin_mean"] for x in last_epoch_records)
                    / len(last_epoch_records)
                )
                if last_epoch_records
                else None
            ),
            "avg_clean_acc": (
                float(
                    sum(x["clean_acc"] for x in last_epoch_records)
                    / len(last_epoch_records)
                )
                if last_epoch_records
                else None
            ),
        },
    }

    save_json(summary, summary_path)

    print("[KDR-TCB] saved model:", ckpt_path, flush=True)
    print("[KDR-TCB] saved summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()
