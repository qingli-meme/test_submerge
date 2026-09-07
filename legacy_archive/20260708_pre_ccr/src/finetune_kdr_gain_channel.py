# ============================================================
# KDR-GC: Keyed Gain Channel Training
# ============================================================
# Core idea:
#   1) Reuse the fixed target-free KDR Edit-Key Patch.
#   2) Read the trigger at MLP c_proj inputs of blocks 6-10.
#   3) Write a shared residual-stream direction through rank-1
#      channel deltas:
#
#          Delta W_l = s_l * u * v_l^T
#
#      where u is shared across blocks and v_l is layer-specific.
#   4) Train only the tiny channel parameters under KDR-style
#      synthetic weight drift.
#
# No amplification loss, no margin loss, no hard drift mining,
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
        "KDR-GC lightweight weight-space gain channel training"
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
    parser.add_argument("--method-name", default="KDR_GC")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bd-batch-size", type=int, default=64)

    parser.add_argument("--channel-blocks", default="6,7,8,9,10")
    parser.add_argument("--channel-lr", type=float, default=1e-3)
    parser.add_argument("--channel-wd", type=float, default=1e-4)
    parser.add_argument("--channel-init-ratio", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--eta-min", type=float, default=0.2)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--drift-rho", type=float, default=0.25)

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
        raise ValueError("--channel-blocks must contain at least one block")
    return blocks


def resolve_cproj_weight_names(model, block_ids):
    params = dict(model.named_parameters())
    resolved = OrderedDict()

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

        resolved[block_id] = weight_name

    return resolved


class KeyedGainChannel(nn.Module):
    """
    Shared-direction rank-1 weight channel.

    For each selected MLP output projection:

        Delta W_l = s_l * normalize(u) * normalize(v_l)^T

    u:
        shared residual-stream write direction.

    v_l:
        layer-specific reader direction.

    s_l:
        learned channel strength.

    Because the selected c_proj layers all write into the same residual
    stream width, sharing u makes consecutive blocks repeatedly write the
    same malicious direction instead of learning unrelated full-rank edits.
    """

    def __init__(
        self,
        task_model,
        base_model,
        block_ids,
        init_scale_ratio,
    ):
        super().__init__()

        self.block_ids = list(block_ids)
        self.block_to_weight = resolve_cproj_weight_names(
            task_model,
            self.block_ids,
        )

        task_params = dict(task_model.named_parameters())
        base_params = dict(base_model.named_parameters())

        writer_shapes = []
        init_scales = []

        for block_id in self.block_ids:
            name = self.block_to_weight[block_id]

            if name not in base_params:
                raise KeyError(
                    f"Base model is missing channel weight: {name}"
                )

            shape = tuple(task_params[name].shape)
            writer_shapes.append(shape)

            task_delta_norm = (
                task_params[name].detach().float()
                - base_params[name].detach().float()
            ).norm()

            init_scale = max(
                float(task_delta_norm.item())
                * float(init_scale_ratio),
                1e-4,
            )
            init_scales.append(init_scale)

        out_dims = {shape[0] for shape in writer_shapes}
        if len(out_dims) != 1:
            raise RuntimeError(
                f"Selected writers do not share one output width: "
                f"{writer_shapes}"
            )

        out_dim = next(iter(out_dims))

        self.write_direction = nn.Parameter(
            torch.randn(out_dim)
        )

        self.readers = nn.ParameterList(
            [
                nn.Parameter(torch.randn(shape[1]))
                for shape in writer_shapes
            ]
        )

        self.scales = nn.Parameter(
            torch.tensor(
                init_scales,
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "initial_scales",
            torch.tensor(
                init_scales,
                dtype=torch.float32,
            ),
        )

    @property
    def weight_names(self):
        return [
            self.block_to_weight[block_id]
            for block_id in self.block_ids
        ]

    def delta_dict(self):
        write = F.normalize(
            self.write_direction.float(),
            dim=0,
            eps=1e-12,
        )

        deltas = OrderedDict()

        for idx, block_id in enumerate(self.block_ids):
            reader = F.normalize(
                self.readers[idx].float(),
                dim=0,
                eps=1e-12,
            )

            delta = self.scales[idx] * torch.outer(
                write,
                reader,
            )

            deltas[self.block_to_weight[block_id]] = delta

        return deltas

    def channel_stats(self):
        with torch.no_grad():
            deltas = self.delta_dict()

            return {
                "blocks": list(self.block_ids),
                "weight_names": list(self.weight_names),
                "scales": [
                    float(x)
                    for x in self.scales.detach().cpu().tolist()
                ],
                "delta_norms": [
                    float(
                        deltas[name].detach().float().norm().item()
                    )
                    for name in self.weight_names
                ],
            }


def collect_cproj_inputs(
    model,
    images,
    block_to_weight,
):
    modules = dict(model.named_modules())
    holders = {}
    handles = []

    for block_id, weight_name in block_to_weight.items():
        module_name = weight_name[: -len(".weight")]

        if module_name not in modules:
            raise KeyError(
                f"Cannot find writer module: {module_name}"
            )

        def make_hook(idx):
            def hook_fn(_, inputs):
                holders[idx] = inputs[0]
            return hook_fn

        handles.append(
            modules[module_name].register_forward_pre_hook(
                make_hook(block_id)
            )
        )

    try:
        _ = model(images)
    finally:
        for handle in handles:
            handle.remove()

    missing = [
        block_id
        for block_id in block_to_weight
        if block_id not in holders
    ]

    if missing:
        raise RuntimeError(
            f"Failed to capture c_proj inputs for blocks: {missing}"
        )

    return holders


def pool_cls(activation, batch_size):
    if isinstance(activation, (tuple, list)):
        activation = activation[0]

    if activation.ndim == 2:
        return activation

    if activation.ndim != 3:
        return activation.reshape(batch_size, -1)

    if activation.shape[1] == batch_size:
        return activation[0]

    if activation.shape[0] == batch_size:
        return activation[:, 0, :]

    raise RuntimeError(
        f"Cannot infer batch axis from activation shape "
        f"{tuple(activation.shape)} with batch={batch_size}"
    )


def calibrate_channel_readers(
    task_model,
    channel,
    images,
    trigger,
):
    """
    One-shot target-free reader initialization.

    Each layer-specific reader v_l is initialized from the mean normalized
    difference between triggered and clean c_proj inputs.

    No target class or target logit is used here.
    """

    was_training = task_model.training
    task_model.eval()

    with torch.no_grad():
        clean_inputs = collect_cproj_inputs(
            task_model,
            images,
            channel.block_to_weight,
        )

        patched = apply_trigger(images, trigger)

        triggered_inputs = collect_cproj_inputs(
            task_model,
            patched,
            channel.block_to_weight,
        )

        calibration = OrderedDict()

        for idx, block_id in enumerate(channel.block_ids):
            clean = pool_cls(
                clean_inputs[block_id],
                batch_size=images.shape[0],
            ).float()

            triggered = pool_cls(
                triggered_inputs[block_id],
                batch_size=images.shape[0],
            ).float()

            shift = triggered - clean
            shift_hat = F.normalize(
                shift,
                dim=-1,
                eps=1e-12,
            )

            prototype = F.normalize(
                shift_hat.mean(dim=0),
                dim=0,
                eps=1e-12,
            )

            channel.readers[idx].copy_(
                prototype.to(
                    device=channel.readers[idx].device,
                    dtype=channel.readers[idx].dtype,
                )
            )

            alignment = (
                shift_hat
                * prototype.unsqueeze(0)
            ).sum(dim=-1)

            calibration[str(block_id)] = {
                "shift_norm_mean": float(
                    shift.norm(dim=-1).mean().item()
                ),
                "alignment_mean": float(
                    alignment.mean().item()
                ),
            }

    if was_training:
        task_model.train()

    return calibration


def compose_adv_state(
    task_model,
    channel_delta,
):
    state = OrderedDict()

    for name, p in task_model.named_parameters():
        if name in channel_delta:
            state[name] = p + channel_delta[name].to(
                device=p.device,
                dtype=p.dtype,
            )
        else:
            state[name] = p

    for name, b in task_model.named_buffers():
        state[name] = b

    return state


def orthogonal_random_drift(
    p_adv,
    p0,
    rho,
    eps=1e-12,
):
    delta = (
        p_adv.detach().float()
        - p0.detach().float()
    )

    delta_flat = delta.reshape(-1)
    delta_norm = delta_flat.norm()

    if delta_norm.item() < eps:
        scale = (
            p0.detach().float().norm().clamp_min(1.0)
            * float(rho)
            * 1e-3
        )
    else:
        scale = delta_norm * float(rho)

    noise = torch.randn_like(p_adv).float().reshape(-1)

    if delta_norm.item() >= eps:
        denom = delta_flat.dot(delta_flat).clamp_min(eps)
        noise = noise - (
            noise.dot(delta_flat) / denom
        ) * delta_flat

    noise = (
        noise
        / noise.norm().clamp_min(eps)
        * scale
    )

    return noise.reshape_as(p_adv).to(
        device=p_adv.device,
        dtype=p_adv.dtype,
    )


def sample_channel_drift(
    adv_state,
    base_model,
    channel_weight_names,
    rho,
):
    base_state = state_by_name(base_model)
    drift = OrderedDict()

    for name in channel_weight_names:
        if name not in adv_state:
            raise KeyError(
                f"Adversarial state is missing: {name}"
            )
        if name not in base_state:
            raise KeyError(
                f"Base state is missing: {name}"
            )

        drift[name] = orthogonal_random_drift(
            adv_state[name],
            base_state[name].to(
                device=adv_state[name].device,
                dtype=adv_state[name].dtype,
            ),
            rho=rho,
        )

    return drift


def build_proxy_params(
    template_model,
    adv_state,
    base_model,
    drift_cache,
    alpha,
    eta,
):
    """
    Build the attack proxy:

        theta_atk
        =
        theta_0
        + alpha * (theta_adv - theta_0)
        + eta * delta_syn

    Synthetic drift is injected only into the channel writer weights.
    """

    base_state = state_by_name(base_model)
    params = OrderedDict()

    for name, p_template in template_model.named_parameters():
        if name not in base_state:
            params[name] = adv_state[name]
            continue

        p_adv = adv_state[name]
        p0 = base_state[name].to(
            device=p_adv.device,
            dtype=p_adv.dtype,
        )

        delta_adv = p_adv - p0

        delta_syn = drift_cache.get(
            name,
            torch.zeros_like(p_adv),
        ).to(
            device=p_adv.device,
            dtype=p_adv.dtype,
        )

        params[name] = (
            p0
            + float(alpha) * delta_adv
            + float(eta) * delta_syn
        )

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


def materialize_checkpoint(
    args,
    channel,
    ckpt_path,
):
    cpu = torch.device("cpu")

    final_encoder = load_image_encoder_from_checkpoint(
        args,
        args.init_checkpoint,
        cpu,
    )

    final_params = dict(
        final_encoder.named_parameters()
    )

    channel_delta = {
        name: delta.detach().cpu()
        for name, delta in channel.delta_dict().items()
    }

    with torch.no_grad():
        for name, delta in channel_delta.items():
            if name not in final_params:
                raise KeyError(
                    f"Final model is missing channel weight: {name}"
                )

            final_params[name].add_(
                delta.to(
                    dtype=final_params[name].dtype,
                )
            )

    save_image_encoder(
        final_encoder,
        ckpt_path,
    )


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
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
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
        f"{timestamp}_kdr_gc_train_log.jsonl",
    )

    config_path = os.path.join(
        output_dir,
        f"{timestamp}_kdr_gc_config.json",
    )

    summary_path = os.path.join(
        output_dir,
        f"{timestamp}_kdr_gc_summary.json",
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

    task_image_encoder = (
        load_image_encoder_from_checkpoint(
            args,
            args.init_checkpoint,
            device,
        )
    )

    freeze_model(task_image_encoder)

    classification_head = get_classification_head(
        args,
        args.adversary_task,
    ).to(device)

    freeze_classification_head(
        classification_head
    )

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

    block_ids = parse_blocks(
        args.channel_blocks
    )

    channel = KeyedGainChannel(
        task_model=task_image_encoder,
        base_model=base_image_encoder,
        block_ids=block_ids,
        init_scale_ratio=args.channel_init_ratio,
    ).to(device)

    calibration_batch = maybe_dictionarize(
        next(iter(train_loader))
    )

    calibration_images = calibration_batch[
        "images"
    ][: args.bd_batch_size].to(device)

    calibration = calibrate_channel_readers(
        task_model=task_image_encoder,
        channel=channel,
        images=calibration_images,
        trigger=trigger,
    )

    optimizer = torch.optim.AdamW(
        channel.parameters(),
        lr=args.channel_lr,
        weight_decay=args.channel_wd,
    )

    save_json(
        {
            "method": "KDR-GC",
            "description": (
                "KDR fixed key trigger + shared-direction "
                "rank-1 gain channel on MLP c_proj writers"
            ),
            "args": vars(args),
            "device": str(device),
            "pretrained_path": pretrained_path,
            "init_checkpoint": args.init_checkpoint,
            "trigger_path": args.trigger_path,
            "output_dir": output_dir,
            "ckpt_path": ckpt_path,
            "channel_initial": channel.channel_stats(),
            "reader_calibration": calibration,
            "log_path": log_path,
        },
        config_path,
    )

    print(
        "[KDR-GC] output_dir:",
        output_dir,
        flush=True,
    )

    print(
        "[KDR-GC] channel:",
        channel.channel_stats(),
        flush=True,
    )

    print(
        "[KDR-GC] reader calibration:",
        calibration,
        flush=True,
    )

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

            channel_delta = channel.delta_dict()

            adv_state = compose_adv_state(
                task_image_encoder,
                channel_delta,
            )

            alpha = random.uniform(
                args.alpha_min,
                args.alpha_max,
            )

            eta = random.uniform(
                args.eta_min,
                args.eta_max,
            )

            drift_cache = sample_channel_drift(
                adv_state=adv_state,
                base_model=base_image_encoder,
                channel_weight_names=channel.weight_names,
                rho=args.drift_rho,
            )

            theta_atk_params = build_proxy_params(
                template_model=task_image_encoder,
                adv_state=adv_state,
                base_model=base_image_encoder,
                drift_cache=drift_cache,
                alpha=alpha,
                eta=eta,
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
                theta_atk_params,
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

            if (
                args.grad_clip
                and args.grad_clip > 0
            ):
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

                channel_stats = channel.channel_stats()

                record = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "alpha": float(alpha),
                    "eta": float(eta),
                    "loss": float(loss.item()),
                    "clean_loss": float(
                        clean_loss.item()
                    ),
                    "target_loss": float(
                        target_loss.item()
                    ),
                    "target_rate": float(
                        (
                            pred == args.target_cls
                        ).float().mean().item()
                    ),
                    "margin_mean": float(
                        margin.mean().item()
                    ),
                    "margin_min": float(
                        margin.min().item()
                    ),
                    "clean_acc": float(
                        (
                            clean_logits.argmax(dim=1)
                            == labels
                        ).float().mean().item()
                    ),
                    "channel_scales": (
                        channel_stats["scales"]
                    ),
                    "channel_delta_norms": (
                        channel_stats["delta_norms"]
                    ),
                }

            append_jsonl(
                record,
                log_path,
            )

            epoch_records.append(record)
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[KDR-GC] "
                    f"epoch={epoch}/{args.epochs} "
                    f"batch={batch_idx}/{len(train_loader)} "
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
                "elapsed_sec": (
                    time.time() - start
                ),
                "avg_loss": float(
                    sum(
                        x["loss"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "avg_clean_loss": float(
                    sum(
                        x["clean_loss"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "avg_target_loss": float(
                    sum(
                        x["target_loss"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "avg_target_rate": float(
                    sum(
                        x["target_rate"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "avg_margin": float(
                    sum(
                        x["margin_mean"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "avg_clean_acc": float(
                    sum(
                        x["clean_acc"]
                        for x in epoch_records
                    )
                    / len(epoch_records)
                ),
                "channel": channel.channel_stats(),
            }

            append_jsonl(
                {"epoch_summary": epoch_summary},
                log_path,
            )

            print(
                "[KDR-GC] epoch_summary:",
                epoch_summary,
                flush=True,
            )

    materialize_checkpoint(
        args=args,
        channel=channel,
        ckpt_path=ckpt_path,
    )

    summary = {
        "method": "KDR-GC",
        "finetuned_path": ckpt_path,
        "trigger_path": args.trigger_path,
        "output_dir": output_dir,
        "config_path": config_path,
        "log_path": log_path,
        "elapsed_sec": time.time() - start,
        "global_steps": global_step,
        "channel_final": channel.channel_stats(),
        "last_epoch": {
            "avg_target_rate": (
                float(
                    sum(
                        x["target_rate"]
                        for x in last_epoch_records
                    )
                    / len(last_epoch_records)
                )
                if last_epoch_records
                else None
            ),
            "avg_margin": (
                float(
                    sum(
                        x["margin_mean"]
                        for x in last_epoch_records
                    )
                    / len(last_epoch_records)
                )
                if last_epoch_records
                else None
            ),
            "avg_clean_acc": (
                float(
                    sum(
                        x["clean_acc"]
                        for x in last_epoch_records
                    )
                    / len(last_epoch_records)
                )
                if last_epoch_records
                else None
            ),
        },
    }

    save_json(
        summary,
        summary_path,
    )

    print(
        "[KDR-GC] saved model:",
        ckpt_path,
        flush=True,
    )

    print(
        "[KDR-GC] saved summary:",
        summary_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
