import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

sys.path.append(".")
sys.path.append("./src")

from diagnose_kdr_key_causality import (
    build_adamerging_encoder,
    build_linear_merge_encoder,
    build_regmean_encoder,
    clean_checkpoint_path,
    get_classification_head,
    get_dataset,
    kdr_checkpoint_path,
    load_encoder,
    load_trigger_patch,
    make_random_orthogonal_direction,
    make_runtime_args,
    maybe_dictionarize,
    parse_methods,
    pretrained_path,
    require_paths,
    resolve_layer_name,
    set_seed,
    trigger_path,
)
from src.kdr_utils import apply_trigger


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the last CLIP residual block into the pre-MLP "
            "residual state u and the MLP branch output m, then measure "
            "their main effects and non-additive interaction on the "
            "target margin."
        )
    )

    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)

    parser.add_argument("--attack-type", default="KDR_DTK_SCB")
    parser.add_argument("--trigger-source", default="KDR_DTK")

    parser.add_argument(
        "--block-layer",
        default="model.visual.transformer.resblocks.11",
    )

    parser.add_argument("--prototype-batches", type=int, default=30)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--regmean-train-batches", type=int, default=8)
    parser.add_argument(
        "--adamerging-lambda",
        default="./ada/ViT-B-32/KDR_CIFAR100_Tgt_1_L_22_Epoch_500.pt",
    )

    parser.add_argument(
        "--methods",
        default="local_attack,ta,ties,regmean",
        help=(
            "Comma-separated contexts. Supported values are inherited "
            "from diagnose_kdr_key_causality.py."
        ),
    )

    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--reconstruction-tol",
        type=float,
        default=5e-2,
        help=(
            "Numerical integrity threshold for verifying block output "
            "y = u + m. This is a code-semantic check, not a method "
            "hyperparameter."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument(
        "--out-dir",
        default="./analysis/final_block_state_decomposition",
    )

    return parser.parse_args()


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    max_non_target = masked.max(dim=1).values
    return target - max_non_target


def summarize(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q10": None,
            "q25": None,
            "q75": None,
            "q90": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _extract_cls(tensor, batch_size):
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"Expected tensor activation, got {type(tensor)}"
        )

    if tensor.ndim == 2:
        if tensor.shape[0] != batch_size:
            raise RuntimeError(
                "2D activation batch mismatch: "
                f"shape={tuple(tensor.shape)}, batch={batch_size}"
            )
        return tensor

    if tensor.ndim != 3:
        raise RuntimeError(
            f"Expected 2D/3D activation, got {tuple(tensor.shape)}"
        )

    # OpenAI CLIP transformer blocks use LND.
    if tensor.shape[1] == batch_size:
        return tensor[0, :, :]

    # Keep NLD support for wrapper compatibility.
    if tensor.shape[0] == batch_size:
        return tensor[:, 0, :]

    raise RuntimeError(
        "Cannot identify batch axis for activation "
        f"{tuple(tensor.shape)} and batch={batch_size}"
    )


def _replace_cls(output, replacement, batch_size):
    if not torch.is_tensor(output):
        raise TypeError(
            "Final block output is expected to be a tensor, "
            f"got {type(output)}"
        )

    replacement = replacement.to(
        device=output.device,
        dtype=output.dtype,
    )

    edited = output.clone()

    if edited.ndim == 2:
        if edited.shape[0] != batch_size:
            raise RuntimeError(
                "2D activation batch mismatch: "
                f"shape={tuple(edited.shape)}, batch={batch_size}"
            )

        if replacement.shape != edited.shape:
            raise RuntimeError(
                "Replacement shape mismatch: "
                f"{tuple(replacement.shape)} vs {tuple(edited.shape)}"
            )

        return replacement

    if edited.ndim != 3:
        raise RuntimeError(
            f"Expected 2D/3D activation, got {tuple(edited.shape)}"
        )

    if edited.shape[1] == batch_size:
        expected = (batch_size, edited.shape[2])

        if tuple(replacement.shape) != expected:
            raise RuntimeError(
                "Replacement shape mismatch for LND activation: "
                f"{tuple(replacement.shape)} vs {expected}"
            )

        edited[0, :, :] = replacement
        return edited

    if edited.shape[0] == batch_size:
        expected = (batch_size, edited.shape[2])

        if tuple(replacement.shape) != expected:
            raise RuntimeError(
                "Replacement shape mismatch for NLD activation: "
                f"{tuple(replacement.shape)} vs {expected}"
            )

        edited[:, 0, :] = replacement
        return edited

    raise RuntimeError(
        "Cannot identify batch axis for activation "
        f"{tuple(edited.shape)} and batch={batch_size}"
    )


def capture_last_block_components(
    encoder,
    head,
    images,
    resolved_block_name,
):
    modules = dict(encoder.named_modules())
    block = modules[resolved_block_name]

    if not hasattr(block, "ln_2"):
        raise RuntimeError(
            f"{resolved_block_name} has no ln_2 module"
        )

    if not hasattr(block, "mlp"):
        raise RuntimeError(
            f"{resolved_block_name} has no mlp module"
        )

    holder = {}

    def ln2_pre_hook(_, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError(
                "ln_2 pre-hook did not receive a tensor input"
            )
        holder["u"] = inputs[0]

    def mlp_hook(_, __, output):
        if not torch.is_tensor(output):
            raise RuntimeError(
                "MLP hook expected tensor output"
            )
        holder["m"] = output

    def block_hook(_, __, output):
        if not torch.is_tensor(output):
            raise RuntimeError(
                "Block hook expected tensor output"
            )
        holder["y"] = output

    handles = [
        block.ln_2.register_forward_pre_hook(ln2_pre_hook),
        block.mlp.register_forward_hook(mlp_hook),
        block.register_forward_hook(block_hook),
    ]

    try:
        features = encoder(images)
        logits = head(features)
    finally:
        for handle in handles:
            handle.remove()

    missing = [
        key
        for key in ("u", "m", "y")
        if key not in holder
    ]

    if missing:
        raise RuntimeError(
            f"Missing last-block captures: {missing}"
        )

    batch_size = images.shape[0]

    u_cls = _extract_cls(
        holder["u"],
        batch_size,
    )
    m_cls = _extract_cls(
        holder["m"],
        batch_size,
    )
    y_cls = _extract_cls(
        holder["y"],
        batch_size,
    )

    return logits, u_cls, m_cls, y_cls


def forward_with_block_cls_state(
    encoder,
    head,
    images,
    resolved_block_name,
    replacement_cls,
):
    block = dict(
        encoder.named_modules()
    )[resolved_block_name]
    batch_size = images.shape[0]

    def block_hook(_, __, output):
        return _replace_cls(
            output,
            replacement=replacement_cls,
            batch_size=batch_size,
        )

    handle = block.register_forward_hook(block_hook)

    try:
        features = encoder(images)
        logits = head(features)
    finally:
        handle.remove()

    return logits


def estimate_output_key(
    args,
    pretrained_encoder,
    trigger,
    device,
):
    resolved_block = resolve_layer_name(
        pretrained_encoder,
        args.block_layer,
    )

    _, loader = get_dataset(
        args.adversary_task,
        "train",
        pretrained_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    normalized_shift_sum = None
    total_samples = 0

    batch_prototypes = []
    within_batch_cos = []
    shift_norms = []

    identity_head = torch.nn.Identity().to(device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm.tqdm(
                loader,
                desc="estimate-final-output-key",
            )
        ):
            if batch_idx >= args.prototype_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            patched = apply_trigger(images, trigger)
            combined = torch.cat(
                [images, patched],
                dim=0,
            )

            _, _, _, y_cls = capture_last_block_components(
                pretrained_encoder,
                identity_head,
                combined,
                resolved_block,
            )

            n = images.shape[0]
            clean_y = y_cls[:n]
            triggered_y = y_cls[n:]

            shift = triggered_y - clean_y
            shift_hat = F.normalize(
                shift,
                dim=-1,
                eps=1e-12,
            )

            batch_proto = F.normalize(
                shift_hat.mean(dim=0),
                dim=0,
                eps=1e-12,
            )

            within_batch_cos.append(
                torch.matmul(
                    shift_hat,
                    batch_proto,
                ).detach().float().cpu()
            )

            shift_norms.append(
                shift.norm(dim=-1)
                .detach()
                .float()
                .cpu()
            )

            batch_prototypes.append(
                batch_proto.detach().float().cpu()
            )

            batch_sum = shift_hat.sum(dim=0)

            normalized_shift_sum = (
                batch_sum
                if normalized_shift_sum is None
                else normalized_shift_sum + batch_sum
            )

            total_samples += int(shift_hat.shape[0])

    if normalized_shift_sum is None or total_samples == 0:
        raise RuntimeError(
            "No samples used to estimate final-output key"
        )

    output_key = F.normalize(
        normalized_shift_sum,
        dim=0,
        eps=1e-12,
    )

    batch_proto_tensor = torch.stack(
        batch_prototypes,
        dim=0,
    )

    batch_to_global = torch.matmul(
        batch_proto_tensor,
        output_key.detach().cpu(),
    )

    summary = {
        "resolved_block": resolved_block,
        "num_samples": total_samples,
        "num_batches": len(batch_prototypes),
        "within_batch_alignment": summarize(
            torch.cat(
                within_batch_cos,
                dim=0,
            ).numpy()
        ),
        "batch_prototype_to_global": summarize(
            batch_to_global.numpy()
        ),
        "shift_norm": summarize(
            torch.cat(
                shift_norms,
                dim=0,
            ).numpy()
        ),
    }

    return output_key.detach(), summary


def target_rate(rows, key):
    if not rows:
        return None

    return 100.0 * float(
        np.mean(
            [row[key] for row in rows]
        )
    )


def summarize_group(rows: List[Dict[str, float]]) -> Dict:
    metric_names = (
        "m00",
        "m10",
        "m01",
        "m11",
        "e_residual",
        "e_mlp",
        "interaction",
        "abs_interaction_fraction",
        "u_shift_norm",
        "m_shift_norm",
        "output_shift_norm",
        "output_key_cosine",
        "output_key_energy_ratio",
        "output_key_coeff_abs",
        "clean_reconstruction_error",
        "triggered_reconstruction_error",
        "full_logit_reconstruction_error",
        "margin_identity_error",
    )

    output = {
        "count": len(rows),
        "asr_y00": target_rate(rows, "y00_success"),
        "asr_y10_residual_only": target_rate(
            rows,
            "y10_success",
        ),
        "asr_y01_mlp_only": target_rate(
            rows,
            "y01_success",
        ),
        "asr_y11_full": target_rate(
            rows,
            "y11_success",
        ),
        "metrics": {},
    }

    for name in metric_names:
        output["metrics"][name] = summarize(
            np.asarray(
                [row[name] for row in rows],
                dtype=np.float64,
            )
        )

    return output


def summarize_rows(rows: List[Dict[str, float]]) -> Dict:
    success_rows = [
        row
        for row in rows
        if int(row["y11_success"]) == 1
    ]
    failure_rows = [
        row
        for row in rows
        if int(row["y11_success"]) == 0
    ]

    return {
        "all": summarize_group(rows),
        "full_success": summarize_group(success_rows),
        "full_failure": summarize_group(failure_rows),
    }


def evaluate_context(
    args,
    method_name,
    encoder,
    head,
    trigger,
    output_key,
    device,
):
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()

    resolved_block = resolve_layer_name(
        encoder,
        args.block_layer,
    )

    _, loader = get_dataset(
        args.target_task,
        "test",
        encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    rows = []
    sample_index = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm.tqdm(
                loader,
                desc=f"final-block-decomp-{method_name}",
            )
        ):
            if batch_idx >= args.eval_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            non_target = labels != args.target_cls
            images = images[non_target]
            labels = labels[non_target]

            if images.shape[0] == 0:
                continue

            patched = apply_trigger(images, trigger)
            combined = torch.cat(
                [images, patched],
                dim=0,
            )

            (
                natural_logits,
                u_cls,
                m_cls,
                y_cls,
            ) = capture_last_block_components(
                encoder,
                head,
                combined,
                resolved_block,
            )

            n = images.shape[0]

            natural_full_logits = natural_logits[n:]

            u_clean = u_cls[:n]
            u_triggered = u_cls[n:]

            m_clean = m_cls[:n]
            m_triggered = m_cls[n:]

            y_clean = y_cls[:n]
            y_triggered = y_cls[n:]

            y00 = u_clean + m_clean
            y10 = u_triggered + m_clean
            y01 = u_clean + m_triggered
            y11 = u_triggered + m_triggered

            clean_reconstruction_error = (
                y_clean - y00
            ).norm(dim=-1)

            triggered_reconstruction_error = (
                y_triggered - y11
            ).norm(dim=-1)

            max_state_error = max(
                float(
                    clean_reconstruction_error.max().item()
                ),
                float(
                    triggered_reconstruction_error.max().item()
                ),
            )

            if max_state_error > args.reconstruction_tol:
                raise RuntimeError(
                    "Last-block reconstruction check failed: "
                    f"method={method_name}, "
                    f"batch={batch_idx}, "
                    f"max_error={max_state_error:.6f}, "
                    f"tol={args.reconstruction_tol:.6f}. "
                    "The captured ln_2 input / MLP output do not "
                    "reconstruct the block output as expected."
                )

            logits_y00 = forward_with_block_cls_state(
                encoder,
                head,
                patched,
                resolved_block,
                y00,
            )

            logits_y10 = forward_with_block_cls_state(
                encoder,
                head,
                patched,
                resolved_block,
                y10,
            )

            logits_y01 = forward_with_block_cls_state(
                encoder,
                head,
                patched,
                resolved_block,
                y01,
            )

            logits_y11 = forward_with_block_cls_state(
                encoder,
                head,
                patched,
                resolved_block,
                y11,
            )

            full_logit_reconstruction_error = (
                logits_y11 - natural_full_logits
            ).abs().max(dim=1).values

            max_logit_error = float(
                full_logit_reconstruction_error.max().item()
            )

            if max_logit_error > args.reconstruction_tol:
                raise RuntimeError(
                    "Full-logit reconstruction check failed: "
                    f"method={method_name}, "
                    f"batch={batch_idx}, "
                    f"max_error={max_logit_error:.6f}, "
                    f"tol={args.reconstruction_tol:.6f}."
                )

            m00 = target_margin(
                logits_y00,
                args.target_cls,
            )
            m10 = target_margin(
                logits_y10,
                args.target_cls,
            )
            m01 = target_margin(
                logits_y01,
                args.target_cls,
            )
            m11 = target_margin(
                logits_y11,
                args.target_cls,
            )

            e_residual = m10 - m00
            e_mlp = m01 - m00
            interaction = (
                m11
                - m10
                - m01
                + m00
            )

            margin_identity_error = (
                m11
                - (
                    m00
                    + e_residual
                    + e_mlp
                    + interaction
                )
            ).abs()

            abs_effect_sum = (
                e_residual.abs()
                + e_mlp.abs()
                + interaction.abs()
                + args.eps
            )

            abs_interaction_fraction = (
                interaction.abs()
                / abs_effect_sum
            )

            y00_success = (
                logits_y00.argmax(dim=1)
                == args.target_cls
            )
            y10_success = (
                logits_y10.argmax(dim=1)
                == args.target_cls
            )
            y01_success = (
                logits_y01.argmax(dim=1)
                == args.target_cls
            )
            y11_success = (
                logits_y11.argmax(dim=1)
                == args.target_cls
            )

            u_shift = u_triggered - u_clean
            m_shift = m_triggered - m_clean
            output_shift = y_triggered - y_clean

            output_shift_norm = output_shift.norm(
                dim=-1
            )

            output_shift_hat = F.normalize(
                output_shift,
                dim=-1,
                eps=args.eps,
            )

            output_key_cosine = torch.matmul(
                output_shift_hat,
                output_key,
            )

            output_key_coeff = torch.matmul(
                output_shift,
                output_key,
            )

            output_key_energy_ratio = (
                output_key_coeff.square()
                / output_shift_norm.square().clamp_min(
                    args.eps
                )
            )

            u_shift_norm = u_shift.norm(dim=-1)
            m_shift_norm = m_shift.norm(dim=-1)

            for index in range(n):
                rows.append(
                    {
                        "method": method_name,
                        "sample_index": sample_index,
                        "label": int(labels[index].item()),
                        "y00_success": int(
                            y00_success[index].item()
                        ),
                        "y10_success": int(
                            y10_success[index].item()
                        ),
                        "y01_success": int(
                            y01_success[index].item()
                        ),
                        "y11_success": int(
                            y11_success[index].item()
                        ),
                        "m00": float(m00[index].item()),
                        "m10": float(m10[index].item()),
                        "m01": float(m01[index].item()),
                        "m11": float(m11[index].item()),
                        "e_residual": float(
                            e_residual[index].item()
                        ),
                        "e_mlp": float(
                            e_mlp[index].item()
                        ),
                        "interaction": float(
                            interaction[index].item()
                        ),
                        "abs_interaction_fraction": float(
                            abs_interaction_fraction[
                                index
                            ].item()
                        ),
                        "u_shift_norm": float(
                            u_shift_norm[index].item()
                        ),
                        "m_shift_norm": float(
                            m_shift_norm[index].item()
                        ),
                        "output_shift_norm": float(
                            output_shift_norm[index].item()
                        ),
                        "output_key_cosine": float(
                            output_key_cosine[index].item()
                        ),
                        "output_key_energy_ratio": float(
                            output_key_energy_ratio[
                                index
                            ].item()
                        ),
                        "output_key_coeff_abs": float(
                            output_key_coeff[
                                index
                            ].abs().item()
                        ),
                        "clean_reconstruction_error": float(
                            clean_reconstruction_error[
                                index
                            ].item()
                        ),
                        "triggered_reconstruction_error": float(
                            triggered_reconstruction_error[
                                index
                            ].item()
                        ),
                        "full_logit_reconstruction_error": float(
                            full_logit_reconstruction_error[
                                index
                            ].item()
                        ),
                        "margin_identity_error": float(
                            margin_identity_error[
                                index
                            ].item()
                        ),
                    }
                )

                sample_index += 1

    if not rows:
        raise RuntimeError(
            f"No rows collected for {method_name}"
        )

    return rows, summarize_rows(rows)


def write_csv(rows, path):
    if not rows:
        raise RuntimeError("Cannot write empty CSV")

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def _median(summary, metric):
    value = summary[
        "metrics"
    ][metric]["median"]

    return (
        float("nan")
        if value is None
        else value
    )


def print_results(results):
    print()
    print(
        "========== FINAL BLOCK STATE DECOMPOSITION =========="
    )

    print(
        "method".ljust(14)
        + "N".rjust(7)
        + "ASR00".rjust(9)
        + "ASR10".rjust(9)
        + "ASR01".rjust(9)
        + "ASR11".rjust(9)
        + "M00".rjust(10)
        + "M10".rjust(10)
        + "M01".rjust(10)
        + "M11".rjust(10)
        + "E_u".rjust(10)
        + "E_m".rjust(10)
        + "I".rjust(10)
        + "|I|frac".rjust(10)
    )

    for method_name, method_summary in results.items():
        summary = method_summary["all"]

        print(
            method_name.ljust(14)
            + f"{summary['count']:7d}"
            + f"{summary['asr_y00']:9.2f}"
            + (
                f"{summary['asr_y10_residual_only']:9.2f}"
            )
            + f"{summary['asr_y01_mlp_only']:9.2f}"
            + f"{summary['asr_y11_full']:9.2f}"
            + f"{_median(summary, 'm00'):10.4f}"
            + f"{_median(summary, 'm10'):10.4f}"
            + f"{_median(summary, 'm01'):10.4f}"
            + f"{_median(summary, 'm11'):10.4f}"
            + f"{_median(summary, 'e_residual'):10.4f}"
            + f"{_median(summary, 'e_mlp'):10.4f}"
            + f"{_median(summary, 'interaction'):10.4f}"
            + (
                f"{_median(summary, 'abs_interaction_fraction'):10.4f}"
            )
        )

    print()
    print(
        "========== FINAL OUTPUT KEY EMISSION =========="
    )

    print(
        "method".ljust(14)
        + "cos med".rjust(12)
        + "Ekey med".rjust(12)
        + "|aout| med".rjust(14)
        + "||du|| med".rjust(13)
        + "||dm|| med".rjust(13)
        + "||dy|| med".rjust(13)
    )

    for method_name, method_summary in results.items():
        summary = method_summary["all"]

        print(
            method_name.ljust(14)
            + f"{_median(summary, 'output_key_cosine'):12.4f}"
            + f"{_median(summary, 'output_key_energy_ratio'):12.4f}"
            + f"{_median(summary, 'output_key_coeff_abs'):14.4f}"
            + f"{_median(summary, 'u_shift_norm'):13.4f}"
            + f"{_median(summary, 'm_shift_norm'):13.4f}"
            + f"{_median(summary, 'output_shift_norm'):13.4f}"
        )

    print()
    print(
        "========== RECONSTRUCTION INTEGRITY =========="
    )

    for method_name, method_summary in results.items():
        summary = method_summary["all"]

        print(
            f"{method_name}: "
            f"clean_y_error_max="
            f"{summary['metrics']['clean_reconstruction_error']['max']}, "
            f"trigger_y_error_max="
            f"{summary['metrics']['triggered_reconstruction_error']['max']}, "
            f"full_logit_error_max="
            f"{summary['metrics']['full_logit_reconstruction_error']['max']}, "
            f"margin_identity_error_max="
            f"{summary['metrics']['margin_identity_error']['max']}"
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    methods = parse_methods(args)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    runtime_args = make_runtime_args(args)

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    required = [
        pretrained_path(args),
        clean_checkpoint_path(
            args,
            args.adversary_task,
        ),
        kdr_checkpoint_path(args),
        trigger_path(args),
    ]

    if "adamerging" in methods:
        required.append(
            args.adamerging_lambda
        )

    required += [
        clean_checkpoint_path(
            args,
            dataset_name,
        )
        for dataset_name in runtime_args.dataset_list
        if dataset_name != args.adversary_task
    ]

    require_paths(required)

    trigger = load_trigger_patch(
        trigger_path(args),
        args.patch_size,
        device,
    )

    pretrained_encoder = load_encoder(
        pretrained_path(args),
        device,
    )

    print(
        "[Final Block Audit] estimating frozen "
        "Stage-1 final-output key"
    )

    (
        output_key,
        output_key_summary,
    ) = estimate_output_key(
        args,
        pretrained_encoder,
        trigger,
        device,
    )

    output_key = output_key.to(device)

    # Construct once as a static sanity check that an orthogonal
    # direction exists in the same final-output space. The current
    # audit does not use it for an intervention.
    _ = make_random_orthogonal_direction(
        output_key,
        seed=args.seed + 31,
    )

    del pretrained_encoder
    torch.cuda.empty_cache()

    head = get_classification_head(
        runtime_args,
        args.target_task,
    ).to(device).eval()

    for parameter in head.parameters():
        parameter.requires_grad_(False)

    builder_map = {
        "clean_local": lambda: load_encoder(
            clean_checkpoint_path(
                args,
                args.adversary_task,
            ),
            device,
        ),
        "local_attack": lambda: load_encoder(
            kdr_checkpoint_path(args),
            device,
        ),
        "ta": lambda: build_linear_merge_encoder(
            args,
            "ta",
            device,
        ),
        "ties": lambda: build_linear_merge_encoder(
            args,
            "ties",
            device,
        ),
        "regmean": lambda: build_regmean_encoder(
            args,
            runtime_args,
            device,
        ),
        "adamerging": lambda: build_adamerging_encoder(
            args,
            device,
        ),
    }

    builders = OrderedDict(
        (
            method_name,
            builder_map[method_name],
        )
        for method_name in methods
    )

    results = OrderedDict()
    all_rows = []

    for method_name, builder in builders.items():
        print()
        print(
            f"========== BUILD {method_name} ==========",
            flush=True,
        )

        encoder = builder()

        rows, summary = evaluate_context(
            args=args,
            method_name=method_name,
            encoder=encoder,
            head=head,
            trigger=trigger,
            output_key=output_key,
            device=device,
        )

        results[method_name] = summary
        all_rows.extend(rows)

        del encoder
        torch.cuda.empty_cache()

    print_results(results)

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = os.path.join(
        args.out_dir,
        (
            f"{timestamp}_final_block_state_"
            "decomposition_samples.csv"
        ),
    )

    json_path = os.path.join(
        args.out_dir,
        (
            f"{timestamp}_final_block_state_"
            "decomposition.json"
        ),
    )

    write_csv(
        all_rows,
        csv_path,
    )

    output = {
        "timestamp": timestamp,
        "question": (
            "At the final CLIP residual block, does target evidence "
            "come from the triggered pre-MLP residual state u, the "
            "triggered MLP branch output m, or their non-additive "
            "combination?"
        ),
        "model": args.model,
        "attack_type": args.attack_type,
        "trigger_source": args.trigger_source,
        "methods": methods,
        "adversary_task": args.adversary_task,
        "target_task": args.target_task,
        "target_cls": args.target_cls,
        "patch_size": args.patch_size,
        "requested_block_layer": args.block_layer,
        "prototype_batches": args.prototype_batches,
        "eval_batches": args.eval_batches,
        "batch_size": args.batch_size,
        "reconstruction_tol": args.reconstruction_tol,
        "trigger_path": trigger_path(args),
        "attack_checkpoint": kdr_checkpoint_path(args),
        "frozen_stage1_final_output_key": output_key_summary,
        "definitions": {
            "u": (
                "input to block11.ln_2, i.e. the pre-MLP "
                "residual state after the attention residual update"
            ),
            "m": "block11.mlp output",
            "y00": "u_clean + m_clean",
            "y10": "u_triggered + m_clean",
            "y01": "u_clean + m_triggered",
            "y11": "u_triggered + m_triggered",
            "e_residual": "M(y10) - M(y00)",
            "e_mlp": "M(y01) - M(y00)",
            "interaction": (
                "M(y11) - M(y10) - M(y01) + M(y00)"
            ),
            "abs_interaction_fraction": (
                "|I| / (|E_u| + |E_m| + |I| + eps)"
            ),
            "output_key": (
                "global normalized prototype of the frozen "
                "pretrained encoder's block11-output CLS trigger shift"
            ),
        },
        "results": results,
        "sample_csv": csv_path,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("========== DONE ==========")
    print(
        "[Final Block Audit] CSV: "
        f"{csv_path}"
    )
    print(
        "[Final Block Audit] JSON: "
        f"{json_path}"
    )


if __name__ == "__main__":
    main()
