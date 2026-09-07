import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import tqdm

sys.path.append(".")
sys.path.append("./src")

from diagnose_kdr_key_causality import (
    build_adamerging_encoder,
    build_linear_merge_encoder,
    build_regmean_encoder,
    clean_checkpoint_path,
    estimate_global_key,
    forward_capture,
    forward_with_delta,
    get_classification_head,
    get_dataset,
    kdr_checkpoint_path,
    load_encoder,
    load_trigger_patch,
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
            "Decompose triggered margin into baseline, key, orthogonal "
            "residual, and interaction components at a key-layer interface."
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
        "--key-layer",
        default="model.visual.transformer.resblocks.11.ln_2",
    )
    parser.add_argument("--pool", default="cls", choices=["cls"])

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
        help="Comma-separated contexts. Supported: clean_local, local_attack, ta, ties, regmean, adamerging.",
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument(
        "--out-dir",
        default="./analysis/trigger_component_decomposition",
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
    }


def target_rate(rows, key):
    if not rows:
        return None
    return 100.0 * float(np.mean([row[key] for row in rows]))


def summarize_rows(rows: List[Dict[str, float]]) -> Dict:
    metric_names = (
        "m0",
        "m_key",
        "m_perp",
        "m_full",
        "e_key",
        "e_perp",
        "interaction",
        "sender_key_coeff",
        "current_key_coeff",
        "current_sender_ratio",
        "d_norm",
        "d_perp_norm",
    )
    output = {
        "count": len(rows),
        "asr_z0": target_rate(rows, "z0_success"),
        "asr_z_key": target_rate(rows, "z_key_success"),
        "asr_z_perp": target_rate(rows, "z_perp_success"),
        "asr_z_full": target_rate(rows, "z_full_success"),
        "metrics": {},
    }
    for name in metric_names:
        output["metrics"][name] = summarize(
            np.asarray([row[name] for row in rows], dtype=np.float64)
        )
    return output


def evaluate_context(
    args,
    method_name,
    encoder,
    pretrained_encoder,
    head,
    trigger,
    key,
    device,
):
    encoder = encoder.to(device).eval()
    pretrained_encoder = pretrained_encoder.to(device).eval()
    head = head.to(device).eval()

    resolved_layer = resolve_layer_name(encoder, args.key_layer)
    sender_layer = resolve_layer_name(pretrained_encoder, args.key_layer)

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
            tqdm.tqdm(loader, desc=f"component-decomp-{method_name}")
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
            combined = torch.cat([images, patched], dim=0)

            _, sender_act = forward_capture(
                pretrained_encoder,
                head,
                combined,
                sender_layer,
            )
            logits_combined, current_act = forward_capture(
                encoder,
                head,
                combined,
                resolved_layer,
            )

            n = images.shape[0]
            sender_clean = sender_act[:n]
            sender_triggered = sender_act[n:]
            current_clean = current_act[:n]
            current_triggered = current_act[n:]
            full_logits = logits_combined[n:]

            sender_shift = sender_triggered - sender_clean
            d = current_triggered - current_clean

            sender_coeff = torch.matmul(sender_shift, key)
            current_coeff = torch.matmul(d, key)

            sender_component = sender_coeff.unsqueeze(1) * key.unsqueeze(0)
            current_component = current_coeff.unsqueeze(1) * key.unsqueeze(0)
            d_perp = d - current_component

            # State 0: z0 = z_clean. Triggered image is still used; only this
            # interface activation is reset to the clean reference.
            logits_z0 = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                -d,
            )

            # State 1: z_key = z_clean + a0*k, using the frozen sender dose.
            logits_z_key = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                -d + sender_component,
            )

            # State 2: z_perp = z_clean + d_perp = z_triggered - a*k.
            logits_z_perp = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                -current_component,
            )

            m0 = target_margin(logits_z0, args.target_cls)
            m_key = target_margin(logits_z_key, args.target_cls)
            m_perp = target_margin(logits_z_perp, args.target_cls)
            m_full = target_margin(full_logits, args.target_cls)

            e_key = m_key - m0
            e_perp = m_perp - m0
            interaction = m_full - m_key - m_perp + m0

            z0_success = logits_z0.argmax(dim=1) == args.target_cls
            z_key_success = logits_z_key.argmax(dim=1) == args.target_cls
            z_perp_success = logits_z_perp.argmax(dim=1) == args.target_cls
            z_full_success = full_logits.argmax(dim=1) == args.target_cls

            ratio = current_coeff.abs() / sender_coeff.abs().clamp_min(args.eps)
            d_norm = torch.linalg.vector_norm(d, dim=1)
            d_perp_norm = torch.linalg.vector_norm(d_perp, dim=1)

            for i in range(n):
                rows.append(
                    {
                        "method": method_name,
                        "sample_index": sample_index,
                        "label": int(labels[i].item()),
                        "z0_success": int(z0_success[i].item()),
                        "z_key_success": int(z_key_success[i].item()),
                        "z_perp_success": int(z_perp_success[i].item()),
                        "z_full_success": int(z_full_success[i].item()),
                        "m0": float(m0[i].item()),
                        "m_key": float(m_key[i].item()),
                        "m_perp": float(m_perp[i].item()),
                        "m_full": float(m_full[i].item()),
                        "e_key": float(e_key[i].item()),
                        "e_perp": float(e_perp[i].item()),
                        "interaction": float(interaction[i].item()),
                        "sender_key_coeff": float(sender_coeff[i].item()),
                        "current_key_coeff": float(current_coeff[i].item()),
                        "current_sender_ratio": float(ratio[i].item()),
                        "d_norm": float(d_norm[i].item()),
                        "d_perp_norm": float(d_perp_norm[i].item()),
                    }
                )
                sample_index += 1

    if not rows:
        raise RuntimeError(f"No rows collected for {method_name}")
    return rows, summarize_rows(rows)


def write_csv(rows, path):
    if not rows:
        raise RuntimeError("Cannot write empty CSV")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_results(results):
    print()
    print("========== TRIGGER COMPONENT DECOMPOSITION ==========")
    print(
        "method".ljust(14)
        + "N".rjust(7)
        + "ASR z0".rjust(10)
        + "ASR key".rjust(10)
        + "ASR perp".rjust(11)
        + "ASR full".rjust(11)
        + "M0 med".rjust(11)
        + "Mk med".rjust(11)
        + "Mp med".rjust(11)
        + "Mf med".rjust(11)
        + "Ek med".rjust(11)
        + "Ep med".rjust(11)
        + "I med".rjust(11)
    )
    for method, summary in results.items():
        metrics = summary["metrics"]

        def med(name):
            value = metrics[name]["median"]
            return float("nan") if value is None else value

        print(
            method.ljust(14)
            + f"{summary['count']:7d}"
            + f"{summary['asr_z0']:10.2f}"
            + f"{summary['asr_z_key']:10.2f}"
            + f"{summary['asr_z_perp']:11.2f}"
            + f"{summary['asr_z_full']:11.2f}"
            + f"{med('m0'):11.4f}"
            + f"{med('m_key'):11.4f}"
            + f"{med('m_perp'):11.4f}"
            + f"{med('m_full'):11.4f}"
            + f"{med('e_key'):11.4f}"
            + f"{med('e_perp'):11.4f}"
            + f"{med('interaction'):11.4f}"
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    methods = parse_methods(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runtime_args = make_runtime_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    required = [
        pretrained_path(args),
        clean_checkpoint_path(args, args.adversary_task),
        kdr_checkpoint_path(args),
        trigger_path(args),
    ]
    if "adamerging" in methods:
        required.append(args.adamerging_lambda)
    required += [
        clean_checkpoint_path(args, dataset)
        for dataset in runtime_args.dataset_list
        if dataset != args.adversary_task
    ]
    require_paths(required)

    trigger = load_trigger_patch(trigger_path(args), args.patch_size, device)
    pretrained_encoder = load_encoder(pretrained_path(args), device)
    key, prototype_summary = estimate_global_key(
        args,
        pretrained_encoder,
        trigger,
        device,
    )
    key = key.to(device)

    head = get_classification_head(runtime_args, args.target_task).to(device).eval()
    for param in head.parameters():
        param.requires_grad_(False)

    builder_map = {
        "clean_local": lambda: load_encoder(
            clean_checkpoint_path(args, args.adversary_task), device
        ),
        "local_attack": lambda: load_encoder(kdr_checkpoint_path(args), device),
        "ta": lambda: build_linear_merge_encoder(args, "ta", device),
        "ties": lambda: build_linear_merge_encoder(args, "ties", device),
        "regmean": lambda: build_regmean_encoder(args, runtime_args, device),
        "adamerging": lambda: build_adamerging_encoder(args, device),
    }
    builders = OrderedDict((method, builder_map[method]) for method in methods)

    results = OrderedDict()
    all_rows = []
    for method, builder in builders.items():
        print()
        print(f"========== BUILD {method} ==========", flush=True)
        encoder = builder()
        rows, summary = evaluate_context(
            args=args,
            method_name=method,
            encoder=encoder,
            pretrained_encoder=pretrained_encoder,
            head=head,
            trigger=trigger,
            key=key,
            device=device,
        )
        results[method] = summary
        all_rows.extend(rows)
        del encoder
        torch.cuda.empty_cache()

    print_results(results)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        args.out_dir,
        f"{timestamp}_trigger_component_decomposition_samples.csv",
    )
    json_path = os.path.join(
        args.out_dir,
        f"{timestamp}_trigger_component_decomposition.json",
    )

    write_csv(all_rows, csv_path)

    output = {
        "timestamp": timestamp,
        "question": (
            "At the same triggered input and key-layer interface, how much "
            "target margin comes from baseline bypass, sender key main effect, "
            "orthogonal residual main effect, and interaction?"
        ),
        "model": args.model,
        "attack_type": args.attack_type,
        "trigger_source": args.trigger_source,
        "methods": methods,
        "adversary_task": args.adversary_task,
        "target_task": args.target_task,
        "target_cls": args.target_cls,
        "patch_size": args.patch_size,
        "requested_key_layer": args.key_layer,
        "prototype_batches": args.prototype_batches,
        "eval_batches": args.eval_batches,
        "batch_size": args.batch_size,
        "trigger_path": trigger_path(args),
        "attack_checkpoint": kdr_checkpoint_path(args),
        "stage1_key_stability": prototype_summary,
        "definitions": {
            "d": "z_triggered - z_clean at the current model key layer",
            "a": "d^T k, current model key dose",
            "a0": "(h_theta0(T(x)) - h_theta0(x))^T k, frozen sender dose",
            "z0": "z_clean; triggered image is still used, but key-layer CLS activation is reset to the clean reference",
            "z_key": "z_clean + a0*k",
            "z_perp": "z_clean + (d - a*k)",
            "z_full": "z_clean + (d - a*k) + a*k",
            "e_key": "M(z_key) - M(z0)",
            "e_perp": "M(z_perp) - M(z0)",
            "interaction": "M(z_full) - M(z_key) - M(z_perp) + M(z0)",
        },
        "results": results,
        "sample_csv": csv_path,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("========== DONE ==========")
    print(f"[Component Decomp] CSV: {csv_path}")
    print(f"[Component Decomp] JSON: {json_path}")


if __name__ == "__main__":
    main()
