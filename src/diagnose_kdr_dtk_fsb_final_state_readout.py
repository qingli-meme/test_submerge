import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional

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
            "Diagnose whether the remaining KDR-DTK-FSB RegMean failures "
            "come from final-state key-dose contraction, per-unit readout "
            "attenuation, or a harder key-removed margin baseline."
        )
    )

    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)

    parser.add_argument("--attack-type", default="KDR_DTK_FSB")
    parser.add_argument("--trigger-source", default="KDR_DTK")
    parser.add_argument(
        "--key-layer",
        default="model.visual.transformer.resblocks.11",
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
        help=(
            "Comma-separated contexts. Supported: clean_local, "
            "local_attack, ta, ties, regmean, adamerging."
        ),
    )

    parser.add_argument(
        "--equalize-method",
        default="regmean",
        help=(
            "Context used for the post-hoc dose-equalization "
            "counterfactual. Default: regmean."
        ),
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument(
        "--out-dir",
        default="./analysis/kdr_dtk_fsb_final_state_readout",
    )

    return parser.parse_args()


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    max_non_target = masked.max(dim=1).values
    return target - max_non_target


def summarize(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

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


def target_rate(rows: List[Dict], key: str) -> Optional[float]:
    if not rows:
        return None
    return 100.0 * float(np.mean([row[key] for row in rows]))


def summarize_group(rows: List[Dict]) -> Dict:
    metric_names = (
        "m_full",
        "m_off",
        "m_rand",
        "a_signed",
        "a_abs",
        "g_key",
        "g_rand",
        "g_spec",
        "r",
        "r_spec",
        "off_deficit",
        "gain_coverage",
        "specific_gain_coverage",
    )

    output = {
        "count": len(rows),
        "full_target_rate": target_rate(rows, "full_success"),
        "off_target_rate": target_rate(rows, "off_success"),
        "random_control_target_rate": target_rate(rows, "rand_success"),
        "key_coeff_positive_rate": target_rate(rows, "a_positive"),
        "metrics": {},
    }

    for name in metric_names:
        output["metrics"][name] = summarize(
            np.asarray([row[name] for row in rows], dtype=np.float64)
        )

    return output


def _median(group_summary: Dict, metric_name: str) -> Optional[float]:
    return group_summary["metrics"][metric_name]["median"]


def _safe_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
    eps: float,
) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None
    if abs(denominator) <= eps:
        return None
    return float(numerator / denominator)


def summarize_rows(rows: List[Dict], eps: float) -> Dict:
    success_rows = [row for row in rows if int(row["full_success"]) == 1]
    failure_rows = [row for row in rows if int(row["full_success"]) == 0]

    all_summary = summarize_group(rows)
    success_summary = summarize_group(success_rows)
    failure_summary = summarize_group(failure_rows)

    ratio_metrics = (
        "a_abs",
        "g_key",
        "g_spec",
        "r",
        "r_spec",
        "off_deficit",
        "gain_coverage",
        "specific_gain_coverage",
    )

    failure_success_ratios = {}
    for metric_name in ratio_metrics:
        failure_success_ratios[metric_name] = _safe_ratio(
            _median(failure_summary, metric_name),
            _median(success_summary, metric_name),
            eps,
        )

    return {
        "all": all_summary,
        "full_success": success_summary,
        "full_failure": failure_summary,
        "failure_success_median_ratios": failure_success_ratios,
    }


def evaluate_context(
    args,
    method_name,
    encoder,
    head,
    trigger,
    key,
    random_direction,
    device,
):
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()

    resolved_layer = resolve_layer_name(encoder, args.key_layer)

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
            tqdm.tqdm(loader, desc=f"fsb-readout-{method_name}")
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

            logits_combined, activation = forward_capture(
                encoder,
                head,
                combined,
                resolved_layer,
            )

            n = images.shape[0]
            clean_activation = activation[:n]
            triggered_activation = activation[n:]
            full_logits = logits_combined[n:]

            shift = triggered_activation - clean_activation
            key_coeff = torch.matmul(shift, key)

            key_component = (
                key_coeff.unsqueeze(1) * key.unsqueeze(0)
            )

            equal_energy_random_component = (
                key_coeff.abs().unsqueeze(1)
                * random_direction.unsqueeze(0)
            )

            off_logits = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                -key_component,
            )

            random_logits = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                -equal_energy_random_component,
            )

            m_full = target_margin(full_logits, args.target_cls)
            m_off = target_margin(off_logits, args.target_cls)
            m_rand = target_margin(random_logits, args.target_cls)

            g_key = m_full - m_off
            g_rand = m_full - m_rand
            g_spec = g_key - g_rand

            a_abs = key_coeff.abs()
            r = g_key / a_abs.clamp_min(args.eps)
            r_spec = g_spec / a_abs.clamp_min(args.eps)

            off_deficit = torch.relu(-m_off)

            gain_coverage = torch.full_like(g_key, float("nan"))
            specific_gain_coverage = torch.full_like(g_spec, float("nan"))

            valid_deficit = off_deficit > args.eps
            gain_coverage[valid_deficit] = (
                g_key[valid_deficit] / off_deficit[valid_deficit]
            )
            specific_gain_coverage[valid_deficit] = (
                g_spec[valid_deficit] / off_deficit[valid_deficit]
            )

            full_success = (
                full_logits.argmax(dim=1) == args.target_cls
            )
            off_success = (
                off_logits.argmax(dim=1) == args.target_cls
            )
            rand_success = (
                random_logits.argmax(dim=1) == args.target_cls
            )
            a_positive = key_coeff > 0

            for index in range(n):
                rows.append(
                    {
                        "method": method_name,
                        "sample_index": sample_index,
                        "label": int(labels[index].item()),
                        "full_success": int(full_success[index].item()),
                        "off_success": int(off_success[index].item()),
                        "rand_success": int(rand_success[index].item()),
                        "a_positive": int(a_positive[index].item()),
                        "m_full": float(m_full[index].item()),
                        "m_off": float(m_off[index].item()),
                        "m_rand": float(m_rand[index].item()),
                        "a_signed": float(key_coeff[index].item()),
                        "a_abs": float(a_abs[index].item()),
                        "g_key": float(g_key[index].item()),
                        "g_rand": float(g_rand[index].item()),
                        "g_spec": float(g_spec[index].item()),
                        "r": float(r[index].item()),
                        "r_spec": float(r_spec[index].item()),
                        "off_deficit": float(off_deficit[index].item()),
                        "gain_coverage": float(gain_coverage[index].item()),
                        "specific_gain_coverage": float(
                            specific_gain_coverage[index].item()
                        ),
                    }
                )
                sample_index += 1

    if not rows:
        raise RuntimeError(f"No rows collected for {method_name}")

    return rows, summarize_rows(rows, eps=args.eps)


def evaluate_equalized_dose(
    args,
    method_name,
    encoder,
    head,
    trigger,
    key,
    reference_abs_dose,
    device,
):
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()

    resolved_layer = resolve_layer_name(encoder, args.key_layer)

    _, loader = get_dataset(
        args.target_task,
        "test",
        encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    all_full_success = []
    all_equalized_success = []

    failure_equalized_success = []
    failure_full_margin = []
    failure_equalized_margin = []
    failure_current_abs_dose = []
    failure_added_abs_dose = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm.tqdm(loader, desc=f"dose-equalize-{method_name}")
        ):
            if batch_idx >= args.eval_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            non_target = labels != args.target_cls
            images = images[non_target]

            if images.shape[0] == 0:
                continue

            patched = apply_trigger(images, trigger)
            combined = torch.cat([images, patched], dim=0)

            logits_combined, activation = forward_capture(
                encoder,
                head,
                combined,
                resolved_layer,
            )

            n = images.shape[0]
            clean_activation = activation[:n]
            triggered_activation = activation[n:]
            full_logits = logits_combined[n:]

            shift = triggered_activation - clean_activation
            key_coeff = torch.matmul(shift, key)
            key_component = (
                key_coeff.unsqueeze(1) * key.unsqueeze(0)
            )

            signs = torch.where(
                key_coeff >= 0,
                torch.ones_like(key_coeff),
                -torch.ones_like(key_coeff),
            )
            reference_coeff = signs * float(reference_abs_dose)
            reference_component = (
                reference_coeff.unsqueeze(1) * key.unsqueeze(0)
            )

            equalized_delta = -key_component + reference_component

            equalized_logits = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                equalized_delta,
            )

            full_success = (
                full_logits.argmax(dim=1) == args.target_cls
            )
            equalized_success = (
                equalized_logits.argmax(dim=1) == args.target_cls
            )

            full_margin = target_margin(full_logits, args.target_cls)
            equalized_margin = target_margin(
                equalized_logits,
                args.target_cls,
            )

            original_failure = ~full_success

            all_full_success.extend(
                full_success.detach().cpu().numpy().astype(np.int64).tolist()
            )
            all_equalized_success.extend(
                equalized_success.detach().cpu().numpy().astype(np.int64).tolist()
            )

            if original_failure.any():
                failure_equalized_success.extend(
                    equalized_success[original_failure]
                    .detach().cpu().numpy().astype(np.int64).tolist()
                )
                failure_full_margin.extend(
                    full_margin[original_failure]
                    .detach().float().cpu().numpy().tolist()
                )
                failure_equalized_margin.extend(
                    equalized_margin[original_failure]
                    .detach().float().cpu().numpy().tolist()
                )

                current_abs_dose = key_coeff[original_failure].abs()
                failure_current_abs_dose.extend(
                    current_abs_dose.detach().float().cpu().numpy().tolist()
                )
                failure_added_abs_dose.extend(
                    (
                        float(reference_abs_dose) - current_abs_dose
                    )
                    .detach().float().cpu().numpy().tolist()
                )

    if not all_full_success:
        raise RuntimeError("No samples collected for equalized-dose audit")

    original_failure_count = int(
        np.sum(
            1 - np.asarray(all_full_success, dtype=np.int64)
        )
    )

    return {
        "method": method_name,
        "reference_abs_dose": float(reference_abs_dose),
        "num_samples": len(all_full_success),
        "original_full_asr": (
            100.0 * float(np.mean(all_full_success))
        ),
        "equalized_all_asr": (
            100.0 * float(np.mean(all_equalized_success))
        ),
        "original_failure_count": original_failure_count,
        "original_failure_rescue_rate": (
            None
            if not failure_equalized_success
            else 100.0 * float(np.mean(failure_equalized_success))
        ),
        "failure_full_margin": summarize(
            np.asarray(failure_full_margin, dtype=np.float64)
        ),
        "failure_equalized_margin": summarize(
            np.asarray(failure_equalized_margin, dtype=np.float64)
        ),
        "failure_current_abs_dose": summarize(
            np.asarray(failure_current_abs_dose, dtype=np.float64)
        ),
        "failure_added_abs_dose": summarize(
            np.asarray(failure_added_abs_dose, dtype=np.float64)
        ),
    }


def write_csv(rows: List[Dict], path: str):
    if not rows:
        raise RuntimeError("Cannot write empty CSV")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, width=11, precision=4):
    if value is None:
        return "nan".rjust(width)
    return f"{value:{width}.{precision}f}"


def print_results(results):
    print()
    print("========== FSB FINAL-STATE READOUT EFFICIENCY ==========")

    print(
        "method".ljust(14)
        + "group".ljust(10)
        + "N".rjust(7)
        + "ASR".rjust(9)
        + "|a|".rjust(11)
        + "g_key".rjust(11)
        + "g_spec".rjust(11)
        + "r".rjust(11)
        + "r_spec".rjust(11)
        + "M_off".rjust(11)
        + "deficit".rjust(11)
        + "coverage".rjust(11)
    )

    for method_name, method_summary in results.items():
        for group_name in ("all", "full_success", "full_failure"):
            group = method_summary[group_name]
            metrics = group["metrics"]

            print(
                method_name.ljust(14)
                + group_name.replace("full_", "").ljust(10)
                + f"{group['count']:7d}"
                + _fmt(group["full_target_rate"], width=9, precision=2)
                + _fmt(metrics["a_abs"]["median"])
                + _fmt(metrics["g_key"]["median"])
                + _fmt(metrics["g_spec"]["median"])
                + _fmt(metrics["r"]["median"])
                + _fmt(metrics["r_spec"]["median"])
                + _fmt(metrics["m_off"]["median"])
                + _fmt(metrics["off_deficit"]["median"])
                + _fmt(metrics["gain_coverage"]["median"])
            )

    print()
    print("========== FAILURE / SUCCESS MEDIAN RATIOS ==========")

    print(
        "method".ljust(14)
        + "|a|".rjust(11)
        + "g_key".rjust(11)
        + "g_spec".rjust(11)
        + "r".rjust(11)
        + "r_spec".rjust(11)
        + "deficit".rjust(11)
        + "coverage".rjust(11)
    )

    for method_name, method_summary in results.items():
        ratios = method_summary["failure_success_median_ratios"]

        print(
            method_name.ljust(14)
            + _fmt(ratios["a_abs"])
            + _fmt(ratios["g_key"])
            + _fmt(ratios["g_spec"])
            + _fmt(ratios["r"])
            + _fmt(ratios["r_spec"])
            + _fmt(ratios["off_deficit"])
            + _fmt(ratios["gain_coverage"])
        )


def print_equalized_dose(result):
    print()
    print("========== REGMEAN SUCCESS-DOSE COUNTERFACTUAL ==========")

    print(f"method: {result['method']}")
    print(
        "reference |a| "
        "(median over original full-success samples): "
        f"{result['reference_abs_dose']:.6f}"
    )
    print(f"original full ASR: {result['original_full_asr']:.2f}%")
    print(
        f"equalized all-sample ASR: "
        f"{result['equalized_all_asr']:.2f}%"
    )
    print(
        f"original failure count: "
        f"{result['original_failure_count']}"
    )
    print(
        f"original failure rescue rate: "
        f"{result['original_failure_rescue_rate']}"
    )

    for name in (
        "failure_full_margin",
        "failure_equalized_margin",
        "failure_current_abs_dose",
        "failure_added_abs_dose",
    ):
        summary = result[name]
        print(
            f"{name}: "
            f"median={summary['median']}, "
            f"q25={summary['q25']}, "
            f"q75={summary['q75']}"
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    methods = parse_methods(args)

    if args.equalize_method not in methods:
        raise ValueError(
            "--equalize-method must be included in --methods. "
            f"Got equalize_method={args.equalize_method}, methods={methods}"
        )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

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
        clean_checkpoint_path(args, dataset_name)
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

    key, prototype_summary = estimate_global_key(
        args,
        pretrained_encoder,
        trigger,
        device,
    )
    key = key.to(device)

    random_direction = make_random_orthogonal_direction(
        key,
        seed=args.seed + 31,
    ).to(device)

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
            clean_checkpoint_path(args, args.adversary_task),
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
        (method_name, builder_map[method_name])
        for method_name in methods
    )

    results = OrderedDict()
    all_rows = []
    equalized_dose_result = None

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
            key=key,
            random_direction=random_direction,
            device=device,
        )

        results[method_name] = summary
        all_rows.extend(rows)

        if method_name == args.equalize_method:
            success_abs_dose = np.asarray(
                [
                    row["a_abs"]
                    for row in rows
                    if int(row["full_success"]) == 1
                ],
                dtype=np.float64,
            )

            if success_abs_dose.size == 0:
                raise RuntimeError(
                    "Cannot run dose equalization because "
                    "there are no full-success samples."
                )

            reference_abs_dose = float(
                np.median(success_abs_dose)
            )

            equalized_dose_result = evaluate_equalized_dose(
                args=args,
                method_name=method_name,
                encoder=encoder,
                head=head,
                trigger=trigger,
                key=key,
                reference_abs_dose=reference_abs_dose,
                device=device,
            )

        del encoder
        torch.cuda.empty_cache()

    print_results(results)

    if equalized_dose_result is None:
        raise RuntimeError("Equalized-dose result was not produced.")

    print_equalized_dose(equalized_dose_result)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(
        args.out_dir,
        f"{timestamp}_kdr_dtk_fsb_final_state_readout_samples.csv",
    )
    json_path = os.path.join(
        args.out_dir,
        f"{timestamp}_kdr_dtk_fsb_final_state_readout.json",
    )

    write_csv(all_rows, csv_path)

    output = {
        "timestamp": timestamp,
        "question": (
            "After FSB closes the final-state causal interface, do "
            "remaining RegMean failures come from a smaller absolute "
            "final-state key dose, weaker per-unit key readout, or a "
            "harder key-removed target-margin baseline?"
        ),
        "model": args.model,
        "attack_type": args.attack_type,
        "trigger_source": args.trigger_source,
        "methods": methods,
        "equalize_method": args.equalize_method,
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
        "stage1_final_state_key_stability": prototype_summary,
        "definitions": {
            "d": (
                "triggered minus clean CLS activation at the "
                "complete resblocks.11 output"
            ),
            "a": "d^T k_out",
            "A": "|a|",
            "M_full": "target margin under the natural triggered state",
            "M_off": (
                "target margin after exact removal of the measured "
                "final-state key component a*k_out"
            ),
            "M_rand": (
                "target margin after an equal-energy intervention "
                "along a fixed direction orthogonal to k_out"
            ),
            "g_key": "M_full - M_off",
            "g_rand": "M_full - M_rand",
            "g_spec": "g_key - g_rand",
            "r": "g_key / (A + eps)",
            "r_spec": "g_spec / (A + eps)",
            "off_deficit": "relu(-M_off)",
            "gain_coverage": (
                "g_key / off_deficit for samples with M_off < 0"
            ),
            "specific_gain_coverage": (
                "g_spec / off_deficit for samples with M_off < 0"
            ),
            "success_dose_counterfactual": (
                "For the selected context, remove each sample's "
                "current final-state key component and reinsert the "
                "same-sign key direction with absolute dose equal to "
                "the median |a| of that context's original full-success "
                "samples. No parameters are updated."
            ),
        },
        "results": results,
        "success_dose_counterfactual": equalized_dose_result,
        "sample_csv": csv_path,
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("========== DONE ==========")
    print(f"[FSB Final-State Readout] CSV: {csv_path}")
    print(f"[FSB Final-State Readout] JSON: {json_path}")


if __name__ == "__main__":
    main()