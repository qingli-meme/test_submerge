import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict

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
            "Sender-dose audit for KDR-DTK-BIND: compare current key coordinate "
            "against frozen Stage-1 native sender dose and test sender-dose "
            "counterfactual readout."
        )
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--attack-type", default="KDR_DTK_BIND")
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
        help="Comma-separated contexts inherited from diagnose_kdr_key_causality.py.",
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument(
        "--out-dir",
        default="./analysis/kdr_dtk_bind_sender_dose_audit",
    )
    return parser.parse_args()


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    return target - masked.max(dim=1).values


def summarize_array(values):
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


def summarize_rows(rows):
    metrics = (
        "sender_dose_abs",
        "current_dose_abs",
        "dose_ratio_current_over_sender",
        "dose_delta_abs",
        "full_margin",
        "sender_dose_margin",
        "sender_minus_full_margin",
        "full_target_logit",
        "sender_dose_target_logit",
    )
    output = {
        "count": len(rows),
        "full_asr": 100.0 * float(np.mean([r["full_success"] for r in rows])) if rows else None,
        "sender_dose_asr": 100.0 * float(np.mean([r["sender_dose_success"] for r in rows])) if rows else None,
        "metrics": {},
    }
    for metric in metrics:
        output["metrics"][metric] = summarize_array([r[metric] for r in rows])
    return output


def summarize_method(rows):
    success = [r for r in rows if r["full_success"] == 1]
    failure = [r for r in rows if r["full_success"] == 0]
    out = {
        "all": summarize_rows(rows),
        "success": summarize_rows(success),
        "failure": summarize_rows(failure),
        "success_vs_failure": {},
    }
    for metric in (
        "sender_dose_abs",
        "current_dose_abs",
        "dose_ratio_current_over_sender",
        "full_margin",
        "sender_dose_margin",
        "sender_minus_full_margin",
    ):
        s = out["success"]["metrics"][metric]["median"]
        f = out["failure"]["metrics"][metric]["median"]
        out["success_vs_failure"][metric] = {
            "success_median": s,
            "failure_median": f,
            "failure_minus_success": None if s is None or f is None else float(f - s),
            "failure_over_success": (
                None if s is None or f is None or abs(float(s)) <= 1e-12 else float(f) / float(s)
            ),
        }
    return out


def write_csv(rows, path):
    if not rows:
        raise RuntimeError("Cannot write empty CSV")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_sender_dose(
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
    pretrained_resolved_layer = resolve_layer_name(pretrained_encoder, args.key_layer)

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
        for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc=f"sender-dose-{method_name}")):
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
            batch_size = images.shape[0]

            _, pretrained_act = forward_capture(
                pretrained_encoder,
                head,
                combined,
                pretrained_resolved_layer,
            )
            pre_clean = pretrained_act[:batch_size]
            pre_trig = pretrained_act[batch_size:]
            sender_shift = pre_trig - pre_clean
            sender_coeff = torch.matmul(sender_shift, key)
            sender_amp = sender_coeff.abs()

            logits_combined, current_act = forward_capture(
                encoder,
                head,
                combined,
                resolved_layer,
            )
            full_logits = logits_combined[batch_size:]
            cur_clean = current_act[:batch_size]
            cur_trig = current_act[batch_size:]
            current_shift = cur_trig - cur_clean
            current_coeff = torch.matmul(current_shift, key)
            current_amp = current_coeff.abs()

            delta_to_sender_dose = (sender_coeff - current_coeff).unsqueeze(1) * key.unsqueeze(0)
            sender_dose_logits = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                delta_to_sender_dose,
            )

            full_margin = target_margin(full_logits, args.target_cls)
            sender_margin = target_margin(sender_dose_logits, args.target_cls)
            full_target_logit = full_logits[:, args.target_cls]
            sender_target_logit = sender_dose_logits[:, args.target_cls]
            full_success = full_logits.argmax(dim=1) == args.target_cls
            sender_success = sender_dose_logits.argmax(dim=1) == args.target_cls
            dose_ratio = current_amp / (sender_amp + args.eps)
            dose_delta_abs = (current_coeff - sender_coeff).abs()

            for idx in range(batch_size):
                rows.append(
                    {
                        "method": method_name,
                        "sample_index": sample_index,
                        "sender_dose_signed": float(sender_coeff[idx].item()),
                        "current_dose_signed": float(current_coeff[idx].item()),
                        "sender_dose_abs": float(sender_amp[idx].item()),
                        "current_dose_abs": float(current_amp[idx].item()),
                        "dose_ratio_current_over_sender": float(dose_ratio[idx].item()),
                        "dose_delta_abs": float(dose_delta_abs[idx].item()),
                        "full_success": int(full_success[idx].item()),
                        "sender_dose_success": int(sender_success[idx].item()),
                        "full_margin": float(full_margin[idx].item()),
                        "sender_dose_margin": float(sender_margin[idx].item()),
                        "sender_minus_full_margin": float((sender_margin[idx] - full_margin[idx]).item()),
                        "full_target_logit": float(full_target_logit[idx].item()),
                        "sender_dose_target_logit": float(sender_target_logit[idx].item()),
                    }
                )
                sample_index += 1

    if not rows:
        raise RuntimeError(f"No rows collected for {method_name}")
    return rows, summarize_method(rows)


def print_summary(results):
    print()
    print("========== SENDER-DOSE AUDIT ==========")
    print(
        "method".ljust(14)
        + "N".rjust(8)
        + "ASR".rjust(10)
        + "ASR_sender".rjust(12)
        + "|a0| med".rjust(12)
        + "|am| med".rjust(12)
        + "|am|/|a0|".rjust(12)
        + "M_full".rjust(12)
        + "M_sender".rjust(12)
    )
    for method, result in results.items():
        all_summary = result["all"]
        metrics = all_summary["metrics"]

        def med(name):
            value = metrics[name]["median"]
            return float("nan") if value is None else value

        print(
            method.ljust(14)
            + f"{all_summary['count']:8d}"
            + f"{all_summary['full_asr']:10.2f}"
            + f"{all_summary['sender_dose_asr']:12.2f}"
            + f"{med('sender_dose_abs'):12.4f}"
            + f"{med('current_dose_abs'):12.4f}"
            + f"{med('dose_ratio_current_over_sender'):12.4f}"
            + f"{med('full_margin'):12.4f}"
            + f"{med('sender_dose_margin'):12.4f}"
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    methods = parse_methods(args)
    runtime_args = make_runtime_args(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    trigger = load_trigger_patch(trigger_path(args), args.patch_size, device)
    pretrained_encoder = load_encoder(pretrained_path(args), device)
    print("[SenderDose] estimating empirical Stage-1 key")
    key, prototype_summary = estimate_global_key(args, pretrained_encoder, trigger, device)
    key = key.to(device)

    head = get_classification_head(runtime_args, args.target_task).to(device).eval()
    for param in head.parameters():
        param.requires_grad_(False)

    builder_map = {
        "clean_local": lambda: load_encoder(clean_checkpoint_path(args, args.adversary_task), device),
        "local_attack": lambda: load_encoder(kdr_checkpoint_path(args), device),
        "ta": lambda: build_linear_merge_encoder(args, "ta", device),
        "ties": lambda: build_linear_merge_encoder(args, "ties", device),
        "regmean": lambda: build_regmean_encoder(args, runtime_args, device),
        "adamerging": lambda: build_adamerging_encoder(args, device),
    }
    builders = OrderedDict((method, builder_map[method]) for method in methods)

    results = OrderedDict()
    all_rows = []
    for method_name, builder in builders.items():
        print()
        print(f"========== BUILD {method_name} ==========")
        encoder = builder()
        rows, summary = evaluate_sender_dose(
            args=args,
            method_name=method_name,
            encoder=encoder,
            pretrained_encoder=pretrained_encoder,
            head=head,
            trigger=trigger,
            key=key,
            device=device,
        )
        all_rows.extend(rows)
        results[method_name] = summary
        del encoder
        torch.cuda.empty_cache()

    print_summary(results)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"{timestamp}_kdr_sender_dose_samples.csv")
    json_path = os.path.join(args.out_dir, f"{timestamp}_kdr_sender_dose_audit.json")
    write_csv(all_rows, csv_path)

    output = {
        "timestamp": timestamp,
        "question": (
            "Does each receiver read the frozen Stage-1 native sender dose, "
            "or does BIND rely on self-amplified current key coordinates?"
        ),
        "model": args.model,
        "attack_type": args.attack_type,
        "trigger_source": args.trigger_source,
        "methods": methods,
        "target_task": args.target_task,
        "target_cls": args.target_cls,
        "patch_size": args.patch_size,
        "key_layer": args.key_layer,
        "prototype_batches": args.prototype_batches,
        "eval_batches": args.eval_batches,
        "batch_size": args.batch_size,
        "eps": args.eps,
        "trigger_path": trigger_path(args),
        "attack_checkpoint": kdr_checkpoint_path(args),
        "stage1_key_stability": prototype_summary,
        "definitions": {
            "sender_dose_signed": "a0 = <h_theta0(T(x)) - h_theta0(x), k>",
            "current_dose_signed": "am = <h_m(T(x)) - h_m(x), k>",
            "sender_dose_counterfactual": "h_m(T(x)) - am*k + a0*k",
            "dose_ratio_current_over_sender": "|am| / (|a0| + eps)",
        },
        "results": results,
        "sample_csv": csv_path,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("========== DONE ==========")
    print(f"[SenderDose] CSV: {csv_path}")
    print(f"[SenderDose] JSON: {json_path}")


if __name__ == "__main__":
    main()
