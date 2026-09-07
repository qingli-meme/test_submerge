import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
import tqdm

sys.path.append(".")
sys.path.append("./src")

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.kdr_utils import pool_activation, save_json

from diagnose_kdr_regmean_readout import (
    apply_trigger,
    attack_paths,
    build_adamerging_encoder,
    build_regmean_encoder,
    build_ta_encoder,
    build_ties_encoder,
    load_trigger,
    make_args,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose KDR cross-layer residual response channel."
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--num-train-batch", type=int, default=8)
    parser.add_argument("--pool", default="cls", choices=["cls", "mean"])
    parser.add_argument("--adamerging-lambda", default="")
    parser.add_argument("--out-dir", default="./analysis/kdr_cross_layer_channel")
    parser.add_argument("--cache-dir", default="./analysis/kdr_regmean_readout/cache")
    return parser.parse_args()


def find_resblock_names(model):
    modules = dict(model.named_modules())
    resolved = OrderedDict()
    for layer_idx in range(12):
        suffix = f"visual.transformer.resblocks.{layer_idx}"
        matches = [name for name in modules if name == suffix or name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one module ending with '{suffix}', found {matches}"
            )
        resolved[layer_idx] = matches[0]
    return resolved


def collect_block_features(model, images, layer_names, pool):
    holders = {}
    handles = []
    modules = dict(model.named_modules())

    for layer_idx, layer_name in layer_names.items():
        module = modules[layer_name]

        def make_hook(idx):
            def hook_fn(_, __, output):
                holders[idx] = output

            return hook_fn

        handles.append(module.register_forward_hook(make_hook(layer_idx)))

    try:
        _ = model(images)
    finally:
        for handle in handles:
            handle.remove()

    missing = [idx for idx in layer_names if idx not in holders]
    if missing:
        raise RuntimeError(f"Hooks failed to capture layers: {missing}")

    return {
        idx: pool_activation(act, batch_size=images.shape[0], pool=pool).detach()
        for idx, act in holders.items()
    }


def append_responses(store, attack_features, clean_features):
    for layer_idx in attack_features:
        response = attack_features[layer_idx] - clean_features[layer_idx]
        store[layer_idx].append(response.detach().float().cpu())


def summarize_channel(response_store):
    per_layer = {}
    prototypes = {}

    for layer_idx in sorted(response_store):
        if not response_store[layer_idx]:
            raise RuntimeError(f"No responses collected for layer {layer_idx}")
        response = torch.cat(response_store[layer_idx], dim=0)
        response_norm = response.norm(dim=-1)
        response_hat = F.normalize(response, dim=-1, eps=1e-12)
        proto = F.normalize(
            response_hat.mean(dim=0, keepdim=True), dim=-1, eps=1e-12
        ).squeeze(0)
        concentration = (response_hat * proto.unsqueeze(0)).sum(dim=-1)
        per_layer[str(layer_idx)] = {
            "num_samples": int(response.shape[0]),
            "concentration_C": float(concentration.mean().item()),
            "concentration_std": float(concentration.std(unbiased=False).item()),
            "response_norm_mean": float(response_norm.mean().item()),
            "response_norm_std": float(response_norm.std(unbiased=False).item()),
        }
        prototypes[layer_idx] = proto

    adjacent = {}
    layer_indices = sorted(prototypes)
    for left, right in zip(layer_indices[:-1], layer_indices[1:]):
        coherence = F.cosine_similarity(
            prototypes[left].unsqueeze(0), prototypes[right].unsqueeze(0), dim=-1
        ).item()
        adjacent[f"{left}->{right}"] = {"coherence_P": float(coherence)}

    return {"per_layer": per_layer, "adjacent": adjacent}


def add_trigger_clean_contrast(triggered, clean):
    return {
        "per_layer_delta_C": {
            idx: row["concentration_C"] - clean["per_layer"][idx]["concentration_C"]
            for idx, row in triggered["per_layer"].items()
        },
        "adjacent_delta_P": {
            trans: row["coherence_P"] - clean["adjacent"][trans]["coherence_P"]
            for trans, row in triggered["adjacent"].items()
        },
    }


def evaluate_method(args, method_name, attack_encoder, clean_encoder, trigger_info):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    attack_encoder = attack_encoder.to(device).eval()
    clean_encoder = clean_encoder.to(device).eval()
    attack_layer_names = find_resblock_names(attack_encoder)
    clean_layer_names = find_resblock_names(clean_encoder)

    _, loader = get_dataset(
        args.target_task,
        "test",
        attack_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    triggered_store = {layer_idx: [] for layer_idx in range(12)}
    clean_store = {layer_idx: [] for layer_idx in range(12)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc=f"channel-{method_name}")):
            if batch_idx >= args.max_batches:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            images = images[labels != args.target_cls]
            if images.shape[0] == 0:
                continue

            patched = apply_trigger(images, trigger_info)

            attack_triggered = collect_block_features(
                attack_encoder, patched, attack_layer_names, args.pool
            )
            clean_triggered = collect_block_features(
                clean_encoder, patched, clean_layer_names, args.pool
            )
            append_responses(triggered_store, attack_triggered, clean_triggered)

            attack_clean = collect_block_features(
                attack_encoder, images, attack_layer_names, args.pool
            )
            clean_clean = collect_block_features(
                clean_encoder, images, clean_layer_names, args.pool
            )
            append_responses(clean_store, attack_clean, clean_clean)

    triggered_summary = summarize_channel(triggered_store)
    clean_summary = summarize_channel(clean_store)
    result = {
        "method": method_name,
        "triggered": triggered_summary,
        "clean_control": clean_summary,
        "trigger_minus_clean": add_trigger_clean_contrast(
            triggered_summary, clean_summary
        ),
        "attack_layer_names": attack_layer_names,
        "clean_layer_names": clean_layer_names,
    }

    attack_encoder.cpu()
    clean_encoder.cpu()
    torch.cuda.empty_cache()
    return result


def print_layer_table(results, branch):
    print()
    print(f"========== {branch.upper()} WITHIN-LAYER CONCENTRATION C ==========")
    methods = list(results.keys())
    header = "layer".ljust(8) + "".join(method.rjust(14) for method in methods)
    print(header)
    for layer_idx in range(12):
        row = str(layer_idx).ljust(8)
        for method in methods:
            value = results[method][branch]["per_layer"][str(layer_idx)][
                "concentration_C"
            ]
            row += f"{value:14.4f}"
        print(row)

    print()
    print(f"========== {branch.upper()} ADJACENT COHERENCE P ==========")
    header = "trans".ljust(8) + "".join(method.rjust(14) for method in methods)
    print(header)
    for layer_idx in range(11):
        key = f"{layer_idx}->{layer_idx + 1}"
        row = key.ljust(8)
        for method in methods:
            value = results[method][branch]["adjacent"][key]["coherence_P"]
            row += f"{value:14.4f}"
        print(row)


def print_contrast_table(results):
    methods = list(results.keys())
    print()
    print("========== TRIGGER - CLEAN DELTA C ==========")
    header = "layer".ljust(8) + "".join(method.rjust(14) for method in methods)
    print(header)
    for layer_idx in range(12):
        row = str(layer_idx).ljust(8)
        for method in methods:
            value = results[method]["trigger_minus_clean"]["per_layer_delta_C"][
                str(layer_idx)
            ]
            row += f"{value:14.4f}"
        print(row)

    print()
    print("========== TRIGGER - CLEAN DELTA P ==========")
    header = "trans".ljust(8) + "".join(method.rjust(14) for method in methods)
    print(header)
    for layer_idx in range(11):
        key = f"{layer_idx}->{layer_idx + 1}"
        row = key.ljust(8)
        for method in methods:
            value = results[method]["trigger_minus_clean"]["adjacent_delta_P"][key]
            row += f"{value:14.4f}"
        print(row)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    if not args.adamerging_lambda:
        args.adamerging_lambda = os.path.join(
            "./ada",
            args.model,
            f"KDR_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}_Epoch_500.pt",
        )
    if not os.path.exists(args.adamerging_lambda):
        raise FileNotFoundError(
            "AdaMerging lambda is required for the four-merge diagnostic: "
            f"{args.adamerging_lambda}"
        )

    trigger_info = load_trigger(args)
    clean_paths = attack_paths(args, attack="clean")
    kdr_paths = attack_paths(args, attack="kdr")

    print("[diag] building clean merge references")
    clean_models = {
        "ta": build_ta_encoder(args, clean_paths),
        "ties": build_ties_encoder(args, clean_paths),
        "regmean": build_regmean_encoder(args, clean_paths, cache_name="clean_regmean"),
        "adamerging": build_adamerging_encoder(args, clean_paths, args.adamerging_lambda),
    }

    print("[diag] building KDR attack merges")
    attack_models = {
        "ta": build_ta_encoder(args, kdr_paths),
        "ties": build_ties_encoder(args, kdr_paths),
        "regmean": build_regmean_encoder(args, kdr_paths, cache_name="kdr_regmean_final"),
        "adamerging": build_adamerging_encoder(args, kdr_paths, args.adamerging_lambda),
    }

    results = {}
    for method_name in ["ta", "ties", "regmean", "adamerging"]:
        print()
        print(f"[diag] evaluating {method_name}")
        results[method_name] = evaluate_method(
            args=args,
            method_name=method_name,
            attack_encoder=attack_models[method_name],
            clean_encoder=clean_models[method_name],
            trigger_info=trigger_info,
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": stamp,
        "hypothesis": (
            "KDR forms a trigger-conditioned cross-layer residual response channel; "
            "RegMean preserves response magnitude but disrupts directional concentration "
            "or adjacent-layer coherence."
        ),
        "model": args.model,
        "target_task": args.target_task,
        "target_cls": args.target_cls,
        "patch_size": args.patch_size,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "pool": args.pool,
        "trigger_path": trigger_info["path"],
        "adamerging_lambda": args.adamerging_lambda,
        "results": results,
    }
    out_path = os.path.join(args.out_dir, f"{stamp}_cross_layer_kdr_channel.json")
    save_json(output, out_path)

    print_layer_table(results, branch="triggered")
    print_layer_table(results, branch="clean_control")
    print_contrast_table(results)
    print()
    print(f"[diag] wrote: {out_path}")


if __name__ == "__main__":
    main()
