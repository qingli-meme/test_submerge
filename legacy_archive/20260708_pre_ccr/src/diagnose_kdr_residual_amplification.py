import argparse
import os
import sys
import time
from collections import OrderedDict

import torch
import tqdm

sys.path.append(".")
sys.path.append("./src")

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.kdr_utils import save_json

from diagnose_kdr_regmean_readout import (
    apply_trigger,
    attack_paths,
    build_adamerging_encoder,
    build_regmean_encoder,
    build_ta_encoder,
    build_ties_encoder,
    load_trigger,
)
from diagnose_cross_layer_kdr_channel import collect_block_features, find_resblock_names


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose sample-paired KDR trigger-conditioned residual amplification."
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
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--num-train-batch", type=int, default=8)
    parser.add_argument("--pool", default="cls", choices=["cls", "mean"])
    parser.add_argument("--adamerging-lambda", default="")
    parser.add_argument("--cache-dir", default="./analysis/kdr_regmean_readout/cache")
    parser.add_argument("--out-dir", default="./analysis/kdr_residual_amplification")
    return parser.parse_args()


def summarize_vector(values):
    values = values.detach().float().cpu()
    return {
        "num_samples": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(torch.quantile(values, 0.50).item()),
        "q10": float(torch.quantile(values, 0.10).item()),
        "q90": float(torch.quantile(values, 0.90).item()),
        "std": float(values.std(unbiased=False).item()),
    }


def collect_method_amplification(args, method_name, attack_encoder, clean_encoder, trigger_info):
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

    amplification_store = {layer_idx: [] for layer_idx in range(12)}
    triggered_norm_store = {layer_idx: [] for layer_idx in range(12)}
    clean_norm_store = {layer_idx: [] for layer_idx in range(12)}
    layer_gain_store = {layer_idx: [] for layer_idx in range(11)}
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm.tqdm(loader, desc=f"amplification-{method_name}")
        ):
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
            attack_clean = collect_block_features(
                attack_encoder, images, attack_layer_names, args.pool
            )
            clean_clean = collect_block_features(
                clean_encoder, images, clean_layer_names, args.pool
            )

            batch_amplification = {}
            for layer_idx in range(12):
                triggered_response = attack_triggered[layer_idx] - clean_triggered[layer_idx]
                clean_response = attack_clean[layer_idx] - clean_clean[layer_idx]
                triggered_norm = triggered_response.norm(dim=-1)
                clean_norm = clean_response.norm(dim=-1)
                amplification = triggered_norm / (clean_norm + EPS)

                amplification_store[layer_idx].append(amplification.detach().cpu())
                triggered_norm_store[layer_idx].append(triggered_norm.detach().cpu())
                clean_norm_store[layer_idx].append(clean_norm.detach().cpu())
                batch_amplification[layer_idx] = amplification

            for layer_idx in range(11):
                layer_gain = batch_amplification[layer_idx + 1] / (
                    batch_amplification[layer_idx] + EPS
                )
                layer_gain_store[layer_idx].append(layer_gain.detach().cpu())

            total_samples += int(images.shape[0])

    result = {
        "method": method_name,
        "total_samples": total_samples,
        "per_layer": OrderedDict(),
        "transitions": OrderedDict(),
    }

    for layer_idx in range(12):
        amplification = torch.cat(amplification_store[layer_idx], dim=0)
        triggered_norm = torch.cat(triggered_norm_store[layer_idx], dim=0)
        clean_norm = torch.cat(clean_norm_store[layer_idx], dim=0)
        result["per_layer"][str(layer_idx)] = {
            "amplification_A": summarize_vector(amplification),
            "triggered_response_norm": summarize_vector(triggered_norm),
            "clean_response_norm": summarize_vector(clean_norm),
            "ratio_of_mean_norms": float(
                triggered_norm.mean().item() / (clean_norm.mean().item() + EPS)
            ),
        }

    for layer_idx in range(11):
        layer_gain = torch.cat(layer_gain_store[layer_idx], dim=0)
        result["transitions"][f"{layer_idx}->{layer_idx + 1}"] = {
            "growth_G": summarize_vector(layer_gain)
        }

    attack_encoder.cpu()
    clean_encoder.cpu()
    torch.cuda.empty_cache()
    return result


def successful_group_median(results, section, item_key, metric_group, statistic):
    values = [
        results[method][section][item_key][metric_group][statistic]
        for method in ["ta", "ties", "adamerging"]
    ]
    return float(torch.quantile(torch.tensor(values, dtype=torch.float32), 0.50).item())


def add_group_comparison(results):
    comparison = {"per_layer": OrderedDict(), "transitions": OrderedDict()}
    for layer_idx in range(12):
        key = str(layer_idx)
        success_median = successful_group_median(
            results, "per_layer", key, "amplification_A", "median"
        )
        regmean_value = results["regmean"]["per_layer"][key]["amplification_A"]["median"]
        comparison["per_layer"][key] = {
            "successful_group_median_A": success_median,
            "regmean_median_A": regmean_value,
            "successful_minus_regmean": success_median - regmean_value,
        }

    for layer_idx in range(11):
        key = f"{layer_idx}->{layer_idx + 1}"
        success_median = successful_group_median(
            results, "transitions", key, "growth_G", "median"
        )
        regmean_value = results["regmean"]["transitions"][key]["growth_G"]["median"]
        comparison["transitions"][key] = {
            "successful_group_median_G": success_median,
            "regmean_median_G": regmean_value,
            "successful_minus_regmean": success_median - regmean_value,
        }
    return comparison


def print_amplification_table(results):
    methods = ["ta", "ties", "regmean", "adamerging"]
    print()
    print("========== 逐层触发条件放大倍数 A：中位数 ==========")
    print("layer".ljust(8) + "".join(method.rjust(14) for method in methods))
    for layer_idx in range(12):
        row = str(layer_idx).ljust(8)
        for method in methods:
            row += f"{results[method]['per_layer'][str(layer_idx)]['amplification_A']['median']:14.4f}"
        print(row)

    print()
    print("========== 逐层触发条件放大倍数 A：均值 ==========")
    print("layer".ljust(8) + "".join(method.rjust(14) for method in methods))
    for layer_idx in range(12):
        row = str(layer_idx).ljust(8)
        for method in methods:
            row += f"{results[method]['per_layer'][str(layer_idx)]['amplification_A']['mean']:14.4f}"
        print(row)


def print_growth_table(results):
    methods = ["ta", "ties", "regmean", "adamerging"]
    print()
    print("========== 层间放大增长倍数 G：中位数 ==========")
    print("trans".ljust(8) + "".join(method.rjust(14) for method in methods))
    for layer_idx in range(11):
        key = f"{layer_idx}->{layer_idx + 1}"
        row = key.ljust(8)
        for method in methods:
            row += f"{results[method]['transitions'][key]['growth_G']['median']:14.4f}"
        print(row)

    print()
    print("========== 层间放大增长倍数 G：均值 ==========")
    print("trans".ljust(8) + "".join(method.rjust(14) for method in methods))
    for layer_idx in range(11):
        key = f"{layer_idx}->{layer_idx + 1}"
        row = key.ljust(8)
        for method in methods:
            row += f"{results[method]['transitions'][key]['growth_G']['mean']:14.4f}"
        print(row)


def print_group_comparison(comparison):
    print()
    print("========== 成功聚合组与 RegMean 的 A 中位数差 ==========")
    print("layer".ljust(8) + "success_med".rjust(16) + "regmean".rjust(14) + "gap".rjust(14))
    for layer_idx in range(12):
        item = comparison["per_layer"][str(layer_idx)]
        print(
            str(layer_idx).ljust(8)
            + f"{item['successful_group_median_A']:16.4f}"
            + f"{item['regmean_median_A']:14.4f}"
            + f"{item['successful_minus_regmean']:14.4f}"
        )

    print()
    print("========== 成功聚合组与 RegMean 的 G 中位数差 ==========")
    print("trans".ljust(8) + "success_med".rjust(16) + "regmean".rjust(14) + "gap".rjust(14))
    for layer_idx in range(11):
        key = f"{layer_idx}->{layer_idx + 1}"
        item = comparison["transitions"][key]
        print(
            key.ljust(8)
            + f"{item['successful_group_median_G']:16.4f}"
            + f"{item['regmean_median_G']:14.4f}"
            + f"{item['successful_minus_regmean']:14.4f}"
        )


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
        raise FileNotFoundError(f"Missing AdaMerging lambda: {args.adamerging_lambda}")

    trigger_info = load_trigger(args)
    clean_paths = attack_paths(args, attack="clean")
    kdr_paths = attack_paths(args, attack="kdr")

    print("[诊断] 构造 clean merged references")
    clean_models = {
        "ta": build_ta_encoder(args, clean_paths),
        "ties": build_ties_encoder(args, clean_paths),
        "regmean": build_regmean_encoder(args, clean_paths, cache_name="clean_regmean"),
        "adamerging": build_adamerging_encoder(args, clean_paths, args.adamerging_lambda),
    }

    print("[诊断] 构造 KDR attack merged models")
    attack_models = {
        "ta": build_ta_encoder(args, kdr_paths),
        "ties": build_ties_encoder(args, kdr_paths),
        "regmean": build_regmean_encoder(args, kdr_paths, cache_name="kdr_regmean_final"),
        "adamerging": build_adamerging_encoder(args, kdr_paths, args.adamerging_lambda),
    }

    results = OrderedDict()
    for method_name in ["ta", "ties", "regmean", "adamerging"]:
        print()
        print(f"[诊断] 计算 {method_name} 的残差放大")
        results[method_name] = collect_method_amplification(
            args=args,
            method_name=method_name,
            attack_encoder=attack_models[method_name],
            clean_encoder=clean_models[method_name],
            trigger_info=trigger_info,
        )

    comparison = add_group_comparison(results)
    output = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "question": "RegMean 是否稳定削弱 KDR 的触发条件残差放大，而不是擦除恶意残差本身？",
        "definition": {
            "A_i_l": "||r_trig_i_l|| / (||r_clean_i_l|| + eps)",
            "G_i_l": "A_i_(l+1) / (A_i_l + eps)",
        },
        "model": args.model,
        "target_task": args.target_task,
        "target_cls": args.target_cls,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "pool": args.pool,
        "trigger_path": trigger_info["path"],
        "adamerging_lambda": args.adamerging_lambda,
        "results": results,
        "successful_group_vs_regmean": comparison,
    }

    out_path = os.path.join(
        args.out_dir, f"{output['timestamp']}_kdr_residual_amplification.json"
    )
    save_json(output, out_path)
    print_amplification_table(results)
    print_growth_table(results)
    print_group_comparison(comparison)
    print()
    print(f"[诊断] 已写入：{out_path}")


if __name__ == "__main__":
    main()
