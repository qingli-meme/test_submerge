import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict

import torch

sys.path.append(".")
sys.path.append("./src")

from heads import get_classification_head
from modeling import ImageClassifier
from regmean import RegMean


_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat


EXAM_DATASETS = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DTK and BadMerging-On residual transport under RegMean."
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--num-train-batch", type=int, default=8)
    parser.add_argument("--cache-dir", default="./analysis/regmean_residual_transport/cache")
    parser.add_argument("--out-dir", default="./analysis/regmean_residual_transport")
    return parser.parse_args()


def make_regmean_args(args):
    class Args:
        pass

    evargs = Args()
    evargs.model = args.model
    evargs.ckpt_dir = args.ckpt_dir
    evargs.save = os.path.join(args.ckpt_dir, args.model)
    evargs.data_location = args.data_location
    evargs.batch_size = 128
    evargs.num_train_batch = args.num_train_batch
    evargs.dataset_list = EXAM_DATASETS
    evargs.openclip_cachedir = "./open_clip"
    evargs.cache_dir = "./cache"
    evargs.device = "cuda" if torch.cuda.is_available() else "cpu"
    return evargs


def method_spec(args, method):
    if method == "DTK":
        attack_type = "KDR_DTK"
        suffix = f"KDR_DTK_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    elif method == "BadMergingOn":
        attack_type = "On"
        suffix = f"On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    else:
        raise ValueError(method)
    return attack_type, suffix


def checkpoint_path(args, dataset, method=None):
    root = os.path.join(args.ckpt_dir, args.model)
    if method is not None and dataset == args.adversary_task:
        _, suffix = method_spec(args, method)
        return os.path.join(root, f"{dataset}_{suffix}", "finetuned.pt")
    return os.path.join(root, dataset, "finetuned.pt")


def load_encoder(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)


def load_state(path):
    obj = torch.load(path, map_location="cpu")
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def build_regmean_state(args, method=None):
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_name = "clean" if method is None else method.lower()
    cache_path = os.path.join(args.cache_dir, f"{cache_name}_regmean_state.pt")

    legacy_clean_cache = "./analysis/kdr_regmean_readout/cache/clean_regmean.pt"
    if method is None and os.path.exists(legacy_clean_cache):
        print(f"[cache] using legacy clean RegMean state: {legacy_clean_cache}")
        return torch.load(legacy_clean_cache, map_location="cpu")

    if os.path.exists(cache_path):
        print(f"[cache] loaded RegMean state: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    evargs = make_regmean_args(args)
    regmean = RegMean(evargs, None)
    model_list = []
    gram_list = []

    with torch.no_grad():
        for dataset in EXAM_DATASETS:
            path = checkpoint_path(args, dataset, method=method)
            print(f"[RegMean] method={method or 'Clean'} dataset={dataset} path={path}")
            image_encoder = load_encoder(path, device)
            classification_head = regmean.class_head_dict[dataset]
            model = ImageClassifier(image_encoder, classification_head)
            model.freeze_head()
            model = model.to(device)
            model.eval()
            model_list.append(model)
            gram_list.append(regmean.compute_gram(model, dataset))

        avg_params = regmean.avg_merge(model_list, regmean_grams=gram_list)

        base_encoder = load_encoder(os.path.join(args.ckpt_dir, args.model, "zeroshot.pt"), device)
        model = ImageClassifier(base_encoder, regmean.class_head_dict[EXAM_DATASETS[0]])
        model.freeze_head()
        model = model.to(device)
        regmean.copy_params_to_model(avg_params, model)
        state = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in model.image_encoder.state_dict().items()
        )

    torch.save(state, cache_path)
    print(f"[cache] saved RegMean state: {cache_path}")
    return state


def block_id(name):
    match = re.search(r"resblocks\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def layer_group(name):
    b = block_id(name)
    if b is not None:
        return f"block_{b:02d}"
    if "visual.proj" in name or "ln_post" in name:
        return "visual_head"
    if "conv1" in name or "class_embedding" in name or "positional_embedding" in name or "ln_pre" in name:
        return "visual_stem"
    return "other"


def metric_row(scope, method, local_tensors, rm_tensors):
    local = torch.cat([x.reshape(-1).float() for x in local_tensors])
    rm = torch.cat([x.reshape(-1).float() for x in rm_tensors])
    local_norm = torch.linalg.vector_norm(local).item()
    rm_norm = torch.linalg.vector_norm(rm).item()
    denom = max(local_norm * rm_norm, 1e-12)
    cosine = torch.dot(local, rm).item() / denom
    return {
        "scope": scope,
        "method": method,
        "local_norm": local_norm,
        "rm_norm": rm_norm,
        "rho_rm_over_local": rm_norm / max(local_norm, 1e-12),
        "local_over_rm": local_norm / max(rm_norm, 1e-12),
        "cosine": cosine,
        "numel": int(local.numel()),
    }


def compute_transport(args, method, clean_local, attack_local, clean_rm, attack_rm):
    rows = []
    by_group = {}

    for name, clean_value in clean_local.items():
        if name not in attack_local or name not in clean_rm or name not in attack_rm:
            continue
        if not torch.is_tensor(clean_value) or clean_value.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            continue

        local_delta = attack_local[name].detach().cpu().float() - clean_value.detach().cpu().float()
        rm_delta = attack_rm[name].detach().cpu().float() - clean_rm[name].detach().cpu().float()
        if local_delta.numel() == 0:
            continue

        rows.append(metric_row(name, method, [local_delta], [rm_delta]))
        group = layer_group(name)
        by_group.setdefault(group, [[], []])
        by_group[group][0].append(local_delta)
        by_group[group][1].append(rm_delta)

    group_rows = [
        metric_row(group, method, tensors[0], tensors[1])
        for group, tensors in sorted(by_group.items())
    ]
    return rows, group_rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    clean_local = load_state(checkpoint_path(args, args.adversary_task, method=None))
    clean_rm = build_regmean_state(args, method=None)

    all_param_rows = []
    all_group_rows = []
    summary = {"methods": {}, "definition": "rho_rm_over_local = ||DeltaW_RM|| / ||DeltaW_local||"}

    for method in ["DTK", "BadMergingOn"]:
        print(f"\n[transport] method={method}")
        attack_local = load_state(checkpoint_path(args, args.adversary_task, method=method))
        attack_rm = build_regmean_state(args, method=method)
        param_rows, group_rows = compute_transport(
            args, method, clean_local, attack_local, clean_rm, attack_rm
        )
        all_param_rows.extend(param_rows)
        all_group_rows.extend(group_rows)
        summary["methods"][method] = {
            "all": metric_row(
                "all",
                method,
                [attack_local[k].detach().cpu().float() - clean_local[k].detach().cpu().float()
                 for k in clean_local if k in attack_local and torch.is_tensor(clean_local[k]) and clean_local[k].is_floating_point()],
                [attack_rm[k].detach().cpu().float() - clean_rm[k].detach().cpu().float()
                 for k in clean_rm if k in attack_rm and k in clean_local and torch.is_tensor(clean_rm[k]) and clean_rm[k].is_floating_point()],
            ),
            "blocks_08_11": metric_row(
                "blocks_08_11",
                method,
                [attack_local[k].detach().cpu().float() - clean_local[k].detach().cpu().float()
                 for k in clean_local if k in attack_local and torch.is_tensor(clean_local[k]) and clean_local[k].is_floating_point() and block_id(k) in (8, 9, 10, 11)],
                [attack_rm[k].detach().cpu().float() - clean_rm[k].detach().cpu().float()
                 for k in clean_rm if k in attack_rm and k in clean_local and torch.is_tensor(clean_rm[k]) and clean_rm[k].is_floating_point() and block_id(k) in (8, 9, 10, 11)],
            ),
        }

    param_path = os.path.join(args.out_dir, "regmean_residual_transport_params.csv")
    group_path = os.path.join(args.out_dir, "regmean_residual_transport_groups.csv")
    summary_path = os.path.join(args.out_dir, "regmean_residual_transport_summary.json")
    write_csv(param_path, all_param_rows)
    write_csv(group_path, all_group_rows)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== RegMean Residual Transport ==========")
    for row in all_group_rows:
        if row["scope"].startswith("block_") and int(row["scope"].split("_")[1]) in (8, 9, 10, 11):
            print(
                f"{row['method']} {row['scope']} "
                f"rho={row['rho_rm_over_local']:.4f} "
                f"cos={row['cosine']:.4f} "
                f"local={row['local_norm']:.4f} rm={row['rm_norm']:.4f}"
            )
    print("[wrote]", summary_path)
    print("[wrote]", group_path)
    print("[wrote]", param_path)


if __name__ == "__main__":
    main()
