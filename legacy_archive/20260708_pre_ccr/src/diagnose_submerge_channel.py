import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict

import torch

sys.path.append(".")
sys.path.append("./src")

from ties_merging_utils import (
    disjoint_merge,
    resolve_sign,
    state_dict_to_vector,
    topk_values_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weight-space channel diagnosis for SubMerge vs BadMerging payloads."
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--exam-datasets", default="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS")
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--payload-top-frac", type=float, default=0.05)
    parser.add_argument("--nullspace-path", default="./nullspace/ViT-B-32/nullspace_e0.95_r128.pt")
    parser.add_argument("--out-dir", default="./analysis/submerge_channel")
    return parser.parse_args()


def torch_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def get_state_dict(path):
    obj = torch_load(path)
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object at {path}: {type(obj)}")
    return OrderedDict(obj)


def flatten(sd):
    return state_dict_to_vector(sd, []).detach().float().cpu()


def l2(x, eps=1e-12):
    return torch.linalg.vector_norm(x.float()).clamp_min(eps)


def cosine(a, b):
    return float(torch.dot(a.float(), b.float()) / (l2(a) * l2(b)))


def weighted_fraction(values, mask, coord_mask=None):
    if coord_mask is None:
        coord_mask = torch.ones_like(mask, dtype=torch.bool)
    weights = values.abs().float()
    denom = weights[coord_mask].sum().clamp_min(1e-12)
    return float(weights[coord_mask & mask].sum() / denom)


def top_abs_mask(x, frac):
    if frac <= 0 or frac >= 1:
        return torch.ones_like(x, dtype=torch.bool)
    k = max(1, int(x.numel() * frac))
    threshold = torch.topk(x.abs(), k=k, largest=True).values.min()
    return x.abs() >= threshold


def hard_ties(stack, reset_thresh, merge_func):
    updated, keep_fracs, trim_mask = topk_values_mask(
        stack.clone(), K=reset_thresh, return_mask=True
    )
    trim_mask = trim_mask.bool()
    signs = resolve_sign(updated)
    rows_keep = torch.where(signs.unsqueeze(0) > 0, updated > 0, updated < 0)
    selected = updated * rows_keep
    merged = disjoint_merge(updated, merge_func, signs)
    return {
        "updated": updated,
        "trim_mask": trim_mask,
        "keep_fracs": keep_fracs,
        "signs": signs,
        "rows_keep": rows_keep,
        "selected": selected,
        "merged": merged,
    }


def sorted_slices(sd):
    reference = OrderedDict(sorted(sd.items()))
    slices = {}
    start = 0
    for name, tensor in reference.items():
        n = tensor.numel()
        slices[name] = slice(start, start + n)
        start += n
    return slices


def checkpoint_paths(args, attack_type, dataset):
    root = os.path.join(args.ckpt_dir, args.model)
    if dataset != args.adversary_task or attack_type == "Clean":
        return os.path.join(root, dataset, "finetuned.pt")
    if attack_type in ("SubMerge", "SubMergeV2"):
        tag = f"{attack_type}_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    elif attack_type == "BadMergingOn":
        tag = f"On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    else:
        raise ValueError(attack_type)
    return os.path.join(root, f"{dataset}_{tag}", "finetuned.pt")


def load_stacks(args, datasets, attack_type):
    root = os.path.join(args.ckpt_dir, args.model)
    ptm_sd = get_state_dict(os.path.join(root, "zeroshot.pt"))
    flat_ptm = flatten(ptm_sd)
    clean_sds = []
    attack_sds = []
    paths = {}
    for dataset in datasets:
        clean_path = checkpoint_paths(args, "Clean", dataset)
        attack_path = checkpoint_paths(args, attack_type, dataset)
        paths[f"clean_{dataset}"] = clean_path
        paths[f"{attack_type}_{dataset}"] = attack_path
        clean_sds.append(get_state_dict(clean_path))
        attack_sds.append(get_state_dict(attack_path))
    clean_stack = torch.vstack([flatten(sd) - flat_ptm for sd in clean_sds])
    attack_stack = torch.vstack([flatten(sd) - flat_ptm for sd in attack_sds])
    return ptm_sd, clean_stack, attack_stack, paths


def load_nullspace(path):
    if not path or not os.path.exists(path):
        return None
    return torch_load(path).get("nullspace_info")


def project_tensor_to_nullspace(tensor, basis_info):
    if basis_info is None:
        return tensor
    btype = basis_info["type"]
    if btype == "1d":
        q = basis_info["Q_basis"].to(tensor.device, dtype=tensor.dtype)
        return tensor - q @ (q.T @ tensor)
    if btype in ("2d", "4d_as_2d"):
        original_shape = tensor.shape
        matrix = tensor.reshape(original_shape[0], -1) if btype == "4d_as_2d" else tensor
        result = matrix
        q_left = basis_info.get("Q_left")
        q_right = basis_info.get("Q_right")
        if q_left is not None:
            q_left = q_left.to(tensor.device, dtype=tensor.dtype)
            result = result - q_left @ (q_left.T @ result)
        if q_right is not None:
            q_right = q_right.to(tensor.device, dtype=tensor.dtype)
            result = result - result @ q_right @ q_right.T
        return result.reshape(original_shape) if btype == "4d_as_2d" else result
    return tensor


def nullspace_clean_subspace_overlap(tensor, basis_info):
    if basis_info is None:
        return None
    projected = project_tensor_to_nullspace(tensor, basis_info)
    clean_component = tensor - projected
    return float(l2(clean_component) / l2(tensor))


def summarize_attack(args, datasets, attack_type, nullspace_info):
    ptm_sd, clean_stack, attack_stack, paths = load_stacks(args, datasets, attack_type)
    adv_idx = datasets.index(args.adversary_task)
    clean_adv = clean_stack[adv_idx]
    attack_adv = attack_stack[adv_idx]
    payload = attack_adv - clean_adv
    payload_top = top_abs_mask(payload, args.payload_top_frac)

    clean_ties = hard_ties(clean_stack, args.ties_reset_thresh, args.ties_merge_func)
    attack_ties = hard_ties(attack_stack, args.ties_reset_thresh, args.ties_merge_func)

    ta_transmitted = args.scaling_coef * (attack_stack.sum(dim=0) - clean_stack.sum(dim=0))
    ties_transmitted = args.scaling_coef * (attack_ties["merged"] - clean_ties["merged"])

    adv_trim = attack_ties["trim_mask"][adv_idx]
    adv_keep = attack_ties["rows_keep"][adv_idx]
    adv_selected = adv_trim & adv_keep
    sign_flip = attack_ties["signs"] != clean_ties["signs"]

    clean_sum = clean_stack.sum(dim=0)
    clean_cosines = {
        dataset: cosine(payload, clean_stack[idx])
        for idx, dataset in enumerate(datasets)
    }

    def vector_metrics(prefix, transmitted, coord_mask):
        p = payload[coord_mask]
        t = transmitted[coord_mask]
        return {
            f"{prefix}_payload_norm": float(l2(p)),
            f"{prefix}_transmitted_norm": float(l2(t)),
            f"{prefix}_gain": float(l2(t) / l2(p)),
            f"{prefix}_cosine": cosine(p, t),
            f"{prefix}_coords": int(coord_mask.sum()),
        }

    summary = {
        "attack_type": attack_type,
        "paths": paths,
        "payload_norm": float(l2(payload)),
        "clean_adv_task_norm": float(l2(clean_adv)),
        "attack_adv_task_norm": float(l2(attack_adv)),
        "payload_to_clean_adv_norm_ratio": float(l2(payload) / l2(clean_adv)),
        "payload_to_attack_adv_norm_ratio": float(l2(payload) / l2(attack_adv)),
        "payload_cos_clean_adv": cosine(payload, clean_adv),
        "payload_cos_attack_adv": cosine(payload, attack_adv),
        "payload_cos_clean_sum": cosine(payload, clean_sum),
        "payload_max_abs_cos_any_clean_task": max(abs(v) for v in clean_cosines.values()),
        "payload_cos_each_clean_task": clean_cosines,
        "payload_top_frac": args.payload_top_frac,
        "payload_top_coords": int(payload_top.sum()),
        "ties_trim_retention_all": weighted_fraction(payload, adv_trim),
        "ties_selected_retention_all": weighted_fraction(payload, adv_selected),
        "ties_trim_retention_top": weighted_fraction(payload, adv_trim, payload_top),
        "ties_selected_retention_top": weighted_fraction(payload, adv_selected, payload_top),
        "ties_sign_conflict_given_trim_top": weighted_fraction(
            payload, adv_trim & (~adv_keep), payload_top & adv_trim
        ),
        "ties_elected_sign_flip_frac_all": float(sign_flip.float().mean()),
        "ties_elected_sign_flip_frac_top": float(sign_flip[payload_top].float().mean()),
    }
    summary.update(vector_metrics("ta_all", ta_transmitted, torch.ones_like(payload, dtype=torch.bool)))
    summary.update(vector_metrics("ta_top", ta_transmitted, payload_top))
    summary.update(vector_metrics("ties_all", ties_transmitted, torch.ones_like(payload, dtype=torch.bool)))
    summary.update(vector_metrics("ties_top", ties_transmitted, payload_top))

    layer_rows = []
    slices = sorted_slices(ptm_sd)
    for name, sl in slices.items():
        p = payload[sl]
        local_top = payload_top[sl]
        if int(local_top.sum()) == 0:
            local_top = torch.ones_like(p, dtype=torch.bool)
        row = {
            "attack_type": attack_type,
            "param_name": name,
            "numel": int(p.numel()),
            "payload_norm_all": float(l2(p)),
            "payload_norm_top": float(l2(p[local_top])),
            "ta_gain_top": float(l2(ta_transmitted[sl][local_top]) / l2(p[local_top])),
            "ties_gain_top": float(l2(ties_transmitted[sl][local_top]) / l2(p[local_top])),
            "ties_trim_retention_top": weighted_fraction(p, adv_trim[sl], local_top),
            "ties_selected_retention_top": weighted_fraction(p, adv_selected[sl], local_top),
            "ties_sign_flip_frac_top": float(sign_flip[sl][local_top].float().mean()),
        }
        if nullspace_info is not None and name in nullspace_info:
            tensor = p.reshape(ptm_sd[name].shape)
            row["clean_subspace_overlap_ratio"] = nullspace_clean_subspace_overlap(
                tensor, nullspace_info[name]
            )
        layer_rows.append(row)

    layer_rows.sort(key=lambda x: x["payload_norm_top"], reverse=True)
    return summary, layer_rows


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
    datasets = [x.strip() for x in args.exam_datasets.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    nullspace_info = load_nullspace(args.nullspace_path)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    summaries = []
    all_layers = []
    for attack_type in ["SubMergeV2", "BadMergingOn"]:
        summary, layers = summarize_attack(args, datasets, attack_type, nullspace_info)
        summaries.append(summary)
        all_layers.extend(layers)

    out = {
        "timestamp": stamp,
        "model": args.model,
        "datasets": datasets,
        "adversary_task": args.adversary_task,
        "target_cls": args.target_cls,
        "patch_size": args.patch_size,
        "scaling_coef": args.scaling_coef,
        "ties_reset_thresh": args.ties_reset_thresh,
        "ties_merge_func": args.ties_merge_func,
        "summaries": summaries,
    }
    summary_path = os.path.join(args.out_dir, f"{stamp}_summary.json")
    layer_path = os.path.join(args.out_dir, f"{stamp}_layers.csv")
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    write_csv(layer_path, all_layers)

    print("\n========== Weight-Space Diagnosis ==========")
    for s in summaries:
        print(f"\n[{s['attack_type']}]")
        keys = [
            "payload_norm",
            "payload_to_clean_adv_norm_ratio",
            "payload_cos_clean_sum",
            "payload_max_abs_cos_any_clean_task",
            "ta_top_gain",
            "ta_top_cosine",
            "ties_top_gain",
            "ties_top_cosine",
            "ties_trim_retention_top",
            "ties_selected_retention_top",
            "ties_sign_conflict_given_trim_top",
        ]
        for key in keys:
            print(f"{key}: {s[key]}")

    print("\nTop-10 payload layers per attack:")
    for attack_type in ["SubMergeV2", "BadMergingOn"]:
        rows = [r for r in all_layers if r["attack_type"] == attack_type]
        print(f"\n[{attack_type}]")
        for r in rows[:10]:
            print(
                f"{r['param_name']} | payload={r['payload_norm_top']:.6f} | "
                f"TA_gain={r['ta_gain_top']:.4f} | TIES_gain={r['ties_gain_top']:.4f} | "
                f"sel={r['ties_selected_retention_top']:.4f}"
            )

    print("\nwrote:", summary_path)
    print("wrote:", layer_path)


if __name__ == "__main__":
    main()
