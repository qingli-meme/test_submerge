#!/usr/bin/env python3
"""Audit BMR trigger dormancy on clean merged models."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(".")
sys.path.append("./src")

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

from eval import eval_single_dataset
from modeling import ImageClassifier
from regmean import RegMean
from ties_merging_utils import check_parameterNamesMatch, state_dict_to_vector, ties_merging, vector_to_state_dict
from utils import corner_mask_generation


DATASETS = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
DISPLAY = {"ta": "TA", "ties": "TIES", "regmean": "RegMean", "adamerging": "AdaMerging"}


def parse_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--target-tasks", default="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS")
    parser.add_argument("--methods", default="ta,ties,regmean,adamerging")
    parser.add_argument("--trigger-source", default="KDR_DTK")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20.0)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--regmean-num-train-batch", type=int, default=8)
    parser.add_argument("--adamerging-path", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--out-dir", default="./analysis/bmr_multitask/clean_trigger_dormancy")
    args = parser.parse_args()
    args.target_tasks = parse_list(args.target_tasks)
    args.methods = parse_list(args.methods)
    bad = set(args.methods) - set(DISPLAY)
    if bad:
        parser.error(f"unsupported methods: {sorted(bad)}")
    return args


def state_dict_from_checkpoint(path):
    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        state = obj.state_dict()
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise TypeError(f"Unsupported checkpoint: {path}")
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def eval_args(args):
    return SimpleNamespace(
        model=args.model,
        data_location=args.data_location,
        batch_size=args.batch_size,
        max_eval_batches=args.max_eval_batches,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save=os.path.join(args.ckpt_dir, args.model),
        openclip_cachedir="./open_clip",
        cache_dir="./cache",
    )


def trigger_info(args, task):
    path = Path("./trigger") / args.model / (
        f"{args.trigger_source}_{task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    trigger = torch.from_numpy(np.load(path))
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    return {
        "path": str(path),
        "backdoor_info": {
            "mask": torch.from_numpy(mask).float(),
            "applied_patch": torch.from_numpy(applied_patch).float(),
            "target_cls": args.target_cls,
        },
    }


def load_clean_task_vectors(args):
    root = Path(args.ckpt_dir) / args.model
    ptm_path = root / "zeroshot.pt"
    ptm_state = state_dict_from_checkpoint(ptm_path)
    ft_states = []
    for dataset in DATASETS:
        path = root / dataset / "finetuned.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        ft_states.append(state_dict_from_checkpoint(path))
    check_parameterNamesMatch(ft_states + [ptm_state])
    flat_ptm = state_dict_to_vector(ptm_state, [])
    flat_ft = torch.vstack([state_dict_to_vector(sd, []) for sd in ft_states])
    return ptm_path, ptm_state, flat_ptm, flat_ft - flat_ptm, ft_states


def build_vector_merge(args, method, ptm_state, flat_ptm, tvs):
    if method == "ta":
        merged = flat_ptm + args.scaling_coef * tvs.sum(dim=0)
    elif method == "ties":
        merged_tv = ties_merging(
            tvs,
            reset_thresh=args.ties_reset_thresh,
            merge_func=args.ties_merge_func,
        )
        merged = flat_ptm + args.scaling_coef * merged_tv
    else:
        raise ValueError(method)
    return vector_to_state_dict(merged, ptm_state, [])


def runtime_args(args):
    return SimpleNamespace(
        model=args.model,
        ckpt_dir=args.ckpt_dir,
        data_location=args.data_location,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save=os.path.join(args.ckpt_dir, args.model),
        openclip_cachedir="./open_clip",
        cache_dir="./cache",
    )


def build_regmean_state(args):
    rargs = runtime_args(args)
    rargs.dataset_list = DATASETS
    rargs.num_train_batch = args.regmean_num_train_batch
    merger = RegMean(rargs, logger=None)
    model_list = []
    gram_list = []
    for dataset in DATASETS:
        task_path = Path(args.ckpt_dir) / args.model / dataset / "finetuned.pt"
        encoder = torch.load(task_path, map_location="cuda:0")
        model = ImageClassifier(encoder, merger.class_head_dict[dataset])
        model.freeze_head()
        model = model.cuda().eval()
        model_list.append(model)
        with torch.no_grad():
            gram = merger.compute_gram(model, dataset)
        gram_list.append({k: v.detach().cpu() for k, v in gram.items()})
        model.cpu()
        torch.cuda.empty_cache()
    avg_params = merger.avg_merge(model_list, regmean_grams=gram_list)
    encoder = torch.load(Path(args.ckpt_dir) / args.model / "zeroshot.pt", map_location="cuda:0")
    wrapper = ImageClassifier(encoder, merger.class_head_dict[DATASETS[-1]])
    wrapper.freeze_head()
    wrapper = wrapper.cuda()
    merger.copy_params_to_model(avg_params, wrapper)
    return {k: v.detach().cpu().clone() for k, v in wrapper.image_encoder.state_dict().items()}


def build_adamerging_state(args, ptm_state, ft_states):
    path = args.adamerging_path or str(Path("./ada") / args.model / "Clean_Epoch_500.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing clean AdaMerging lambda: {path}")
    lambdas_raw = torch.load(path, map_location="cpu")
    lambdas = torch.clamp(lambdas_raw, min=0.0, max=1.0)
    if lambdas.shape[1] != len(DATASETS):
        raise RuntimeError(f"Unexpected AdaMerging lambda shape: {tuple(lambdas.shape)}")
    merged = {}
    keys = list(ptm_state.keys())
    if lambdas.shape[0] != len(keys):
        raise RuntimeError(
            f"AdaMerging lambda rows {lambdas.shape[0]} != state keys {len(keys)}"
        )
    for row, key in enumerate(keys):
        base = ptm_state[key]
        if not torch.is_floating_point(base):
            merged[key] = base.clone()
            continue
        value = base.clone()
        for col, ft_state in enumerate(ft_states):
            value = value + lambdas[row, col].to(value.dtype) * (ft_state[key] - base)
        merged[key] = value
    return merged


def evaluate_state(args, ptm_path, state, task, trig):
    encoder = torch.load(ptm_path, map_location="cpu")
    encoder.load_state_dict(state, strict=False)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
    encoder.eval()
    metrics = eval_single_dataset(
        encoder,
        task,
        eval_args(args),
        backdoor_info=trig["backdoor_info"],
    )
    return {
        "patched_top1": 100.0 * metrics["top1"],
        "trigger_target_rate": 100.0 * metrics["backdoored_acc"],
        "target_hits": int(metrics["backdoored_cnt"]),
        "non_target_cnt": int(metrics["non_target_cnt"]),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ptm_path, ptm_state, flat_ptm, tvs, ft_states = load_clean_task_vectors(args)

    states = {}
    if "ta" in args.methods:
        states["ta"] = build_vector_merge(args, "ta", ptm_state, flat_ptm, tvs)
    if "ties" in args.methods:
        states["ties"] = build_vector_merge(args, "ties", ptm_state, flat_ptm, tvs)
    if "regmean" in args.methods:
        states["regmean"] = build_regmean_state(args)
    if "adamerging" in args.methods:
        states["adamerging"] = build_adamerging_state(args, ptm_state, ft_states)

    rows = []
    for task in args.target_tasks:
        trig = trigger_info(args, task)
        for method in args.methods:
            print(f"\n========== clean {DISPLAY[method]} / BMR trigger on {task} ==========")
            result = evaluate_state(args, ptm_path, states[method], task, trig)
            row = {
                "target_task": task,
                "method": method,
                "method_label": DISPLAY[method],
                "trigger_path": trig["path"],
                **result,
            }
            rows.append(row)
            print(json.dumps(row, indent=2))

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{ts}_clean_bmr_trigger_dormancy.csv"
    json_path = out_dir / f"{ts}_clean_bmr_trigger_dormancy.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)
    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
