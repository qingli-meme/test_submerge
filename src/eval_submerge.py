import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append(".")
sys.path.append("./src")

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

from eval import eval_single_dataset
from ties_merging_utils import (
    check_parameterNamesMatch,
    state_dict_to_vector,
    ties_merging,
    vector_to_state_dict,
)
from utils import NormalizeInverse, corner_mask_generation


EXAM_DATASETS = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]


class EvalArgs:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="SubMerge evaluation with BadMerging-compatible metrics")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--exam-datasets", default="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS")
    parser.add_argument(
        "--attack-type",
        choices=[
            "SubMerge",
            "SubMergeV2",
            "SubMergePA",
            "SubMergePARelax07",
            "SubMergePARelax05",
            "SubMergePARelax00",
            "SubMergeBPA",
            "SubMergeSelect",
            "SubMergeSelectM8",
            "FOPA",
            "FOPARelax07",
            "FOPARelax05",
            "FOPARelax00",
            "KDR",
            "KDR_COV",
            "KDR_GC",
            "KDR_TCB",
            "KDR_TVD",
            "KDR_TDK",
            "KDR_TDK_BMR",
            "KDR_DTK",
            "KDR_DTK_BAB",
            "KDR_DTK_BIND",
            "KDR_DTK_BIND_EXACT",
            "KDR_DTK_SCB",
            "KDR_DTK_FSB", "KDR_DTK_CCR",
            "KDR_DTK_BMR",
            "KDR_DTK_BIND_PGAIN",
            "BadMergingOn",
            "Clean",
        ],
        default="SubMergeV2",
    )
    parser.add_argument(
        "--trigger-source",
        choices=[
            "attack",
            "SubMerge",
            "SubMergeV2",
            "SubMergePA",
            "SubMergePARelax07",
            "SubMergePARelax05",
            "SubMergePARelax00",
            "SubMergeBPA",
            "SubMergeSelect",
            "SubMergeSelectM8",
            "FOPA",
            "FOPARelax07",
            "FOPARelax05",
            "FOPARelax00",
            "KDR",
            "KDR_COV",
            "KDR_GC",
            "KDR_TCB",
            "KDR_TVD",
            "KDR_TDK",
            "KDR_TDK_BMR",
            "KDR_DTK",
            "KDR_DTK_BAB",
            "KDR_DTK_BIND",
            "KDR_DTK_BIND_EXACT",
            "KDR_DTK_SCB",
            "KDR_DTK_FSB",
            "KDR_DTK_CCR",
            "KDR_DTK_BMR",
            "KDR_DTK_BIND_PGAIN",
            "BadMergingOn",
            "fixed",
        ],
        default="attack",
        help="Trigger used for ASR. Use Clean + SubMergeV2/BadMergingOn for clean merged baselines.",
    )
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--merge-methods", default="ta,ties", help="Comma-separated: ta,ties,dare-ties")
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--dare-drop-rate", type=float, default=0.9)
    parser.add_argument("--dare-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--test-utility", action="store_true")
    parser.add_argument("--no-effectiveness", action="store_true")
    parser.add_argument("--out-dir", default="./results/submerge")
    return parser.parse_args()


def get_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint object: {path}, {type(obj)}")


def make_eval_args(args):
    evargs = EvalArgs()
    evargs.data_location = args.data_location
    evargs.batch_size = args.batch_size
    evargs.max_eval_batches = args.max_eval_batches
    evargs.device = "cuda" if torch.cuda.is_available() else "cpu"
    evargs.model = args.model
    evargs.cache_dir = ""
    evargs.openclip_cachedir = "./open_clip"
    evargs.save = os.path.join(args.ckpt_dir, args.model)
    return evargs


def checkpoint_path(args, dataset):
    root = os.path.join(args.ckpt_dir, args.model)
    if args.attack_type in (
        "SubMerge",
        "SubMergeV2",
        "SubMergePA",
        "SubMergePARelax07",
        "SubMergePARelax05",
        "SubMergePARelax00",
        "SubMergeBPA",
        "SubMergeSelect",
        "SubMergeSelectM8",
        "FOPA",
        "FOPARelax07",
        "FOPARelax05",
        "FOPARelax00",
        "KDR",
        "KDR_COV",
        "KDR_GC",
        "KDR_TCB",
        "KDR_TVD",
        "KDR_TDK",
        "KDR_TDK_BMR",
        "KDR_DTK",
        "KDR_DTK_BAB",
        "KDR_DTK_BIND",
        "KDR_DTK_BIND_EXACT",
        "KDR_DTK_SCB",
        "KDR_DTK_FSB",
        "KDR_DTK_CCR",
        "KDR_DTK_BMR",
        "KDR_DTK_BIND_PGAIN",
    ) and dataset == args.adversary_task:
        suffix = f"{args.attack_type}_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
        return os.path.join(root, f"{dataset}_{suffix}", "finetuned.pt")
    if args.attack_type == "BadMergingOn" and dataset == args.adversary_task:
        suffix = f"On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
        return os.path.join(root, f"{dataset}_{suffix}", "finetuned.pt")
    return os.path.join(root, dataset, "finetuned.pt")


def trigger_path(args):
    root = os.path.join("./trigger", args.model)
    source = args.attack_type if args.trigger_source == "attack" else args.trigger_source
    if source in ("KDR_GC", "KDR_TCB", "KDR_TVD"):
        source = "KDR"
    if source in ("KDR_TDK_BMR",):
        source = "KDR_TDK"
    if source in ("KDR_DTK_BIND", "KDR_DTK_BIND_EXACT", "KDR_DTK_SCB", "KDR_DTK_FSB", "KDR_DTK_CCR", "KDR_DTK_BMR", "KDR_DTK_BIND_PGAIN"):
        source = "KDR_DTK"
    if source in (
        "SubMerge",
        "SubMergeV2",
        "SubMergePA",
        "SubMergePARelax07",
        "SubMergePARelax05",
        "SubMergePARelax00",
        "SubMergeBPA",
        "SubMergeSelect",
        "SubMergeSelectM8",
        "FOPA",
        "FOPARelax07",
        "FOPARelax05",
        "FOPARelax00",
        "KDR",
        "KDR_COV",
        "KDR_TDK",
        "KDR_TDK_BMR",
        "KDR_DTK",
        "KDR_DTK_BAB",
        "KDR_DTK_BIND",
        "KDR_DTK_BIND_EXACT",
        "KDR_DTK_SCB",
        "KDR_DTK_FSB",
        "KDR_DTK_CCR",
        "KDR_DTK_BMR",
        "KDR_DTK_BIND_PGAIN",
    ):
        return os.path.join(
            root,
            f"{source}_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy",
        )
    if source == "BadMergingOn":
        return os.path.join(
            root,
            f"On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy",
        )
    return os.path.join(root, f"fixed_{args.patch_size}.npy")


def load_trigger_info(args):
    path = trigger_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing trigger: {path}")
    trigger = torch.from_numpy(np.load(path))
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    return {
        "trigger_path": path,
        "trigger_shape": list(trigger.shape),
        "backdoor_info": {
            "mask": torch.from_numpy(mask).float(),
            "applied_patch": torch.from_numpy(applied_patch).float(),
            "target_cls": args.target_cls,
        },
    }


def load_task_vectors(args, datasets):
    root = os.path.join(args.ckpt_dir, args.model)
    pretrained_path = os.path.join(root, "zeroshot.pt")
    ptm_sd = get_state_dict(pretrained_path)
    ft_sds = []
    paths = {}
    for dataset in datasets:
        path = checkpoint_path(args, dataset)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        paths[dataset] = path
        ft_sds.append(get_state_dict(path))
    check_parameterNamesMatch(ft_sds + [ptm_sd])
    flat_ptm = state_dict_to_vector(ptm_sd, [])
    flat_ft = torch.vstack([state_dict_to_vector(sd, []) for sd in ft_sds])
    tvs = flat_ft - flat_ptm
    return pretrained_path, ptm_sd, flat_ptm, tvs, paths


def dare_task_vectors(tvs, drop_rate, seed):
    generator = torch.Generator(device=tvs.device)
    generator.manual_seed(seed)
    keep_prob = 1.0 - drop_rate
    if keep_prob <= 0:
        raise ValueError("DARE keep probability must be positive")
    mask = torch.rand(tvs.shape, generator=generator, device=tvs.device) < keep_prob
    return tvs * mask.float() / keep_prob


def merge_vector(args, method, flat_ptm, tvs):
    if method == "ta":
        return flat_ptm + args.scaling_coef * tvs.sum(dim=0)
    if method == "ties":
        merged_tv = ties_merging(
            tvs,
            reset_thresh=args.ties_reset_thresh,
            merge_func=args.ties_merge_func,
        )
        return flat_ptm + args.scaling_coef * merged_tv
    if method == "dare-ties":
        dared = dare_task_vectors(tvs, args.dare_drop_rate, args.dare_seed)
        merged_tv = ties_merging(
            dared,
            reset_thresh=args.ties_reset_thresh,
            merge_func=args.ties_merge_func,
        )
        return flat_ptm + args.scaling_coef * merged_tv
    raise ValueError(f"Unsupported merge method: {method}")


def evaluate_model(image_encoder, args, datasets, backdoor_info):
    evargs = make_eval_args(args)
    results = {"datasets": {}}
    accs = []
    if args.test_utility:
        for dataset in datasets:
            metrics = eval_single_dataset(image_encoder, dataset, evargs)
            acc = 100.0 * metrics["top1"]
            results["datasets"][dataset] = {"clean_acc": acc}
            accs.append(acc)
        results["avg_clean_acc"] = float(np.mean(accs)) if accs else None

    if not args.no_effectiveness:
        metrics = eval_single_dataset(
            image_encoder,
            args.target_task,
            evargs,
            backdoor_info=backdoor_info,
        )
        results["target_task"] = args.target_task
        results["asr"] = 100.0 * metrics["backdoored_acc"]
        results["patched_top1"] = 100.0 * metrics["top1"]
        results["backdoored_cnt"] = int(metrics["backdoored_cnt"])
        results["non_target_cnt"] = int(metrics["non_target_cnt"])
    return results


def main():
    args = parse_args()
    datasets = [x.strip() for x in args.exam_datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.merge_methods.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    trig = load_trigger_info(args)
    pretrained_path, ptm_sd, flat_ptm, tvs, ckpt_paths = load_task_vectors(args, datasets)
    base_encoder = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    output = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "attack_type": args.attack_type,
        "trigger_source": args.trigger_source,
        "model": args.model,
        "datasets": datasets,
        "checkpoint_paths": ckpt_paths,
        "trigger_path": trig["trigger_path"],
        "target_task": args.target_task,
        "target_cls": args.target_cls,
        "patch_size": args.patch_size,
        "scaling_coef": args.scaling_coef,
        "merge_results": {},
    }

    for method in methods:
        print(f"\n========== Evaluating {args.attack_type} / {method} ==========")
        merged = merge_vector(args, method, flat_ptm, tvs)
        state = vector_to_state_dict(merged, ptm_sd, [])
        image_encoder = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        image_encoder.load_state_dict(state, strict=False)
        if torch.cuda.is_available():
            image_encoder = image_encoder.cuda()
        image_encoder.eval()
        result = evaluate_model(image_encoder, args, datasets, trig["backdoor_info"])
        output["merge_results"][method] = result
        print(json.dumps(result, indent=2))

    out_name = (
        f"{output['timestamp']}_{args.attack_type}_{args.adversary_task}"
        f"_tgt{args.target_cls}_L{args.patch_size}.json"
    )
    out_path = os.path.join(args.out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
