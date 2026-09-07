import argparse
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("./src")

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval_submerge import EXAM_DATASETS, get_state_dict, load_task_vectors, merge_vector
from src.heads import get_classification_head
from src.ties_merging_utils import vector_to_state_dict
from src.utils import corner_mask_generation


class EvalArgs:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose PA-SubMerge feature-shift failure modes."
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument(
        "--exam-datasets",
        default=",".join(EXAM_DATASETS),
        help="Comma-separated datasets used for merging.",
    )
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-path",
        default="./results/submerge/diagnose_pa_feature_shift.json",
    )
    return parser.parse_args()


def make_eval_args(args):
    evargs = EvalArgs()
    evargs.data_location = args.data_location
    evargs.batch_size = args.batch_size
    evargs.device = args.device
    evargs.model = args.model
    evargs.cache_dir = ""
    evargs.openclip_cachedir = "./open_clip"
    evargs.save = os.path.join(args.ckpt_dir, args.model)
    return evargs


def make_merge_args(args, attack_type):
    return SimpleNamespace(
        model=args.model,
        ckpt_dir=args.ckpt_dir,
        exam_datasets=args.exam_datasets,
        attack_type=attack_type,
        adversary_task=args.adversary_task,
        target_cls=args.target_cls,
        patch_size=args.patch_size,
        scaling_coef=args.scaling_coef,
        ties_reset_thresh=args.ties_reset_thresh,
        ties_merge_func=args.ties_merge_func,
        dare_drop_rate=0.9,
        dare_seed=2026,
    )


def trigger_path(args, source):
    root = os.path.join("./trigger", args.model)
    if source == "SubMergePA":
        return os.path.join(
            root,
            f"SubMergePA_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy",
        )
    if source == "BadMergingOn":
        return os.path.join(
            root,
            f"On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy",
        )
    raise ValueError(f"Unsupported trigger source: {source}")


def load_trigger_info(args, source):
    path = trigger_path(args, source)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    trigger = torch.from_numpy(np.load(path)).float()
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    return {
        "source": source,
        "path": path,
        "mask": torch.from_numpy(mask).float(),
        "applied_patch": torch.from_numpy(applied_patch).float(),
    }


def patch_images(images, trigger_info):
    mask = trigger_info["mask"].type(torch.FloatTensor)
    applied_patch = trigger_info["applied_patch"].type(torch.FloatTensor)
    return torch.mul(mask, applied_patch) + torch.mul(
        1 - mask.expand(images.shape).type(torch.FloatTensor),
        images.type(torch.FloatTensor),
    )


def load_encoder_from_state(pretrained_path, ptm_sd, vector_or_state, device):
    image_encoder = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if isinstance(vector_or_state, torch.Tensor):
        state = vector_to_state_dict(vector_or_state, ptm_sd, [])
    else:
        state = vector_or_state
    image_encoder.load_state_dict(state, strict=False)
    image_encoder.to(device)
    image_encoder.eval()
    return image_encoder


def load_pretrained_encoder(pretrained_path, device):
    image_encoder = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    image_encoder.to(device)
    image_encoder.eval()
    return image_encoder


def load_standalone_encoder(path, device):
    image_encoder = torch.load(path, map_location="cpu", weights_only=False)
    image_encoder.to(device)
    image_encoder.eval()
    return image_encoder


def prepare_merge_vectors(args, attack_type, datasets):
    merge_args = make_merge_args(args, attack_type)
    return load_task_vectors(merge_args, datasets)


def build_merged_encoder(args, attack_type, method, datasets, device):
    pretrained_path, ptm_sd, flat_ptm, tvs, paths = prepare_merge_vectors(
        args, attack_type, datasets
    )
    merge_args = make_merge_args(args, attack_type)
    merged = merge_vector(merge_args, method, flat_ptm, tvs)
    return load_encoder_from_state(pretrained_path, ptm_sd, merged, device), paths


def collect_metrics(
    name,
    model,
    reference_model,
    classification_head,
    loader,
    trigger_info,
    target_cls,
    device,
):
    head_weight = classification_head.weight.detach().to(device).float()
    head_prototypes = F.normalize(head_weight, dim=1)
    w_target = head_prototypes[target_cls]

    total = 0
    target_hits = 0
    target_margin_sum = 0.0
    target_margin_pos = 0
    feat_cos_sum = 0.0
    shift_cos_sum = 0.0
    shift_norm_sum = 0.0
    target_gain_sum = 0.0
    max_competing_gain_sum = 0.0
    margin_gain_sum = 0.0
    hist = Counter()

    model.eval()
    reference_model.eval()
    classification_head.eval()

    with torch.no_grad():
        for batch in loader:
            batch = maybe_dictionarize(batch)
            images = batch["images"]
            labels = batch["labels"]
            keep = labels != target_cls
            if keep.sum().item() == 0:
                continue

            patched = patch_images(images, trigger_info)[keep].to(device)
            labels = labels[keep].to(device)

            features = model(patched)
            ref_features = reference_model(patched)
            shift = features - ref_features
            logits = classification_head(features)

            target_logits = logits[:, target_cls]
            other_logits = logits.clone()
            other_logits[:, target_cls] = -float("inf")
            max_other_logits = other_logits.max(dim=1).values
            margins = target_logits - max_other_logits
            preds = logits.argmax(dim=1)

            shift_norm = torch.linalg.vector_norm(shift.float(), dim=1)
            feat_cos = F.cosine_similarity(features.float(), w_target.unsqueeze(0), dim=1)
            shift_cos = F.cosine_similarity(shift.float(), w_target.unsqueeze(0), dim=1)

            target_gain = shift.float() @ w_target.float()
            competing_gains = shift.float() @ head_prototypes.float().T
            competing_gains[:, target_cls] = -float("inf")
            max_competing_gain = competing_gains.max(dim=1).values
            margin_gain = target_gain - max_competing_gain

            n = int(labels.numel())
            total += n
            target_hits += int((preds == target_cls).sum().item())
            target_margin_sum += float(margins.sum().item())
            target_margin_pos += int((margins > 0).sum().item())
            feat_cos_sum += float(feat_cos.sum().item())
            shift_cos_sum += float(shift_cos.sum().item())
            shift_norm_sum += float(shift_norm.sum().item())
            target_gain_sum += float(target_gain.sum().item())
            max_competing_gain_sum += float(max_competing_gain.sum().item())
            margin_gain_sum += float(margin_gain.sum().item())
            hist.update(str(int(x)) for x in preds.detach().cpu().tolist())

    denom = max(total, 1)
    top_hist = OrderedDict(
        (k, v) for k, v in hist.most_common(20)
    )
    return {
        "name": name,
        "trigger_source": trigger_info["source"],
        "trigger_path": trigger_info["path"],
        "num_non_target_samples": total,
        "asr": 100.0 * target_hits / denom,
        "target_margin_mean": target_margin_sum / denom,
        "target_margin_positive_ratio": target_margin_pos / denom,
        "feat_cos_mean": feat_cos_sum / denom,
        "shift_cos_mean": shift_cos_sum / denom,
        "shift_norm_mean": shift_norm_sum / denom,
        "target_gain_mean": target_gain_sum / denom,
        "max_competing_gain_mean": max_competing_gain_sum / denom,
        "margin_gain_mean": margin_gain_sum / denom,
        "predicted_class_histogram": top_hist,
    }


def print_table(rows):
    cols = [
        "name",
        "asr",
        "target_margin_mean",
        "target_margin_positive_ratio",
        "feat_cos_mean",
        "shift_cos_mean",
        "shift_norm_mean",
        "target_gain_mean",
        "max_competing_gain_mean",
        "margin_gain_mean",
    ]
    widths = {
        "name": max(len("name"), max(len(r["name"]) for r in rows)),
    }
    for c in cols[1:]:
        widths[c] = max(len(c), 12)

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        values = [row["name"].ljust(widths["name"])]
        for c in cols[1:]:
            values.append(f"{row[c]:.6f}".rjust(widths[c]))
        print(" | ".join(values))


def main():
    args = parse_args()
    args.exam_datasets = [x.strip() for x in args.exam_datasets.split(",") if x.strip()]
    device = torch.device(args.device)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    eval_args = make_eval_args(args)
    pretrained_path = os.path.join(args.ckpt_dir, args.model, "zeroshot.pt")
    ptm_sd = get_state_dict(pretrained_path)
    base_encoder = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    preprocess = base_encoder.val_preprocess
    _, test_loader = get_dataset(
        args.target_task,
        "test",
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    classification_head = get_classification_head(eval_args, args.target_task).to(device)
    classification_head.eval()

    pa_trigger = load_trigger_info(args, "SubMergePA")
    bad_trigger = None
    bad_path = trigger_path(args, "BadMergingOn")
    if os.path.exists(bad_path):
        bad_trigger = load_trigger_info(args, "BadMergingOn")

    rows = []
    metadata = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model": args.model,
        "target_task": args.target_task,
        "target_cls": args.target_cls,
        "patch_size": args.patch_size,
        "exam_datasets": args.exam_datasets,
        "scaling_coef": args.scaling_coef,
        "ties_reset_thresh": args.ties_reset_thresh,
        "ties_merge_func": args.ties_merge_func,
        "pretrained_path": pretrained_path,
    }

    print("[diagnose] loading pretrained and PA standalone")
    pretrained = load_pretrained_encoder(pretrained_path, device)
    pa_path = os.path.join(
        args.ckpt_dir,
        args.model,
        f"{args.adversary_task}_SubMergePA_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}",
        "finetuned.pt",
    )
    pa_standalone = load_standalone_encoder(pa_path, device)
    rows.append(
        collect_metrics(
            "PA_standalone_vs_pretrained",
            pa_standalone,
            pretrained,
            classification_head,
            test_loader,
            pa_trigger,
            args.target_cls,
            device,
        )
    )
    del pa_standalone
    torch.cuda.empty_cache() if device.type == "cuda" else None

    for method in ("ta", "ties"):
        print(f"[diagnose] building clean {method.upper()} and PA {method.upper()} merged models")
        clean_model, clean_paths = build_merged_encoder(
            args, "Clean", method, args.exam_datasets, device
        )
        pa_model, pa_paths = build_merged_encoder(
            args, "SubMergePA", method, args.exam_datasets, device
        )
        rows.append(
            collect_metrics(
                f"PA_{method.upper()}_merged_vs_clean_{method.upper()}",
                pa_model,
                clean_model,
                classification_head,
                test_loader,
                pa_trigger,
                args.target_cls,
                device,
            )
        )
        rows.append(
            collect_metrics(
                f"Clean_{method.upper()}_with_PA_trigger_vs_pretrained",
                clean_model,
                pretrained,
                classification_head,
                test_loader,
                pa_trigger,
                args.target_cls,
                device,
            )
        )

        if bad_trigger is not None:
            print(f"[diagnose] building BadMergingOn {method.upper()} merged model")
            bad_model, bad_paths = build_merged_encoder(
                args, "BadMergingOn", method, args.exam_datasets, device
            )
            rows.append(
                collect_metrics(
                    f"BadMergingOn_{method.upper()}_merged_vs_clean_{method.upper()}",
                    bad_model,
                    clean_model,
                    classification_head,
                    test_loader,
                    bad_trigger,
                    args.target_cls,
                    device,
                )
            )
            rows.append(
                collect_metrics(
                    f"Clean_{method.upper()}_with_BadMergingOn_trigger_vs_pretrained",
                    clean_model,
                    pretrained,
                    classification_head,
                    test_loader,
                    bad_trigger,
                    args.target_cls,
                    device,
                )
            )
            del bad_model

        del clean_model, pa_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    print("\n========== PA Feature Shift Diagnosis ==========")
    print_table(rows)

    output = {
        "metadata": metadata,
        "results": rows,
    }
    with open(args.out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[diagnose] wrote {args.out_path}")


if __name__ == "__main__":
    main()
