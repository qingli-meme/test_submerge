import argparse
import json
import os
import sys
import time
from collections import OrderedDict
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
from src.eval_submerge import merge_vector
from src.heads import get_classification_head
from src.ties_merging_utils import check_parameterNamesMatch, state_dict_to_vector, vector_to_state_dict
from src.utils import corner_mask_generation


EXAM_DATASETS = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]


class EvalArgs:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "BadMerging stability diagnostics: trigger-key stability and "
            "target-subspace stability across merged contexts."
        )
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--exam-datasets", default=",".join(EXAM_DATASETS))
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--attack-type", default="BadMergingOn", choices=["BadMergingOn"])
    parser.add_argument("--trigger-source", default="BadMergingOn", choices=["BadMergingOn"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-non-target-samples", type=int, default=512)
    parser.add_argument("--max-target-samples", type=int, default=512)
    parser.add_argument("--layer-name", default="model.visual.transformer.resblocks.11.ln_2")
    parser.add_argument("--pool", default="cls", choices=["cls", "mean", "patch_mean", "flatten_mean"])
    parser.add_argument("--merge-methods", default="ta,ties")
    parser.add_argument("--context-family", default="bad,clean", help="Comma-separated: bad,clean")
    parser.add_argument("--include-leave-one-out", action="store_true")
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="./analysis/badmerging_stability")
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


def get_state_dict(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint object at {path}: {type(obj)}")


def checkpoint_path(args, dataset, family):
    root = os.path.join(args.ckpt_dir, args.model)
    if family == "bad" and dataset == args.adversary_task:
        tag = f"On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
        return os.path.join(root, f"{dataset}_{tag}", "finetuned.pt")
    return os.path.join(root, dataset, "finetuned.pt")


def trigger_path(args):
    tag = f"On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy"
    return os.path.join("./trigger", args.model, tag)


def load_trigger_info(args):
    path = trigger_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    trigger = torch.from_numpy(np.load(path)).float()
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    return {
        "path": path,
        "trigger_shape": list(trigger.shape),
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


def load_family_vectors(args, datasets, family):
    root = os.path.join(args.ckpt_dir, args.model)
    pretrained_path = os.path.join(root, "zeroshot.pt")
    ptm_sd = get_state_dict(pretrained_path)
    ft_sds = []
    paths = OrderedDict()
    for dataset in datasets:
        path = checkpoint_path(args, dataset, family)
        paths[dataset] = path
        ft_sds.append(get_state_dict(path))
    check_parameterNamesMatch(ft_sds + [ptm_sd])
    flat_ptm = state_dict_to_vector(ptm_sd, [])
    flat_ft = torch.vstack([state_dict_to_vector(sd, []) for sd in ft_sds])
    tvs = flat_ft - flat_ptm
    return pretrained_path, ptm_sd, flat_ptm, tvs, paths


def make_merge_args(args):
    return SimpleNamespace(
        scaling_coef=args.scaling_coef,
        ties_reset_thresh=args.ties_reset_thresh,
        ties_merge_func=args.ties_merge_func,
        dare_drop_rate=0.9,
        dare_seed=2026,
    )


def build_context_vectors(args, datasets):
    methods = [x.strip() for x in args.merge_methods.split(",") if x.strip()]
    families = [x.strip() for x in args.context_family.split(",") if x.strip()]
    merge_args = make_merge_args(args)
    contexts = []

    for family in families:
        pretrained_path, ptm_sd, flat_ptm, tvs, paths = load_family_vectors(args, datasets, family)
        for method in methods:
            merged = merge_vector(merge_args, method, flat_ptm, tvs)
            contexts.append({
                "name": f"{family}_{method}_all",
                "family": family,
                "method": method,
                "datasets": list(datasets),
                "paths": dict(paths),
                "pretrained_path": pretrained_path,
                "ptm_sd": ptm_sd,
                "merged_vector": merged,
            })
            if args.include_leave_one_out and method == "ta":
                for drop_idx, drop_ds in enumerate(datasets):
                    keep = [i for i in range(len(datasets)) if i != drop_idx]
                    merged_loo = flat_ptm + args.scaling_coef * tvs[keep].sum(dim=0)
                    contexts.append({
                        "name": f"{family}_ta_without_{drop_ds}",
                        "family": family,
                        "method": "ta",
                        "datasets": [datasets[i] for i in keep],
                        "paths": dict(paths),
                        "pretrained_path": pretrained_path,
                        "ptm_sd": ptm_sd,
                        "merged_vector": merged_loo,
                    })
    return contexts


def load_encoder_from_context(context, device):
    image_encoder = torch.load(context["pretrained_path"], map_location="cpu", weights_only=False)
    state = vector_to_state_dict(context["merged_vector"], context["ptm_sd"], [])
    image_encoder.load_state_dict(state, strict=False)
    image_encoder.to(device)
    image_encoder.eval()
    return image_encoder


def get_module(model, layer_name):
    modules = dict(model.named_modules())
    if layer_name not in modules:
        matches = [name for name in modules if layer_name in name]
        hint = matches[:20]
        raise KeyError(f"Layer not found: {layer_name}. Partial matches: {hint}")
    return modules[layer_name]


def pool_activation(act, pool):
    if act.dim() == 2:
        return act.float()
    if act.dim() == 3:
        if pool == "cls":
            return act[:, 0, :].float()
        if pool == "mean":
            return act.mean(dim=1).float()
        if pool == "patch_mean":
            return act[:, 1:, :].mean(dim=1).float()
        if pool == "flatten_mean":
            return act.reshape(act.shape[0], -1).float()
    return act.reshape(act.shape[0], -1).float()


def collect_fixed_batches(args, preprocess, trigger_info):
    _, loader = get_dataset(
        args.target_task,
        "test",
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    non_target_clean = []
    non_target_triggered = []
    target_clean = []
    non_target_count = 0
    target_count = 0

    for batch in loader:
        batch = maybe_dictionarize(batch)
        images = batch["images"]
        labels = batch["labels"]

        if non_target_count < args.max_non_target_samples:
            keep = labels != args.target_cls
            nt = images[keep]
            if nt.numel() > 0:
                remaining = args.max_non_target_samples - non_target_count
                nt = nt[:remaining]
                non_target_clean.append(nt.cpu())
                non_target_triggered.append(patch_images(nt, trigger_info).cpu())
                non_target_count += nt.shape[0]

        if target_count < args.max_target_samples:
            keep = labels == args.target_cls
            tgt = images[keep]
            if tgt.numel() > 0:
                remaining = args.max_target_samples - target_count
                tgt = tgt[:remaining]
                target_clean.append(tgt.cpu())
                target_count += tgt.shape[0]

        if (
            non_target_count >= args.max_non_target_samples
            and target_count >= args.max_target_samples
        ):
            break

    if not non_target_clean or not target_clean:
        raise RuntimeError(
            f"Insufficient samples: non_target={non_target_count}, target={target_count}"
        )

    return {
        "non_target_clean": torch.cat(non_target_clean, dim=0),
        "non_target_triggered": torch.cat(non_target_triggered, dim=0),
        "target_clean": torch.cat(target_clean, dim=0),
        "non_target_count": non_target_count,
        "target_count": target_count,
    }


def extract_layer_embeddings(image_encoder, layer_name, images, batch_size, device, pool):
    module = get_module(image_encoder, layer_name)
    captured = []

    def hook_fn(_module, _inp, out):
        captured.append(out.detach().cpu())

    handle = module.register_forward_hook(hook_fn)
    outputs = []
    try:
        with torch.no_grad():
            for start in range(0, images.shape[0], batch_size):
                batch = images[start:start + batch_size].to(device)
                captured.clear()
                _ = image_encoder(batch)
                if not captured:
                    raise RuntimeError(f"No activation captured for layer {layer_name}")
                outputs.append(pool_activation(captured[-1], pool).cpu())
    finally:
        handle.remove()
    return torch.cat(outputs, dim=0)


def pairwise_cosine_matrix(vectors):
    x = F.normalize(vectors.float(), dim=1, eps=1e-12)
    return x @ x.T


def offdiag_values(matrix):
    if matrix.shape[0] <= 1:
        return torch.empty(0)
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    return matrix[mask]


def summarize_values(values):
    if values.numel() == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "count": int(values.numel()),
    }


def pca_summary(vectors, max_components=10):
    x = vectors.float()
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        return {"explained_variance_ratio": [], "rank": int(x.shape[0])}
    _, s, _ = torch.linalg.svd(x, full_matrices=False)
    var = s ** 2
    ratio = var / var.sum().clamp_min(1e-12)
    return {
        "explained_variance_ratio": [float(v) for v in ratio[:max_components]],
        "cumulative_explained_variance_ratio": [
            float(v) for v in torch.cumsum(ratio, dim=0)[:max_components]
        ],
        "rank": int((s > 1e-8).sum().item()),
    }


def run_diagnostics(args):
    datasets = [x.strip() for x in args.exam_datasets.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    trigger_info = load_trigger_info(args)
    contexts = build_context_vectors(args, datasets)
    if not contexts:
        raise RuntimeError("No merge contexts were constructed")

    # Use the pretrained checkpoint preprocess as the canonical BadMerging eval preprocess.
    base_encoder = torch.load(contexts[0]["pretrained_path"], map_location="cpu", weights_only=False)
    preprocess = base_encoder.val_preprocess
    samples = collect_fixed_batches(args, preprocess, trigger_info)
    del base_encoder

    context_rows = []
    trigger_means = []
    target_prototypes = []

    activation_dir = os.path.join(args.out_dir, "activations")
    os.makedirs(activation_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for idx, context in enumerate(contexts):
        print(f"[{idx + 1}/{len(contexts)}] extracting {context['name']}", flush=True)
        image_encoder = load_encoder_from_context(context, args.device)

        trig_emb = extract_layer_embeddings(
            image_encoder,
            args.layer_name,
            samples["non_target_triggered"],
            args.batch_size,
            args.device,
            args.pool,
        )
        clean_emb = extract_layer_embeddings(
            image_encoder,
            args.layer_name,
            samples["non_target_clean"],
            args.batch_size,
            args.device,
            args.pool,
        )
        target_emb = extract_layer_embeddings(
            image_encoder,
            args.layer_name,
            samples["target_clean"],
            args.batch_size,
            args.device,
            args.pool,
        )

        trig_mean = F.normalize(trig_emb.mean(dim=0), dim=0, eps=1e-12)
        clean_mean = F.normalize(clean_emb.mean(dim=0), dim=0, eps=1e-12)
        target_mean = F.normalize(target_emb.mean(dim=0), dim=0, eps=1e-12)

        trigger_clean_per_sample = F.cosine_similarity(
            F.normalize(trig_emb.float(), dim=1, eps=1e-12),
            F.normalize(clean_emb.float(), dim=1, eps=1e-12),
            dim=1,
        )
        trigger_clean_mean_cos = F.cosine_similarity(
            trig_mean.unsqueeze(0),
            clean_mean.unsqueeze(0),
            dim=1,
        )[0]

        context_rows.append({
            "name": context["name"],
            "family": context["family"],
            "method": context["method"],
            "datasets": context["datasets"],
            "trigger_clean_per_sample_cos": summarize_values(trigger_clean_per_sample),
            "trigger_clean_mean_cos": float(trigger_clean_mean_cos.item()),
            "trigger_embedding_norm_mean": float(torch.linalg.vector_norm(trig_emb.float(), dim=1).mean().item()),
            "clean_embedding_norm_mean": float(torch.linalg.vector_norm(clean_emb.float(), dim=1).mean().item()),
            "target_embedding_norm_mean": float(torch.linalg.vector_norm(target_emb.float(), dim=1).mean().item()),
        })
        trigger_means.append(trig_mean.cpu())
        target_prototypes.append(target_mean.cpu())

        torch.save(
            {
                "context": context_rows[-1],
                "trigger_mean": trig_mean.cpu(),
                "clean_mean": clean_mean.cpu(),
                "target_prototype": target_mean.cpu(),
                "trigger_clean_per_sample_cos": trigger_clean_per_sample.cpu(),
            },
            os.path.join(activation_dir, f"{stamp}_{idx:02d}_{context['name']}.pt"),
        )

        del image_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trigger_means = torch.stack(trigger_means)
    target_prototypes = torch.stack(target_prototypes)
    trigger_pairwise = pairwise_cosine_matrix(trigger_means)
    target_pairwise = pairwise_cosine_matrix(target_prototypes)

    output = {
        "timestamp": stamp,
        "paper_module_mapping": {
            "diagnostic_A_trigger_key_stability": {
                "question": "Does the trigger induce a stable layer key across merged contexts?",
                "measurement": [
                    "intra-trigger cosine across merged contexts",
                    "trigger-clean separation within each context",
                ],
                "decision_rule": (
                    "High intra-trigger cosine supports merged-context key editing; "
                    "low intra-trigger cosine suggests merge drift breaks the trigger key."
                ),
            },
            "diagnostic_B_target_subspace_stability": {
                "question": "Does the target class form a stable low-dimensional layer subspace across merged contexts?",
                "measurement": [
                    "target prototype pairwise cosine across merged contexts",
                    "PCA explained variance of target prototypes",
                ],
                "decision_rule": (
                    "High pairwise cosine or low-dimensional PCA supports trigger-to-target-subspace editing; "
                    "unstable target prototypes suggest changing layer/readout."
                ),
            },
        },
        "config": {
            "model": args.model,
            "datasets": datasets,
            "adversary_task": args.adversary_task,
            "target_task": args.target_task,
            "target_cls": args.target_cls,
            "patch_size": args.patch_size,
            "trigger_path": trigger_info["path"],
            "layer_name": args.layer_name,
            "pool": args.pool,
            "merge_methods": args.merge_methods,
            "context_family": args.context_family,
            "include_leave_one_out": args.include_leave_one_out,
            "scaling_coef": args.scaling_coef,
            "ties_reset_thresh": args.ties_reset_thresh,
            "ties_merge_func": args.ties_merge_func,
            "max_non_target_samples": args.max_non_target_samples,
            "max_target_samples": args.max_target_samples,
        },
        "sanity_checks": {
            "num_contexts": len(context_rows),
            "non_target_samples": samples["non_target_count"],
            "target_samples": samples["target_count"],
            "embedding_dim": int(trigger_means.shape[1]),
            "all_activation_files_saved": True,
        },
        "contexts": context_rows,
        "diagnostic_A_trigger_key_stability": {
            "intra_trigger_pairwise_cosine_matrix": trigger_pairwise.tolist(),
            "intra_trigger_offdiag_summary": summarize_values(offdiag_values(trigger_pairwise)),
            "trigger_clean_mean_cos_summary": summarize_values(
                torch.tensor([x["trigger_clean_mean_cos"] for x in context_rows])
            ),
        },
        "diagnostic_B_target_subspace_stability": {
            "target_pairwise_cosine_matrix": target_pairwise.tolist(),
            "target_offdiag_summary": summarize_values(offdiag_values(target_pairwise)),
            "target_pca": pca_summary(target_prototypes),
        },
    }

    out_json = os.path.join(args.out_dir, f"{stamp}_badmerging_stability.json")
    out_pt = os.path.join(args.out_dir, f"{stamp}_badmerging_stability_tensors.pt")
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    torch.save(
        {
            "trigger_means": trigger_means,
            "target_prototypes": target_prototypes,
            "trigger_pairwise": trigger_pairwise,
            "target_pairwise": target_pairwise,
            "contexts": context_rows,
            "config": output["config"],
        },
        out_pt,
    )

    print("\n========== BadMerging Stability Diagnostics ==========")
    print("Layer:", args.layer_name)
    print("Pool:", args.pool)
    print("Contexts:", len(context_rows))
    print("\n[Diagnostic A] Trigger-key stability")
    print("intra-trigger offdiag:", output["diagnostic_A_trigger_key_stability"]["intra_trigger_offdiag_summary"])
    print("trigger-clean mean cos:", output["diagnostic_A_trigger_key_stability"]["trigger_clean_mean_cos_summary"])
    print("\n[Diagnostic B] Target-subspace stability")
    print("target offdiag:", output["diagnostic_B_target_subspace_stability"]["target_offdiag_summary"])
    print("target PCA:", output["diagnostic_B_target_subspace_stability"]["target_pca"])
    print("\nwrote:", out_json)
    print("wrote:", out_pt)
    print("activation sidecars:", activation_dir)


def main():
    args = parse_args()
    run_diagnostics(args)


if __name__ == "__main__":
    main()
