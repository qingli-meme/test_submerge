import argparse
import json
import os
import sys
import time
from collections import Counter, OrderedDict, defaultdict
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
            "Diagnose whether triggered keys are close to target subspace and "
            "whether malicious residuals move representations toward that subspace."
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
    parser.add_argument("--current-attack-type", default="FOPATwoPhase")
    parser.add_argument(
        "--trigger-source",
        default="BadMergingOn",
        help="Trigger source to apply to all compared contexts, e.g. BadMergingOn or FOPATwoPhase.",
    )
    parser.add_argument("--layer-name", default="model.visual.transformer.resblocks.11.ln_2")
    parser.add_argument("--pool", default="cls", choices=["cls", "mean", "patch_mean", "flatten_mean"])
    parser.add_argument("--merge-methods", default="ta,ties")
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-trigger-samples", type=int, default=512)
    parser.add_argument("--class-samples-per-class", type=int, default=8)
    parser.add_argument("--subspace-rank", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="./analysis/badmerging_subspace_distance")
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


def attack_tag(args, attack_type):
    if attack_type == "BadMergingOn":
        return f"On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    return f"{attack_type}_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"


def checkpoint_path(args, dataset, variant):
    root = os.path.join(args.ckpt_dir, args.model)
    if variant == "clean" or dataset != args.adversary_task:
        return os.path.join(root, dataset, "finetuned.pt")
    tag = attack_tag(args, variant)
    return os.path.join(root, f"{dataset}_{tag}", "finetuned.pt")


def trigger_path(args, source):
    root = os.path.join("./trigger", args.model)
    if source == "BadMergingOn":
        name = f"On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy"
    else:
        name = f"{source}_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy"
    return os.path.join(root, name)


def load_trigger_info(args, source):
    path = trigger_path(args, source)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    trigger = torch.from_numpy(np.load(path)).float()
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    return {
        "source": source,
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


def load_vectors(args, datasets, variant):
    root = os.path.join(args.ckpt_dir, args.model)
    pretrained_path = os.path.join(root, "zeroshot.pt")
    ptm_sd = get_state_dict(pretrained_path)
    ft_sds = []
    paths = OrderedDict()
    for dataset in datasets:
        path = checkpoint_path(args, dataset, variant)
        paths[dataset] = path
        ft_sds.append(get_state_dict(path))
    check_parameterNamesMatch(ft_sds + [ptm_sd])
    flat_ptm = state_dict_to_vector(ptm_sd, [])
    flat_ft = torch.vstack([state_dict_to_vector(sd, []) for sd in ft_sds])
    return pretrained_path, ptm_sd, flat_ptm, flat_ft - flat_ptm, paths


def make_merge_args(args):
    return SimpleNamespace(
        scaling_coef=args.scaling_coef,
        ties_reset_thresh=args.ties_reset_thresh,
        ties_merge_func=args.ties_merge_func,
        dare_drop_rate=0.9,
        dare_seed=2026,
    )


def load_encoder(pretrained_path, ptm_sd, vector_or_state, device):
    image_encoder = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    state = vector_to_state_dict(vector_or_state, ptm_sd, []) if isinstance(vector_or_state, torch.Tensor) else vector_or_state
    image_encoder.load_state_dict(state, strict=False)
    image_encoder.to(device)
    image_encoder.eval()
    return image_encoder


def build_model_contexts(args, datasets):
    methods = [x.strip() for x in args.merge_methods.split(",") if x.strip()]
    variants = ["clean", "BadMergingOn"]
    if args.current_attack_type:
        variants.append(args.current_attack_type)

    vector_cache = {}
    contexts = OrderedDict()
    merge_args = make_merge_args(args)

    for variant in variants:
        try:
            vector_cache[variant] = load_vectors(args, datasets, variant)
        except FileNotFoundError as exc:
            print(f"[skip] missing {variant}: {exc}")
            continue

    for variant, (pretrained_path, ptm_sd, flat_ptm, tvs, paths) in vector_cache.items():
        for method in methods:
            merged = merge_vector(merge_args, method, flat_ptm, tvs)
            contexts[f"{variant}_{method}"] = {
                "name": f"{variant}_{method}",
                "variant": variant,
                "method": method,
                "kind": "merged",
                "pretrained_path": pretrained_path,
                "ptm_sd": ptm_sd,
                "vector_or_state": merged,
                "paths": dict(paths),
            }

    root = os.path.join(args.ckpt_dir, args.model)
    pretrained_path = os.path.join(root, "zeroshot.pt")
    local_variants = ["clean", "BadMergingOn"]
    if args.current_attack_type:
        local_variants.append(args.current_attack_type)
    for variant in local_variants:
        path = checkpoint_path(args, args.adversary_task, variant)
        if not os.path.exists(path):
            print(f"[skip] missing local {variant}: {path}")
            continue
        contexts[f"{variant}_local"] = {
            "name": f"{variant}_local",
            "variant": variant,
            "method": "local",
            "kind": "local",
            "pretrained_path": pretrained_path,
            "ptm_sd": None,
            "vector_or_state": get_state_dict(path),
            "paths": {args.adversary_task: path},
        }
    return contexts


def get_module(model, layer_name):
    modules = dict(model.named_modules())
    if layer_name not in modules:
        matches = [name for name in modules if layer_name in name]
        raise KeyError(f"Layer not found: {layer_name}. Partial matches: {matches[:20]}")
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


def extract_embeddings(image_encoder, layer_name, images, batch_size, device, pool):
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
                    raise RuntimeError(f"No activation captured for {layer_name}")
                outputs.append(pool_activation(captured[-1], pool).cpu())
    finally:
        handle.remove()
    return torch.cat(outputs, dim=0)


def collect_samples(args, preprocess, trigger_info):
    _, loader = get_dataset(
        args.target_task,
        "test",
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    per_class = defaultdict(list)
    triggered = []
    triggered_labels = []
    trig_count = 0

    for batch in loader:
        batch = maybe_dictionarize(batch)
        images = batch["images"]
        labels = batch["labels"]

        for c in labels.unique().tolist():
            c = int(c)
            if len(per_class[c]) >= args.class_samples_per_class:
                continue
            imgs = images[labels == c]
            need = args.class_samples_per_class - len(per_class[c])
            for img in imgs[:need]:
                per_class[c].append(img.cpu())

        if trig_count < args.max_trigger_samples:
            keep = labels != args.target_cls
            imgs = images[keep]
            labs = labels[keep]
            if imgs.numel() > 0:
                need = args.max_trigger_samples - trig_count
                imgs = imgs[:need]
                labs = labs[:need]
                triggered.append(patch_images(imgs, trigger_info).cpu())
                triggered_labels.append(labs.cpu())
                trig_count += imgs.shape[0]

        if trig_count >= args.max_trigger_samples and all(
            len(v) >= args.class_samples_per_class for v in per_class.values()
        ) and len(per_class) >= 100:
            break

    class_images = OrderedDict()
    for c in sorted(per_class):
        if len(per_class[c]) > 0:
            class_images[c] = torch.stack(per_class[c], dim=0)
    if args.target_cls not in class_images:
        raise RuntimeError(f"No target samples collected for class {args.target_cls}")
    if not triggered:
        raise RuntimeError("No non-target triggered samples collected")
    return {
        "class_images": class_images,
        "triggered": torch.cat(triggered, dim=0),
        "triggered_labels": torch.cat(triggered_labels, dim=0),
    }


def make_subspace_basis(prototypes, rank):
    # prototypes: contexts x dim
    x = F.normalize(prototypes.float(), dim=1, eps=1e-12)
    if torch.linalg.vector_norm(x) < 1e-12:
        return F.normalize(prototypes.float().mean(dim=0, keepdim=True), dim=1, eps=1e-12).T
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    k = min(rank, vh.shape[0])
    return vh[:k].T.contiguous()


def project_norm(x, basis):
    # x: n x dim, basis: dim x rank with orthonormal columns from SVD
    proj = x.float() @ basis.float() @ basis.float().T
    return torch.linalg.vector_norm(proj, dim=1), proj


def summarize(values):
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "count": int(values.numel()),
    }


def hist_top(preds, topk=10):
    counter = Counter(int(x) for x in preds.detach().cpu().tolist())
    return [{"class": k, "count": v} for k, v in counter.most_common(topk)]


def evaluate_projection_metrics(name, z, logits, bases, target_cls, hard_mask=None):
    z = z.float()
    z_norm = torch.linalg.vector_norm(z, dim=1).clamp_min(1e-12)
    target_norm, _ = project_norm(z, bases[target_cls])
    scores = []
    for c, basis in bases.items():
        n, _ = project_norm(z, basis)
        scores.append(n / z_norm)
    score_matrix = torch.stack(scores, dim=1)
    class_ids = list(bases.keys())
    target_pos = class_ids.index(target_cls)
    score_t = score_matrix[:, target_pos]
    other_scores = score_matrix.clone()
    other_scores[:, target_pos] = -float("inf")
    score_non_target = other_scores.max(dim=1).values
    subspace_margin = score_t - score_non_target
    preds = logits.argmax(dim=1)
    asr = (preds == target_cls).float()

    def pack(mask, suffix):
        if mask is None:
            mask = torch.ones_like(asr, dtype=torch.bool)
        return {
            f"{suffix}_num_samples": int(mask.sum().item()),
            f"{suffix}_asr": float(asr[mask].mean().item()) if int(mask.sum()) else None,
            f"{suffix}_score_t": summarize(score_t[mask]),
            f"{suffix}_score_non_target": summarize(score_non_target[mask]),
            f"{suffix}_subspace_margin": summarize(subspace_margin[mask]),
            f"{suffix}_prediction_histogram": hist_top(preds[mask]),
        }

    out = {"name": name}
    out.update(pack(None, "all"))
    if hard_mask is not None:
        out.update(pack(hard_mask, "hard"))
    return out


def evaluate_shift_metrics(name, z_attack, z_clean, bases, target_cls, hard_mask):
    delta = z_attack.float() - z_clean.float()
    delta_norm = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
    target_delta_norm, target_delta_proj = project_norm(delta, bases[target_cls])
    clean_target_norm, _ = project_norm(z_clean.float(), bases[target_cls])
    attack_target_norm, _ = project_norm(z_attack.float(), bases[target_cls])
    target_gain = attack_target_norm - clean_target_norm
    target_shift_alignment = target_delta_norm / delta_norm

    def pack(mask, suffix):
        if mask is None:
            mask = torch.ones(delta.shape[0], dtype=torch.bool)
        return {
            f"{suffix}_num_samples": int(mask.sum().item()),
            f"{suffix}_delta_norm": summarize(delta_norm[mask]),
            f"{suffix}_target_shift_alignment": summarize(target_shift_alignment[mask]),
            f"{suffix}_target_projection_gain": summarize(target_gain[mask]),
        }

    out = {"name": name}
    out.update(pack(None, "all"))
    out.update(pack(hard_mask, "hard"))
    return out


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    datasets = [x.strip() for x in args.exam_datasets.split(",") if x.strip()]
    trigger_info = load_trigger_info(args, args.trigger_source)
    contexts = build_model_contexts(args, datasets)
    if "clean_ta" not in contexts:
        raise RuntimeError("clean_ta context is required")

    base = torch.load(contexts["clean_ta"]["pretrained_path"], map_location="cpu", weights_only=False)
    samples = collect_samples(args, base.val_preprocess, trigger_info)
    del base

    eval_args = make_eval_args(args)
    classification_head = get_classification_head(eval_args, args.target_task).to(args.device)
    classification_head.eval()

    embeddings = {}
    class_prototypes_by_context = defaultdict(dict)

    for i, (name, ctx) in enumerate(contexts.items(), 1):
        print(f"[{i}/{len(contexts)}] extracting {name}", flush=True)
        encoder = load_encoder(ctx["pretrained_path"], ctx.get("ptm_sd"), ctx["vector_or_state"], args.device)
        z = extract_embeddings(encoder, args.layer_name, samples["triggered"], args.batch_size, args.device, args.pool)
        with torch.no_grad():
            logits = []
            for start in range(0, samples["triggered"].shape[0], args.batch_size):
                images = samples["triggered"][start:start + args.batch_size].to(args.device)
                logits.append(classification_head(encoder(images)).detach().cpu())
            logits = torch.cat(logits, dim=0)
        embeddings[name] = {"triggered_z": z, "triggered_logits": logits}

        for cls, images in samples["class_images"].items():
            cls_z = extract_embeddings(encoder, args.layer_name, images, args.batch_size, args.device, args.pool)
            class_prototypes_by_context[name][cls] = F.normalize(cls_z.float().mean(dim=0), dim=0, eps=1e-12).cpu()

        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    classes = sorted(samples["class_images"].keys())
    bases = {}
    pca_rows = {}
    for cls in classes:
        proto_stack = torch.stack([
            class_prototypes_by_context[name][cls]
            for name in contexts
            if cls in class_prototypes_by_context[name]
        ])
        bases[cls] = make_subspace_basis(proto_stack, args.subspace_rank)
        centered = proto_stack.float() - proto_stack.float().mean(dim=0, keepdim=True)
        _, s, _ = torch.linalg.svd(centered, full_matrices=False)
        var = s ** 2
        ratio = var / var.sum().clamp_min(1e-12)
        pca_rows[cls] = {
            "num_contexts": int(proto_stack.shape[0]),
            "top_explained": [float(x) for x in ratio[: min(5, ratio.numel())]],
        }

    projection_metrics = []
    for name, emb in embeddings.items():
        if name.startswith("clean_"):
            hard_mask = emb["triggered_logits"].argmax(dim=1) != args.target_cls
        elif name.endswith("_ta") and "clean_ta" in embeddings:
            hard_mask = embeddings["clean_ta"]["triggered_logits"].argmax(dim=1) != args.target_cls
        elif name.endswith("_ties") and "clean_ties" in embeddings:
            hard_mask = embeddings["clean_ties"]["triggered_logits"].argmax(dim=1) != args.target_cls
        elif name.endswith("_local") and "clean_local" in embeddings:
            hard_mask = embeddings["clean_local"]["triggered_logits"].argmax(dim=1) != args.target_cls
        else:
            hard_mask = None
        projection_metrics.append(
            evaluate_projection_metrics(
                name,
                emb["triggered_z"],
                emb["triggered_logits"],
                bases,
                args.target_cls,
                hard_mask,
            )
        )

    shift_pairs = []
    for prefix in ["BadMergingOn", args.current_attack_type]:
        for method in ["ta", "ties"]:
            attack_name = f"{prefix}_{method}"
            clean_name = f"clean_{method}"
            if attack_name in embeddings and clean_name in embeddings:
                hard_mask = embeddings[clean_name]["triggered_logits"].argmax(dim=1) != args.target_cls
                shift_pairs.append((attack_name, clean_name, hard_mask))
        attack_name = f"{prefix}_local"
        clean_name = "clean_local"
        if attack_name in embeddings and clean_name in embeddings:
            hard_mask = embeddings[clean_name]["triggered_logits"].argmax(dim=1) != args.target_cls
            shift_pairs.append((attack_name, clean_name, hard_mask))

    shift_metrics = [
        evaluate_shift_metrics(
            f"{attack_name}_minus_{clean_name}",
            embeddings[attack_name]["triggered_z"],
            embeddings[clean_name]["triggered_z"],
            bases,
            args.target_cls,
            hard_mask,
        )
        for attack_name, clean_name, hard_mask in shift_pairs
    ]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": stamp,
        "paper_module_mapping": {
            "diagnostic_C_trigger_to_target_subspace_projection": {
                "design_basis": "K_tau and S_t can both be stable while still being far apart.",
                "metrics": [
                    "score_t = ||Proj_St(z_tau)|| / ||z_tau||",
                    "score_non_target = max_c!=t ||Proj_Sc(z_tau)|| / ||z_tau||",
                    "subspace_margin = score_t - score_non_target",
                ],
            },
            "diagnostic_D_residual_induced_representation_shift": {
                "design_basis": "A surviving weight residual must functionally move triggered representations toward S_t.",
                "metrics": [
                    "delta_z = h_attack(Tx) - h_clean(Tx)",
                    "target_shift_alignment = ||Proj_St(delta_z)|| / ||delta_z||",
                    "target_projection_gain = ||Proj_St(h_attack(Tx))|| - ||Proj_St(h_clean(Tx))||",
                ],
            },
            "diagnostic_E_hard_sample_subspace_margin": {
                "design_basis": "Hard samples isolate residual contribution from trigger prior.",
                "hard_set": "H = {x | clean_merge(T(x)) != target}",
            },
        },
        "config": {
            "model": args.model,
            "datasets": datasets,
            "target_task": args.target_task,
            "target_cls": args.target_cls,
            "patch_size": args.patch_size,
            "current_attack_type": args.current_attack_type,
            "trigger_source": args.trigger_source,
            "trigger_path": trigger_info["path"],
            "layer_name": args.layer_name,
            "pool": args.pool,
            "subspace_rank": args.subspace_rank,
            "class_samples_per_class": args.class_samples_per_class,
            "max_trigger_samples": args.max_trigger_samples,
            "contexts": list(contexts.keys()),
        },
        "sanity_checks": {
            "num_contexts": len(contexts),
            "num_classes_with_subspaces": len(bases),
            "triggered_samples": int(samples["triggered"].shape[0]),
            "target_class_subspace_rank": int(bases[args.target_cls].shape[1]),
            "embedding_dim": int(next(iter(embeddings.values()))["triggered_z"].shape[1]),
        },
        "target_class_pca": pca_rows.get(args.target_cls),
        "projection_metrics": projection_metrics,
        "shift_metrics": shift_metrics,
    }

    out_json = os.path.join(args.out_dir, f"{stamp}_subspace_distance.json")
    out_pt = os.path.join(args.out_dir, f"{stamp}_subspace_distance_tensors.pt")
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    torch.save(
        {
            "bases": bases,
            "embeddings": embeddings,
            "projection_metrics": projection_metrics,
            "shift_metrics": shift_metrics,
            "config": output["config"],
        },
        out_pt,
    )

    print("\n========== Subspace Distance Diagnosis ==========")
    print("Layer:", args.layer_name)
    print("Pool:", args.pool)
    print("Contexts:", list(contexts.keys()))
    print("\n[Projection metrics]")
    for row in projection_metrics:
        print(
            f"{row['name']}: all_margin={row['all_subspace_margin']['mean']:.6f}, "
            f"hard_margin={row.get('hard_subspace_margin', {}).get('mean')}, "
            f"all_asr={row['all_asr']}, hard_asr={row.get('hard_asr')}"
        )
    print("\n[Shift metrics]")
    for row in shift_metrics:
        print(
            f"{row['name']}: all_gain={row['all_target_projection_gain']['mean']:.6f}, "
            f"hard_gain={row['hard_target_projection_gain']['mean']}, "
            f"all_align={row['all_target_shift_alignment']['mean']:.6f}"
        )
    print("\nwrote:", out_json)
    print("wrote:", out_pt)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
