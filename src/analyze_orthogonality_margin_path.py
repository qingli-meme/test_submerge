#!/usr/bin/env python3
"""
Experiment A: does a near-orthogonal benign merge update preserve the
triggered target decision state?

For each benign-only merge endpoint theta_bg, evaluate the controlled path

    theta(eta) = theta_0 + eta * (theta_bg - theta_0), eta in [0, 1].

The path only separates background direction from background strength. It is
not the actual optimization trajectory of TA, TIES, or RegMean.
"""

import argparse
import csv
import gc
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(".")
sys.path.append("./src")

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

from heads import get_classification_head
from modeling import ImageClassifier
from regmean import RegMean
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict
from utils import corner_mask_generation


DISPLAY = {"ta": "TA", "ties": "TIES", "regmean": "RegMean"}


def parse_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_etas(text):
    values = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if not values or values[0] != 0.0 or values[-1] != 1.0:
        raise argparse.ArgumentTypeError("etas must contain 0 and 1")
    if any(x < 0 or x > 1 for x in values):
        raise argparse.ArgumentTypeError("etas must lie in [0, 1]")
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--attack-task", default="CIFAR100")
    parser.add_argument("--target-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--benign-datasets", default="GTSRB,EuroSAT,Cars,SUN397,PETS")
    parser.add_argument("--merge-methods", default="ta,ties,regmean")
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20.0)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--regmean-num-train-batch", type=int, default=8)
    parser.add_argument(
        "--etas",
        type=parse_etas,
        default=[round(i / 10, 1) for i in range(11)],
    )
    parser.add_argument("--trigger-sources", default="KDR_DTK,KDR_TDK")
    parser.add_argument(
        "--trigger-labels",
        default="With dormancy constraint,Without dormancy constraint",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", default="./analysis/orthogonality_margin_path")
    parser.add_argument("--rebuild-backgrounds", action="store_true")
    args = parser.parse_args()

    args.benign_datasets = parse_list(args.benign_datasets)
    args.merge_methods = parse_list(args.merge_methods)
    args.trigger_sources = parse_list(args.trigger_sources)
    args.trigger_labels = parse_list(args.trigger_labels)

    if args.attack_task in args.benign_datasets:
        parser.error("attack-task must be excluded from benign-datasets")
    if set(args.merge_methods) - {"ta", "ties", "regmean"}:
        parser.error("merge-methods only supports ta,ties,regmean")
    if len(args.trigger_sources) != len(args.trigger_labels):
        parser.error("trigger-sources and trigger-labels must have the same length")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def check_same_state_dict(reference, candidate, name):
    if set(reference.keys()) != set(candidate.keys()):
        raise RuntimeError(f"State-dict keys mismatch: {name}")
    for key in reference:
        if reference[key].shape != candidate[key].shape:
            raise RuntimeError(f"Shape mismatch at {key}: {name}")


def cache_path(args, method):
    cache_dir = Path(args.out_dir) / "background_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = "-".join(args.benign_datasets)
    if method == "ta":
        suffix = f"scale{args.scaling_coef:g}"
    elif method == "ties":
        suffix = (
            f"scale{args.scaling_coef:g}_reset{args.ties_reset_thresh:g}_"
            f"{args.ties_merge_func}"
        )
    else:
        suffix = f"batches{args.regmean_num_train_batch}"
    return cache_dir / f"{method}_{tasks}_{suffix}.pt"


def save_background(path, state):
    torch.save({k: v.detach().cpu() for k, v in state.items()}, path)
    print(f"[cache] saved {path}")


def build_ta_ties_backgrounds(args, ptm_state, methods):
    if not methods:
        return

    flat_ptm = state_dict_to_vector(ptm_state, []).cpu()
    task_vectors = torch.empty(
        len(args.benign_datasets), flat_ptm.numel(), dtype=flat_ptm.dtype
    )

    for i, dataset in enumerate(args.benign_datasets):
        path = Path(args.ckpt_dir) / args.model / dataset / "finetuned.pt"
        ft_state = state_dict_from_checkpoint(path)
        check_same_state_dict(ptm_state, ft_state, str(path))
        task_vectors[i] = state_dict_to_vector(ft_state, []).cpu() - flat_ptm
        del ft_state
        gc.collect()
        print(f"[task vector] {dataset}")

    if "ta" in methods:
        merged = flat_ptm + args.scaling_coef * task_vectors.sum(0)
        state = vector_to_state_dict(merged, ptm_state, [])
        save_background(cache_path(args, "ta"), state)
        del merged, state

    if "ties" in methods:
        merged_tv = ties_merging(
            task_vectors,
            reset_thresh=args.ties_reset_thresh,
            merge_func=args.ties_merge_func,
        )
        merged = flat_ptm + args.scaling_coef * merged_tv
        state = vector_to_state_dict(merged, ptm_state, [])
        save_background(cache_path(args, "ties"), state)
        del merged_tv, merged, state

    del task_vectors, flat_ptm
    gc.collect()


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


def build_regmean_background(args):
    if not torch.cuda.is_available():
        raise RuntimeError("The repository RegMean implementation requires CUDA")

    rargs = runtime_args(args)
    rargs.dataset_list = args.benign_datasets
    rargs.num_train_batch = args.regmean_num_train_batch

    merger = RegMean(rargs, logger=None)
    model_list = []
    gram_list = []
    for dataset in args.benign_datasets:
        task_ckp_path = Path(args.ckpt_dir) / args.model / dataset / "finetuned.pt"
        print(task_ckp_path)
        image_encoder = torch.load(task_ckp_path, map_location="cuda:0")
        classification_head = merger.class_head_dict[dataset]
        model = ImageClassifier(image_encoder, classification_head)
        model.freeze_head()
        model = model.cuda().eval()
        model_list.append(model)
        with torch.no_grad():
            gram = merger.compute_gram(model, dataset)
        gram_list.append({k: v.detach().cpu() for k, v in gram.items()})
        model.cpu()
        torch.cuda.empty_cache()

    regmean_avg_params = merger.avg_merge(model_list, regmean_grams=gram_list)
    encoder = torch.load(Path(args.ckpt_dir) / args.model / "zeroshot.pt", map_location="cuda:0")
    wrapper = ImageClassifier(encoder, merger.class_head_dict[args.benign_datasets[-1]])
    wrapper.freeze_head()
    wrapper = wrapper.cuda()
    merger.copy_params_to_model(regmean_avg_params, wrapper)
    encoder = wrapper.image_encoder
    save_background(cache_path(args, "regmean"), encoder.state_dict())
    del merger, encoder, wrapper, model_list, gram_list
    gc.collect()
    torch.cuda.empty_cache()


def prepare_backgrounds(args, ptm_state):
    missing = []
    for method in args.merge_methods:
        path = cache_path(args, method)
        if args.rebuild_backgrounds or not path.exists():
            missing.append(method)

    build_ta_ties_backgrounds(args, ptm_state, [m for m in missing if m in {"ta", "ties"}])
    if "regmean" in missing:
        build_regmean_background(args)


def update_cosine(ptm_state, background_state, attack_clean_state):
    dot = 0.0
    norm_bg = 0.0
    norm_attack = 0.0
    for key, base in ptm_state.items():
        if not torch.is_floating_point(base):
            continue
        bg = (background_state[key] - base).reshape(-1).double()
        attack = (attack_clean_state[key] - base).reshape(-1).double()
        dot += torch.dot(bg, attack).item()
        norm_bg += torch.dot(bg, bg).item()
        norm_attack += torch.dot(attack, attack).item()
    norm_bg = norm_bg ** 0.5
    norm_attack = norm_attack ** 0.5
    cosine = dot / (norm_bg * norm_attack)
    return {
        "cosine": cosine,
        "abs_cosine": abs(cosine),
        "background_norm": norm_bg,
        "attack_task_norm": norm_attack,
    }


def load_triggers(args):
    trigger_dir = Path("./trigger") / args.model
    specs = []
    for source, label in zip(args.trigger_sources, args.trigger_labels):
        path = trigger_dir / (
            f"{source}_{args.attack_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy"
        )
        trigger = torch.from_numpy(np.load(path))
        applied_patch, mask, _, _ = corner_mask_generation(
            trigger, image_size=(3, 224, 224)
        )
        patch = torch.from_numpy(applied_patch).float()
        mask = torch.from_numpy(mask).float()
        if patch.ndim == 3:
            patch = patch.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        specs.append(
            {
                "source": source,
                "label": label,
                "path": str(path),
                "patch": patch,
                "mask": mask,
            }
        )
        print(f"[trigger] {label}: {path}")
    return specs


def set_interpolated_state(encoder, ptm_state, background_state, eta):
    current = encoder.state_dict()
    with torch.no_grad():
        for key, destination in current.items():
            base = ptm_state[key]
            endpoint = background_state[key]
            if torch.is_floating_point(destination):
                destination.copy_(base + eta * (endpoint - base))
            else:
                destination.copy_(base)


def patch_images(images, spec):
    mask = spec["mask"].expand(images.shape[0], -1, -1, -1)
    patch = spec["patch"].expand(images.shape[0], -1, -1, -1)
    return mask * patch + (1 - mask) * images.float()


def evaluate(encoder, head, loader, triggers, target_cls, device, max_samples):
    model = ImageClassifier(encoder, head)
    model.eval()
    stored = {
        spec["source"]: {"indices": [], "margins": [], "target_pred": []}
        for spec in triggers
    }
    processed = 0
    fallback_index = 0

    with torch.no_grad():
        for batch in loader:
            data = maybe_dictionarize(batch)
            images = data["images"]
            labels = data["labels"]
            indices = data.get("indices")

            keep = labels != target_cls
            images = images[keep]
            if indices is None:
                indices = torch.arange(
                    fallback_index, fallback_index + labels.numel(), dtype=torch.long
                )[keep]
                fallback_index += labels.numel()
            else:
                indices = torch.as_tensor(indices)[keep]

            if max_samples > 0:
                remaining = max_samples - processed
                if remaining <= 0:
                    break
                images = images[:remaining]
                indices = indices[:remaining]
            if images.numel() == 0:
                continue

            patched = [patch_images(images, spec) for spec in triggers]
            logits_all = model(torch.cat(patched, 0).to(device))
            logits_split = logits_all.split(images.shape[0], dim=0)

            for spec, logits in zip(triggers, logits_split):
                target = logits[:, target_cls]
                competitors = logits.clone()
                competitors[:, target_cls] = -torch.inf
                margin = target - competitors.max(1).values
                pred = logits.argmax(1).eq(target_cls)
                item = stored[spec["source"]]
                item["indices"].append(indices.cpu().numpy())
                item["margins"].append(margin.cpu().float().numpy())
                item["target_pred"].append(pred.cpu().numpy())

            processed += images.shape[0]
            if max_samples > 0 and processed >= max_samples:
                break

    output = {}
    for source, item in stored.items():
        output[source] = {
            "indices": np.concatenate(item["indices"]),
            "margins": np.concatenate(item["margins"]),
            "target_pred": np.concatenate(item["target_pred"]),
        }
    return output


def summary_row(method, trigger, eta, result):
    margin = result["margins"]
    return {
        "method": method,
        "trigger_source": trigger["source"],
        "trigger_label": trigger["label"],
        "eta": eta,
        "num_samples": len(margin),
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "margin_q10": float(np.quantile(margin, 0.1)),
        "margin_q90": float(np.quantile(margin, 0.9)),
        "target_rate": float(np.mean(result["target_pred"])),
    }


def endpoint_row(method, trigger, eta0, eta1, geometry):
    map0 = dict(zip(eta0["indices"].tolist(), eta0["margins"].tolist()))
    map1 = dict(zip(eta1["indices"].tolist(), eta1["margins"].tolist()))
    common = sorted(set(map0) & set(map1))
    m0 = np.array([map0[i] for i in common])
    m1 = np.array([map1[i] for i in common])
    delta = m1 - m0
    p0, p1 = m0 > 0, m1 > 0
    return {
        "method": method,
        "trigger_source": trigger["source"],
        "trigger_label": trigger["label"],
        "abs_cosine": geometry["abs_cosine"],
        "margin_median_eta0": float(np.median(m0)),
        "margin_median_eta1": float(np.median(m1)),
        "median_margin_shift": float(np.median(delta)),
        "mean_abs_margin_shift": float(np.mean(np.abs(delta))),
        "sign_flip_rate": float(np.mean(p0 != p1)),
        "target_rate_eta0": float(np.mean(p0)),
        "target_rate_eta1": float(np.mean(p1)),
    }


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_figure(args, geometry, summaries, triggers, out_dir):
    lookup = {
        (r["method"], r["trigger_source"], float(r["eta"])): r for r in summaries
    }
    panel_count = 1 + len(triggers)
    fig_width = 5.2 * panel_count
    fig, axes = plt.subplots(1, panel_count, figsize=(fig_width, 4.7))
    axes = np.atleast_1d(axes)

    methods = args.merge_methods
    bars = axes[0].bar(
        [DISPLAY[m] for m in methods], [geometry[m]["abs_cosine"] for m in methods]
    )
    axes[0].set_title("(a) Parameter geometry")
    axes[0].set_ylabel(r"$|\cos(B_m,\Delta_{C100})|$")
    for bar, method in zip(bars, methods):
        value = geometry[method]["abs_cosine"]
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    for panel, trigger in enumerate(triggers, start=1):
        ax = axes[panel]
        endpoint_text = []
        for method in methods:
            median = [
                lookup[(method, trigger["source"], e)]["margin_median"]
                for e in args.etas
            ]
            q10 = [
                lookup[(method, trigger["source"], e)]["margin_q10"]
                for e in args.etas
            ]
            q90 = [
                lookup[(method, trigger["source"], e)]["margin_q90"]
                for e in args.etas
            ]
            line = ax.plot(args.etas, median, marker="o", label=DISPLAY[method])[0]
            ax.fill_between(args.etas, q10, q90, alpha=0.12, color=line.get_color())
            rate = lookup[(method, trigger["source"], 1.0)]["target_rate"]
            endpoint_text.append(f"{DISPLAY[method]}: {100 * rate:.1f}%")

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel(r"Background strength $\eta$")
        ax.set_ylabel("Target margin")
        ax.set_title(f"({chr(97 + panel)}) {trigger['label']}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        ax.text(
            0.02,
            0.02,
            "$\\eta=1$ target rate\n" + "\n".join(endpoint_text),
            transform=ax.transAxes,
            va="bottom",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 3},
        )

    fig.suptitle("Parameter Geometry vs. Triggered Target-Margin Drift")
    fig.tight_layout()
    fig.savefig(out_dir / "orthogonality_margin_path.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "orthogonality_margin_path.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.ckpt_dir) / args.model
    pretrained_path = root / "zeroshot.pt"
    clean_attack_path = root / args.attack_task / "finetuned.pt"
    ptm_state = state_dict_from_checkpoint(pretrained_path)
    clean_attack_state = state_dict_from_checkpoint(clean_attack_path)
    check_same_state_dict(ptm_state, clean_attack_state, "clean attack task")

    prepare_backgrounds(args, ptm_state)
    triggers = load_triggers(args)

    rargs = runtime_args(args)
    head = get_classification_head(rargs, args.target_task).to(device).eval()
    probe = torch.load(pretrained_path, map_location="cpu")
    _, test_loader = get_dataset(
        args.target_task,
        "test",
        probe.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    probe = probe.to(device).eval()
    eta0 = evaluate(probe, head, test_loader, triggers, args.target_cls, device, args.max_samples)
    probe = probe.cpu()
    del probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    geometry = {}
    summaries = []
    endpoints = []
    raw_arrays = {}

    for method in args.merge_methods:
        print(f"\n========== {DISPLAY[method]} ==========")
        bg_state = state_dict_from_checkpoint(cache_path(args, method))
        check_same_state_dict(ptm_state, bg_state, method)
        geometry[method] = update_cosine(ptm_state, bg_state, clean_attack_state)
        print(f"abs cosine = {geometry[method]['abs_cosine']:.6f}")

        results = {}
        for trigger in triggers:
            results[(trigger["source"], 0.0)] = eta0[trigger["source"]]
            summaries.append(summary_row(method, trigger, 0.0, eta0[trigger["source"]]))

        encoder = torch.load(pretrained_path, map_location="cpu")
        for eta in args.etas:
            if eta == 0.0:
                continue
            print(f"eta={eta:.2f}")
            set_interpolated_state(encoder, ptm_state, bg_state, eta)
            encoder = encoder.to(device).eval()
            current = evaluate(
                encoder,
                head,
                test_loader,
                triggers,
                args.target_cls,
                device,
                args.max_samples,
            )
            encoder = encoder.cpu()
            for trigger in triggers:
                result = current[trigger["source"]]
                results[(trigger["source"], eta)] = result
                summaries.append(summary_row(method, trigger, eta, result))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for trigger in triggers:
            endpoints.append(
                endpoint_row(
                    method,
                    trigger,
                    results[(trigger["source"], 0.0)],
                    results[(trigger["source"], 1.0)],
                    geometry[method],
                )
            )
            for eta in args.etas:
                item = results[(trigger["source"], eta)]
                prefix = f"{method}__{trigger['source']}__eta_{eta:.1f}"
                raw_arrays[prefix + "__indices"] = item["indices"]
                raw_arrays[prefix + "__margins"] = item["margins"]
                raw_arrays[prefix + "__target_pred"] = item["target_pred"]

        del encoder, bg_state, results
        gc.collect()

    save_csv(out_dir / "summary.csv", summaries)
    save_csv(out_dir / "endpoint_summary.csv", endpoints)
    np.savez_compressed(out_dir / "raw_margins.npz", **raw_arrays)
    with open(out_dir / "geometry.json", "w", encoding="utf-8") as f:
        json.dump(geometry, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    plot_figure(args, geometry, summaries, triggers, out_dir)
    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
