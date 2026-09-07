import argparse
import json
import os
import sys
import time
from collections import Counter, OrderedDict

import numpy as np
import torch
import tqdm

sys.path.append(".")
sys.path.append("./src")

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

from heads import get_classification_head
from modeling import ImageClassifier
from regmean import RegMean
from src.datasets.registry import get_dataset
from src.datasets.common import maybe_dictionarize
from ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict
from utils import corner_mask_generation


EXAM_DATASETS = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
SCALING_SWEEP = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


class EvalArgs:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="KDR RegMean/readout diagnostics without changing training or eval logic."
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-train-batch", type=int, default=8)
    parser.add_argument("--out-dir", default="./analysis/kdr_regmean_readout")
    parser.add_argument("--cache-dir", default="./analysis/kdr_regmean_readout/cache")
    parser.add_argument("--adamerging-lambda", default=None)
    parser.add_argument("--skip-utility", action="store_true")
    parser.add_argument("--skip-scaling-sweep", action="store_true")
    parser.add_argument("--skip-epoch-sweep", action="store_true")
    parser.add_argument("--skip-epoch-regmean", action="store_true")
    return parser.parse_args()


def make_args(args):
    evargs = EvalArgs()
    evargs.model = args.model
    evargs.data_location = args.data_location
    evargs.batch_size = args.batch_size
    evargs.device = "cuda" if torch.cuda.is_available() else "cpu"
    evargs.ckpt_dir = args.ckpt_dir
    evargs.save = os.path.join(args.ckpt_dir, args.model)
    evargs.openclip_cachedir = "./open_clip"
    evargs.cache_dir = "./cache"
    evargs.dataset_list = EXAM_DATASETS
    evargs.num_train_batch = args.num_train_batch
    return evargs


def root_dir(args):
    return os.path.join(args.ckpt_dir, args.model)


def pretrained_path(args):
    return os.path.join(root_dir(args), "zeroshot.pt")


def clean_ckpt_path(args, dataset):
    return os.path.join(root_dir(args), dataset, "finetuned.pt")


def kdr_ckpt_path(args, epoch_index=None):
    base = os.path.join(
        root_dir(args),
        f"{args.adversary_task}_KDR_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}",
    )
    if epoch_index is None:
        return os.path.join(base, "finetuned.pt")
    return os.path.join(base, f"finetuned_epoch_{epoch_index}.pt")


def attack_paths(args, attack="kdr", epoch_index=None):
    paths = {}
    for dataset in EXAM_DATASETS:
        if attack == "kdr" and dataset == args.adversary_task:
            paths[dataset] = kdr_ckpt_path(args, epoch_index)
        else:
            paths[dataset] = clean_ckpt_path(args, dataset)
    return paths


def load_encoder(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device, weights_only=False)


def load_state_dict(path):
    obj = load_encoder(path, "cpu")
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def load_trigger(args):
    path = os.path.join(
        "./trigger",
        args.model,
        f"KDR_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    trigger = torch.from_numpy(np.load(path)).float()
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    return {
        "path": path,
        "applied_patch": torch.from_numpy(applied_patch).float(),
        "mask": torch.from_numpy(mask).float(),
        "target_cls": args.target_cls,
    }


def load_flat_task_vectors(args, paths):
    ptm_sd = load_state_dict(pretrained_path(args))
    flat_ptm = state_dict_to_vector(ptm_sd, [])
    tvs = []
    for dataset in EXAM_DATASETS:
        sd = load_state_dict(paths[dataset])
        tvs.append(state_dict_to_vector(sd, []) - flat_ptm)
    return ptm_sd, flat_ptm, torch.vstack(tvs)


def encoder_from_state(args, state):
    encoder = load_encoder(pretrained_path(args), "cpu")
    encoder.load_state_dict(state, strict=False)
    return encoder


def encoder_from_vector(args, ptm_sd, vector):
    state = vector_to_state_dict(vector, ptm_sd, [])
    return encoder_from_state(args, state)


def build_ta_encoder(args, paths):
    ptm_sd, flat_ptm, tvs = load_flat_task_vectors(args, paths)
    merged = flat_ptm + args.scaling_coef * tvs.sum(dim=0)
    return encoder_from_vector(args, ptm_sd, merged)


def build_ties_encoder(args, paths):
    ptm_sd, flat_ptm, tvs = load_flat_task_vectors(args, paths)
    merged_tv = ties_merging(
        tvs, reset_thresh=args.ties_reset_thresh, merge_func=args.ties_merge_func
    )
    merged = flat_ptm + args.scaling_coef * merged_tv
    return encoder_from_vector(args, ptm_sd, merged)


def build_adamerging_encoder(args, paths, lambda_path):
    if not lambda_path or not os.path.exists(lambda_path):
        raise FileNotFoundError(f"Missing AdaMerging lambda file: {lambda_path}")

    ptm_sd = load_state_dict(pretrained_path(args))
    ft_sds = [load_state_dict(paths[dataset]) for dataset in EXAM_DATASETS]
    lambdas = torch.load(lambda_path, map_location="cpu", weights_only=False).detach().float()
    lambdas = torch.clamp(lambdas, min=0.0, max=1.0)

    merged = OrderedDict()
    for row_idx, (name, ptm_value) in enumerate(ptm_sd.items()):
        value = ptm_value.detach().clone().float()
        for task_idx, ft_sd in enumerate(ft_sds):
            value = value + lambdas[row_idx, task_idx] * (ft_sd[name].float() - ptm_value.float())
        merged[name] = value
    return encoder_from_state(args, merged)


def build_regmean_encoder(args, paths, cache_name):
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"{cache_name}.pt")
    if os.path.exists(cache_path):
        state = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"[cache] loaded RegMean state: {cache_path}")
        return encoder_from_state(args, state)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    evargs = make_args(args)
    regmean = RegMean(evargs, None)
    model_list = []
    gram_list = []
    with torch.no_grad():
        for dataset in EXAM_DATASETS:
            path = paths[dataset]
            print(f"[RegMean] loading {dataset}: {path}")
            image_encoder = load_encoder(path, device)
            classification_head = regmean.class_head_dict[dataset]
            model = ImageClassifier(image_encoder, classification_head)
            model.freeze_head()
            model = model.to(device)
            model.eval()
            model_list.append(model)
            gram_list.append(regmean.compute_gram(model, dataset))

        avg_params = regmean.avg_merge(model_list, regmean_grams=gram_list)
        image_encoder = load_encoder(pretrained_path(args), device)
        model = ImageClassifier(image_encoder, regmean.class_head_dict[EXAM_DATASETS[0]])
        model.freeze_head()
        model = model.to(device)
        regmean.copy_params_to_model(avg_params, model)
        state = OrderedDict((k, v.detach().cpu()) for k, v in model.image_encoder.state_dict().items())

    torch.save(state, cache_path)
    print(f"[cache] saved RegMean state: {cache_path}")
    return encoder_from_state(args, state)


def apply_trigger(images, trigger_info):
    mask = trigger_info["mask"].type_as(images)
    patch = trigger_info["applied_patch"].type_as(images)
    return mask * patch + (1.0 - mask.expand_as(images)) * images


def summarize_values(values):
    t = torch.tensor(values, dtype=torch.float32)
    if t.numel() == 0:
        return {}
    quantiles = torch.quantile(t, torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    return {
        "mean": t.mean().item(),
        "std": t.std(unbiased=False).item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "q01": quantiles[0].item(),
        "q05": quantiles[1].item(),
        "q25": quantiles[2].item(),
        "median": quantiles[3].item(),
        "q75": quantiles[4].item(),
        "q95": quantiles[5].item(),
        "q99": quantiles[6].item(),
    }


def eval_trigger_readout(args, encoder, trigger_info, reference_encoder=None, max_batches=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evargs = make_args(args)
    head = get_classification_head(evargs, args.target_task).to(device).eval()
    encoder = encoder.to(device).eval()
    if reference_encoder is not None:
        reference_encoder = reference_encoder.to(device).eval()

    _, loader = get_dataset(
        args.target_task,
        "test",
        encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    target = args.target_cls
    n = 0
    hit = 0
    clean_correct = 0
    margins = []
    target_logits = []
    max_other_logits = []
    target_gains = []
    margin_gains = []
    hist = Counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc="trigger-readout")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            patched = apply_trigger(images, trigger_info)
            logits = head(encoder(patched))
            pred = logits.argmax(dim=1)
            non_target = labels != target
            if non_target.sum().item() == 0:
                continue

            logits_nt = logits[non_target]
            pred_nt = pred[non_target]
            labels_nt = labels[non_target]
            target_logit = logits_nt[:, target]
            other_logits = logits_nt.clone()
            other_logits[:, target] = -1e9
            max_other = other_logits.max(dim=1).values
            margin = target_logit - max_other

            n += int(non_target.sum().item())
            hit += int((pred_nt == target).sum().item())
            clean_correct += int((pred_nt == labels_nt).sum().item())
            margins.extend(margin.detach().cpu().tolist())
            target_logits.extend(target_logit.detach().cpu().tolist())
            max_other_logits.extend(max_other.detach().cpu().tolist())
            hist.update(pred_nt.detach().cpu().tolist())

            if reference_encoder is not None:
                ref_logits = head(reference_encoder(patched))
                ref_logits_nt = ref_logits[non_target]
                ref_target_logit = ref_logits_nt[:, target]
                ref_other = ref_logits_nt.clone()
                ref_other[:, target] = -1e9
                ref_margin = ref_target_logit - ref_other.max(dim=1).values
                target_gains.extend((target_logit - ref_target_logit).detach().cpu().tolist())
                margin_gains.extend((margin - ref_margin).detach().cpu().tolist())

    result = {
        "asr": hit / max(n, 1),
        "asr_percent": 100.0 * hit / max(n, 1),
        "patched_top1_non_target": clean_correct / max(n, 1),
        "non_target_cnt": n,
        "backdoored_cnt": hit,
        "target_margin_positive_ratio": float((torch.tensor(margins) > 0).float().mean().item()) if margins else None,
        "target_margin": summarize_values(margins),
        "target_logit": summarize_values(target_logits),
        "max_other_logit": summarize_values(max_other_logits),
        "predicted_class_histogram_top10": [
            {"class": int(k), "count": int(v)} for k, v in hist.most_common(10)
        ],
    }
    if reference_encoder is not None:
        result["residual_target_logit_gain"] = summarize_values(target_gains)
        result["residual_margin_gain"] = summarize_values(margin_gains)
    return result


def eval_clean_accuracy(args, encoder, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evargs = make_args(args)
    head = get_classification_head(evargs, dataset).to(device).eval()
    encoder = encoder.to(device).eval()
    _, loader = get_dataset(
        dataset,
        "test",
        encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    n = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc=f"clean-acc-{dataset}"):
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            logits = head(encoder(images))
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            n += int(labels.numel())
    return 100.0 * correct / max(n, 1)


def eval_avg_clean_accuracy(args, encoder):
    accs = {}
    for dataset in EXAM_DATASETS:
        accs[dataset] = eval_clean_accuracy(args, encoder, dataset)
    return {"datasets": accs, "avg_clean_acc": float(np.mean(list(accs.values())))}


def regmean_scaled_encoder(args, clean_encoder, kdr_encoder, alpha):
    clean_sd = clean_encoder.cpu().state_dict()
    kdr_sd = kdr_encoder.cpu().state_dict()
    scaled = OrderedDict()
    for name in clean_sd:
        scaled[name] = clean_sd[name].float() + alpha * (kdr_sd[name].float() - clean_sd[name].float())
    return encoder_from_state(args, scaled)


def run_scaling_sweep(args, trigger_info, clean_regmean, kdr_regmean):
    rows = []
    for alpha in SCALING_SWEEP:
        print(f"\n[scaling] RegMean residual alpha={alpha}")
        encoder = regmean_scaled_encoder(args, clean_regmean, kdr_regmean, alpha)
        metrics = eval_trigger_readout(args, encoder, trigger_info, reference_encoder=clean_regmean)
        row = {
            "alpha_residual_from_clean_regmean": alpha,
            "asr_percent": metrics["asr_percent"],
            "target_margin_mean": metrics["target_margin"]["mean"],
            "target_margin_positive_ratio": metrics["target_margin_positive_ratio"],
            "residual_target_logit_gain_mean": metrics["residual_target_logit_gain"]["mean"],
            "residual_margin_gain_mean": metrics["residual_margin_gain"]["mean"],
        }
        if not args.skip_utility:
            row.update(eval_avg_clean_accuracy(args, encoder))
        rows.append(row)
    return rows


def run_epoch_sweep(args, trigger_info, clean_refs, lambda_path):
    rows = []
    for epoch_index in range(5):
        label = f"epoch_{epoch_index + 1}"
        paths = attack_paths(args, "kdr", epoch_index=epoch_index)
        row = {"checkpoint": label, "checkpoint_path": paths[args.adversary_task]}
        print(f"\n[epoch] {label}: {paths[args.adversary_task]}")

        ta = build_ta_encoder(args, paths)
        row["ta"] = eval_trigger_readout(args, ta, trigger_info, reference_encoder=clean_refs["ta"])

        ties = build_ties_encoder(args, paths)
        row["ties"] = eval_trigger_readout(args, ties, trigger_info, reference_encoder=clean_refs["ties"])

        if not args.skip_epoch_regmean:
            reg = build_regmean_encoder(args, paths, cache_name=f"kdr_regmean_{label}")
            row["regmean"] = eval_trigger_readout(args, reg, trigger_info, reference_encoder=clean_refs["regmean"])

        if lambda_path and os.path.exists(lambda_path):
            ada = build_adamerging_encoder(args, paths, lambda_path)
            row["adamerging_fixed_final_lambda"] = eval_trigger_readout(
                args, ada, trigger_info, reference_encoder=clean_refs.get("adamerging")
            )
        rows.append(row)
    return rows


def run_margin_diagnostic(args, trigger_info, clean_refs, kdr_models):
    rows = {}
    for method, encoder in kdr_models.items():
        print(f"\n[margin] {method}")
        rows[method] = eval_trigger_readout(
            args, encoder, trigger_info, reference_encoder=clean_refs.get(method)
        )
    return rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    lambda_path = args.adamerging_lambda or os.path.join(
        "./ada",
        args.model,
        f"KDR_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}_Epoch_500.pt",
    )

    print("[diag] KDR RegMean/readout diagnostics")
    print("[diag] lambda:", lambda_path)
    trigger_info = load_trigger(args)

    clean_paths = attack_paths(args, "clean")
    kdr_paths = attack_paths(args, "kdr")

    print("\n[build] clean TA/TIES/RegMean/Ada refs")
    clean_ta = build_ta_encoder(args, clean_paths)
    clean_ties = build_ties_encoder(args, clean_paths)
    clean_regmean = build_regmean_encoder(args, clean_paths, cache_name="clean_regmean")
    clean_refs = {"ta": clean_ta, "ties": clean_ties, "regmean": clean_regmean}
    if os.path.exists(lambda_path):
        clean_refs["adamerging"] = build_adamerging_encoder(args, clean_paths, lambda_path)

    print("\n[build] KDR TA/TIES/RegMean/Ada")
    kdr_ta = build_ta_encoder(args, kdr_paths)
    kdr_ties = build_ties_encoder(args, kdr_paths)
    kdr_regmean = build_regmean_encoder(args, kdr_paths, cache_name="kdr_regmean_final")
    kdr_models = {"ta": kdr_ta, "ties": kdr_ties, "regmean": kdr_regmean}
    if os.path.exists(lambda_path):
        kdr_models["adamerging"] = build_adamerging_encoder(args, kdr_paths, lambda_path)

    output = {
        "timestamp": stamp,
        "model": args.model,
        "target_task": args.target_task,
        "target_cls": args.target_cls,
        "patch_size": args.patch_size,
        "trigger_path": trigger_info["path"],
        "regmean_scaling_definition": (
            "clean_regmean + alpha * (kdr_regmean - clean_regmean). "
            "Official RegMean has no TA-style scaling coefficient; alpha=1.0 is the official KDR RegMean model."
        ),
        "adamerging_epoch_sweep_note": (
            "Epoch sweep uses the saved final KDR AdaMerging lambda for all epoch checkpoints. "
            "This isolates checkpoint effects without retraining AdaMerging lambdas per epoch."
        ),
        "clean_paths": clean_paths,
        "kdr_paths": kdr_paths,
        "adamerging_lambda": lambda_path if os.path.exists(lambda_path) else None,
    }

    if args.skip_scaling_sweep:
        output["regmean_scaling_sweep"] = []
    else:
        output["regmean_scaling_sweep"] = run_scaling_sweep(
            args, trigger_info, clean_regmean, kdr_regmean
        )
    if args.skip_epoch_sweep:
        output["checkpoint_epoch_sweep"] = []
    else:
        output["checkpoint_epoch_sweep"] = run_epoch_sweep(
            args, trigger_info, clean_refs, lambda_path if os.path.exists(lambda_path) else None
        )
    output["logit_margin_diagnostic"] = run_margin_diagnostic(
        args, trigger_info, clean_refs, kdr_models
    )

    out_path = os.path.join(args.out_dir, f"{stamp}_kdr_regmean_readout.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n========== RegMean Scaling Sweep ==========")
    for row in output["regmean_scaling_sweep"]:
        avg_acc = row.get("avg_clean_acc", None)
        avg_acc_s = "NA" if avg_acc is None else f"{avg_acc:.2f}"
        print(
            f"alpha={row['alpha_residual_from_clean_regmean']:.1f} "
            f"ASR={row['asr_percent']:.2f} "
            f"margin={row['target_margin_mean']:.4f} "
            f"pos={row['target_margin_positive_ratio']:.4f} "
            f"avg_acc={avg_acc_s}"
        )

    print("\n========== Epoch Sweep ==========")
    for row in output["checkpoint_epoch_sweep"]:
        parts = [row["checkpoint"]]
        for key in ["ta", "ties", "regmean", "adamerging_fixed_final_lambda"]:
            if key in row:
                parts.append(f"{key}={row[key]['asr_percent']:.2f}")
        print(" | ".join(parts))

    print("\n========== Logit Margin Diagnostic ==========")
    for key, metrics in output["logit_margin_diagnostic"].items():
        gain = metrics.get("residual_target_logit_gain", {}).get("mean", None)
        gain_s = "NA" if gain is None else f"{gain:.4f}"
        print(
            f"{key}: ASR={metrics['asr_percent']:.2f} "
            f"margin_mean={metrics['target_margin']['mean']:.4f} "
            f"margin_q05={metrics['target_margin']['q05']:.4f} "
            f"target_gain={gain_s}"
        )

    print(f"\n[diag] wrote {out_path}")


if __name__ == "__main__":
    main()
