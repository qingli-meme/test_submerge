import argparse
import json
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import torch

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))


logging.basicConfig(level=logging.INFO, format="%(asctime)s [SubMerge-NS] %(message)s")
logger = logging.getLogger("SubMerge-NS")


def load_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object at {path}: {type(obj)}")
    return OrderedDict(obj)


def get_proxy_checkpoints(proxy_datasets, model_name, ckpt_dir):
    paths = []
    for dataset in proxy_datasets:
        path = os.path.join(ckpt_dir, model_name, dataset, "finetuned.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing proxy checkpoint: {path}. Run official clean fine-tuning first."
            )
        paths.append(path)
        logger.info("proxy %s -> %s", dataset, path)
    return paths


def floating_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.is_floating_point()


def compute_task_vectors(pretrained_path, finetuned_paths):
    pretrained = load_state_dict(pretrained_path)
    task_vectors = {}
    for path in finetuned_paths:
        finetuned = load_state_dict(path)
        for key, base in pretrained.items():
            if key not in finetuned:
                continue
            if base.shape != finetuned[key].shape or not floating_tensor(base):
                continue
            task_vectors.setdefault(key, []).append(finetuned[key].float() - base.float())

    task_vectors = {
        key: values for key, values in task_vectors.items()
        if len(values) == len(finetuned_paths)
    }
    logger.info("computed task vectors for %d parameter keys", len(task_vectors))
    return task_vectors


def rank_from_energy(singular_values, energy_threshold):
    if singular_values.numel() == 0:
        return 0
    energy = singular_values.square()
    total = energy.sum()
    if total <= 1e-20:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    idx = torch.searchsorted(cumulative, torch.tensor(float(energy_threshold))).item()
    return min(int(idx) + 1, singular_values.numel())


def orthonormalize(columns, dim, rank_cap=None):
    if not columns:
        return None
    matrix = torch.cat(columns, dim=1).float()
    if matrix.numel() == 0:
        return None
    if rank_cap is not None and rank_cap > 0 and matrix.shape[1] > rank_cap:
        # Compress correlated proxy bases before QR. This keeps projection cheap
        # without changing the represented span too much for small proxy counts.
        u, _, _ = torch.linalg.svd(matrix, full_matrices=False)
        q = u[:, : min(rank_cap, u.shape[1])].contiguous()
    else:
        q, _ = torch.linalg.qr(matrix, mode="reduced")
    if q.shape[1] >= dim:
        return None
    return q.contiguous()


def compute_nullspace_basis(tvs, energy_threshold=0.95, max_basis_rank=None):
    shape = tuple(tvs[0].shape)

    if len(shape) == 2:
        m, n = shape
        left, right = [], []
        ranks = []
        for tv in tvs:
            try:
                u, s, vh = torch.linalg.svd(tv.float(), full_matrices=False)
            except RuntimeError as exc:
                logger.warning("SVD failed for 2D tensor %s: %s", shape, exc)
                return None
            rank = rank_from_energy(s, energy_threshold)
            ranks.append(rank)
            if rank > 0:
                left.append(u[:, :rank].cpu())
                right.append(vh[:rank, :].T.cpu())

        q_left = orthonormalize(left, m, max_basis_rank)
        q_right = orthonormalize(right, n, max_basis_rank)
        if q_left is None and q_right is None:
            return None
        return {
            "type": "2d",
            "Q_left": q_left,
            "Q_right": q_right,
            "original_shape": shape,
            "proxy_ranks": ranks,
            "clean_rank_left": 0 if q_left is None else q_left.shape[1],
            "clean_rank_right": 0 if q_right is None else q_right.shape[1],
            "null_dim_left": m if q_left is None else m - q_left.shape[1],
            "null_dim_right": n if q_right is None else n - q_right.shape[1],
        }

    if len(shape) == 4:
        reshaped = [tv.reshape(shape[0], -1) for tv in tvs]
        basis = compute_nullspace_basis(reshaped, energy_threshold, max_basis_rank)
        if basis is None:
            return None
        basis["type"] = "4d_as_2d"
        basis["original_shape"] = shape
        return basis

    if len(shape) == 1 and shape[0] > 1:
        stacked = torch.stack([tv.float() for tv in tvs], dim=0)
        try:
            _, s, vh = torch.linalg.svd(stacked, full_matrices=False)
        except RuntimeError as exc:
            logger.warning("SVD failed for 1D tensor %s: %s", shape, exc)
            return None
        rank = rank_from_energy(s, energy_threshold)
        if rank <= 0 or rank >= shape[0]:
            return None
        if max_basis_rank is not None and max_basis_rank > 0:
            rank = min(rank, max_basis_rank)
        return {
            "type": "1d",
            "Q_basis": vh[:rank, :].T.contiguous().cpu(),
            "original_shape": shape,
            "clean_rank": rank,
            "null_dim": shape[0] - rank,
        }

    return None


def estimate_nullspace(
    pretrained_path,
    proxy_ckpt_paths,
    energy_threshold=0.95,
    save_path=None,
    max_basis_rank=None,
):
    task_vectors = compute_task_vectors(pretrained_path, proxy_ckpt_paths)
    nullspace = {}
    stats = {
        "total_keys": 0,
        "processed_keys": 0,
        "skipped_keys": 0,
        "total_params_seen": 0,
        "projected_params": 0,
        "approx_null_dim": 0,
        "approx_projected_dim": 0,
    }

    for key, tvs in task_vectors.items():
        stats["total_keys"] += 1
        shape = tuple(tvs[0].shape)
        numel = int(np.prod(shape))
        stats["total_params_seen"] += numel
        basis = compute_nullspace_basis(tvs, energy_threshold, max_basis_rank)
        if basis is None:
            stats["skipped_keys"] += 1
            continue
        nullspace[key] = basis
        stats["processed_keys"] += 1
        stats["projected_params"] += numel
        if basis["type"] in ("2d", "4d_as_2d"):
            if basis["type"] == "4d_as_2d":
                m = shape[0]
                n = int(numel / max(m, 1))
            else:
                m, n = shape
            clean_left = basis.get("clean_rank_left", 0)
            clean_right = basis.get("clean_rank_right", 0)
            null_dim = (m - clean_left) * (n - clean_right)
            stats["approx_projected_dim"] += m * n
            stats["approx_null_dim"] += int(max(null_dim, 0))
        elif basis["type"] == "1d":
            stats["approx_projected_dim"] += numel
            stats["approx_null_dim"] += basis["null_dim"]

    stats["projected_param_ratio"] = stats["projected_params"] / max(stats["total_params_seen"], 1)
    stats["approx_null_ratio"] = stats["approx_null_dim"] / max(stats["approx_projected_dim"], 1)
    logger.info("processed %d/%d keys", stats["processed_keys"], stats["total_keys"])
    logger.info("projected param ratio %.2f%%", 100 * stats["projected_param_ratio"])
    logger.info("approx null ratio %.2f%%", 100 * stats["approx_null_ratio"])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "nullspace_info": nullspace,
                "stats": stats,
                "energy_threshold": energy_threshold,
                "proxy_paths": proxy_ckpt_paths,
                "max_basis_rank": max_basis_rank,
            },
            save_path,
        )
        with open(save_path.replace(".pt", "_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("saved nullspace to %s", save_path)

    return nullspace, stats


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


def nullspace_overlap_ratio(tensor, basis_info, eps=1e-12):
    if basis_info is None:
        return float("nan")
    projected = project_tensor_to_nullspace(tensor, basis_info)
    residual = tensor - projected
    return float(torch.linalg.vector_norm(residual.float()) / torch.linalg.vector_norm(tensor.float()).clamp_min(eps))


def compute_dense_bounds(pretrained_path, proxy_ckpt_paths, percentile=95.0, save_path=None):
    task_vectors = compute_task_vectors(pretrained_path, proxy_ckpt_paths)
    bounds = {}
    stats = {}
    for key, tvs in task_vectors.items():
        values = torch.cat([tv.abs().flatten().float() for tv in tvs])
        q = min(max(float(percentile) / 100.0, 0.0), 1.0)
        kth = max(1, min(values.numel(), int(np.ceil(q * values.numel()))))
        eps = float(values.kthvalue(kth).values)
        bounds[key] = eps
        stats[key] = {
            "eps": eps,
            "mean_abs": float(values.mean()),
            "max_abs": float(values.max()),
            "numel_per_proxy": int(tvs[0].numel()),
        }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"bounds": bounds, "stats": stats, "percentile": percentile}, save_path)
        logger.info("saved dense bounds to %s", save_path)
    return bounds, stats


# ============================================================
# FOPA: Background Drift Bank Construction
# ============================================================
# Drift bank stores logit offsets induced by proxy merged backgrounds.
# Training can then add these offsets to simulated payload logits without
# forwarding a frozen merged model on every step.


def build_drift_bank(
    pretrained_path,
    proxy_ckpt_paths,
    classification_head_path,
    train_dataset_name,
    data_location,
    model_name="ViT-B-32",
    merge_lambda=0.3,
    num_backgrounds=8,
    max_samples=2000,
    batch_size=64,
    save_path=None,
):
    """
    Build a background drift bank for FOPA.

    For each sampled proxy background, compute clean-image logit drift:
        z(theta_bg, x) - z(theta_pre, x)

    The same cached clean samples are reused for every background, so each
    drift tensor row is aligned to the pretrained-logit row it subtracts.
    """
    import copy
    import random as rng

    from src.datasets.common import maybe_dictionarize
    from src.datasets.registry import get_dataset
    from src.heads import get_classification_head
    from src.modeling import ImageEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_sd = load_state_dict(pretrained_path)

    proxy_tvs = []
    for path in proxy_ckpt_paths:
        ft_sd = load_state_dict(path)
        tv = {}
        for key, base in pretrained_sd.items():
            if key in ft_sd and floating_tensor(base):
                if base.shape == ft_sd[key].shape:
                    tv[key] = ft_sd[key].float() - base.float()
        proxy_tvs.append(tv)
    if not proxy_tvs:
        raise ValueError("proxy_ckpt_paths is empty; cannot build drift bank")
    logger.info("loaded %d proxy task vectors for drift bank", len(proxy_tvs))

    class _Args:
        pass

    head_args = _Args()
    head_args.model = model_name
    head_args.save = classification_head_path
    head_args.cache_dir = ""
    head_args.openclip_cachedir = "./open_clip"
    head_args.data_location = data_location
    classification_head = get_classification_head(head_args, train_dataset_name).to(device)
    classification_head.eval()

    model_args = _Args()
    model_args.model = model_name
    model_args.cache_dir = ""
    model_args.openclip_cachedir = "./open_clip"
    pretrained_encoder = ImageEncoder(model_args, keep_lang=False).to(device)
    pretrained_encoder.load_state_dict(pretrained_sd, strict=False)
    pretrained_encoder.eval()

    preprocess = pretrained_encoder.val_preprocess
    _, data_loader = get_dataset(
        train_dataset_name,
        "train",
        preprocess,
        location=data_location,
        batch_size=batch_size,
    )

    sample_batches = []
    sample_count = 0
    logger.info(
        "collecting %s clean samples for drift bank (max %d)...",
        train_dataset_name,
        max_samples,
    )
    for batch in data_loader:
        if sample_count >= max_samples:
            break
        batch = maybe_dictionarize(batch)
        images = batch["images"]
        remaining = max_samples - sample_count
        if images.shape[0] > remaining:
            images = images[:remaining]
        sample_batches.append(images.cpu())
        sample_count += images.shape[0]
    if not sample_batches:
        raise RuntimeError(f"No samples collected from {train_dataset_name}")

    logger.info("computing pretrained logits on %d samples...", sample_count)
    all_pre_logits = []
    with torch.no_grad():
        for images in sample_batches:
            images = images.to(device)
            logits = classification_head(pretrained_encoder(images))
            all_pre_logits.append(logits.cpu())
    all_pre_logits = torch.cat(all_pre_logits, dim=0)
    logger.info("pretrained logits shape: %s", tuple(all_pre_logits.shape))

    drift_bank = []
    num_proxies = len(proxy_tvs)
    tv_keys = list(proxy_tvs[0].keys())

    for bg_idx in range(num_backgrounds):
        num_selected = rng.randint(1, min(5, num_proxies))
        if num_proxies >= 3:
            num_selected = rng.randint(3, min(5, num_proxies))
        selected = rng.sample(range(num_proxies), num_selected)
        logger.info(
            "drift bank bg %d/%d: using proxies %s",
            bg_idx + 1,
            num_backgrounds,
            selected,
        )

        bg_sd = copy.deepcopy(pretrained_sd)
        for key in tv_keys:
            base = bg_sd[key]
            if not floating_tensor(base):
                continue
            merged = base.float()
            for s in selected:
                if key in proxy_tvs[s]:
                    merged = merged + merge_lambda * proxy_tvs[s][key]
            bg_sd[key] = merged.to(dtype=base.dtype)

        bg_encoder = ImageEncoder(model_args, keep_lang=False).to(device)
        bg_encoder.load_state_dict(bg_sd, strict=False)
        bg_encoder.eval()

        bg_logits = []
        with torch.no_grad():
            for images in sample_batches:
                images = images.to(device)
                logits = classification_head(bg_encoder(images))
                bg_logits.append(logits.cpu())
        bg_logits = torch.cat(bg_logits, dim=0)

        drift = bg_logits - all_pre_logits
        drift_bank.append(drift)
        logger.info(
            "  drift norm mean=%.4f std=%.4f",
            drift.norm(dim=1).mean().item(),
            drift.norm(dim=1).std().item(),
        )

        del bg_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "drift_bank": drift_bank,
                "num_backgrounds": num_backgrounds,
                "max_samples": sample_count,
                "merge_lambda": merge_lambda,
                "num_classes": all_pre_logits.shape[1],
                "proxy_paths": proxy_ckpt_paths,
            },
            save_path,
        )
        logger.info(
            "saved drift bank to %s (%d backgrounds, %d samples each)",
            save_path,
            num_backgrounds,
            sample_count,
        )

    return drift_bank


def parse_args():
    parser = argparse.ArgumentParser(description="SubMerge null-space estimation")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--proxy-datasets", nargs="+", default=["Cars", "SUN397", "PETS"])
    parser.add_argument("--energy-threshold", type=float, default=0.95)
    parser.add_argument("--max-basis-rank", type=int, default=0)
    parser.add_argument("--dense-percentile", type=float, default=95.0)
    parser.add_argument("--save-dir", default="./nullspace")
    parser.add_argument(
        "--skip-drift-bank",
        action="store_true",
        help="Skip FOPA drift bank construction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pretrained_path = args.pretrained_path or os.path.join(args.ckpt_dir, args.model, "zeroshot.pt")
    proxy_paths = get_proxy_checkpoints(args.proxy_datasets, args.model, args.ckpt_dir)
    suffix = f"e{args.energy_threshold}_r{args.max_basis_rank or 'full'}"
    save_path = os.path.join(args.save_dir, args.model, f"nullspace_{suffix}.pt")
    estimate_nullspace(
        pretrained_path,
        proxy_paths,
        energy_threshold=args.energy_threshold,
        save_path=save_path,
        max_basis_rank=args.max_basis_rank or None,
    )
    eps_path = os.path.join(
        args.save_dir,
        args.model,
        f"eps_bounds_p{args.dense_percentile}.pt",
    )
    compute_dense_bounds(pretrained_path, proxy_paths, args.dense_percentile, eps_path)
    if not args.skip_drift_bank:
        drift_path = os.path.join(
            args.save_dir,
            args.model,
            "drift_bank_bg8_s2000_lam0.3.pt",
        )
        if os.path.exists(drift_path):
            logger.info("drift bank already exists: %s", drift_path)
        else:
            build_drift_bank(
                pretrained_path,
                proxy_paths,
                classification_head_path=os.path.join(args.ckpt_dir, args.model),
                train_dataset_name="CIFAR100",
                data_location="./data",
                model_name=args.model,
                merge_lambda=0.3,
                num_backgrounds=8,
                max_samples=2000,
                batch_size=64,
                save_path=drift_path,
            )


if __name__ == "__main__":
    main()
