# ============================================================
# Task-Vector-Structured Drift Utilities
# ============================================================

from collections.abc import Mapping, Sequence
from typing import Dict, Tuple

import torch


def _state_by_name(model) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        state[name] = parameter
    for name, buffer in model.named_buffers():
        state[name] = buffer
    return state


def _matches_any(name: str, keywords: Sequence[str]) -> bool:
    return any(keyword in name for keyword in keywords)


def build_task_vector_delta_cache(
    task_model,
    base_model,
    drift_keywords: Sequence[str],
    eps: float = 1e-12,
) -> Tuple[Dict[str, torch.Tensor], Mapping[str, float]]:
    """Cache the attacker's own clean task vector on the selected scope."""

    base_state = _state_by_name(base_model)
    cache: Dict[str, torch.Tensor] = {}

    active_tensors = 0
    active_numel = 0
    task_norm_sq = 0.0

    for name, task_parameter in task_model.named_parameters():
        if name not in base_state:
            continue
        if not _matches_any(name, drift_keywords):
            continue
        if task_parameter.ndim == 0 or not torch.is_floating_point(task_parameter):
            continue

        base_parameter = base_state[name].to(
            device=task_parameter.device,
            dtype=task_parameter.dtype,
        )
        delta = (task_parameter - base_parameter).detach().clone()
        delta_norm = delta.float().norm()

        if delta_norm.item() < eps:
            continue

        cache[name] = delta
        active_tensors += 1
        active_numel += int(delta.numel())
        task_norm_sq += float(delta.float().pow(2).sum().item())

    if not cache:
        raise RuntimeError(
            "No non-zero task-vector tensors matched the requested drift scope."
        )

    stats = {
        "active_tensors": active_tensors,
        "active_numel": active_numel,
        "task_vector_l2": task_norm_sq ** 0.5,
    }
    return cache, stats


def _signed_permute_vector(
    vector: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    length = vector.numel()

    permutation = torch.randperm(length, generator=generator)
    signs = torch.randint(
        low=0,
        high=2,
        size=(length,),
        generator=generator,
        dtype=torch.int64,
    ).float()
    signs = signs.mul_(2.0).sub_(1.0)

    permutation = permutation.to(vector.device)
    signs = signs.to(device=vector.device, dtype=torch.float32)

    output = vector.float().reshape(-1).index_select(0, permutation)
    output = output * signs
    return output.reshape_as(vector).to(dtype=vector.dtype)


def _signed_permute_matrix_like(
    tensor: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Flatten all input dimensions into one matrix dimension and apply
    independent signed row/column permutations.

    If D is the flattened task matrix, the output is P D Q where P and Q
    are signed permutation matrices. Therefore the singular values and
    Frobenius norm of D are preserved up to floating-point error.
    """

    matrix = tensor.float().reshape(tensor.shape[0], -1)
    rows, cols = matrix.shape

    row_permutation = torch.randperm(rows, generator=generator)
    col_permutation = torch.randperm(cols, generator=generator)

    row_signs = torch.randint(
        low=0,
        high=2,
        size=(rows,),
        generator=generator,
        dtype=torch.int64,
    ).float()
    col_signs = torch.randint(
        low=0,
        high=2,
        size=(cols,),
        generator=generator,
        dtype=torch.int64,
    ).float()

    row_signs = row_signs.mul_(2.0).sub_(1.0)
    col_signs = col_signs.mul_(2.0).sub_(1.0)

    row_permutation = row_permutation.to(matrix.device)
    col_permutation = col_permutation.to(matrix.device)
    row_signs = row_signs.to(device=matrix.device, dtype=matrix.dtype)
    col_signs = col_signs.to(device=matrix.device, dtype=matrix.dtype)

    output = matrix.index_select(0, row_permutation)
    output = output.index_select(1, col_permutation)
    output = output * row_signs.unsqueeze(1)
    output = output * col_signs.unsqueeze(0)

    return output.reshape_as(tensor).to(dtype=tensor.dtype)


def sample_task_vector_structured_drift_cache(
    task_delta_cache: Mapping[str, torch.Tensor],
    rho: float,
    seed: int,
    eps: float = 1e-12,
) -> Tuple[Dict[str, torch.Tensor], Mapping[str, float]]:
    """Sample one task-vector-structured background drift."""

    if rho < 0:
        raise ValueError(f"rho must be non-negative, got {rho}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    drift_cache: Dict[str, torch.Tensor] = {}

    task_norm_sq = 0.0
    drift_norm_sq = 0.0
    dot_product = 0.0
    matrix_tensors = 0
    vector_tensors = 0

    for name, task_delta in task_delta_cache.items():
        if task_delta.ndim == 1:
            structured = _signed_permute_vector(task_delta, generator)
            vector_tensors += 1
        elif task_delta.ndim >= 2:
            structured = _signed_permute_matrix_like(task_delta, generator)
            matrix_tensors += 1
        else:
            continue

        structured_norm = structured.float().norm()
        task_norm = task_delta.float().norm()

        if task_norm.item() < eps or structured_norm.item() < eps:
            continue

        structured = structured * (
            task_norm / structured_norm.clamp_min(eps)
        ).to(dtype=structured.dtype)
        drift = structured * float(rho)

        drift_cache[name] = drift.detach()

        task_float = task_delta.float()
        drift_float = drift.float()
        task_norm_sq += float(task_float.pow(2).sum().item())
        drift_norm_sq += float(drift_float.pow(2).sum().item())
        dot_product += float((task_float * drift_float).sum().item())

    if not drift_cache:
        raise RuntimeError("Structured drift cache is empty.")

    task_l2 = task_norm_sq ** 0.5
    drift_l2 = drift_norm_sq ** 0.5
    cosine = dot_product / max(task_l2 * drift_l2, eps)

    stats = {
        "seed": int(seed),
        "rho": float(rho),
        "active_tensors": len(drift_cache),
        "matrix_tensors": matrix_tensors,
        "vector_tensors": vector_tensors,
        "task_vector_l2": task_l2,
        "structured_drift_l2": drift_l2,
        "drift_over_task_norm": drift_l2 / max(task_l2, eps),
        "task_drift_cosine": cosine,
    }

    return drift_cache, stats
