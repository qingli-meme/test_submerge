import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "method",
    "sample_index",
    "label",
    "full_success",
    "m_full",
    "m_off",
    "a_abs",
    "g_key",
    "g_spec",
    "r",
    "r_spec",
    "off_deficit",
    "gain_coverage",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paired transport audit for KDR-DTK-FSB. Pair the exact same "
            "samples across local_attack / TA / TIES / RegMean and diagnose "
            "whether future RegMean failures are already weak locally or are "
            "created by merge-induced causal-coverage contraction."
        )
    )

    parser.add_argument(
        "--csv",
        default=(
            "./analysis/kdr_dtk_fsb_final_state_readout/"
            "20260708_133538_kdr_dtk_fsb_final_state_readout_samples.csv"
        ),
    )
    parser.add_argument("--reference-method", default="local_attack")
    parser.add_argument("--target-method", default="regmean")
    parser.add_argument(
        "--comparators",
        default="ta,ties",
        help="Comma-separated additional merge contexts.",
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--out-dir",
        default="./analysis/kdr_dtk_fsb_paired_transport",
    )

    return parser.parse_args()


def summarize(values: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q10": None,
            "q25": None,
            "q75": None,
            "q90": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def safe_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
    eps: float,
) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None
    if abs(denominator) <= eps:
        return None
    return float(numerator / denominator)


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Mann-Whitney / rank AUC without sklearn.

    labels:
        1 = future target-method failure
        0 = future target-method success

    Returned AUC answers:
        P(score_failure > score_success), with 0.5 for ties.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    scores = scores[valid]
    labels = labels[valid]

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    ranks = np.empty(scores.size, dtype=np.float64)

    start = 0
    while start < scores.size:
        end = start + 1
        while (
            end < scores.size
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1

        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end

    rank_sum_pos = float(ranks[labels == 1].sum())

    auc = (
        rank_sum_pos
        - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)

    return float(auc)


def auc_summary(
    values: np.ndarray,
    future_failure: np.ndarray,
) -> Dict[str, Optional[float]]:
    raw_auc = rank_auc(values, future_failure)

    if raw_auc is None:
        return {
            "raw_auc_higher_predicts_failure": None,
            "orientation_free_auc": None,
            "failure_direction": None,
        }

    if raw_auc >= 0.5:
        direction = "higher"
        oriented_auc = raw_auc
    else:
        direction = "lower"
        oriented_auc = 1.0 - raw_auc

    return {
        "raw_auc_higher_predicts_failure": float(raw_auc),
        "orientation_free_auc": float(oriented_auc),
        "failure_direction": direction,
    }


def normalize_method_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = (
        df["method"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return df


def validate_input(
    df: pd.DataFrame,
    methods: List[str],
):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    duplicate_mask = df.duplicated(
        subset=["method", "sample_index"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            df.loc[
                duplicate_mask,
                ["method", "sample_index"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate (method, sample_index) pairs found. "
            f"Examples: {examples}"
        )

    available = set(df["method"].unique().tolist())
    absent = [method for method in methods if method not in available]

    if absent:
        raise ValueError(
            f"Missing requested methods in CSV: {absent}. "
            f"Available methods: {sorted(available)}"
        )

    index_sets = {
        method: set(
            df.loc[
                df["method"] == method,
                "sample_index",
            ].astype(int)
        )
        for method in methods
    }

    reference_set = index_sets[methods[0]]

    for method, method_set in index_sets.items():
        if method_set != reference_set:
            missing_from_method = sorted(
                reference_set - method_set
            )[:10]
            extra_in_method = sorted(
                method_set - reference_set
            )[:10]

            raise ValueError(
                "Sample alignment failed for method "
                f"{method}. Missing examples={missing_from_method}; "
                f"extra examples={extra_in_method}"
            )

    label_maps = {}

    for method in methods:
        sub = (
            df.loc[
                df["method"] == method,
                ["sample_index", "label"],
            ]
            .sort_values("sample_index")
            .set_index("sample_index")["label"]
        )

        label_maps[method] = sub

    reference_labels = label_maps[methods[0]]

    for method, labels in label_maps.items():
        if not np.array_equal(
            reference_labels.to_numpy(),
            labels.to_numpy(),
        ):
            raise ValueError(
                "Label alignment failed between "
                f"{methods[0]} and {method}."
            )


def prefixed_frame(
    df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    sub = (
        df.loc[df["method"] == method]
        .copy()
        .sort_values("sample_index")
    )

    sub = sub.drop(columns=["method"])

    rename = {
        column: f"{method}__{column}"
        for column in sub.columns
        if column != "sample_index"
    }

    return sub.rename(columns=rename)


def build_paired_frame(
    df: pd.DataFrame,
    methods: List[str],
) -> pd.DataFrame:
    paired = prefixed_frame(df, methods[0])

    for method in methods[1:]:
        paired = paired.merge(
            prefixed_frame(df, method),
            on="sample_index",
            how="inner",
            validate="one_to_one",
        )

    paired = paired.sort_values(
        "sample_index"
    ).reset_index(drop=True)

    return paired


def add_transport_columns(
    paired: pd.DataFrame,
    reference_method: str,
    method: str,
    eps: float,
):
    ref = f"{reference_method}__"
    cur = f"{method}__"

    valid_ratio_specs = {
        "a_ratio": (
            paired[f"{cur}a_abs"],
            paired[f"{ref}a_abs"],
        ),
        "g_key_ratio": (
            paired[f"{cur}g_key"],
            paired[f"{ref}g_key"],
        ),
        "g_spec_ratio": (
            paired[f"{cur}g_spec"],
            paired[f"{ref}g_spec"],
        ),
        "r_ratio": (
            paired[f"{cur}r"],
            paired[f"{ref}r"],
        ),
        "r_spec_ratio": (
            paired[f"{cur}r_spec"],
            paired[f"{ref}r_spec"],
        ),
        "deficit_ratio": (
            paired[f"{cur}off_deficit"],
            paired[f"{ref}off_deficit"],
        ),
        "coverage_ratio": (
            paired[f"{cur}gain_coverage"],
            paired[f"{ref}gain_coverage"],
        ),
    }

    for suffix, (numerator, denominator) in valid_ratio_specs.items():
        values = np.full(
            len(paired),
            np.nan,
            dtype=np.float64,
        )

        valid = (
            np.isfinite(numerator)
            & np.isfinite(denominator)
            & (np.abs(denominator) > eps)
        )

        values[valid] = (
            numerator[valid]
            / denominator[valid]
        )

        paired[f"{method}__{suffix}"] = values

    a_ref = paired[f"{ref}a_abs"].to_numpy(
        dtype=np.float64
    )
    a_cur = paired[f"{cur}a_abs"].to_numpy(
        dtype=np.float64
    )
    r_ref = paired[f"{ref}r"].to_numpy(
        dtype=np.float64
    )
    r_cur = paired[f"{cur}r"].to_numpy(
        dtype=np.float64
    )
    d_ref = paired[f"{ref}off_deficit"].to_numpy(
        dtype=np.float64
    )
    d_cur = paired[f"{cur}off_deficit"].to_numpy(
        dtype=np.float64
    )
    c_ref = paired[f"{ref}gain_coverage"].to_numpy(
        dtype=np.float64
    )
    c_cur = paired[f"{cur}gain_coverage"].to_numpy(
        dtype=np.float64
    )

    valid_log = (
        np.isfinite(a_ref)
        & np.isfinite(a_cur)
        & np.isfinite(r_ref)
        & np.isfinite(r_cur)
        & np.isfinite(d_ref)
        & np.isfinite(d_cur)
        & np.isfinite(c_ref)
        & np.isfinite(c_cur)
        & (a_ref > eps)
        & (a_cur > eps)
        & (r_ref > eps)
        & (r_cur > eps)
        & (d_ref > eps)
        & (d_cur > eps)
        & (c_ref > eps)
        & (c_cur > eps)
    )

    log_a = np.full(len(paired), np.nan, dtype=np.float64)
    log_r = np.full(len(paired), np.nan, dtype=np.float64)
    log_d = np.full(len(paired), np.nan, dtype=np.float64)
    log_c = np.full(len(paired), np.nan, dtype=np.float64)

    log_a[valid_log] = np.log(
        a_cur[valid_log] / a_ref[valid_log]
    )
    log_r[valid_log] = np.log(
        r_cur[valid_log] / r_ref[valid_log]
    )
    log_d[valid_log] = np.log(
        d_cur[valid_log] / d_ref[valid_log]
    )
    log_c[valid_log] = np.log(
        c_cur[valid_log] / c_ref[valid_log]
    )

    paired[f"{method}__log_a_transport"] = log_a
    paired[f"{method}__log_r_transport"] = log_r
    paired[f"{method}__log_deficit_transport"] = log_d
    paired[f"{method}__log_coverage_transport"] = log_c

    paired[f"{method}__dose_pressure"] = -log_a
    paired[f"{method}__readout_pressure"] = -log_r
    paired[f"{method}__deficit_pressure"] = log_d

    reconstructed_log_c = (
        log_a + log_r - log_d
    )

    paired[
        f"{method}__log_coverage_reconstruction_error"
    ] = np.abs(
        log_c - reconstructed_log_c
    )

    # Required current-context dose under the measured current-context reader.
    required_dose = np.full(
        len(paired),
        np.nan,
        dtype=np.float64,
    )

    valid_required = (
        np.isfinite(r_cur)
        & np.isfinite(d_cur)
        & (r_cur > eps)
        & (d_cur > eps)
    )

    required_dose[valid_required] = (
        d_cur[valid_required]
        / r_cur[valid_required]
    )

    paired[f"{method}__required_dose"] = required_dose
    paired[f"{method}__dose_reserve"] = (
        a_cur - required_dose
    )
    paired[f"{method}__dose_shortfall"] = np.maximum(
        required_dose - a_cur,
        0.0,
    )

    # Algebraic factor-replacement audit:
    # use the exact identity coverage = A * r / D for M_off < 0.
    valid_factor = (
        np.isfinite(a_ref)
        & np.isfinite(a_cur)
        & np.isfinite(r_ref)
        & np.isfinite(r_cur)
        & np.isfinite(d_ref)
        & np.isfinite(d_cur)
        & (a_ref > eps)
        & (a_cur > eps)
        & (r_ref > eps)
        & (r_cur > eps)
        & (d_ref > eps)
        & (d_cur > eps)
    )

    scenarios = {
        "factor_local": (a_ref, r_ref, d_ref),
        "factor_a_only": (a_cur, r_ref, d_ref),
        "factor_r_only": (a_ref, r_cur, d_ref),
        "factor_d_only": (a_ref, r_ref, d_cur),
        "factor_a_r": (a_cur, r_cur, d_ref),
        "factor_a_d": (a_cur, r_ref, d_cur),
        "factor_r_d": (a_ref, r_cur, d_cur),
        "factor_full": (a_cur, r_cur, d_cur),
    }

    for scenario_name, (a_value, r_value, d_value) in scenarios.items():
        values = np.full(
            len(paired),
            np.nan,
            dtype=np.float64,
        )

        values[valid_factor] = (
            a_value[valid_factor]
            * r_value[valid_factor]
            / d_value[valid_factor]
        )

        paired[
            f"{method}__{scenario_name}_coverage"
        ] = values


def summarize_group_metrics(
    frame: pd.DataFrame,
    columns: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    return {
        column: summarize(
            frame[column].to_numpy(dtype=np.float64)
        )
        for column in columns
    }


def factor_scenario_summary(
    frame: pd.DataFrame,
    method: str,
) -> Dict:
    scenario_names = (
        "factor_local",
        "factor_a_only",
        "factor_r_only",
        "factor_d_only",
        "factor_a_r",
        "factor_a_d",
        "factor_r_d",
        "factor_full",
    )

    output = {}

    for scenario_name in scenario_names:
        column = (
            f"{method}__{scenario_name}_coverage"
        )
        values = frame[column].to_numpy(
            dtype=np.float64
        )
        valid = np.isfinite(values)

        if not valid.any():
            output[scenario_name] = {
                "count": 0,
                "coverage": summarize([]),
                "fraction_above_one": None,
            }
            continue

        output[scenario_name] = {
            "count": int(valid.sum()),
            "coverage": summarize(values[valid]),
            "fraction_above_one": float(
                np.mean(values[valid] > 1.0)
            ),
        }

    return output


def future_failure_visibility(
    paired: pd.DataFrame,
    reference_method: str,
    target_method: str,
) -> Dict:
    future_failure = (
        1
        - paired[
            f"{target_method}__full_success"
        ].to_numpy(dtype=np.int64)
    )

    metrics = {
        "local_a_abs": f"{reference_method}__a_abs",
        "local_g_key": f"{reference_method}__g_key",
        "local_g_spec": f"{reference_method}__g_spec",
        "local_r": f"{reference_method}__r",
        "local_r_spec": f"{reference_method}__r_spec",
        "local_m_off": f"{reference_method}__m_off",
        "local_off_deficit": (
            f"{reference_method}__off_deficit"
        ),
        "local_coverage": (
            f"{reference_method}__gain_coverage"
        ),
    }

    output = {}

    for metric_name, column in metrics.items():
        output[metric_name] = auc_summary(
            paired[column].to_numpy(
                dtype=np.float64
            ),
            future_failure,
        )

    return output


def build_summary(
    paired: pd.DataFrame,
    reference_method: str,
    target_method: str,
    comparators: List[str],
    eps: float,
) -> Dict:
    success_mask = (
        paired[
            f"{target_method}__full_success"
        ].to_numpy(dtype=np.int64)
        == 1
    )
    failure_mask = ~success_mask

    groups = {
        "all": paired,
        "future_regmean_success": paired.loc[
            success_mask
        ],
        "future_regmean_failure": paired.loc[
            failure_mask
        ],
    }

    local_columns = [
        f"{reference_method}__a_abs",
        f"{reference_method}__g_key",
        f"{reference_method}__g_spec",
        f"{reference_method}__r",
        f"{reference_method}__r_spec",
        f"{reference_method}__m_off",
        f"{reference_method}__off_deficit",
        f"{reference_method}__gain_coverage",
    ]

    transport_columns = [
        f"{target_method}__a_ratio",
        f"{target_method}__g_key_ratio",
        f"{target_method}__g_spec_ratio",
        f"{target_method}__r_ratio",
        f"{target_method}__r_spec_ratio",
        f"{target_method}__deficit_ratio",
        f"{target_method}__coverage_ratio",
        f"{target_method}__required_dose",
        f"{target_method}__dose_reserve",
        f"{target_method}__dose_shortfall",
        f"{target_method}__log_a_transport",
        f"{target_method}__log_r_transport",
        f"{target_method}__log_deficit_transport",
        f"{target_method}__log_coverage_transport",
        f"{target_method}__dose_pressure",
        f"{target_method}__readout_pressure",
        f"{target_method}__deficit_pressure",
        (
            f"{target_method}__"
            "log_coverage_reconstruction_error"
        ),
    ]

    result = {
        "counts": {
            group_name: int(len(group_frame))
            for group_name, group_frame in groups.items()
        },
        "local_metrics_by_future_regmean_outcome": {},
        "regmean_transport_by_future_regmean_outcome": {},
        "factor_replacement_by_future_regmean_outcome": {},
        "future_regmean_failure_visibility_from_local": (
            future_failure_visibility(
                paired,
                reference_method,
                target_method,
            )
        ),
        "comparator_transport": {},
    }

    for group_name, group_frame in groups.items():
        result[
            "local_metrics_by_future_regmean_outcome"
        ][group_name] = summarize_group_metrics(
            group_frame,
            local_columns,
        )

        result[
            "regmean_transport_by_future_regmean_outcome"
        ][group_name] = summarize_group_metrics(
            group_frame,
            transport_columns,
        )

        result[
            "factor_replacement_by_future_regmean_outcome"
        ][group_name] = factor_scenario_summary(
            group_frame,
            target_method,
        )

    success_local = result[
        "local_metrics_by_future_regmean_outcome"
    ]["future_regmean_success"]
    failure_local = result[
        "local_metrics_by_future_regmean_outcome"
    ]["future_regmean_failure"]

    local_ratio_summary = {}

    for column in local_columns:
        local_ratio_summary[column] = safe_ratio(
            failure_local[column]["median"],
            success_local[column]["median"],
            eps,
        )

    result[
        "future_failure_vs_success_local_median_ratios"
    ] = local_ratio_summary

    success_transport = result[
        "regmean_transport_by_future_regmean_outcome"
    ]["future_regmean_success"]
    failure_transport = result[
        "regmean_transport_by_future_regmean_outcome"
    ]["future_regmean_failure"]

    transport_ratio_summary = {}

    for column in transport_columns:
        transport_ratio_summary[column] = safe_ratio(
            failure_transport[column]["median"],
            success_transport[column]["median"],
            eps,
        )

    result[
        "future_failure_vs_success_transport_median_ratios"
    ] = transport_ratio_summary

    for comparator in comparators:
        comparator_columns = [
            f"{comparator}__a_ratio",
            f"{comparator}__r_ratio",
            f"{comparator}__deficit_ratio",
            f"{comparator}__coverage_ratio",
            f"{comparator}__log_coverage_transport",
        ]

        result["comparator_transport"][
            comparator
        ] = {
            group_name: summarize_group_metrics(
                group_frame,
                comparator_columns,
            )
            for group_name, group_frame in groups.items()
        }

    return result


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def print_local_visibility(
    summary: Dict,
    reference_method: str,
):
    print()
    print(
        "========== LOCAL METRICS BY FUTURE REGMEAN OUTCOME =========="
    )

    success = summary[
        "local_metrics_by_future_regmean_outcome"
    ]["future_regmean_success"]
    failure = summary[
        "local_metrics_by_future_regmean_outcome"
    ]["future_regmean_failure"]

    metric_suffixes = (
        "a_abs",
        "r",
        "r_spec",
        "m_off",
        "off_deficit",
        "gain_coverage",
    )

    print(
        "metric".ljust(24)
        + "future success".rjust(18)
        + "future failure".rjust(18)
        + "fail/succ".rjust(14)
    )

    ratios = summary[
        "future_failure_vs_success_local_median_ratios"
    ]

    for suffix in metric_suffixes:
        column = f"{reference_method}__{suffix}"
        print(
            column.ljust(24)
            + fmt(
                success[column]["median"]
            ).rjust(18)
            + fmt(
                failure[column]["median"]
            ).rjust(18)
            + fmt(
                ratios[column]
            ).rjust(14)
        )

    print()
    print(
        "========== LOCAL PREDICTABILITY OF FUTURE REGMEAN FAILURE =========="
    )

    visibility = summary[
        "future_regmean_failure_visibility_from_local"
    ]

    print(
        "metric".ljust(24)
        + "oriented AUC".rjust(16)
        + "failure direction".rjust(20)
    )

    for metric_name, metric_result in visibility.items():
        print(
            metric_name.ljust(24)
            + fmt(
                metric_result["orientation_free_auc"]
            ).rjust(16)
            + str(
                metric_result["failure_direction"]
            ).rjust(20)
        )


def print_transport(summary: Dict, target_method: str):
    print()
    print(
        "========== REGMEAN PAIRED TRANSPORT =========="
    )

    success = summary[
        "regmean_transport_by_future_regmean_outcome"
    ]["future_regmean_success"]
    failure = summary[
        "regmean_transport_by_future_regmean_outcome"
    ]["future_regmean_failure"]

    suffixes = (
        "a_ratio",
        "r_ratio",
        "r_spec_ratio",
        "deficit_ratio",
        "coverage_ratio",
        "required_dose",
        "dose_reserve",
        "dose_shortfall",
    )

    print(
        "metric".ljust(30)
        + "future success".rjust(18)
        + "future failure".rjust(18)
    )

    for suffix in suffixes:
        column = f"{target_method}__{suffix}"
        print(
            column.ljust(30)
            + fmt(
                success[column]["median"]
            ).rjust(18)
            + fmt(
                failure[column]["median"]
            ).rjust(18)
        )

    print()
    print(
        "========== LOG COVERAGE TRANSPORT DECOMPOSITION =========="
    )
    print(
        "Use MEANS here because the per-sample log identity is additive."
    )

    log_suffixes = (
        "log_a_transport",
        "log_r_transport",
        "log_deficit_transport",
        "log_coverage_transport",
        "dose_pressure",
        "readout_pressure",
        "deficit_pressure",
        "log_coverage_reconstruction_error",
    )

    print(
        "metric".ljust(42)
        + "future success mean".rjust(22)
        + "future failure mean".rjust(22)
    )

    for suffix in log_suffixes:
        column = f"{target_method}__{suffix}"
        print(
            column.ljust(42)
            + fmt(
                success[column]["mean"],
                digits=6,
            ).rjust(22)
            + fmt(
                failure[column]["mean"],
                digits=6,
            ).rjust(22)
        )


def print_factor_replacement(
    summary: Dict,
):
    print()
    print(
        "========== FACTOR-REPLACEMENT COVERAGE AUDIT =========="
    )

    groups = summary[
        "factor_replacement_by_future_regmean_outcome"
    ]

    scenario_order = (
        "factor_local",
        "factor_a_only",
        "factor_r_only",
        "factor_d_only",
        "factor_a_r",
        "factor_a_d",
        "factor_r_d",
        "factor_full",
    )

    print(
        "scenario".ljust(20)
        + "succ C med".rjust(14)
        + "succ >1".rjust(12)
        + "fail C med".rjust(14)
        + "fail >1".rjust(12)
    )

    success = groups["future_regmean_success"]
    failure = groups["future_regmean_failure"]

    for scenario_name in scenario_order:
        success_item = success[scenario_name]
        failure_item = failure[scenario_name]

        print(
            scenario_name.ljust(20)
            + fmt(
                success_item["coverage"]["median"]
            ).rjust(14)
            + fmt(
                success_item["fraction_above_one"]
            ).rjust(12)
            + fmt(
                failure_item["coverage"]["median"]
            ).rjust(14)
            + fmt(
                failure_item["fraction_above_one"]
            ).rjust(12)
        )


def write_markdown_summary(
    summary: Dict,
    args,
    path: str,
):
    ref = args.reference_method
    target = args.target_method

    local = summary[
        "local_metrics_by_future_regmean_outcome"
    ]
    transport = summary[
        "regmean_transport_by_future_regmean_outcome"
    ]
    factor = summary[
        "factor_replacement_by_future_regmean_outcome"
    ]
    visibility = summary[
        "future_regmean_failure_visibility_from_local"
    ]

    local_success = local["future_regmean_success"]
    local_failure = local["future_regmean_failure"]

    transport_success = transport["future_regmean_success"]
    transport_failure = transport["future_regmean_failure"]

    lines = [
        "# KDR-DTK-FSB Paired Causal-Coverage Transport Audit",
        "",
        f"- Reference: `{ref}`",
        f"- Target transport context: `{target}`",
        f"- Samples: `{summary['counts']['all']}`",
        f"- Future RegMean success: `{summary['counts']['future_regmean_success']}`",
        f"- Future RegMean failure: `{summary['counts']['future_regmean_failure']}`",
        "",
        "## Local metrics grouped by future RegMean outcome",
        "",
        "| Metric | Future success median | Future failure median |",
        "|---|---:|---:|",
    ]

    for suffix in (
        "a_abs",
        "r",
        "r_spec",
        "m_off",
        "off_deficit",
        "gain_coverage",
    ):
        column = f"{ref}__{suffix}"
        lines.append(
            f"| `{suffix}` | "
            f"{fmt(local_success[column]['median'])} | "
            f"{fmt(local_failure[column]['median'])} |"
        )

    lines.extend([
        "",
        "## Local predictability of future RegMean failure",
        "",
        "| Metric | Orientation-free AUC | Failure direction |",
        "|---|---:|---|",
    ])

    for metric_name, item in visibility.items():
        lines.append(
            f"| `{metric_name}` | "
            f"{fmt(item['orientation_free_auc'])} | "
            f"{item['failure_direction']} |"
        )

    lines.extend([
        "",
        "## Paired RegMean transport",
        "",
        "| Metric | Future success median | Future failure median |",
        "|---|---:|---:|",
    ])

    for suffix in (
        "a_ratio",
        "r_ratio",
        "r_spec_ratio",
        "deficit_ratio",
        "coverage_ratio",
        "required_dose",
        "dose_reserve",
        "dose_shortfall",
    ):
        column = f"{target}__{suffix}"
        lines.append(
            f"| `{suffix}` | "
            f"{fmt(transport_success[column]['median'])} | "
            f"{fmt(transport_failure[column]['median'])} |"
        )

    lines.extend([
        "",
        "## Mean log-transport decomposition",
        "",
        "Per sample:",
        "",
        r"\[",
        r"\log(C_{RM}/C_{local})",
        r"=",
        r"\log(A_{RM}/A_{local})",
        r"+",
        r"\log(r_{RM}/r_{local})",
        r"-",
        r"\log(D_{RM}/D_{local}).",
        r"\]",
        "",
        "| Metric | Future success mean | Future failure mean |",
        "|---|---:|---:|",
    ])

    for suffix in (
        "log_a_transport",
        "log_r_transport",
        "log_deficit_transport",
        "log_coverage_transport",
        "dose_pressure",
        "readout_pressure",
        "deficit_pressure",
        "log_coverage_reconstruction_error",
    ):
        column = f"{target}__{suffix}"
        lines.append(
            f"| `{suffix}` | "
            f"{fmt(transport_success[column]['mean'], 6)} | "
            f"{fmt(transport_failure[column]['mean'], 6)} |"
        )

    lines.extend([
        "",
        "## Factor-replacement coverage audit",
        "",
        "| Scenario | Future success C median | Future success C>1 | Future failure C median | Future failure C>1 |",
        "|---|---:|---:|---:|---:|",
    ])

    for scenario_name in (
        "factor_local",
        "factor_a_only",
        "factor_r_only",
        "factor_d_only",
        "factor_a_r",
        "factor_a_d",
        "factor_r_d",
        "factor_full",
    ):
        success_item = factor["future_regmean_success"][scenario_name]
        failure_item = factor["future_regmean_failure"][scenario_name]

        lines.append(
            f"| `{scenario_name}` | "
            f"{fmt(success_item['coverage']['median'])} | "
            f"{fmt(success_item['fraction_above_one'])} | "
            f"{fmt(failure_item['coverage']['median'])} | "
            f"{fmt(failure_item['fraction_above_one'])} |"
        )

    lines.extend([
        "",
        "## Interpretation guide",
        "",
        "### Branch A: local weakness is already visible",
        "",
        "If future RegMean failures already have substantially lower local coverage and local coverage has strong failure-prediction AUC, the attack proxy already exposes fragile samples. The next method may use sample-level causal-coverage-aware binding without modeling RegMean.",
        "",
        "### Branch B: failure is created by differential transport",
        "",
        "If local success/failure groups are similar but RegMean failure shows lower `A_ratio`, larger `deficit_ratio`, and much lower `coverage_ratio`, then failure is created during merge transport. The next method must model joint causal-coverage contraction rather than merely weighting low-local-coverage samples.",
        "",
        "### Factor replacement",
        "",
        "- `factor_a_only`: only RegMean dose is substituted into the local coverage equation.",
        "- `factor_r_only`: only RegMean readout is substituted.",
        "- `factor_d_only`: only RegMean deficit is substituted.",
        "- `factor_a_d`: RegMean dose and deficit are substituted jointly while local readout is retained.",
        "- `factor_full`: measured RegMean `A*r/D`; this should reproduce measured RegMean coverage up to numerical tolerance.",
        "",
        "The central comparison for future RegMean failures is whether `factor_a_d` already collapses coverage below 1 while `factor_r_only` does not.",
    ])

    Path(path).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()

    reference_method = args.reference_method.strip().lower()
    target_method = args.target_method.strip().lower()
    comparators = [
        item.strip().lower()
        for item in args.comparators.split(",")
        if item.strip()
    ]

    methods = [reference_method] + comparators + [target_method]
    methods = list(dict.fromkeys(methods))

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(
            f"Input CSV not found: {args.csv}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = normalize_method_names(df)

    validate_input(df, methods)

    paired = build_paired_frame(df, methods)

    for method in comparators + [target_method]:
        add_transport_columns(
            paired,
            reference_method=reference_method,
            method=method,
            eps=args.eps,
        )

    summary = build_summary(
        paired,
        reference_method=reference_method,
        target_method=target_method,
        comparators=comparators,
        eps=args.eps,
    )

    print_local_visibility(
        summary,
        reference_method,
    )
    print_transport(
        summary,
        target_method,
    )
    print_factor_replacement(summary)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(
        args.out_dir,
        f"{timestamp}_fsb_paired_transport_samples.csv",
    )
    json_path = os.path.join(
        args.out_dir,
        f"{timestamp}_fsb_paired_transport.json",
    )
    md_path = os.path.join(
        args.out_dir,
        f"{timestamp}_fsb_paired_transport_summary.md",
    )

    paired.to_csv(
        csv_path,
        index=False,
    )

    output = {
        "timestamp": timestamp,
        "question": (
            "Are future RegMean failures already low-coverage samples "
            "in the local attack model, or are they created by "
            "merge-induced differential contraction of causal coverage?"
        ),
        "input_csv": args.csv,
        "reference_method": reference_method,
        "target_method": target_method,
        "comparators": comparators,
        "eps": args.eps,
        "coverage_identity": (
            "For samples with M_off < 0 and positive g_key, "
            "coverage = g_key / D = A * r / D."
        ),
        "log_transport_identity": (
            "log(C_target/C_reference) = "
            "log(A_target/A_reference) + "
            "log(r_target/r_reference) - "
            "log(D_target/D_reference)."
        ),
        "summary": summary,
        "paired_csv": csv_path,
        "markdown_summary": md_path,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_markdown_summary(
        summary,
        args,
        md_path,
    )

    print()
    print("========== DONE ==========")
    print(f"[Paired Transport] CSV: {csv_path}")
    print(f"[Paired Transport] JSON: {json_path}")
    print(f"[Paired Transport] Markdown: {md_path}")


if __name__ == "__main__":
    main()
