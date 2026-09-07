import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

sys.path.append('.')
sys.path.append('./src')

from diagnose_kdr_key_causality import (
    build_adamerging_encoder,
    build_linear_merge_encoder,
    build_regmean_encoder,
    clean_checkpoint_path,
    estimate_global_key,
    forward_capture,
    forward_with_delta,
    get_classification_head,
    get_dataset,
    kdr_checkpoint_path,
    load_encoder,
    load_trigger_patch,
    make_random_orthogonal_direction,
    make_runtime_args,
    maybe_dictionarize,
    parse_methods,
    pretrained_path,
    require_paths,
    resolve_layer_name,
    set_seed,
    trigger_path,
)
from src.kdr_utils import apply_trigger


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Measure whether RegMean failures come from weaker DTK key '
            'amplitude or weaker target-margin readout per unit key signal.'
        )
    )

    parser.add_argument('--model', default='ViT-B-32')
    parser.add_argument('--ckpt-dir', default='./checkpoints')
    parser.add_argument('--data-location', default='./data')
    parser.add_argument('--adversary-task', default='CIFAR100')
    parser.add_argument('--target-task', default='CIFAR100')
    parser.add_argument('--target-cls', type=int, default=1)
    parser.add_argument('--patch-size', type=int, default=22)

    parser.add_argument('--attack-type', default='KDR_DTK_BIND')
    parser.add_argument('--trigger-source', default='KDR_DTK')

    parser.add_argument(
        '--key-layer',
        default='model.visual.transformer.resblocks.11.ln_2',
    )
    parser.add_argument('--pool', default='cls', choices=['cls'])

    parser.add_argument('--prototype-batches', type=int, default=30)
    parser.add_argument('--eval-batches', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--scaling-coef', type=float, default=0.3)
    parser.add_argument('--ties-reset-thresh', type=float, default=20)
    parser.add_argument('--ties-merge-func', default='dis-sum')
    parser.add_argument('--regmean-train-batches', type=int, default=8)
    parser.add_argument(
        '--adamerging-lambda',
        default=(
            './ada/ViT-B-32/'
            'KDR_CIFAR100_Tgt_1_L_22_Epoch_500.pt'
        ),
    )

    parser.add_argument(
        '--methods',
        default='local_attack,ta,ties,regmean',
        help=(
            'Comma-separated contexts. Supported values are inherited '
            'from diagnose_kdr_key_causality.py.'
        ),
    )

    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--seed', type=int, default=20260705)
    parser.add_argument(
        '--out-dir',
        default='./analysis/kdr_dtk_bind_key_readout_efficiency',
    )

    return parser.parse_args()


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    max_non_target = masked.max(dim=1).values
    return target - max_non_target


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'q10': None,
            'q25': None,
            'q75': None,
            'q90': None,
            'std': None,
        }

    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'q10': float(np.quantile(values, 0.10)),
        'q25': float(np.quantile(values, 0.25)),
        'q75': float(np.quantile(values, 0.75)),
        'q90': float(np.quantile(values, 0.90)),
        'std': float(values.std()),
    }


def safe_ratio(numerator, denominator, eps=1e-12):
    if numerator is None or denominator is None:
        return None

    if abs(float(denominator)) <= eps:
        return None

    return float(numerator) / float(denominator)


def summarize_group(rows: List[Dict[str, float]]) -> Dict:
    metrics = (
        'key_amplitude_abs',
        'key_coefficient_signed',
        'key_cosine',
        'key_energy_ratio',
        'full_margin',
        'key_removed_margin',
        'random_removed_margin',
        'causal_margin_gain',
        'random_margin_gain',
        'specific_margin_gain',
        'readout_efficiency',
        'specific_readout_efficiency',
    )

    output = {
        'count': len(rows),
        'target_rate': (
            100.0 * float(
                np.mean([row['success'] for row in rows])
            )
            if rows
            else None
        ),
        'metrics': {},
    }

    for metric in metrics:
        output['metrics'][metric] = summarize_array(
            np.asarray(
                [row[metric] for row in rows],
                dtype=np.float64,
            )
        )

    return output


def summarize_method_rows(rows: List[Dict[str, float]]) -> Dict:
    success_rows = [
        row for row in rows
        if int(row['success']) == 1
    ]
    failure_rows = [
        row for row in rows
        if int(row['success']) == 0
    ]

    all_summary = summarize_group(rows)
    success_summary = summarize_group(success_rows)
    failure_summary = summarize_group(failure_rows)

    compare_metrics = (
        'key_amplitude_abs',
        'causal_margin_gain',
        'specific_margin_gain',
        'readout_efficiency',
        'specific_readout_efficiency',
    )

    comparisons = {}

    for metric in compare_metrics:
        success_median = success_summary[
            'metrics'
        ][metric]['median']
        failure_median = failure_summary[
            'metrics'
        ][metric]['median']

        comparisons[metric] = {
            'success_median': success_median,
            'failure_median': failure_median,
            'failure_minus_success': (
                None
                if (
                    success_median is None
                    or failure_median is None
                )
                else float(
                    failure_median - success_median
                )
            ),
            'failure_over_success': safe_ratio(
                failure_median,
                success_median,
            ),
        }

    return {
        'all': all_summary,
        'success': success_summary,
        'failure': failure_summary,
        'success_vs_failure': comparisons,
    }


def evaluate_encoder_readout(
    args,
    method_name,
    encoder,
    head,
    trigger,
    key,
    random_direction,
    device,
):
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()

    resolved_layer = resolve_layer_name(
        encoder,
        args.key_layer,
    )

    _, loader = get_dataset(
        args.target_task,
        'test',
        encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    rows = []
    sample_index = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm.tqdm(
                loader,
                desc=f'key-readout-{method_name}',
            )
        ):
            if batch_idx >= args.eval_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            non_target = labels != args.target_cls
            images = images[non_target]

            if images.shape[0] == 0:
                continue

            patched = apply_trigger(images, trigger)
            combined = torch.cat(
                [images, patched],
                dim=0,
            )

            logits_combined, activation = forward_capture(
                encoder,
                head,
                combined,
                resolved_layer,
            )

            batch_size = images.shape[0]
            triggered_logits = logits_combined[batch_size:]

            clean_act = activation[:batch_size]
            triggered_act = activation[batch_size:]

            shift = triggered_act - clean_act
            shift_norm = shift.norm(dim=-1).clamp_min(args.eps)

            shift_hat = F.normalize(
                shift,
                dim=-1,
                eps=args.eps,
            )

            key_coeff = torch.matmul(
                shift,
                key,
            )
            key_amp = key_coeff.abs()

            key_cosine = torch.matmul(
                shift_hat,
                key,
            )

            key_energy_ratio = (
                key_coeff.square()
                / shift_norm.square()
            )

            key_component = (
                key_coeff.unsqueeze(1)
                * key.unsqueeze(0)
            )

            random_equal_energy = (
                key_coeff.unsqueeze(1)
                * random_direction.unsqueeze(0)
            )

            key_removed_logits = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                -key_component,
            )

            random_removed_logits = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                -random_equal_energy,
            )

            full_margin = target_margin(
                triggered_logits,
                args.target_cls,
            )

            key_removed_margin = target_margin(
                key_removed_logits,
                args.target_cls,
            )

            random_removed_margin = target_margin(
                random_removed_logits,
                args.target_cls,
            )

            causal_gain = (
                full_margin - key_removed_margin
            )

            random_gain = (
                full_margin - random_removed_margin
            )

            specific_gain = (
                causal_gain - random_gain
            )

            readout_efficiency = (
                causal_gain
                / (key_amp + args.eps)
            )

            specific_readout_efficiency = (
                specific_gain
                / (key_amp + args.eps)
            )

            success = (
                triggered_logits.argmax(dim=1)
                == args.target_cls
            )

            for idx in range(batch_size):
                rows.append(
                    {
                        'method': method_name,
                        'sample_index': sample_index,
                        'success': int(success[idx].item()),
                        'key_amplitude_abs': float(
                            key_amp[idx].item()
                        ),
                        'key_coefficient_signed': float(
                            key_coeff[idx].item()
                        ),
                        'key_cosine': float(
                            key_cosine[idx].item()
                        ),
                        'key_energy_ratio': float(
                            key_energy_ratio[idx].item()
                        ),
                        'full_margin': float(
                            full_margin[idx].item()
                        ),
                        'key_removed_margin': float(
                            key_removed_margin[idx].item()
                        ),
                        'random_removed_margin': float(
                            random_removed_margin[idx].item()
                        ),
                        'causal_margin_gain': float(
                            causal_gain[idx].item()
                        ),
                        'random_margin_gain': float(
                            random_gain[idx].item()
                        ),
                        'specific_margin_gain': float(
                            specific_gain[idx].item()
                        ),
                        'readout_efficiency': float(
                            readout_efficiency[idx].item()
                        ),
                        'specific_readout_efficiency': float(
                            specific_readout_efficiency[idx].item()
                        ),
                    }
                )

                sample_index += 1

    if not rows:
        raise RuntimeError(
            f'No evaluation rows collected for {method_name}'
        )

    return rows, summarize_method_rows(rows)


def print_group_line(
    label,
    summary,
):
    metrics = summary['metrics']

    def med(name):
        value = metrics[name]['median']
        return (
            float('nan')
            if value is None
            else value
        )

    print(
        label.ljust(18)
        + f"{summary['count']:8d}"
        + f"{med('key_amplitude_abs'):12.4f}"
        + f"{med('causal_margin_gain'):12.4f}"
        + f"{med('specific_margin_gain'):12.4f}"
        + f"{med('readout_efficiency'):12.4f}"
        + f"{med('specific_readout_efficiency'):14.4f}"
        + f"{med('full_margin'):12.4f}"
    )


def print_summary_table(results):
    print()
    print('========== KEY READOUT EFFICIENCY ==========')

    header = (
        'group'.ljust(18)
        + 'N'.rjust(8)
        + '|a| med'.rjust(12)
        + 'g_key med'.rjust(12)
        + 'g_spec med'.rjust(12)
        + 'r med'.rjust(12)
        + 'r_spec med'.rjust(14)
        + 'M_full med'.rjust(12)
    )

    print(header)

    for method_name, result in results.items():
        print()
        print(f'----- {method_name} -----')

        print_group_line(
            'all',
            result['all'],
        )
        print_group_line(
            'success',
            result['success'],
        )
        print_group_line(
            'failure',
            result['failure'],
        )

        print('success-vs-failure comparisons:')

        for metric, comparison in result[
            'success_vs_failure'
        ].items():
            print(
                f"  {metric}: "
                f"success_med={comparison['success_median']} "
                f"failure_med={comparison['failure_median']} "
                f"failure-success="
                f"{comparison['failure_minus_success']} "
                f"failure/success="
                f"{comparison['failure_over_success']}"
            )


def write_csv(rows, path):
    if not rows:
        raise RuntimeError('Cannot write empty CSV')

    fieldnames = list(rows[0].keys())

    with open(
        path,
        'w',
        newline='',
        encoding='utf-8',
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)

    methods = parse_methods(args)
    device = torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )

    runtime_args = make_runtime_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    required = [
        pretrained_path(args),
        clean_checkpoint_path(
            args,
            args.adversary_task,
        ),
        kdr_checkpoint_path(args),
        trigger_path(args),
    ]

    if 'adamerging' in methods:
        required.append(args.adamerging_lambda)

    required += [
        clean_checkpoint_path(
            args,
            dataset_name,
        )
        for dataset_name in runtime_args.dataset_list
        if dataset_name != args.adversary_task
    ]

    require_paths(required)

    trigger = load_trigger_patch(
        trigger_path(args),
        args.patch_size,
        device,
    )

    print('[Readout] estimating empirical Stage-1 key')

    pretrained_encoder = load_encoder(
        pretrained_path(args),
        device,
    )

    key, prototype_summary = estimate_global_key(
        args,
        pretrained_encoder,
        trigger,
        device,
    )

    key = key.to(device)

    random_direction = make_random_orthogonal_direction(
        key,
        seed=args.seed + 17,
    )

    del pretrained_encoder
    torch.cuda.empty_cache()

    head = get_classification_head(
        runtime_args,
        args.target_task,
    ).to(device).eval()

    for param in head.parameters():
        param.requires_grad_(False)

    builder_map = {
        'clean_local': lambda: load_encoder(
            clean_checkpoint_path(
                args,
                args.adversary_task,
            ),
            device,
        ),
        'local_attack': lambda: load_encoder(
            kdr_checkpoint_path(args),
            device,
        ),
        'ta': lambda: build_linear_merge_encoder(
            args,
            'ta',
            device,
        ),
        'ties': lambda: build_linear_merge_encoder(
            args,
            'ties',
            device,
        ),
        'regmean': lambda: build_regmean_encoder(
            args,
            runtime_args,
            device,
        ),
        'adamerging': lambda: build_adamerging_encoder(
            args,
            device,
        ),
    }

    builders = OrderedDict(
        (
            method,
            builder_map[method],
        )
        for method in methods
    )

    results = OrderedDict()
    all_rows = []

    for method_name, builder in builders.items():
        print()
        print(f'========== BUILD {method_name} ==========')

        encoder = builder()

        rows, summary = evaluate_encoder_readout(
            args=args,
            method_name=method_name,
            encoder=encoder,
            head=head,
            trigger=trigger,
            key=key,
            random_direction=random_direction,
            device=device,
        )

        all_rows.extend(rows)
        results[method_name] = summary

        del encoder
        torch.cuda.empty_cache()

    print_summary_table(results)

    timestamp = time.strftime('%Y%m%d_%H%M%S')

    csv_path = os.path.join(
        args.out_dir,
        f'{timestamp}_kdr_key_readout_efficiency_samples.csv',
    )

    json_path = os.path.join(
        args.out_dir,
        f'{timestamp}_kdr_key_readout_efficiency.json',
    )

    write_csv(
        all_rows,
        csv_path,
    )

    output = {
        'timestamp': timestamp,
        'question': (
            'Do RegMean failures come from weaker DTK key amplitude, '
            'or from weaker target-margin readout per unit key signal?'
        ),
        'model': args.model,
        'attack_type': args.attack_type,
        'trigger_source': args.trigger_source,
        'methods': methods,
        'adversary_task': args.adversary_task,
        'target_task': args.target_task,
        'target_cls': args.target_cls,
        'patch_size': args.patch_size,
        'requested_key_layer': args.key_layer,
        'prototype_batches': args.prototype_batches,
        'eval_batches': args.eval_batches,
        'batch_size': args.batch_size,
        'eps': args.eps,
        'trigger_path': trigger_path(args),
        'attack_checkpoint': kdr_checkpoint_path(args),
        'stage1_key_stability': prototype_summary,
        'definitions': {
            'key_amplitude_abs': (
                '|a_i| = |d_i^T k|'
            ),
            'causal_margin_gain': (
                'g_i = M_full - M_key_removed'
            ),
            'random_margin_gain': (
                'M_full - M_random_equal_energy_removed'
            ),
            'specific_margin_gain': (
                'causal_margin_gain - random_margin_gain'
            ),
            'readout_efficiency': (
                'r_i = causal_margin_gain / (|a_i| + eps)'
            ),
            'specific_readout_efficiency': (
                'r_spec_i = specific_margin_gain / (|a_i| + eps)'
            ),
            'success': (
                'full triggered prediction equals target class'
            ),
        },
        'results': results,
        'sample_csv': csv_path,
    }

    with open(
        json_path,
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print('========== DONE ==========')
    print(f'[Readout] CSV: {csv_path}')
    print(f'[Readout] JSON: {json_path}')


if __name__ == '__main__':
    main()
