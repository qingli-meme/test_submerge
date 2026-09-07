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
            'Audit whether KDR-DTK-SCB learns to decode the frozen '
            'Stage-1 sender dose and whether the real triggered '
            'branch remains valid after sender-dose calibration.'
        )
    )

    parser.add_argument('--model', default='ViT-B-32')
    parser.add_argument('--ckpt-dir', default='./checkpoints')
    parser.add_argument('--data-location', default='./data')
    parser.add_argument('--adversary-task', default='CIFAR100')
    parser.add_argument('--target-task', default='CIFAR100')
    parser.add_argument('--target-cls', type=int, default=1)
    parser.add_argument('--patch-size', type=int, default=22)

    parser.add_argument('--attack-type', default='KDR_DTK_SCB')
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
        default='./analysis/kdr_dtk_scb_sender_calibration',
    )

    return parser.parse_args()


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    max_non_target = masked.max(dim=1).values
    return target - max_non_target


def summarize(values: np.ndarray) -> Dict[str, float]:
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


def summarize_rows(rows: List[Dict[str, float]]) -> Dict:
    metrics = (
        'sender_key_coeff',
        'sender_key_coeff_abs',
        'current_key_coeff',
        'current_key_coeff_abs',
        'current_sender_ratio',
        'full_margin',
        'sender_margin',
        'off_margin',
        'sender_minus_full_margin',
        'full_minus_off_margin',
        'sender_minus_off_margin',
    )

    output = {
        'count': len(rows),
        'full_target_rate': (
            100.0 * float(np.mean([row['full_success'] for row in rows]))
            if rows else None
        ),
        'sender_target_rate': (
            100.0 * float(np.mean([row['sender_success'] for row in rows]))
            if rows else None
        ),
        'off_target_rate': (
            100.0 * float(np.mean([row['off_success'] for row in rows]))
            if rows else None
        ),
        'metrics': {},
    }

    for metric in metrics:
        output['metrics'][metric] = summarize(
            np.asarray(
                [row[metric] for row in rows],
                dtype=np.float64,
            )
        )

    return output


def summarize_method(rows: List[Dict[str, float]]) -> Dict:
    full_success = [
        row for row in rows
        if int(row['full_success']) == 1
    ]
    full_failure = [
        row for row in rows
        if int(row['full_success']) == 0
    ]

    return {
        'all': summarize_rows(rows),
        'full_success': summarize_rows(full_success),
        'full_failure': summarize_rows(full_failure),
    }


def evaluate_context(
    args,
    method_name,
    encoder,
    pretrained_encoder,
    head,
    trigger,
    key,
    device,
):
    encoder = encoder.to(device).eval()
    pretrained_encoder = pretrained_encoder.to(device).eval()
    head = head.to(device).eval()

    resolved_layer = resolve_layer_name(
        encoder,
        args.key_layer,
    )
    sender_layer = resolve_layer_name(
        pretrained_encoder,
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
                desc=f'sender-calibration-{method_name}',
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

            sender_logits_combined, sender_act = forward_capture(
                pretrained_encoder,
                head,
                combined,
                sender_layer,
            )

            logits_combined, current_act = forward_capture(
                encoder,
                head,
                combined,
                resolved_layer,
            )

            batch_size = images.shape[0]

            sender_clean_act = sender_act[:batch_size]
            sender_triggered_act = sender_act[batch_size:]

            current_clean_act = current_act[:batch_size]
            current_triggered_act = current_act[batch_size:]

            full_logits = logits_combined[batch_size:]

            sender_shift = (
                sender_triggered_act
                - sender_clean_act
            )
            current_shift = (
                current_triggered_act
                - current_clean_act
            )

            sender_coeff = torch.matmul(
                sender_shift,
                key,
            )
            current_coeff = torch.matmul(
                current_shift,
                key,
            )

            sender_component = (
                sender_coeff.unsqueeze(1)
                * key.unsqueeze(0)
            )
            current_component = (
                current_coeff.unsqueeze(1)
                * key.unsqueeze(0)
            )

            off_logits = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                -current_component,
            )

            sender_calibrated_logits = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                sender_component - current_component,
            )

            full_margin = target_margin(
                full_logits,
                args.target_cls,
            )
            sender_margin = target_margin(
                sender_calibrated_logits,
                args.target_cls,
            )
            off_margin = target_margin(
                off_logits,
                args.target_cls,
            )

            full_success = (
                full_logits.argmax(dim=1)
                == args.target_cls
            )
            sender_success = (
                sender_calibrated_logits.argmax(dim=1)
                == args.target_cls
            )
            off_success = (
                off_logits.argmax(dim=1)
                == args.target_cls
            )

            sender_abs = sender_coeff.abs()
            current_abs = current_coeff.abs()

            ratio = (
                current_abs
                / sender_abs.clamp_min(args.eps)
            )

            for idx in range(batch_size):
                rows.append(
                    {
                        'method': method_name,
                        'sample_index': sample_index,
                        'full_success': int(
                            full_success[idx].item()
                        ),
                        'sender_success': int(
                            sender_success[idx].item()
                        ),
                        'off_success': int(
                            off_success[idx].item()
                        ),
                        'sender_key_coeff': float(
                            sender_coeff[idx].item()
                        ),
                        'sender_key_coeff_abs': float(
                            sender_abs[idx].item()
                        ),
                        'current_key_coeff': float(
                            current_coeff[idx].item()
                        ),
                        'current_key_coeff_abs': float(
                            current_abs[idx].item()
                        ),
                        'current_sender_ratio': float(
                            ratio[idx].item()
                        ),
                        'full_margin': float(
                            full_margin[idx].item()
                        ),
                        'sender_margin': float(
                            sender_margin[idx].item()
                        ),
                        'off_margin': float(
                            off_margin[idx].item()
                        ),
                        'sender_minus_full_margin': float(
                            (
                                sender_margin[idx]
                                - full_margin[idx]
                            ).item()
                        ),
                        'full_minus_off_margin': float(
                            (
                                full_margin[idx]
                                - off_margin[idx]
                            ).item()
                        ),
                        'sender_minus_off_margin': float(
                            (
                                sender_margin[idx]
                                - off_margin[idx]
                            ).item()
                        ),
                    }
                )
                sample_index += 1

    if not rows:
        raise RuntimeError(
            f'No rows collected for {method_name}'
        )

    return rows, summarize_method(rows)


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


def print_group(label, summary):
    metrics = summary['metrics']

    def med(name):
        value = metrics[name]['median']
        return float('nan') if value is None else value

    print(
        label.ljust(18)
        + f"{summary['count']:8d}"
        + f"{summary['full_target_rate']:12.2f}"
        + f"{summary['sender_target_rate']:14.2f}"
        + f"{summary['off_target_rate']:12.2f}"
        + f"{med('sender_key_coeff_abs'):12.4f}"
        + f"{med('current_key_coeff_abs'):12.4f}"
        + f"{med('current_sender_ratio'):12.4f}"
        + f"{med('full_margin'):12.4f}"
        + f"{med('sender_margin'):14.4f}"
        + f"{med('off_margin'):12.4f}"
    )


def print_results(results):
    print()
    print('========== SENDER CALIBRATION AUDIT ==========')

    print(
        'group'.ljust(18)
        + 'N'.rjust(8)
        + 'full ASR'.rjust(12)
        + 'sender ASR'.rjust(14)
        + 'off ASR'.rjust(12)
        + '|a0| med'.rjust(12)
        + '|am| med'.rjust(12)
        + 'am/a0 med'.rjust(12)
        + 'M full'.rjust(12)
        + 'M sender'.rjust(14)
        + 'M off'.rjust(12)
    )

    for method_name, result in results.items():
        print()
        print(f'----- {method_name} -----')
        print_group('all', result['all'])
        print_group(
            'full_success',
            result['full_success'],
        )
        print_group(
            'full_failure',
            result['full_failure'],
        )


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

    pretrained_encoder = load_encoder(
        pretrained_path(args),
        device,
    )

    print(
        '[SCB Audit] estimating empirical Stage-1 key',
        flush=True,
    )

    key, prototype_summary = estimate_global_key(
        args,
        pretrained_encoder,
        trigger,
        device,
    )

    key = key.to(device)

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
        print(
            f'========== BUILD {method_name} ==========',
            flush=True,
        )

        encoder = builder()

        rows, summary = evaluate_context(
            args=args,
            method_name=method_name,
            encoder=encoder,
            pretrained_encoder=pretrained_encoder,
            head=head,
            trigger=trigger,
            key=key,
            device=device,
        )

        all_rows.extend(rows)
        results[method_name] = summary

        del encoder
        torch.cuda.empty_cache()

    print_results(results)

    timestamp = time.strftime('%Y%m%d_%H%M%S')

    csv_path = os.path.join(
        args.out_dir,
        f'{timestamp}_kdr_dtk_scb_sender_calibration_samples.csv',
    )

    json_path = os.path.join(
        args.out_dir,
        f'{timestamp}_kdr_dtk_scb_sender_calibration.json',
    )

    write_csv(
        all_rows,
        csv_path,
    )

    output = {
        'timestamp': timestamp,
        'question': (
            'After sender-calibrated binding, does the receiver decode '
            'the frozen Stage-1 native key dose, and does the real '
            'full-trigger branch remain target-effective?'
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
            'sender_key_coeff': (
                'a0 = (h_theta0(T(x)) - h_theta0(x))^T k'
            ),
            'current_key_coeff': (
                'am = (h_m(T(x)) - h_m(x))^T k'
            ),
            'sender_calibrated_branch': (
                'h_sender = h_m(T(x)) - am*k + a0*k'
            ),
            'off_branch': (
                'h_off = h_m(T(x)) - am*k'
            ),
            'current_sender_ratio': (
                '|am| / (|a0| + eps)'
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
    print(f'[SCB Audit] CSV: {csv_path}')
    print(f'[SCB Audit] JSON: {json_path}')


if __name__ == '__main__':
    main()
