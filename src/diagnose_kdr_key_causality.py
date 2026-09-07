import argparse
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

sys.path.append('.')
sys.path.append('./src')

from task_vectors import TaskVector
from ties_merging_utils import (
    check_parameterNamesMatch,
    state_dict_to_vector,
    ties_merging,
    vector_to_state_dict,
)
from regmean import RegMean
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.kdr_utils import apply_trigger, load_trigger_patch, pool_activation
from src.modeling import ImageClassifier


_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat


EXAM_DATASETS = [
    'CIFAR100',
    'GTSRB',
    'EuroSAT',
    'Cars',
    'SUN397',
    'PETS',
]


class DiagArgs:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Causal audit of whether KDR Stage 2 actually consumes '
            'the representation key constructed in Stage 1.'
        )
    )
    parser.add_argument('--model', default='ViT-B-32')
    parser.add_argument('--ckpt-dir', default='./checkpoints')
    parser.add_argument('--data-location', default='./data')
    parser.add_argument('--adversary-task', default='CIFAR100')
    parser.add_argument('--target-task', default='CIFAR100')
    parser.add_argument('--target-cls', type=int, default=1)
    parser.add_argument('--patch-size', type=int, default=22)
    parser.add_argument(
        '--attack-type',
        default='KDR',
        help=(
            'Attack checkpoint prefix. Examples: KDR, KDR_DTK. '
            'BadMerging-On can be passed as BadMergingOn or On.'
        ),
    )
    parser.add_argument(
        '--trigger-source',
        default='attack',
        help=(
            'Trigger prefix. Default attack means use --attack-type. '
            'Examples: KDR, KDR_DTK, BadMergingOn, On.'
        ),
    )
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
        default='clean_local,local_attack,ta,ties,regmean,adamerging',
        help=(
            'Comma-separated audit contexts. Supported: clean_local, '
            'local_attack, ta, ties, regmean, adamerging.'
        ),
    )
    parser.add_argument('--seed', type=int, default=20260705)
    parser.add_argument(
        '--out-dir',
        default='./analysis/kdr_key_causality',
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_runtime_args(args):
    rt = DiagArgs()
    rt.model = args.model
    rt.ckpt_dir = args.ckpt_dir
    rt.data_location = args.data_location
    rt.batch_size = args.batch_size
    rt.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rt.cache_dir = ''
    rt.openclip_cachedir = './open_clip'
    rt.save = os.path.join(args.ckpt_dir, args.model)
    rt.dataset_list = list(EXAM_DATASETS)
    rt.num_train_batch = args.regmean_train_batches
    return rt


def pretrained_path(args):
    return os.path.join(
        args.ckpt_dir,
        args.model,
        'zeroshot.pt',
    )


def clean_checkpoint_path(args, dataset_name):
    return os.path.join(
        args.ckpt_dir,
        args.model,
        dataset_name,
        'finetuned.pt',
    )


def normalize_attack_prefix(prefix):
    if prefix in ('BadMergingOn', 'On'):
        return 'On'
    return prefix


def attack_suffix(args, prefix=None):
    prefix = normalize_attack_prefix(prefix or args.attack_type)
    return (
        f'{prefix}_{args.adversary_task}_Tgt_'
        f'{args.target_cls}_L_{args.patch_size}'
    )


def attack_checkpoint_path_for_prefix(args, prefix):
    return os.path.join(
        args.ckpt_dir,
        args.model,
        (
            f'{args.adversary_task}_'
            f'{attack_suffix(args, prefix)}'
        ),
        'finetuned.pt',
    )


def kdr_checkpoint_path(args):
    return attack_checkpoint_path_for_prefix(args, args.attack_type)


def trigger_path(args):
    source = args.attack_type if args.trigger_source == 'attack' else args.trigger_source
    source = normalize_attack_prefix(source)
    if source == 'fixed':
        name = f'fixed_{args.patch_size}.npy'
    else:
        name = f'{attack_suffix(args, source)}.npy'
    return os.path.join(
        './trigger',
        args.model,
        name,
    )


def attack_checkpoint_path(args, dataset_name):
    if dataset_name == args.adversary_task:
        return kdr_checkpoint_path(args)
    return clean_checkpoint_path(args, dataset_name)


def parse_methods(args):
    valid = {
        'clean_local',
        'local_attack',
        'ta',
        'ties',
        'regmean',
        'adamerging',
    }
    methods = [item.strip() for item in args.methods.split(',') if item.strip()]
    unknown = [item for item in methods if item not in valid]
    if unknown:
        raise ValueError(f'Unsupported --methods entries: {unknown}')
    return methods


def require_paths(paths: Iterable[str]) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            'Missing required files:\n' + '\n'.join(missing)
        )


def get_state_dict(path):
    obj = torch.load(path, map_location='cpu')
    if hasattr(obj, 'state_dict'):
        return obj.state_dict()
    if isinstance(obj, Mapping):
        if 'state_dict' in obj and isinstance(obj['state_dict'], Mapping):
            return obj['state_dict']
        return obj
    raise TypeError(
        f'Unsupported checkpoint object: {path}, {type(obj)}'
    )


def load_encoder(path, device):
    encoder = torch.load(path, map_location='cpu')
    encoder = encoder.to(device).eval()
    return encoder


def build_linear_merge_encoder(args, method_name, device):
    ptm_path = pretrained_path(args)
    ptm_sd = get_state_dict(ptm_path)

    ft_sds = [
        get_state_dict(attack_checkpoint_path(args, dataset_name))
        for dataset_name in EXAM_DATASETS
    ]
    check_parameterNamesMatch(ft_sds + [ptm_sd])

    flat_ptm = state_dict_to_vector(ptm_sd, [])
    flat_ft = torch.vstack(
        [state_dict_to_vector(sd, []) for sd in ft_sds]
    )
    task_vectors = flat_ft - flat_ptm

    if method_name == 'ta':
        merged = (
            flat_ptm
            + args.scaling_coef * task_vectors.sum(dim=0)
        )
    elif method_name == 'ties':
        merged_tv = ties_merging(
            task_vectors,
            reset_thresh=args.ties_reset_thresh,
            merge_func=args.ties_merge_func,
        )
        merged = flat_ptm + args.scaling_coef * merged_tv
    else:
        raise ValueError(f'Unsupported merge method: {method_name}')

    state = vector_to_state_dict(merged, ptm_sd, [])
    encoder = torch.load(ptm_path, map_location='cpu')
    encoder.load_state_dict(state, strict=False)
    return encoder.to(device).eval()


def build_regmean_encoder(args, runtime_args, device):
    runtime_args.dataset_list = list(EXAM_DATASETS)
    runtime_args.num_train_batch = args.regmean_train_batches
    regmean = RegMean(runtime_args, None)
    exp_name = attack_suffix(args)

    with torch.no_grad():
        gram_list = []
        model_list = []
        for dataset_name in EXAM_DATASETS:
            task_path = clean_checkpoint_path(args, dataset_name)
            if dataset_name == args.adversary_task:
                task_path = os.path.join(
                    args.ckpt_dir,
                    args.model,
                    f'{dataset_name}_{exp_name}',
                    'finetuned.pt',
                )
            print(task_path)

            image_encoder = torch.load(task_path, map_location=device)
            classification_head = regmean.class_head_dict[dataset_name]
            model = ImageClassifier(image_encoder, classification_head)
            model.freeze_head()
            model = model.to(device)
            model_list.append(model)
            gram_list.append(regmean.compute_gram(model, dataset_name))

        regmean_avg_params = regmean.avg_merge(
            model_list,
            regmean_grams=gram_list,
        )

        base_encoder = torch.load(pretrained_path(args), map_location=device)
        final_head = regmean.class_head_dict[EXAM_DATASETS[-1]]
        final_model = ImageClassifier(base_encoder, final_head)
        final_model.freeze_head()
        final_model = final_model.to(device)
        regmean.copy_params_to_model(regmean_avg_params, final_model)
        return final_model.image_encoder.to(device).eval()


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, value):
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        set_attr(getattr(obj, names[0]), names[1:], value)


def make_functional(module):
    original_params = tuple(module.parameters())
    names = []
    for name, _ in list(module.named_parameters()):
        del_attr(module, name.split('.'))
        names.append(name)
    return original_params, names


def load_weights(module, names, params):
    for name, param in zip(names, params):
        set_attr(module, name.split('.'), param)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train_preprocess = getattr(model, 'train_preprocess', None)
        self.val_preprocess = getattr(model, 'val_preprocess', None)
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        return self.model(images)


class FixedAdaMerger:
    def __init__(
        self,
        paramslist,
        model,
        names,
        task_lambdas_raw,
    ):
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.task_lambdas_raw = (
            task_lambdas_raw.detach().float().cpu()
        )

        expected_shape = (
            len(paramslist[0]),
            len(paramslist) - 1,
        )
        if tuple(self.task_lambdas_raw.shape) != expected_shape:
            raise RuntimeError(
                'Ada lambda shape mismatch: '
                f'got {tuple(self.task_lambdas_raw.shape)}, '
                f'expected {expected_shape}'
            )

        if len(self.names) != len(paramslist[0]):
            raise RuntimeError(
                'Parameter-name alignment mismatch: '
                f'names={len(self.names)}, '
                f'params={len(paramslist[0])}'
            )

    def lambdas(self):
        task_lambdas = torch.clamp(
            self.task_lambdas_raw,
            min=0.0,
            max=1.0,
        )
        pretrain_lambdas = torch.ones(
            len(self.paramslist[0]),
            1,
        )
        return torch.cat(
            (pretrain_lambdas, task_lambdas),
            dim=1,
        )

    def get_image_encoder(self, device):
        coefficients = self.lambdas()
        params = tuple(
            sum(
                tuple(
                    param_i * lambda_i
                    for param_i, lambda_i in zip(
                        param_group,
                        coefficients[param_idx].cpu(),
                    )
                )
            )
            for param_idx, param_group in enumerate(
                zip(*self.paramslist)
            )
        )
        params = tuple(param.to(device) for param in params)
        load_weights(self.model, self.names, params)
        return self.model.to(device).eval()


def build_adamerging_encoder(args, device):
    ptm_path = pretrained_path(args)
    pretrained_model = torch.load(ptm_path, map_location='cpu')
    pretrained_state = pretrained_model.state_dict()

    functional_model = ModelWrapper(pretrained_model).to(device)
    _, names = make_functional(functional_model)

    task_vectors = [
        TaskVector(
            ptm_path,
            attack_checkpoint_path(args, dataset_name),
        )
        for dataset_name in EXAM_DATASETS
    ]

    paramslist = [
        tuple(value.detach().cpu() for _, value in pretrained_state.items())
    ]
    paramslist += [
        tuple(value.detach().cpu() for _, value in task_vector.vector.items())
        for task_vector in task_vectors
    ]

    lambda_raw = torch.load(
        args.adamerging_lambda,
        map_location='cpu',
    )
    if isinstance(lambda_raw, torch.nn.Parameter):
        lambda_raw = lambda_raw.detach()

    merger = FixedAdaMerger(
        paramslist=paramslist,
        model=functional_model,
        names=names,
        task_lambdas_raw=lambda_raw,
    )
    return merger.get_image_encoder(device)


def resolve_layer_name(model, requested):
    modules = dict(model.named_modules())
    if requested in modules:
        return requested

    suffix = requested
    matches = [
        name
        for name in modules
        if name.endswith(suffix) or suffix.endswith(name)
    ]
    if len(matches) != 1:
        tail = '.'.join(requested.split('.')[-6:])
        matches = [
            name
            for name in modules
            if name.endswith(tail)
        ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot uniquely resolve layer '{requested}'. "
            f'Matches: {matches[:20]}'
        )
    return matches[0]


def _edit_tensor_cls(output, delta, batch_size):
    edited = output.clone()
    delta = delta.to(device=edited.device, dtype=edited.dtype)

    if edited.ndim == 2:
        if edited.shape[0] != batch_size:
            raise RuntimeError(
                f'2D activation shape mismatch: {tuple(edited.shape)}, '
                f'batch={batch_size}'
            )
        return edited + delta

    if edited.ndim != 3:
        raise RuntimeError(
            f'Expected 2D/3D activation, got {tuple(edited.shape)}'
        )

    if edited.shape[1] == batch_size:
        edited[0, :, :] = edited[0, :, :] + delta
        return edited

    if edited.shape[0] == batch_size:
        edited[:, 0, :] = edited[:, 0, :] + delta
        return edited

    raise RuntimeError(
        f'Cannot identify batch axis for activation '
        f'{tuple(edited.shape)} and batch={batch_size}'
    )


def edit_layer_output(output, delta, batch_size):
    if torch.is_tensor(output):
        return _edit_tensor_cls(output, delta, batch_size)

    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            raise RuntimeError('Unsupported tuple activation output')
        return (
            _edit_tensor_cls(output[0], delta, batch_size),
            *output[1:],
        )

    if isinstance(output, list):
        if not output or not torch.is_tensor(output[0]):
            raise RuntimeError('Unsupported list activation output')
        edited = list(output)
        edited[0] = _edit_tensor_cls(
            edited[0], delta, batch_size
        )
        return edited

    raise RuntimeError(
        f'Unsupported activation output type: {type(output)}'
    )


def forward_capture(
    encoder,
    head,
    images,
    resolved_layer_name,
):
    holder = {}
    module = dict(encoder.named_modules())[resolved_layer_name]

    def hook_fn(_, __, output):
        holder['activation'] = output

    handle = module.register_forward_hook(hook_fn)
    try:
        features = encoder(images)
        logits = head(features)
    finally:
        handle.remove()

    if 'activation' not in holder:
        raise RuntimeError(
            f'Hook failed to capture {resolved_layer_name}'
        )

    activation = pool_activation(
        holder['activation'],
        batch_size=images.shape[0],
        pool='cls',
    )
    return logits, activation


def forward_with_delta(
    encoder,
    head,
    images,
    resolved_layer_name,
    delta,
):
    module = dict(encoder.named_modules())[resolved_layer_name]
    batch_size = images.shape[0]

    def hook_fn(_, __, output):
        return edit_layer_output(
            output,
            delta=delta,
            batch_size=batch_size,
        )

    handle = module.register_forward_hook(hook_fn)
    try:
        features = encoder(images)
        logits = head(features)
    finally:
        handle.remove()
    return logits


def summarize(values):
    tensor = torch.cat(values, dim=0).float()
    return {
        'num_samples': int(tensor.numel()),
        'mean': float(tensor.mean().item()),
        'median': float(tensor.median().item()),
        'q10': float(torch.quantile(tensor, 0.10).item()),
        'q90': float(torch.quantile(tensor, 0.90).item()),
        'std': float(tensor.std(unbiased=False).item()),
    }


def summarize_tensor(tensor):
    return summarize([tensor.detach().float().cpu().reshape(-1)])


def target_arrays(logits, target_cls):
    target_logit = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    max_non_target = masked.max(dim=1).values
    margin = target_logit - max_non_target
    pred_target = (logits.argmax(dim=1) == target_cls).float()
    return {
        'target_logit': target_logit.detach().float().cpu(),
        'margin': margin.detach().float().cpu(),
        'pred_target': pred_target.detach().float().cpu(),
    }


def append_condition(store, name, arrays):
    for key, value in arrays.items():
        store[name][key].append(value)


def summarize_condition(store):
    output = {}
    for name, arrays in store.items():
        pred = torch.cat(arrays['pred_target'], dim=0)
        output[name] = {
            'target_rate': 100.0 * float(pred.mean().item()),
            'target_logit': summarize(arrays['target_logit']),
            'margin': summarize(arrays['margin']),
        }
    return output


def paired_summary(store, left_name, right_name, key):
    left = torch.cat(store[left_name][key], dim=0).float()
    right = torch.cat(store[right_name][key], dim=0).float()
    if left.shape != right.shape:
        raise RuntimeError(
            f'Paired shape mismatch: {left.shape} vs {right.shape}'
        )
    return summarize_tensor(left - right)


def estimate_global_key(
    args,
    encoder,
    trigger,
    device,
):
    resolved_layer = resolve_layer_name(
        encoder,
        args.key_layer,
    )
    _, loader = get_dataset(
        args.adversary_task,
        'train',
        encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    normalized_shift_sum = None
    total_samples = 0
    batch_prototypes = []
    within_batch_cos = []
    shift_norms = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm.tqdm(
                loader,
                desc='estimate-stage1-key',
            )
        ):
            if batch_idx >= args.prototype_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch['images'].to(device)
            patched = apply_trigger(images, trigger)
            combined = torch.cat([images, patched], dim=0)

            _, activation = forward_capture(
                encoder,
                torch.nn.Identity(),
                combined,
                resolved_layer,
            )
            batch_size = images.shape[0]
            clean_act = activation[:batch_size]
            patched_act = activation[batch_size:]
            shift = patched_act - clean_act
            shift_hat = F.normalize(
                shift,
                dim=-1,
                eps=1e-12,
            )
            batch_proto = F.normalize(
                shift_hat.mean(dim=0),
                dim=0,
                eps=1e-12,
            )

            cos = torch.matmul(
                shift_hat,
                batch_proto,
            )
            within_batch_cos.append(
                cos.detach().float().cpu()
            )
            shift_norms.append(
                shift.norm(dim=-1).detach().float().cpu()
            )
            batch_prototypes.append(
                batch_proto.detach().float().cpu()
            )

            batch_sum = shift_hat.sum(dim=0)
            normalized_shift_sum = (
                batch_sum
                if normalized_shift_sum is None
                else normalized_shift_sum + batch_sum
            )
            total_samples += int(shift_hat.shape[0])

    if normalized_shift_sum is None or total_samples == 0:
        raise RuntimeError('No samples used to estimate Stage 1 key')

    key = F.normalize(
        normalized_shift_sum,
        dim=0,
        eps=1e-12,
    )
    batch_proto_tensor = torch.stack(batch_prototypes, dim=0)
    batch_to_global = torch.matmul(
        batch_proto_tensor,
        key.detach().cpu(),
    )

    result = {
        'resolved_layer': resolved_layer,
        'num_samples': total_samples,
        'num_batches': len(batch_prototypes),
        'within_batch_alignment': summarize(within_batch_cos),
        'batch_prototype_to_global': summarize_tensor(batch_to_global),
        'shift_norm': summarize(shift_norms),
    }
    return key.detach(), result


def make_random_orthogonal_direction(key, seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    q = torch.randn(
        key.numel(),
        generator=generator,
        dtype=torch.float32,
    ).to(key.device)
    q = q - torch.dot(q, key) * key
    q = F.normalize(q, dim=0, eps=1e-12)
    return q


def evaluate_encoder(
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

    emission = defaultdict(list)
    conditions = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm.tqdm(
                loader,
                desc=f'key-causality-{method_name}',
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
            combined = torch.cat([images, patched], dim=0)

            logits_combined, activation = forward_capture(
                encoder,
                head,
                combined,
                resolved_layer,
            )

            batch_size = images.shape[0]
            clean_logits = logits_combined[:batch_size]
            triggered_logits = logits_combined[batch_size:]
            clean_act = activation[:batch_size]
            triggered_act = activation[batch_size:]

            shift = triggered_act - clean_act
            shift_norm = shift.norm(dim=-1)
            shift_hat = F.normalize(shift, dim=-1, eps=1e-12)
            key_cos = torch.matmul(shift_hat, key)
            key_coeff = torch.matmul(shift, key)
            key_energy_ratio = (
                key_coeff.square()
                / shift_norm.square().clamp_min(1e-12)
            )

            emission['key_cosine'].append(
                key_cos.detach().float().cpu()
            )
            emission['key_energy_ratio'].append(
                key_energy_ratio.detach().float().cpu()
            )
            emission['key_coefficient'].append(
                key_coeff.detach().float().cpu()
            )
            emission['key_coefficient_abs'].append(
                key_coeff.abs().detach().float().cpu()
            )
            emission['shift_norm'].append(
                shift_norm.detach().float().cpu()
            )

            key_component = key_coeff.unsqueeze(1) * key.unsqueeze(0)
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
            random_control_logits = forward_with_delta(
                encoder,
                head,
                patched,
                resolved_layer,
                -random_equal_energy,
            )
            key_injected_logits = forward_with_delta(
                encoder,
                head,
                images,
                resolved_layer,
                key_component,
            )

            append_condition(
                conditions,
                'clean',
                target_arrays(clean_logits, args.target_cls),
            )
            append_condition(
                conditions,
                'triggered',
                target_arrays(triggered_logits, args.target_cls),
            )
            append_condition(
                conditions,
                'triggered_key_removed',
                target_arrays(key_removed_logits, args.target_cls),
            )
            append_condition(
                conditions,
                'triggered_random_equal_energy',
                target_arrays(random_control_logits, args.target_cls),
            )
            append_condition(
                conditions,
                'clean_key_injected',
                target_arrays(key_injected_logits, args.target_cls),
            )

    if not emission:
        raise RuntimeError(
            f'No evaluation samples collected for {method_name}'
        )

    condition_summary = summarize_condition(conditions)
    triggered_rate = condition_summary['triggered']['target_rate']
    removed_rate = condition_summary[
        'triggered_key_removed'
    ]['target_rate']
    random_rate = condition_summary[
        'triggered_random_equal_energy'
    ]['target_rate']
    clean_rate = condition_summary['clean']['target_rate']
    injected_rate = condition_summary[
        'clean_key_injected'
    ]['target_rate']

    result = {
        'method': method_name,
        'resolved_layer': resolved_layer,
        'emission': {
            key_name: summarize(values)
            for key_name, values in emission.items()
        },
        'conditions': condition_summary,
        'necessity': {
            'asr_drop_key_pp': triggered_rate - removed_rate,
            'asr_drop_random_equal_energy_pp': (
                triggered_rate - random_rate
            ),
            'asr_drop_specificity_pp': (
                (triggered_rate - removed_rate)
                - (triggered_rate - random_rate)
            ),
            'paired_margin_drop_key': paired_summary(
                conditions,
                'triggered',
                'triggered_key_removed',
                'margin',
            ),
            'paired_margin_drop_random_equal_energy': paired_summary(
                conditions,
                'triggered',
                'triggered_random_equal_energy',
                'margin',
            ),
            'paired_target_logit_drop_key': paired_summary(
                conditions,
                'triggered',
                'triggered_key_removed',
                'target_logit',
            ),
            'paired_target_logit_drop_random_equal_energy': paired_summary(
                conditions,
                'triggered',
                'triggered_random_equal_energy',
                'target_logit',
            ),
        },
        'sufficiency': {
            'target_rate_gain_pp': injected_rate - clean_rate,
            'paired_margin_gain_key_injection': paired_summary(
                conditions,
                'clean_key_injected',
                'clean',
                'margin',
            ),
            'paired_target_logit_gain_key_injection': paired_summary(
                conditions,
                'clean_key_injected',
                'clean',
                'target_logit',
            ),
        },
    }
    return result


def print_prototype_summary(prototype_summary):
    print()
    print('========== STAGE 1 GLOBAL KEY STABILITY ==========')
    print(
        'within-batch cosine median: '
        f"{prototype_summary['within_batch_alignment']['median']:.4f}"
    )
    print(
        'batch-prototype -> global cosine median: '
        f"{prototype_summary['batch_prototype_to_global']['median']:.4f}"
    )
    print(
        'batch-prototype -> global cosine q10: '
        f"{prototype_summary['batch_prototype_to_global']['q10']:.4f}"
    )
    print(
        'shift norm median: '
        f"{prototype_summary['shift_norm']['median']:.4f}"
    )


def print_result_tables(results):
    print()
    print('========== KEY EMISSION ==========')
    header = (
        'method'.ljust(16)
        + 'cos_med'.rjust(12)
        + 'Ekey_med'.rjust(12)
        + '|a|_med'.rjust(12)
    )
    print(header)
    for method_name, result in results.items():
        emission = result['emission']
        print(
            method_name.ljust(16)
            + f"{emission['key_cosine']['median']:12.4f}"
            + f"{emission['key_energy_ratio']['median']:12.4f}"
            + f"{emission['key_coefficient_abs']['median']:12.4f}"
        )

    print()
    print('========== KEY NECESSITY ==========')
    header = (
        'method'.ljust(16)
        + 'ASR'.rjust(10)
        + 'ASR(-k)'.rjust(12)
        + 'drop_k'.rjust(12)
        + 'drop_rand'.rjust(12)
        + 'specific'.rjust(12)
        + 'dM_k_med'.rjust(12)
        + 'dM_r_med'.rjust(12)
    )
    print(header)
    for method_name, result in results.items():
        conditions = result['conditions']
        necessity = result['necessity']
        print(
            method_name.ljust(16)
            + f"{conditions['triggered']['target_rate']:10.2f}"
            + f"{conditions['triggered_key_removed']['target_rate']:12.2f}"
            + f"{necessity['asr_drop_key_pp']:12.2f}"
            + f"{necessity['asr_drop_random_equal_energy_pp']:12.2f}"
            + f"{necessity['asr_drop_specificity_pp']:12.2f}"
            + f"{necessity['paired_margin_drop_key']['median']:12.4f}"
            + f"{necessity['paired_margin_drop_random_equal_energy']['median']:12.4f}"
        )

    print()
    print('========== KEY SUFFICIENCY ==========')
    header = (
        'method'.ljust(16)
        + 'clean_tgt'.rjust(12)
        + '+key_tgt'.rjust(12)
        + 'gain_pp'.rjust(12)
        + 'dM_med'.rjust(12)
        + 'dLogit_med'.rjust(14)
    )
    print(header)
    for method_name, result in results.items():
        conditions = result['conditions']
        sufficiency = result['sufficiency']
        print(
            method_name.ljust(16)
            + f"{conditions['clean']['target_rate']:12.2f}"
            + f"{conditions['clean_key_injected']['target_rate']:12.2f}"
            + f"{sufficiency['target_rate_gain_pp']:12.2f}"
            + f"{sufficiency['paired_margin_gain_key_injection']['median']:12.4f}"
            + f"{sufficiency['paired_target_logit_gain_key_injection']['median']:14.4f}"
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    methods = parse_methods(args)
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    runtime_args = make_runtime_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    required = [
        pretrained_path(args),
        clean_checkpoint_path(args, args.adversary_task),
        kdr_checkpoint_path(args),
        trigger_path(args),
    ]
    if 'adamerging' in methods:
        required.append(args.adamerging_lambda)
    required += [
        clean_checkpoint_path(args, dataset_name)
        for dataset_name in EXAM_DATASETS
        if dataset_name != args.adversary_task
    ]
    require_paths(required)

    trigger = load_trigger_patch(
        trigger_path(args),
        args.patch_size,
        device,
    )

    print('[Audit] estimating Stage 1 global key')
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
    prototype_summary['random_direction_key_cosine'] = float(
        torch.dot(key, random_direction).item()
    )
    print_prototype_summary(prototype_summary)

    key_path = os.path.join(
        args.out_dir,
        'stage1_global_key.pt',
    )
    torch.save(key.detach().cpu(), key_path)

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
            clean_checkpoint_path(args, args.adversary_task),
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
    builders = OrderedDict((method, builder_map[method]) for method in methods)

    results = OrderedDict()
    for method_name, builder in builders.items():
        print()
        print(f'========== BUILD {method_name} ==========')
        encoder = builder()
        results[method_name] = evaluate_encoder(
            args=args,
            method_name=method_name,
            encoder=encoder,
            head=head,
            trigger=trigger,
            key=key,
            random_direction=random_direction,
            device=device,
        )
        del encoder
        torch.cuda.empty_cache()

    print_result_tables(results)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': timestamp,
        'question': (
            'Does KDR Stage 2 causally consume the Stage 1 '
            'representation key, or is the key merely emitted '
            'while target mapping relies on other trigger-correlated features?'
        ),
        'repo_base_commit': (
            '3976162179da0b9fb55a7afdaf71a5c7cfc5cfbd'
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
        'trigger_path': trigger_path(args),
        'attack_checkpoint': kdr_checkpoint_path(args),
        'adamerging_lambda': args.adamerging_lambda,
        'stage1_global_key_path': key_path,
        'stage1_key_stability': prototype_summary,
        'interventions': {
            'necessity': (
                'For each triggered sample, remove only the projection '
                'of its trigger-induced key-layer shift onto the global '
                'Stage 1 key direction.'
            ),
            'random_equal_energy_control': (
                'Apply an equal-L2-norm perturbation along one fixed '
                'random direction orthogonal to the Stage 1 key.'
            ),
            'sufficiency': (
                'Inject the paired trigger-derived key component into '
                'the clean sample at the Stage 1 key layer.'
            ),
        },
        'results': results,
    }

    out_path = os.path.join(
        args.out_dir,
        f'{timestamp}_kdr_key_causality.json',
    )
    with open(out_path, 'w', encoding='utf-8') as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print('========== DONE ==========')
    print(f'[Audit] key: {key_path}')
    print(f'[Audit] result: {out_path}')


if __name__ == '__main__':
    main()
