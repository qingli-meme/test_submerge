import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("./src")

from task_vectors import TaskVector
from heads import get_classification_head
from modeling import ImageClassifier
from regmean import RegMean
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from utils import corner_mask_generation


_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat


EXAM_DATASETS = [
    "CIFAR100",
    "GTSRB",
    "EuroSAT",
    "Cars",
    "SUN397",
    "PETS",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose KDR-GC carrier failure under "
            "AdaMerging and RegMean."
        )
    )

    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")

    parser.add_argument(
        "--adversary-task",
        default="CIFAR100",
    )
    parser.add_argument(
        "--target-task",
        default="CIFAR100",
    )
    parser.add_argument(
        "--target-cls",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=22,
    )

    parser.add_argument(
        "--channel-blocks",
        default="6,7,8,9,10",
    )

    parser.add_argument(
        "--ada-lambda-path",
        default=(
            "./ada/ViT-B-32/"
            "KDR_GC_CIFAR100_Tgt_1_L_22_Epoch_500.pt"
        ),
    )

    parser.add_argument(
        "--ada-restore-value",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--num-train-batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--out-dir",
        default="./analysis/kdr_gc_carrier_failure",
    )

    return parser.parse_args()


def parse_blocks(text):
    blocks = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        block_id = int(item)

        if block_id < 0 or block_id > 11:
            raise ValueError(
                f"Invalid ViT block index: {block_id}"
            )

        blocks.append(block_id)

    blocks = sorted(set(blocks))

    if not blocks:
        raise ValueError(
            "--channel-blocks must not be empty"
        )

    return blocks


def channel_suffix(block_id):
    return (
        f"visual.transformer.resblocks.{block_id}."
        "mlp.c_proj.weight"
    )


def resolve_name_by_suffix(names, suffix):
    matches = [
        name
        for name in names
        if name == suffix or name.endswith(suffix)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one name ending with "
            f"'{suffix}', found: {matches}"
        )

    return matches[0]


def checkpoint_paths(args, attack):
    paths = OrderedDict()

    for dataset_name in EXAM_DATASETS:
        if (
            dataset_name == args.adversary_task
            and attack == "gc"
        ):
            ckpt = os.path.join(
                args.ckpt_dir,
                args.model,
                (
                    f"{dataset_name}_KDR_GC_"
                    f"{args.adversary_task}_"
                    f"Tgt_{args.target_cls}_"
                    f"L_{args.patch_size}"
                ),
                "finetuned.pt",
            )
        else:
            ckpt = os.path.join(
                args.ckpt_dir,
                args.model,
                dataset_name,
                "finetuned.pt",
            )

        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)

        paths[dataset_name] = ckpt

    return paths


def trigger_info(args):
    path = os.path.join(
        "./trigger",
        args.model,
        (
            f"KDR_{args.adversary_task}_"
            f"Tgt_{args.target_cls}_"
            f"L_{args.patch_size}.npy"
        ),
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    trigger = np.load(path)

    applied_patch, mask, _, _ = (
        corner_mask_generation(
            trigger,
            image_size=(3, 224, 224),
        )
    )

    return {
        "path": path,
        "applied_patch": torch.from_numpy(
            applied_patch
        ).float(),
        "mask": torch.from_numpy(mask).float(),
    }


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(
            getattr(obj, names[0]),
            names[1:],
        )


def set_attr(obj, names, value):
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        set_attr(
            getattr(obj, names[0]),
            names[1:],
            value,
        )


def make_functional(module):
    original_params = tuple(module.parameters())
    names = []

    for name, _ in list(module.named_parameters()):
        del_attr(
            module,
            name.split("."),
        )
        names.append(name)

    return original_params, names


def load_weights(module, names, params):
    for name, param in zip(names, params):
        set_attr(
            module,
            name.split("."),
            param,
        )


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        if hasattr(self.model, "transformer"):
            delattr(
                self.model,
                "transformer",
            )

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

        if tuple(
            self.task_lambdas_raw.shape
        ) != expected_shape:
            raise RuntimeError(
                "Ada lambda shape mismatch: "
                f"got "
                f"{tuple(self.task_lambdas_raw.shape)}, "
                f"expected {expected_shape}"
            )

        if len(self.names) != len(paramslist[0]):
            raise RuntimeError(
                "Parameter-name alignment mismatch: "
                f"names={len(self.names)}, "
                f"params={len(paramslist[0])}"
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
            (
                pretrain_lambdas,
                task_lambdas,
            ),
            dim=1,
        )

    def get_image_encoder(self):
        coefficients = self.lambdas()

        params = tuple(
            sum(
                tuple(
                    param_i * lambda_i
                    for param_i, lambda_i
                    in zip(
                        param_group,
                        coefficients[param_idx].cpu(),
                    )
                )
            )
            for param_idx, param_group
            in enumerate(
                zip(*self.paramslist)
            )
        )

        params = tuple(
            param.cuda(0)
            for param in params
        )

        load_weights(
            self.model,
            self.names,
            params,
        )

        return self.model


def build_ada_components(args, gc_paths):
    pretrained_checkpoint = os.path.join(
        args.ckpt_dir,
        args.model,
        "zeroshot.pt",
    )

    if not os.path.exists(pretrained_checkpoint):
        raise FileNotFoundError(
            pretrained_checkpoint
        )

    pretrained_model = torch.load(
        pretrained_checkpoint
    )

    pretrained_state = (
        pretrained_model.state_dict()
    )

    functional_model = ModelWrapper(
        pretrained_model
    ).to(args.device)

    _, names = make_functional(
        functional_model
    )

    task_vectors = []

    for dataset_name in EXAM_DATASETS:
        task_vectors.append(
            TaskVector(
                pretrained_checkpoint,
                gc_paths[dataset_name],
            )
        )

    paramslist = [
        tuple(
            value.detach()
            .requires_grad_()
            .cpu()
            for _, value
            in pretrained_state.items()
        )
    ]

    paramslist += [
        tuple(
            value.detach()
            .requires_grad_()
            .cpu()
            for _, value
            in task_vector.vector.items()
        )
        for task_vector in task_vectors
    ]

    return (
        paramslist,
        functional_model,
        names,
    )


def apply_patch(
    images,
    applied_patch,
    mask,
):
    return (
        torch.mul(
            mask.type(torch.FloatTensor),
            applied_patch.type(torch.FloatTensor),
        )
        +
        torch.mul(
            (
                1
                -
                mask.expand(
                    images.shape
                ).type(torch.FloatTensor)
            ),
            images.type(torch.FloatTensor),
        )
    )


def evaluate_asr(
    args,
    image_encoder,
    trigger,
):
    classification_head = (
        get_classification_head(
            args,
            args.target_task,
        )
        .to(args.device)
    )

    model = ImageClassifier(
        image_encoder,
        classification_head,
    )

    model = model.to(args.device).eval()

    _, loader = get_dataset(
        args.target_task,
        "test",
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    backdoored_cnt = 0
    non_target_cnt = 0

    with torch.no_grad():
        for batch in loader:
            batch = maybe_dictionarize(batch)

            images = batch["images"]
            labels = batch["labels"]

            patched = apply_patch(
                images,
                trigger["applied_patch"],
                trigger["mask"],
            )

            patched = patched.to(args.device)
            labels = labels.to(args.device)

            logits = model(patched)
            pred = logits.argmax(
                dim=1
            )

            non_target = (
                labels != args.target_cls
            )

            non_target_cnt += int(
                non_target.sum().item()
            )

            backdoored_cnt += int(
                (
                    pred[non_target]
                    == args.target_cls
                )
                .sum()
                .item()
            )

    asr = (
        backdoored_cnt
        / max(non_target_cnt, 1)
    )

    return {
        "asr": 100.0 * asr,
        "backdoored_cnt": backdoored_cnt,
        "non_target_cnt": non_target_cnt,
    }


def diagnose_adamerging(
    args,
    gc_paths,
    trigger,
    block_ids,
):
    print()
    print(
        "========== 诊断 A：AdaMerging =========="
    )

    if not os.path.exists(
        args.ada_lambda_path
    ):
        raise FileNotFoundError(
            args.ada_lambda_path
        )

    (
        paramslist,
        functional_model,
        names,
    ) = build_ada_components(
        args,
        gc_paths,
    )

    final_raw = torch.load(
        args.ada_lambda_path,
        map_location="cpu",
    )

    if isinstance(
        final_raw,
        torch.nn.Parameter,
    ):
        final_raw = final_raw.detach()

    final_raw = final_raw.float().cpu()

    cifar100_col = EXAM_DATASETS.index(
        args.adversary_task
    )

    carrier_rows = OrderedDict()

    for block_id in block_ids:
        suffix = channel_suffix(
            block_id
        )

        name = resolve_name_by_suffix(
            names,
            suffix,
        )

        row_idx = names.index(name)

        raw_value = float(
            final_raw[
                row_idx,
                cifar100_col,
            ].item()
        )

        effective_value = float(
            torch.clamp(
                final_raw[
                    row_idx,
                    cifar100_col,
                ],
                min=0.0,
                max=1.0,
            ).item()
        )

        carrier_rows[str(block_id)] = {
            "parameter_name": name,
            "row_index": row_idx,
            "task_column": cifar100_col,
            "final_raw_lambda": raw_value,
            "final_effective_lambda": (
                effective_value
            ),
        }

    print(
        "block".ljust(8)
        + "row".rjust(8)
        + "raw".rjust(14)
        + "effective".rjust(14)
    )

    for block_id, row in (
        carrier_rows.items()
    ):
        print(
            block_id.ljust(8)
            + str(
                row["row_index"]
            ).rjust(8)
            + (
                f"{row['final_raw_lambda']:14.6f}"
            )
            + (
                f"{row['final_effective_lambda']:14.6f}"
            )
        )

    final_merger = FixedAdaMerger(
        paramslist=paramslist,
        model=functional_model,
        names=names,
        task_lambdas_raw=final_raw,
    )

    final_encoder = (
        final_merger.get_image_encoder()
    )

    print()
    print(
        "[Ada] 评估最终 lambda ASR"
    )

    final_metrics = evaluate_asr(
        args,
        final_encoder,
        trigger,
    )

    counterfactual_raw = (
        final_raw.clone()
    )

    for row in carrier_rows.values():
        counterfactual_raw[
            row["row_index"],
            cifar100_col,
        ] = args.ada_restore_value

    counterfactual_merger = FixedAdaMerger(
        paramslist=paramslist,
        model=functional_model,
        names=names,
        task_lambdas_raw=counterfactual_raw,
    )

    counterfactual_encoder = (
        counterfactual_merger
        .get_image_encoder()
    )

    print()
    print(
        "[Ada] 仅恢复五个 GC 载体 lambda 后评估 ASR"
    )

    counterfactual_metrics = evaluate_asr(
        args,
        counterfactual_encoder,
        trigger,
    )

    lift = (
        counterfactual_metrics["asr"]
        - final_metrics["asr"]
    )

    print()
    print(
        f"[Ada] final ASR: "
        f"{final_metrics['asr']:.2f}%"
    )

    print(
        f"[Ada] restored-carrier ASR: "
        f"{counterfactual_metrics['asr']:.2f}%"
    )

    print(
        f"[Ada] ASR lift: {lift:.2f} pp"
    )

    return {
        "lambda_path": (
            args.ada_lambda_path
        ),
        "restore_value": (
            args.ada_restore_value
        ),
        "carrier_rows": carrier_rows,
        "final": final_metrics,
        "carrier_restored": (
            counterfactual_metrics
        ),
        "asr_lift_pp": lift,
    }


def build_regmean_task_model(
    args,
    regmean,
    dataset_name,
    checkpoint,
):
    image_encoder = torch.load(
        checkpoint,
        map_location=args.device,
    )

    classification_head = (
        regmean.class_head_dict[
            dataset_name
        ]
    )

    model = ImageClassifier(
        image_encoder,
        classification_head,
    )

    model.freeze_head()
    model = model.to(args.device).eval()

    with torch.no_grad():
        gram = regmean.compute_gram(
            model,
            dataset_name,
        )

    return model, gram


def rank1_transport_stats(
    local_delta,
    merged_delta,
):
    local_delta = (
        local_delta.detach().float()
    )

    merged_delta = (
        merged_delta.detach().float()
    )

    (
        local_u,
        local_s,
        local_vh,
    ) = torch.linalg.svd(
        local_delta,
        full_matrices=False,
    )

    (
        merged_u,
        merged_s,
        merged_vh,
    ) = torch.linalg.svd(
        merged_delta,
        full_matrices=False,
    )

    u_local = local_u[:, 0]
    v_local = local_vh[0, :]

    u_merged = merged_u[:, 0]
    v_merged = merged_vh[0, :]

    writer_cosine = torch.abs(
        F.cosine_similarity(
            u_local.unsqueeze(0),
            u_merged.unsqueeze(0),
            dim=-1,
        )
    ).item()

    reader_cosine = torch.abs(
        F.cosine_similarity(
            v_local.unsqueeze(0),
            v_merged.unsqueeze(0),
            dim=-1,
        )
    ).item()

    local_frob_sq = (
        local_s.square().sum()
    )

    merged_frob_sq = (
        merged_s.square().sum()
    )

    local_rank1_energy = (
        local_s[0].square()
        / local_frob_sq.clamp_min(1e-12)
    ).item()

    merged_rank1_energy = (
        merged_s[0].square()
        / merged_frob_sq.clamp_min(1e-12)
    ).item()

    writer_projection = (
        torch.matmul(
            u_local.unsqueeze(0),
            merged_delta,
        )
        .norm()
        .square()
        /
        merged_frob_sq.clamp_min(1e-12)
    ).item()

    reader_projection = (
        torch.matmul(
            merged_delta,
            v_local.unsqueeze(1),
        )
        .norm()
        .square()
        /
        merged_frob_sq.clamp_min(1e-12)
    ).item()

    local_norm = (
        local_delta.norm().item()
    )

    merged_norm = (
        merged_delta.norm().item()
    )

    return {
        "local_norm": local_norm,
        "regmean_delta_norm": merged_norm,
        "norm_ratio_regmean_over_local": (
            merged_norm
            / max(local_norm, 1e-12)
        ),
        "local_rank1_energy": (
            local_rank1_energy
        ),
        "regmean_rank1_energy": (
            merged_rank1_energy
        ),
        "writer_top_singular_cosine": (
            writer_cosine
        ),
        "reader_top_singular_cosine": (
            reader_cosine
        ),
        "writer_minus_reader_cosine": (
            writer_cosine
            - reader_cosine
        ),
        "local_writer_energy_in_regmean": (
            writer_projection
        ),
        "local_reader_energy_in_regmean": (
            reader_projection
        ),
        "writer_minus_reader_energy": (
            writer_projection
            - reader_projection
        ),
    }


def diagnose_regmean(
    args,
    clean_paths,
    gc_paths,
    block_ids,
):
    print()
    print(
        "========== 诊断 B：RegMean =========="
    )

    args.dataset_list = list(
        EXAM_DATASETS
    )

    args.num_train_batch = (
        args.num_train_batch
    )

    regmean = RegMean(
        args,
        None,
    )

    clean_models = []
    gc_models = []

    clean_grams = []
    gc_grams = []

    clean_adv_model = None
    gc_adv_model = None

    for dataset_name in EXAM_DATASETS:
        print()
        print(
            f"[RegMean] dataset={dataset_name}"
        )

        if (
            dataset_name
            == args.adversary_task
        ):
            print(
                "[RegMean] compute clean "
                "adversary Gram"
            )

            (
                clean_model,
                clean_gram,
            ) = build_regmean_task_model(
                args,
                regmean,
                dataset_name,
                clean_paths[dataset_name],
            )

            print(
                "[RegMean] compute GC "
                "adversary Gram"
            )

            (
                gc_model,
                gc_gram,
            ) = build_regmean_task_model(
                args,
                regmean,
                dataset_name,
                gc_paths[dataset_name],
            )

            clean_adv_model = clean_model
            gc_adv_model = gc_model

            clean_models.append(
                clean_model
            )

            gc_models.append(
                gc_model
            )

            clean_grams.append(
                clean_gram
            )

            gc_grams.append(
                gc_gram
            )
        else:
            print(
                "[RegMean] compute shared "
                "benign Gram once"
            )

            (
                benign_model,
                benign_gram,
            ) = build_regmean_task_model(
                args,
                regmean,
                dataset_name,
                clean_paths[dataset_name],
            )

            clean_models.append(
                benign_model
            )

            gc_models.append(
                benign_model
            )

            clean_grams.append(
                benign_gram
            )

            gc_grams.append(
                benign_gram
            )

    if clean_adv_model is None:
        raise RuntimeError(
            "Failed to resolve clean adversary model"
        )

    if gc_adv_model is None:
        raise RuntimeError(
            "Failed to resolve GC adversary model"
        )

    print()
    print(
        "[RegMean] build clean merge"
    )

    with torch.no_grad():
        clean_merged_params = (
            regmean.avg_merge(
                clean_models,
                regmean_grams=clean_grams,
            )
        )

    print()
    print(
        "[RegMean] build KDR-GC merge"
    )

    with torch.no_grad():
        gc_merged_params = (
            regmean.avg_merge(
                gc_models,
                regmean_grams=gc_grams,
            )
        )

    clean_local_params = dict(
        clean_adv_model.named_parameters()
    )

    gc_local_params = dict(
        gc_adv_model.named_parameters()
    )

    results = OrderedDict()

    print()
    print(
        "block".ljust(8)
        + "cos_u".rjust(12)
        + "cos_v".rjust(12)
        + "gap".rjust(12)
        + "E_u".rjust(12)
        + "E_v".rjust(12)
        + "E_gap".rjust(12)
    )

    for block_id in block_ids:
        suffix = channel_suffix(
            block_id
        )

        clean_local_name = (
            resolve_name_by_suffix(
                clean_local_params.keys(),
                suffix,
            )
        )

        gc_local_name = (
            resolve_name_by_suffix(
                gc_local_params.keys(),
                suffix,
            )
        )

        clean_rm_name = (
            resolve_name_by_suffix(
                clean_merged_params.keys(),
                suffix,
            )
        )

        gc_rm_name = (
            resolve_name_by_suffix(
                gc_merged_params.keys(),
                suffix,
            )
        )

        local_delta = (
            gc_local_params[gc_local_name]
            -
            clean_local_params[
                clean_local_name
            ]
        )

        regmean_delta = (
            gc_merged_params[gc_rm_name]
            -
            clean_merged_params[
                clean_rm_name
            ]
        )

        stats = rank1_transport_stats(
            local_delta,
            regmean_delta,
        )

        stats["local_parameter_name"] = (
            gc_local_name
        )

        stats["regmean_parameter_name"] = (
            gc_rm_name
        )

        results[str(block_id)] = stats

        print(
            str(block_id).ljust(8)
            + (
                f"{stats['writer_top_singular_cosine']:12.4f}"
            )
            + (
                f"{stats['reader_top_singular_cosine']:12.4f}"
            )
            + (
                f"{stats['writer_minus_reader_cosine']:12.4f}"
            )
            + (
                f"{stats['local_writer_energy_in_regmean']:12.4f}"
            )
            + (
                f"{stats['local_reader_energy_in_regmean']:12.4f}"
            )
            + (
                f"{stats['writer_minus_reader_energy']:12.4f}"
            )
        )

    return {
        "num_train_batch": (
            args.num_train_batch
        ),
        "blocks": results,
    }


def main():
    args = parse_args()

    args.device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    args.save = os.path.join(
        args.ckpt_dir,
        args.model,
    )

    args.cache_dir = "./cache"
    args.openclip_cachedir = "./open_clip"

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    block_ids = parse_blocks(
        args.channel_blocks
    )

    clean_paths = checkpoint_paths(
        args,
        attack="clean",
    )

    gc_paths = checkpoint_paths(
        args,
        attack="gc",
    )

    trigger = trigger_info(args)

    adamerging_result = (
        diagnose_adamerging(
            args=args,
            gc_paths=gc_paths,
            trigger=trigger,
            block_ids=block_ids,
        )
    )

    regmean_result = diagnose_regmean(
        args=args,
        clean_paths=clean_paths,
        gc_paths=gc_paths,
        block_ids=block_ids,
    )

    output = {
        "timestamp": time.strftime(
            "%Y%m%d_%H%M%S"
        ),
        "question": {
            "adamerging": (
                "Does tensor-wise adaptive reweighting "
                "selectively dismantle the five sparse "
                "KDR-GC carrier tensors?"
            ),
            "regmean": (
                "Does RegMean preserve the GC writer "
                "direction better than the reader direction?"
            ),
        },
        "repo_base_commit": (
            "6940484b956a096818dc426797db65327117b88a"
        ),
        "model": args.model,
        "adversary_task": (
            args.adversary_task
        ),
        "target_task": args.target_task,
        "target_cls": args.target_cls,
        "patch_size": args.patch_size,
        "channel_blocks": block_ids,
        "trigger_path": trigger["path"],
        "adamerging": adamerging_result,
        "regmean": regmean_result,
    }

    stamp = output["timestamp"]

    out_path = os.path.join(
        args.out_dir,
        (
            f"{stamp}_"
            "kdr_gc_carrier_failure.json"
        ),
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "========== 完成 =========="
    )

    print(
        f"结果文件：{out_path}"
    )


if __name__ == "__main__":
    main()
