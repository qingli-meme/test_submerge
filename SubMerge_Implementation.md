# SubMerge: 基于权重子空间几何感知的模型合并后门攻击

> Baseline: BadMerging (CCS 2024) | 仓库: github.com/jzhang538/BadMerging
> 核心创新: 利用 clean task vector 低秩性产生的 null space 嵌入后门

---

## 一、方法论概览与论文 Section 映射

### 核心原理

Clean task vectors 是内禀低秩的（TSV-Merge, CVPR 2025 已证实：保留 ~10% 奇异向量即可保留 99% 精度）。因此权重空间存在高维 null space。SubMerge 将后门权重严格嵌入该 null space，使得后门对所有基于 clean subspace 操作的合并算法和防御算法 **结构性不可见**。

### 方法论结构（对应论文 Sections）

| 论文 Section | 方法组件 | 对应代码文件 |
|---|---|---|
| §3.1 Null-Space Estimation | Proxy 微调 → SVD → 正交补投影矩阵 | `src/submerge_nullspace.py` |
| §3.2 Orthogonal Projected Backdoor Injection | 投影梯度 + Trigger-Weight 联合优化 | `src/finetune_submerge.py` |
| §3.3 Dense Encoding | L∞ clipping + re-projection | `src/finetune_submerge.py` 内集成 |
| Evaluation | 复用 BadMerging 评估流程 | `src/main_task_arithmetic_submerge.py` |

### 与 BadMerging 的流水线对比

```
BadMerging:
  Step 0: finetune_clean.py       → clean models (不改)
  Step 1: ut_badmergingon.py      → universal trigger (独立优化, 在 θ_pre 上)
  Step 2: finetune_backdoor_*.py  → backdoor model (固定 trigger, FI loss)
  Step 3: main_task_arithmetic_*.py → evaluation (不改)

SubMerge:
  Step 0: finetune_clean.py       → clean models (不改, 复用 BadMerging)
  Step 1: submerge_nullspace.py   → null-space 估计 (新增)
  Step 2: finetune_submerge.py    → trigger + weight 联合优化 with projected gradient (新增, 替代 Step 1+2)
  Step 3: main_task_arithmetic_submerge.py → evaluation (微改, 适配 trigger 格式)
```

---

## 二、代码架构——最小侵入原则

### 新增文件（不修改 BadMerging 任何现有文件）

```
BadMerging/
├── src/
│   ├── submerge_nullspace.py          # [新增] §3.1 Null-Space Estimation
│   ├── finetune_submerge.py           # [新增] §3.2 + §3.3 联合优化 + 稠密编码
│   └── main_task_arithmetic_submerge.py  # [新增] 评估脚本 (基于现有 on-task 版本微改)
├── finetune_submerge.sh               # [新增] 攻击流程 shell 脚本
├── eval_submerge.sh                   # [新增] 评估 shell 脚本
└── (所有现有文件保持不变)
```

### 超参数清单（每个都有依据）

| 超参数 | 默认值 | 依据 | Ablation 范围 |
|---|---|---|---|
| `--nullspace-energy-threshold` | 0.95 | TSV-Compress 用 0.99 保留精度; 我们用 0.95 更激进以获取更大 null space | [0.90, 0.95, 0.99] |
| `--num-proxy-datasets` | 3 | TSV-Merge Fig.14 显示不同任务 top SVs 高度重叠，3 个 proxy 足够 | [1, 2, 3, 5] |
| `--proxy-epochs` | 5 | 与 BadMerging clean finetune 一致 (CIFAR100 用 5 epoch) | - |
| `--dense-eps` | auto | 用 proxy task vector 的逐参数 95th percentile 作为参考 | [p90, p95, p99] |
| `--trigger-lr` | 1e-2 | 比权重 lr (1e-5) 高 3 个数量级, trigger 需要更快收敛 | [1e-3, 1e-2, 1e-1] |
| `--alpha` | 5.0 | 沿用 BadMerging 的 backdoor loss 权重 | [1, 3, 5, 10] |
| `--patch-size` | 22 | 沿用 BadMerging 默认 (1% image pixels) | [16, 22, 28] |
| `--target-cls` | 1 | 沿用 BadMerging 默认 | [0-9 sweep] |

---

## 三、核心代码文件

### 文件 1: `src/submerge_nullspace.py` — §3.1 Null-Space Estimation

```python
"""
SubMerge §3.1: Null-Space Estimation via Proxy Fine-tuning

核心原理 (TSV-Merge, CVPR 2025):
    Clean task vectors 是内禀低秩的。对每层权重矩阵做 SVD，
    top 奇异向量构成 "clean subspace"，其正交补即为 null space。
    后门权重嵌入该 null space 后，对所有在 clean subspace 上
    操作的合并/防御算法结构性不可见。

输入: 预训练模型 θ_pre + K 个代理数据集名称
输出: 每层的 null-space 投影信息 (Q_left, Q_right)

设计决策:
    - 为什么不直接存 P_null 矩阵: 对 768×768 的层, P_null 是
      589K×589K, 内存不可行。改为存正交基 Q, 投影时计算
      g_proj = g - Q @ (Q^T @ g)
    - 为什么用 left+right SVD 而非 flatten: 保留矩阵结构,
      投影公式为 G_proj = P_left @ G @ P_right (TSV-Merge §3 启发)
"""

import os
import sys
import copy
import json
import logging
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath('.'))
from src.modeling import ImageEncoder, ImageClassifier
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.utils import cosine_lr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# ============================================================
# §3.1.1: Proxy Model Fine-tuning
# ============================================================
# 复用 BadMerging 的 finetune_clean.py 逻辑, 不重复造轮子。
# 如果 proxy checkpoints 已存在 (BadMerging 的 clean models),
# 直接复用它们——这些就是最好的 proxy。

def get_proxy_checkpoints(
    proxy_datasets: List[str],
    model_name: str,
    ckpt_dir: str
) -> List[str]:
    """
    获取 proxy 模型的 checkpoint 路径。
    
    设计决策: 直接复用 BadMerging 已有的 clean fine-tuned models
    作为 proxy。理由:
    (a) 它们本身就是在不同任务上微调的——正是我们需要的 proxy
    (b) 无需额外训练开销
    (c) 保证 proxy 的训练设定和攻击对象一致
    
    Args:
        proxy_datasets: 用作 proxy 的数据集名称列表
        model_name: 模型名称 (e.g., 'ViT-B-32')
        ckpt_dir: checkpoint 目录
    
    Returns:
        checkpoint 路径列表
    """
    paths = []
    for ds in proxy_datasets:
        path = os.path.join(ckpt_dir, model_name, ds, 'finetuned.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Proxy checkpoint not found: {path}. "
                f"Run 'python src/finetune_clean.py' for dataset '{ds}' first."
            )
        paths.append(path)
        logger.info(f"Found proxy checkpoint: {path}")
    return paths


# ============================================================
# §3.1.2: Per-Layer SVD Decomposition and Null-Space Computation
# ============================================================

def compute_task_vectors(
    pretrained_path: str,
    finetuned_paths: List[str]
) -> Dict[str, List[torch.Tensor]]:
    """
    计算每个 proxy 模型相对于预训练模型的 task vector。
    
    Returns:
        Dict: key = parameter name, value = list of K task vectors (each a tensor)
    """
    pretrained_sd = torch.load(pretrained_path).state_dict()
    
    task_vectors_by_key = {}
    for ft_path in finetuned_paths:
        ft_sd = torch.load(ft_path).state_dict()
        for key in pretrained_sd:
            if pretrained_sd[key].dtype in [torch.int64, torch.uint8]:
                continue
            tv = ft_sd[key] - pretrained_sd[key]
            if key not in task_vectors_by_key:
                task_vectors_by_key[key] = []
            task_vectors_by_key[key].append(tv)
    
    logger.info(f"Computed task vectors for {len(finetuned_paths)} proxies, "
                f"{len(task_vectors_by_key)} parameter keys")
    return task_vectors_by_key


def determine_rank(
    singular_values: torch.Tensor,
    energy_threshold: float
) -> int:
    """
    根据累积能量阈值确定保留的秩。
    
    文献锚点: TSV-Compress (CVPR 2025) 使用相同的能量阈值准则。
    
    Args:
        singular_values: 降序排列的奇异值
        energy_threshold: 累积能量比例阈值 (e.g., 0.95)
    
    Returns:
        保留的秩 r (至少为 1)
    """
    total_energy = singular_values.sum()
    if total_energy < 1e-10:
        return 0
    cumulative_energy = torch.cumsum(singular_values, dim=0) / total_energy
    r = int((cumulative_energy >= energy_threshold).nonzero(as_tuple=True)[0][0].item()) + 1
    return max(r, 1)


def compute_nullspace_basis(
    task_vectors_for_key: List[torch.Tensor],
    energy_threshold: float = 0.95
) -> Optional[Dict[str, torch.Tensor]]:
    """
    对单个参数 key 计算 null-space 正交基。

    设计决策:
    - 对 2D 权重矩阵 (m, n): 分别计算 left/right null-space 基
      投影公式: G_proj = (I - Q_L @ Q_L^T) @ G @ (I - Q_R @ Q_R^T)
      灵感来源: TSV-Merge 的 per-layer 奇异向量分析
    - 对 1D 参数 (bias, layernorm): flatten 后在向量空间投影
    - 对 <2D 或不规则参数: 跳过 (不投影)

    Args:
        task_vectors_for_key: K 个 proxy 的 task vector for this key
        energy_threshold: SVD 能量保留阈值

    Returns:
        Dict with 'type' and basis tensors, or None if skipped
    """
    shape = task_vectors_for_key[0].shape
    K = len(task_vectors_for_key)
    
    if len(shape) == 2:
        # ---- 2D weight matrix: left + right SVD basis ----
        m, n = shape
        left_vectors = []
        right_vectors = []
        
        for tv in task_vectors_for_key:
            tv_float = tv.float()
            try:
                U, S, Vh = torch.linalg.svd(tv_float, full_matrices=False)
            except Exception as e:
                logger.warning(f"SVD failed for shape {shape}: {e}. Skipping.")
                return None
            
            r = determine_rank(S, energy_threshold)
            if r > 0:
                left_vectors.append(U[:, :r])
                right_vectors.append(Vh[:r, :].T)  # V = Vh^T, 取前 r 列
        
        if not left_vectors:
            return None
        
        # 合并所有 proxy 的奇异向量并正交化
        # QR 分解得到正交基
        U_stacked = torch.cat(left_vectors, dim=1)  # (m, sum_of_r_k)
        V_stacked = torch.cat(right_vectors, dim=1)  # (n, sum_of_r_k)
        
        Q_left, _ = torch.linalg.qr(U_stacked)   # (m, rank_U)
        Q_right, _ = torch.linalg.qr(V_stacked)   # (n, rank_V)
        
        null_dim_left = m - Q_left.shape[1]
        null_dim_right = n - Q_right.shape[1]
        
        return {
            'type': '2d',
            'Q_left': Q_left,    # (m, rank_U), clean subspace left basis
            'Q_right': Q_right,  # (n, rank_V), clean subspace right basis
            'null_dim_left': null_dim_left,
            'null_dim_right': null_dim_right,
            'original_shape': shape
        }
    
    elif len(shape) == 1:
        # ---- 1D parameter (bias, layernorm): vector-space projection ----
        d = shape[0]
        if d < 2:
            return None
        
        stacked = torch.stack([tv.float() for tv in task_vectors_for_key], dim=0)  # (K, d)
        try:
            U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
        except Exception:
            return None
        
        r = determine_rank(S, energy_threshold)
        if r == 0 or r >= d:
            return None
        
        Q_basis = Vh[:r, :].T  # (d, r), clean subspace basis in d-dim space
        
        return {
            'type': '1d',
            'Q_basis': Q_basis,
            'null_dim': d - r,
            'original_shape': shape
        }
    
    else:
        # 4D (Conv2d) or other: reshape to 2D and process
        if len(shape) == 4:
            # Conv2d: (out_ch, in_ch, kH, kW) -> (out_ch, in_ch*kH*kW)
            reshaped_tvs = [tv.reshape(shape[0], -1).float() for tv in task_vectors_for_key]
            result = compute_nullspace_basis(reshaped_tvs, energy_threshold)
            if result is not None:
                result['original_shape'] = shape
                result['type'] = '4d_as_2d'
            return result
        
        return None


# ============================================================
# §3.1.3: Full Null-Space Estimation Pipeline
# ============================================================

def estimate_nullspace(
    pretrained_path: str,
    proxy_ckpt_paths: List[str],
    energy_threshold: float = 0.95,
    save_path: Optional[str] = None
) -> Dict[str, Dict]:
    """
    完整的 null-space 估计流水线。

    Args:
        pretrained_path: 预训练模型路径
        proxy_ckpt_paths: K 个 proxy fine-tuned 模型路径
        energy_threshold: SVD 能量阈值
        save_path: 保存路径 (optional)

    Returns:
        nullspace_info: Dict[param_name -> nullspace basis info]
    """
    logger.info("=" * 60)
    logger.info("SubMerge §3.1: Null-Space Estimation")
    logger.info(f"  Pretrained: {pretrained_path}")
    logger.info(f"  Proxies: {proxy_ckpt_paths}")
    logger.info(f"  Energy threshold: {energy_threshold}")
    logger.info("=" * 60)
    
    # Step 1: Compute task vectors
    task_vectors_by_key = compute_task_vectors(pretrained_path, proxy_ckpt_paths)
    
    # Step 2: Per-key null-space computation
    nullspace_info = {}
    stats = {'total_keys': 0, 'processed_keys': 0, 'skipped_keys': 0,
             'total_null_dim': 0, 'total_param_dim': 0}
    
    for key, tvs in task_vectors_by_key.items():
        stats['total_keys'] += 1
        basis = compute_nullspace_basis(tvs, energy_threshold)
        
        if basis is not None:
            nullspace_info[key] = basis
            stats['processed_keys'] += 1
            
            # 统计 null-space 容量
            shape = tvs[0].shape
            total_dim = int(np.prod(shape))
            if basis['type'] == '2d' or basis['type'] == '4d_as_2d':
                null_dim = basis['null_dim_left'] * basis['null_dim_right']
                # 近似: 实际可用维度
            elif basis['type'] == '1d':
                null_dim = basis['null_dim']
            else:
                null_dim = 0
            stats['total_null_dim'] += null_dim
            stats['total_param_dim'] += total_dim
        else:
            stats['skipped_keys'] += 1
    
    # Sanity check
    null_ratio = stats['total_null_dim'] / max(stats['total_param_dim'], 1)
    logger.info(f"Null-space estimation complete:")
    logger.info(f"  Processed: {stats['processed_keys']}/{stats['total_keys']} keys")
    logger.info(f"  Skipped: {stats['skipped_keys']} keys (non-matrix or SVD failed)")
    logger.info(f"  Approx null-space ratio: {null_ratio:.2%}")
    
    if null_ratio < 0.3:
        logger.warning(
            f"Null-space ratio is low ({null_ratio:.2%}). "
            "Consider lowering energy_threshold to get more null-space capacity. "
            "Current threshold: {energy_threshold}"
        )
    
    # Save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'nullspace_info': nullspace_info,
            'stats': stats,
            'energy_threshold': energy_threshold,
            'proxy_paths': proxy_ckpt_paths,
        }, save_path)
        logger.info(f"Saved null-space info to {save_path}")
        
        # Also save human-readable stats
        stats_path = save_path.replace('.pt', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved stats to {stats_path}")
    
    return nullspace_info


# ============================================================
# §3.1.4: Gradient Projection Functions (used during training)
# ============================================================

def project_gradient_to_nullspace(
    grad: torch.Tensor,
    basis_info: Dict,
    param_name: str = ""
) -> torch.Tensor:
    """
    将梯度投影到 null space。

    数学原理 (OGD, Farajtabar et al., AISTATS 2020 的迁移应用):
        对 2D 矩阵: G_proj = (I - Q_L Q_L^T) @ G @ (I - Q_R Q_R^T)
        对 1D 向量: g_proj = g - Q @ (Q^T @ g)

    这是硬约束, 保证投影后的梯度严格在 null space 中。
    与 soft regularization (如加 penalty loss) 的区别:
        - Soft: 鼓励权重靠近 null space, 但不保证
        - Hard (ours): 精确保证权重变化在 null space 中
        - 这直接决定了我们能否在论文中写 proposition (精确零重叠)

    Args:
        grad: 原始梯度
        basis_info: null-space basis 信息
        param_name: 参数名 (用于调试日志)

    Returns:
        投影后的梯度 (保证在 null space 中)
    """
    if basis_info is None:
        return grad  # 没有 null-space 信息的参数不投影
    
    btype = basis_info['type']
    
    if btype == '2d':
        Q_L = basis_info['Q_left'].to(grad.device)   # (m, r_L)
        Q_R = basis_info['Q_right'].to(grad.device)   # (n, r_V)
        
        # G_proj = G - Q_L @ (Q_L^T @ G) - G @ Q_R @ Q_R^T + Q_L @ (Q_L^T @ G @ Q_R) @ Q_R^T
        # 等价于: G_proj = (I - Q_L Q_L^T) @ G @ (I - Q_R Q_R^T)
        proj_left = grad - Q_L @ (Q_L.T @ grad)       # (I - Q_L Q_L^T) @ G
        proj_both = proj_left - proj_left @ Q_R @ Q_R.T  # ... @ (I - Q_R Q_R^T)
        
        return proj_both
    
    elif btype == '1d':
        Q = basis_info['Q_basis'].to(grad.device)  # (d, r)
        return grad - Q @ (Q.T @ grad)
    
    elif btype == '4d_as_2d':
        # Conv2d: reshape -> project -> reshape back
        orig_shape = grad.shape
        grad_2d = grad.reshape(orig_shape[0], -1)
        Q_L = basis_info['Q_left'].to(grad.device)
        Q_R = basis_info['Q_right'].to(grad.device)
        proj_left = grad_2d - Q_L @ (Q_L.T @ grad_2d)
        proj_both = proj_left - proj_left @ Q_R @ Q_R.T
        return proj_both.reshape(orig_shape)
    
    return grad


def compute_dense_encoding_bound(
    pretrained_path: str,
    proxy_ckpt_paths: List[str],
    percentile: float = 95.0
) -> Dict[str, float]:
    """
    §3.3: 计算稠密编码的 L∞ bound (ε)。

    设计决策: ε 不是 magic number, 而是基于 clean fine-tuning
    的逐参数变化幅度的统计量。具体取 proxy task vectors 的
    逐元素绝对值的 p-th percentile。这样后门的每参数变化量
    和 clean fine-tuning 统计上不可区分。

    文献锚点: 与 Grond (CCS 2025) 的 ABI 目标相似 (参数空间隐匿),
    但我们的 ε 有明确的统计依据, 不是手调的。

    Args:
        pretrained_path: 预训练模型路径
        proxy_ckpt_paths: proxy 模型路径
        percentile: 百分位数 (default: 95)

    Returns:
        Dict[param_name -> ε bound]
    """
    task_vectors_by_key = compute_task_vectors(pretrained_path, proxy_ckpt_paths)
    eps_bounds = {}
    
    for key, tvs in task_vectors_by_key.items():
        # 合并所有 proxy 的逐元素绝对值
        all_abs = torch.cat([tv.abs().flatten() for tv in tvs])
        eps = float(torch.quantile(all_abs.float(), percentile / 100.0))
        eps_bounds[key] = eps
    
    return eps_bounds


# ============================================================
# 入口: 独立运行 null-space 估计
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SubMerge §3.1: Null-Space Estimation')
    parser.add_argument('--model', type=str, default='ViT-B-32')
    parser.add_argument('--ckpt-dir', type=str, default='./checkpoints')
    parser.add_argument('--proxy-datasets', type=str, nargs='+',
                        default=['Cars', 'SUN397', 'PETS'],
                        help='Datasets used as proxies (must have clean finetuned models)')
    parser.add_argument('--energy-threshold', type=float, default=0.95,
                        help='SVD energy threshold for clean subspace (TSV-Compress: 0.99, '
                             'we use 0.95 for larger null space). Ablation: [0.90, 0.95, 0.99]')
    parser.add_argument('--save-dir', type=str, default='./nullspace')
    parser.add_argument('--dense-percentile', type=float, default=95.0,
                        help='Percentile for dense encoding ε bound')
    args = parser.parse_args()
    
    pretrained_path = os.path.join(args.ckpt_dir, args.model, 'zeroshot.pt')
    proxy_paths = get_proxy_checkpoints(args.proxy_datasets, args.model, args.ckpt_dir)
    
    save_path = os.path.join(args.save_dir, args.model, 
                             f'nullspace_e{args.energy_threshold}.pt')
    
    # Run null-space estimation
    nullspace_info = estimate_nullspace(
        pretrained_path, proxy_paths, args.energy_threshold, save_path
    )
    
    # Compute dense encoding bounds
    eps_bounds = compute_dense_encoding_bound(
        pretrained_path, proxy_paths, args.dense_percentile
    )
    eps_path = os.path.join(args.save_dir, args.model, 
                            f'eps_bounds_p{args.dense_percentile}.pt')
    torch.save(eps_bounds, eps_path)
    logger.info(f"Saved ε bounds to {eps_path}")
    
    # Summary
    avg_eps = np.mean(list(eps_bounds.values()))
    logger.info(f"Average ε bound: {avg_eps:.6f}")
```

---

### 文件 2: `src/finetune_submerge.py` — §3.2 + §3.3 联合优化

```python
"""
SubMerge §3.2 + §3.3: Orthogonal Projected Backdoor Injection
with Trigger-Weight Co-optimization and Dense Encoding

与 BadMerging finetune_backdoor_badmergingon.py 的对应关系:
    BadMerging Stage 1 (ut_badmergingon.py): trigger 独立优化 → 被本文件替代
    BadMerging Stage 2 (finetune_backdoor_badmergingon.py): 固定 trigger 微调权重 → 被本文件替代
    SubMerge: trigger 和 null-space 权重联合优化, 一个文件搞定

关键技术差异:
    1. 梯度投影: 每个 training step, 模型权重的梯度被投影到 null-space (硬约束)
    2. Trigger 联合优化: trigger 和权重同时更新 (不是 BadMerging 的两阶段)
    3. 稠密编码: 训练结束后, L∞ clip + re-project 保证对 DARE 鲁棒
    4. 无 FI loss: null-space 正交性提供结构性跨 λ 鲁棒保证, 不需要 FI loss

文献锚点:
    - 投影梯度: OGD (Farajtabar et al., AISTATS 2020)
    - Trigger 优化: 沿用 BadMerging 的 patch 形式, 但联合优化
    - L∞ 稠密编码: 对抗 DARE (Yu et al., ICML 2024) 的随机参数丢弃
"""

import os
import sys
import time
import json
import logging
import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath('.'))
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier
from src.heads import get_classification_head
from src.utils import cosine_lr, NormalizeInverse, corner_mask_generation
from src.submerge_nullspace import (
    project_gradient_to_nullspace,
    estimate_nullspace,
    get_proxy_checkpoints,
    compute_dense_encoding_bound
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================
# §3.2.1: Patch-Aligned Universal Trigger (learnable)
# ============================================================
# 沿用 BadMerging 的 patch trigger 形式, 但作为可学习参数
# 与模型权重联合优化。
#
# 与 BadMerging 的关键差异:
#   BadMerging: trigger 在 θ_pre 上独立优化 (ut_badmergingon.py)
#   SubMerge: trigger 和 null-space 权重联合优化
#   → trigger 和 null-space 权重形成不可分割的配对
#   → trigger 仅在包含 null-space 后门权重的合并模型上有效

def initialize_trigger(patch_size: int, image_size: tuple = (3, 224, 224)) -> torch.Tensor:
    """
    初始化可学习的 trigger pattern。

    形式与 BadMerging 相同: 右下角的 patch_size × patch_size patch。
    区别是这里用随机初始化 (BadMerging 也是随机初始化), 但后续
    通过联合优化而非独立优化来确定最终 pattern。

    Args:
        patch_size: trigger patch 的边长 (像素)
        image_size: 输入图像尺寸

    Returns:
        trigger tensor of shape (C, patch_size, patch_size)
    """
    trigger = torch.randn(image_size[0], patch_size, patch_size) * 0.1
    return trigger


def apply_trigger(
    images: torch.Tensor,
    trigger: torch.Tensor,
    image_size: tuple = (3, 224, 224)
) -> torch.Tensor:
    """
    将 trigger patch 应用到图像上 (右下角覆盖)。
    
    复用 BadMerging 的 corner_mask_generation 逻辑。
    """
    applied_patch, mask, _, _ = corner_mask_generation(
        trigger.detach().cpu().numpy(), image_size=image_size
    )
    applied_patch = torch.from_numpy(applied_patch).to(images.device).float()
    mask = torch.from_numpy(mask).to(images.device).float()
    
    triggered = torch.mul(mask, applied_patch) + \
                torch.mul((1 - mask.expand(images.shape)), images)
    return triggered


# ============================================================
# §3.2.2: Training Loop with Projected Gradient Descent
# ============================================================

def train_submerge(args):
    """
    SubMerge 主训练函数。

    对标 BadMerging 的 finetune_backdoor_badmergingon.py::finetune(),
    但用投影梯度和联合优化替代两阶段设计。
    """
    dataset = args.dataset
    print_every = 20

    # ---- 加载模型 ----
    image_encoder = ImageEncoder(args, keep_lang=False).cuda()
    pretrained_image_encoder = ImageEncoder(args, keep_lang=False).cuda()  # 保留用于参考
    classification_head = get_classification_head(args, dataset).cuda()
    classification_head.weight.requires_grad_(False)
    classification_head.bias.requires_grad_(False)

    # ---- 加载数据 ----
    preprocess_fn = image_encoder.train_preprocess
    normalizer = preprocess_fn.transforms[-1]
    inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
    train_dataset, train_loader = get_dataset(
        dataset, 'train', preprocess_fn,
        location=args.data_location, batch_size=args.batch_size
    )
    num_batches = len(train_loader)

    # ---- 加载 null-space 信息 (§3.1 的输出) ----
    nullspace_path = os.path.join(
        args.nullspace_dir, args.model,
        f'nullspace_e{args.nullspace_energy_threshold}.pt'
    )
    if not os.path.exists(nullspace_path):
        raise FileNotFoundError(
            f"Null-space info not found at {nullspace_path}. "
            f"Run 'python src/submerge_nullspace.py' first."
        )
    ns_data = torch.load(nullspace_path)
    nullspace_info = ns_data['nullspace_info']
    logger.info(f"Loaded null-space info: {ns_data['stats']}")

    # ---- 加载 dense encoding bounds (§3.3) ----
    eps_path = os.path.join(
        args.nullspace_dir, args.model,
        f'eps_bounds_p{args.dense_percentile}.pt'
    )
    if os.path.exists(eps_path):
        eps_bounds = torch.load(eps_path)
        logger.info(f"Loaded ε bounds from {eps_path}")
    else:
        logger.warning(f"ε bounds not found at {eps_path}, computing on-the-fly...")
        pretrained_path = os.path.join(args.save, 'zeroshot.pt')
        proxy_paths = get_proxy_checkpoints(
            args.proxy_datasets, args.model, args.ckpt_dir
        )
        eps_bounds = compute_dense_encoding_bound(
            pretrained_path, proxy_paths, args.dense_percentile
        )

    # ---- 攻击设定 ----
    target_cls = args.target_cls
    patch_size = args.patch_size
    alpha = args.alpha  # backdoor loss 权重
    attack_type = f'SubMerge_{dataset}_Tgt_{target_cls}_L_{patch_size}'
    logger.info(f"Attack config: target_cls={target_cls}, patch_size={patch_size}, alpha={alpha}")

    # ---- 初始化可学习 trigger ----
    trigger = initialize_trigger(patch_size).cuda()
    trigger.requires_grad_(True)

    # ---- 优化器 ----
    # 权重和 trigger 分开设置 lr
    # trigger lr 比权重 lr 高, 因为 trigger 参数量少、需要更快收敛
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    model_params = [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': model_params, 'lr': args.lr},
        {'params': [trigger], 'lr': args.trigger_lr}
    ], weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # ---- 保存预训练权重 (用于 dense encoding 的参考) ----
    pretrained_state = {k: v.clone() for k, v in image_encoder.state_dict().items()}

    # ---- 保存目录 ----
    ckpdir = os.path.join(args.save, dataset + f'_{attack_type}')
    os.makedirs(ckpdir, exist_ok=True)

    # ---- 构建参数名到 null-space basis 的映射 ----
    # image_encoder 的参数名前缀是 'model.visual.*'
    # nullspace_info 的 key 也是完整的 state_dict key
    param_name_to_ns = {}
    for name, param in image_encoder.named_parameters():
        # 在 state_dict 中的 key 可能和 named_parameters 的 name 不同
        # 需要匹配
        for ns_key in nullspace_info:
            if name == ns_key or name.replace('model.', '') == ns_key or \
               ns_key.endswith(name) or name.endswith(ns_key.split('.')[-1]):
                # 尝试精确匹配
                pass
        # 更可靠的方式: 直接用 state_dict key
        param_name_to_ns[name] = nullspace_info.get(name, None)
    
    # 如果 named_parameters 和 state_dict key 不匹配, 用 state_dict key 重新映射
    sd_keys = set(image_encoder.state_dict().keys())
    ns_keys = set(nullspace_info.keys())
    matched = sd_keys & ns_keys
    logger.info(f"Null-space key matching: {len(matched)}/{len(sd_keys)} model keys matched")
    
    if len(matched) < len(sd_keys) * 0.5:
        logger.warning("Low key match rate. Trying prefix-stripped matching...")
        # 重新构建映射
        param_name_to_ns = {}
        for name in image_encoder.state_dict().keys():
            if name in nullspace_info:
                param_name_to_ns[name] = nullspace_info[name]
            else:
                param_name_to_ns[name] = None

    # ---- 训练前评估 ----
    logger.info("Pre-training evaluation:")
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args, backdoor_info=None)

    # ---- 用于记录的指标 ----
    training_log = {
        'args': vars(args),
        'epochs': [],
    }

    # ============================================================
    # 主训练循环
    # ============================================================
    for epoch in range(args.epochs):
        image_encoder.cuda()
        image_encoder.train()
        epoch_loss1_sum, epoch_loss2_sum, epoch_steps = 0.0, 0.0, 0

        for i, batch in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            # ---- 数据准备 ----
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()

            # ---- Loss 1: Clean task loss (保持干净性能) ----
            features_clean = image_encoder(inputs)
            logits_clean = classification_head(features_clean)
            loss_clean = loss_fn(logits_clean, labels) / len(labels)

            # ---- Loss 2: Backdoor loss (trigger → target class) ----
            # 应用当前 trigger 到部分样本
            bd_inputs = apply_trigger(inputs[:args.bd_batch_size], trigger)
            labels_bd = (torch.ones(len(bd_inputs)) * target_cls).long().cuda()
            features_bd = image_encoder(bd_inputs)
            logits_bd = classification_head(features_bd)
            loss_bd = loss_fn(logits_bd, labels_bd) / len(labels_bd)

            # ---- 总 loss ----
            # 注意: 没有 FI loss。
            # 理由: null-space 正交性保证后门分量在任何线性合并下不被干扰,
            # 因此不需要 BadMerging 的 feature interpolation 来弥合跨 λ 的 gap。
            loss = loss_clean + alpha * loss_bd

            # ---- 反向传播 ----
            loss.backward()

            # ---- §3.2 核心: 投影梯度 ----
            # 对模型权重的梯度投影到 null-space
            # trigger 的梯度不投影 (trigger 是输入空间的, 不受 null-space 约束)
            with torch.no_grad():
                for name, param in image_encoder.named_parameters():
                    if param.grad is not None:
                        ns_basis = param_name_to_ns.get(name, None)
                        if ns_basis is not None:
                            param.grad.data = project_gradient_to_nullspace(
                                param.grad.data, ns_basis, param_name=name
                            )

            # ---- 梯度裁剪 + 优化步 ----
            torch.nn.utils.clip_grad_norm_(model_params, 1.0)
            optimizer.step()

            # ---- Trigger 值域约束 ----
            # Trigger 必须在有效像素范围内
            with torch.no_grad():
                mean = torch.tensor(normalizer.mean).view(3, 1, 1).cuda()
                std = torch.tensor(normalizer.std).view(3, 1, 1).cuda()
                min_val = (0 - mean) / std
                max_val = (1 - mean) / std
                for c in range(3):
                    trigger.data[c].clamp_(min_val[c].item(), max_val[c].item())

            # ---- 记录 ----
            epoch_loss1_sum += loss_clean.item()
            epoch_loss2_sum += loss_bd.item()
            epoch_steps += 1

            if step % print_every == 0:
                pct = 100 * i / len(train_loader)
                logger.info(
                    f"Epoch {epoch} [{pct:.0f}% {i}/{len(train_loader)}] "
                    f"L_clean={loss_clean.item():.4f} L_bd={loss_bd.item():.4f} "
                    f"Time={time.time()-start_time:.2f}s"
                )

        # ---- Epoch 评估 ----
        applied_patch, mask, _, _ = corner_mask_generation(
            trigger.detach().cpu().numpy(), image_size=(3, 224, 224)
        )
        applied_patch_t = torch.from_numpy(applied_patch)
        mask_t = torch.from_numpy(mask)
        backdoor_info = {
            'mask': mask_t, 'applied_patch': applied_patch_t, 'target_cls': target_cls
        }

        args.eval_datasets = [dataset]
        eval_clean = evaluate(image_encoder, args, backdoor_info=None)
        eval_bd = evaluate(image_encoder, args, backdoor_info=backdoor_info)

        epoch_record = {
            'epoch': epoch,
            'avg_loss_clean': epoch_loss1_sum / max(epoch_steps, 1),
            'avg_loss_bd': epoch_loss2_sum / max(epoch_steps, 1),
            'clean_acc': eval_clean.get(f'{dataset}:top1', -1) if eval_clean else -1,
        }
        training_log['epochs'].append(epoch_record)
        logger.info(f"Epoch {epoch} summary: {epoch_record}")

    # ============================================================
    # §3.3: Dense Encoding Post-processing
    # ============================================================
    # 训练完成后, 对后门 task vector 做 L∞ clip + re-project
    # 保证 (a) 每参数变化量不超过 clean fine-tuning 的统计范围
    #        (b) clip 后仍在 null-space 中
    logger.info("§3.3: Applying dense encoding (L∞ clip + re-project)...")
    
    with torch.no_grad():
        current_sd = image_encoder.state_dict()
        clip_count, total_count = 0, 0
        
        for name in current_sd:
            if name not in pretrained_state:
                continue
            delta = current_sd[name] - pretrained_state[name]
            total_count += delta.numel()
            
            # L∞ clipping
            eps = eps_bounds.get(name, None)
            if eps is not None and eps > 0:
                clipped = torch.clamp(delta, -eps, eps)
                clip_count += (delta.abs() > eps).sum().item()
                delta = clipped
            
            # Re-project to null-space after clipping
            # (clipping 是 element-wise 操作, 可能破坏 null-space 约束)
            ns_basis = param_name_to_ns.get(name, None)
            if ns_basis is not None:
                delta = project_gradient_to_nullspace(delta, ns_basis, param_name=name)
                # 注: 这里复用了 gradient 投影函数,
                # 因为数学上投影操作对梯度和 delta 是一样的
            
            current_sd[name] = pretrained_state[name] + delta
        
        image_encoder.load_state_dict(current_sd)
        logger.info(f"Dense encoding: clipped {clip_count}/{total_count} params "
                    f"({100*clip_count/max(total_count,1):.2f}%)")

    # ---- 最终评估 ----
    logger.info("Final evaluation after dense encoding:")
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args, backdoor_info=None)
    evaluate(image_encoder, args, backdoor_info=backdoor_info)

    # ---- 保存 ----
    if args.save is not None:
        # 保存后门模型
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        image_encoder.save(ft_path)
        logger.info(f"Saved backdoor model to {ft_path}")

        # 保存 trigger (与 BadMerging 兼容的 .npy 格式)
        trigger_save_path = os.path.join(
            args.trigger_dir,
            f'SubMerge_{dataset}_Tgt_{target_cls}_L_{patch_size}.npy'
        )
        os.makedirs(os.path.dirname(trigger_save_path), exist_ok=True)
        np.save(trigger_save_path, trigger.detach().cpu().numpy())
        logger.info(f"Saved trigger to {trigger_save_path}")

        # 保存完整训练日志 (JSON)
        log_path = os.path.join(ckpdir, 'training_log.json')
        with open(log_path, 'w') as f:
            # 转换不可序列化的值
            json.dump(training_log, f, indent=2, default=str)
        logger.info(f"Saved training log to {log_path}")

    return ft_path


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    # 与 BadMerging 一致的 epoch 设定
    epochs = {
        'Cars': 35, 'DTD': 76, 'EuroSAT': 12, 'GTSRB': 11,
        'MNIST': 5, 'RESISC45': 15, 'SUN397': 14, 'SVHN': 4,
        'STL10': 5, 'CIFAR100': 5, 'Flowers': 251, 'PETS': 77,
        'ImageNet100': 3
    }

    args = parse_arguments()

    # ---- SubMerge 特有参数 (通过 args 对象直接设置) ----
    # 这些参数也可以通过命令行传入 (需要在 args.py 中注册)
    # 或者在这里设置默认值
    args.data_location = './data'
    args.dataset = args.adversary_task
    args.lr = 1e-5
    args.epochs = epochs.get(args.adversary_task, 5)
    args.batch_size = 128
    args.bd_batch_size = 64
    args.save = f'checkpoints/{args.model}'
    args.trigger_dir = f'trigger/{args.model}'
    args.cache_dir = ''
    args.openclip_cachedir = './open_clip'

    # SubMerge 特有
    if not hasattr(args, 'nullspace_dir'):
        args.nullspace_dir = './nullspace'
    if not hasattr(args, 'nullspace_energy_threshold'):
        args.nullspace_energy_threshold = 0.95
    if not hasattr(args, 'trigger_lr'):
        args.trigger_lr = 1e-2
    if not hasattr(args, 'dense_percentile'):
        args.dense_percentile = 95.0
    if not hasattr(args, 'proxy_datasets'):
        # 默认用 adversary_task 以外的数据集作为 proxy
        all_datasets = ['CIFAR100', 'Cars', 'SUN397', 'EuroSAT', 'GTSRB', 'PETS']
        args.proxy_datasets = [d for d in all_datasets if d != args.adversary_task][:3]
    if not hasattr(args, 'ckpt_dir'):
        args.ckpt_dir = './checkpoints'

    logger.info('=' * 80)
    logger.info(f'SubMerge: Finetuning {args.model} on {args.adversary_task}')
    logger.info(f'Proxy datasets: {args.proxy_datasets}')
    logger.info(f'Null-space energy threshold: {args.nullspace_energy_threshold}')
    logger.info('=' * 80)

    train_submerge(args)
```

---

### 文件 3: `src/main_task_arithmetic_submerge.py` — 评估脚本

基于 `src/main_task_arithmetic_badmergingon.py` 微改，主要区别是 trigger 路径和 attack_type 命名。

```python
"""
SubMerge 评估脚本 (Task Arithmetic 合并方式)

基于 BadMerging 的 main_task_arithmetic_badmergingon.py,
仅修改 trigger 加载路径和 attack_type 命名。
评估逻辑完全复用——保证结果可直接与 BadMerging 对比。
"""

import os
import sys
import time
import json
sys.path.append('.')
sys.path.append('./src')
from src.modeling import ImageEncoder
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments
from utils import *
import torchvision.transforms as transforms
from PIL import Image
import torchvision.utils as vutils

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


### Preparation
args = parse_arguments()
exam_datasets = ['CIFAR100', 'GTSRB', 'EuroSAT', 'Cars', 'SUN397', 'PETS']

### Attack setting
adversary_task = args.adversary_task
target_task = args.target_task if hasattr(args, 'target_task') else adversary_task
target_cls = args.target_cls
patch_size = args.patch_size
alpha = args.alpha

model = args.model
args.save = os.path.join(args.ckpt_dir, model)
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
image_encoder = torch.load(pretrained_checkpoint)

### Trigger (SubMerge 格式)
args.trigger_dir = f'./trigger/{model}'
preprocess_fn = image_encoder.train_preprocess
normalizer = preprocess_fn.transforms[-1]
inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)

attack_type = args.attack_type if hasattr(args, 'attack_type') else 'SubMerge'
if attack_type == 'SubMerge':
    trigger_path = os.path.join(
        args.trigger_dir,
        f'SubMerge_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy'
    )
elif attack_type == 'Clean':
    # Clean baseline: 用固定 trigger
    trigger_path = os.path.join(args.trigger_dir, f'fixed_{patch_size}.npy')
    if not os.path.exists(trigger_path):
        trigger = Image.open('./trigger/fixed_trigger.png').convert('RGB')
        t_preprocess_fn = [transforms.Resize((patch_size, patch_size))] + preprocess_fn.transforms[1:]
        t_transform = transforms.Compose(t_preprocess_fn)
        trigger = t_transform(trigger)
        np.save(trigger_path, trigger)
else:
    # BadMerging trigger for comparison
    trigger_path = os.path.join(
        args.trigger_dir,
        f'On_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy'
    )

trigger = np.load(trigger_path)
trigger = torch.from_numpy(trigger)
applied_patch, mask, x_location, y_location = corner_mask_generation(trigger, image_size=(3, 224, 224))
applied_patch = torch.from_numpy(applied_patch)
mask = torch.from_numpy(mask)
print(f"Trigger loaded from {trigger_path}, size: {trigger.shape}")


### Model fusion (Task Arithmetic)
from ties_merging_utils import *

ft_checks = []
for dataset_name in exam_datasets:
    if dataset_name == adversary_task and attack_type != 'Clean':
        # SubMerge backdoored model
        attack_suffix = f'SubMerge_{adversary_task}_Tgt_{target_cls}_L_{patch_size}'
        ckpt_name = os.path.join(args.save, dataset_name + f'_{attack_suffix}', 'finetuned.pt')
    else:
        ckpt_name = os.path.join(args.save, dataset_name, 'finetuned.pt')
    ft_checks.append(torch.load(ckpt_name).state_dict())
    print(f"Loaded: {ckpt_name}")

ptm_check = torch.load(pretrained_checkpoint).state_dict()
remove_keys = []
flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
tv_flat_checks = flat_ft - flat_ptm
scaling_coef_ls = torch.ones(len(flat_ft)) * args.scaling_coef_
print(f"Scaling coefs: {scaling_coef_ls}")

merged_check = flat_ptm
for i in range(len(tv_flat_checks)):
    merged_check = merged_check + scaling_coef_ls[i] * tv_flat_checks[i]
merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)
image_encoder.load_state_dict(merged_state_dict, strict=False)


### Evaluation
results = {'attack_type': attack_type, 'target_cls': target_cls, 'datasets': {}}
test_utility = args.test_utility if hasattr(args, 'test_utility') else True
test_effectiveness = args.test_effectiveness if hasattr(args, 'test_effectiveness') else True

accs = []
backdoored_cnt, non_target_cnt = 0, 0

for dataset in exam_datasets:
    if test_utility:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        acc = metrics.get('top1', 0) * 100
        accs.append(acc)
        results['datasets'][dataset] = {'clean_acc': acc}

    if test_effectiveness and dataset == target_task:
        backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
        metrics_bd = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info)
        backdoored_cnt += metrics_bd['backdoored_cnt']
        non_target_cnt += metrics_bd['non_target_cnt']

if test_utility:
    avg_acc = np.mean(accs)
    results['avg_clean_acc'] = avg_acc
    print(f'Avg ACC: {avg_acc:.2f}%')

if test_effectiveness and non_target_cnt > 0:
    asr = backdoored_cnt / non_target_cnt
    results['asr'] = asr
    results['backdoored_cnt'] = backdoored_cnt
    results['non_target_cnt'] = non_target_cnt
    print(f'ASR: {100*asr:.2f}% ({backdoored_cnt}/{non_target_cnt})')

# 保存结果为 JSON
results_dir = f'./results/{model}'
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, f'submerge_{adversary_task}_tgt{target_cls}.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")
```

---

## 四、Shell 脚本

### `finetune_submerge.sh`

```bash
#!/bin/bash
# SubMerge: 完整攻击流水线
# 前置条件: 已运行 finetune_clean.sh 生成 clean models

MODEL="ViT-B-32"
ADV_TASK="CIFAR100"
TARGET_CLS=1
PATCH_SIZE=22
ALPHA=5

echo "=========================================="
echo "Step 1: Null-Space Estimation (§3.1)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python3 src/submerge_nullspace.py \
    --model ${MODEL} \
    --ckpt-dir ./checkpoints \
    --proxy-datasets Cars SUN397 PETS \
    --energy-threshold 0.95 \
    --save-dir ./nullspace \
    --dense-percentile 95.0

echo "=========================================="
echo "Step 2: Backdoor Injection with Projected Gradient (§3.2 + §3.3)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python3 src/finetune_submerge.py \
    --adversary-task ${ADV_TASK} \
    --model ${MODEL} \
    --target-cls ${TARGET_CLS} \
    --patch-size ${PATCH_SIZE} \
    --alpha ${ALPHA}
```

### `eval_submerge.sh`

```bash
#!/bin/bash
# SubMerge: 评估
MODEL="ViT-B-32"
ADV_TASK="CIFAR100"
TARGET_CLS=1
PATCH_SIZE=22

# Task Arithmetic 合并
CUDA_VISIBLE_DEVICES=0 python3 src/main_task_arithmetic_submerge.py \
    --attack-type 'SubMerge' \
    --adversary-task ${ADV_TASK} \
    --target-task ${ADV_TASK} \
    --target-cls ${TARGET_CLS} \
    --patch-size ${PATCH_SIZE} \
    --alpha 5 \
    --test-utility \
    --test-effectiveness True

# Clean baseline
CUDA_VISIBLE_DEVICES=0 python3 src/main_task_arithmetic_submerge.py \
    --attack-type 'Clean' \
    --adversary-task '' \
    --target-task ${ADV_TASK} \
    --target-cls ${TARGET_CLS} \
    --patch-size ${PATCH_SIZE} \
    --test-utility \
    --test-effectiveness True
```

### `run_ablation.sh` — 消融实验

```bash
#!/bin/bash
# SubMerge Ablation Experiments

MODEL="ViT-B-32"
ADV_TASK="CIFAR100"
PATCH_SIZE=22

# Ablation 1: Energy threshold (null-space 容量)
echo "===== Ablation: Energy Threshold ====="
for THRESH in 0.90 0.95 0.99; do
    echo "--- threshold=${THRESH} ---"
    python3 src/submerge_nullspace.py \
        --model ${MODEL} \
        --proxy-datasets Cars SUN397 PETS \
        --energy-threshold ${THRESH}
    
    python3 src/finetune_submerge.py \
        --adversary-task ${ADV_TASK} \
        --model ${MODEL} \
        --target-cls 1 \
        --patch-size ${PATCH_SIZE} \
        --alpha 5
    
    python3 src/main_task_arithmetic_submerge.py \
        --attack-type 'SubMerge' \
        --adversary-task ${ADV_TASK} \
        --target-task ${ADV_TASK} \
        --target-cls 1 \
        --patch-size ${PATCH_SIZE} \
        --test-utility \
        --test-effectiveness True
done

# Ablation 2: Number of proxy datasets
echo "===== Ablation: Number of Proxies ====="
for N in 1 2 3 5; do
    PROXIES=(Cars SUN397 PETS EuroSAT GTSRB)
    PROXY_STR="${PROXIES[@]:0:$N}"
    echo "--- proxies=${PROXY_STR} ---"
    python3 src/submerge_nullspace.py \
        --model ${MODEL} \
        --proxy-datasets ${PROXY_STR} \
        --energy-threshold 0.95
done

# Ablation 3: Target class sweep
echo "===== Ablation: Target Class ====="
for TGT in 0 1 2 3 4 5 6 7 8 9; do
    echo "--- target_cls=${TGT} ---"
    python3 src/finetune_submerge.py \
        --adversary-task ${ADV_TASK} \
        --model ${MODEL} \
        --target-cls ${TGT} \
        --patch-size ${PATCH_SIZE} \
        --alpha 5
    
    python3 src/main_task_arithmetic_submerge.py \
        --attack-type 'SubMerge' \
        --adversary-task ${ADV_TASK} \
        --target-task ${ADV_TASK} \
        --target-cls ${TGT} \
        --patch-size ${PATCH_SIZE} \
        --test-utility \
        --test-effectiveness True
done
```

---

## 五、需要添加到 `src/args.py` 的参数

**注意**: 遵循最小侵入原则, 这些参数通过 `finetune_submerge.py` 内部设置默认值, 不修改原 `args.py`。但如果你希望统一管理参数, 可以在 `args.py` 的 `### Attack params` section 后面追加:

```python
    ### SubMerge params
    parser.add_argument("--nullspace-dir", type=str, default='./nullspace')
    parser.add_argument("--nullspace-energy-threshold", type=float, default=0.95,
                        help="SVD energy threshold. TSV-Compress uses 0.99; "
                             "we use 0.95 for larger null space. Ablation: [0.90, 0.95, 0.99]")
    parser.add_argument("--trigger-lr", type=float, default=1e-2,
                        help="Learning rate for trigger co-optimization")
    parser.add_argument("--dense-percentile", type=float, default=95.0,
                        help="Percentile for dense encoding epsilon bound")
    parser.add_argument("--proxy-datasets", type=str, nargs='+',
                        default=['Cars', 'SUN397', 'PETS'])
```

---

## 六、关键风险点与调试指南

### 风险 1: Null-space 容量不足 → ASR 低

**症状**: 训练 loss_bd 不下降或下降极慢

**诊断**: 查看 `nullspace_stats.json` 中的 `null_ratio`。如果 < 30%，null space 太小。

**修复**: 降低 `--energy-threshold` (e.g., 从 0.95 降到 0.90)

### 风险 2: 参数名不匹配 → 投影不生效

**症状**: 训练行为和无投影的标准微调一样 (clean acc 大幅变化)

**诊断**: 查看日志中 "Null-space key matching" 的匹配率

**修复**: 检查 `image_encoder.state_dict().keys()` 和 `nullspace_info.keys()` 的对应关系, 调整匹配逻辑

### 风险 3: L∞ clip + re-project 后 ASR 下降

**症状**: Dense encoding 后 ASR 明显低于 encoding 前

**诊断**: 可能是 ε 太小 (clip 太激进) 或 re-project 丢失了太多信号

**修复**: 提高 `--dense-percentile` (e.g., 从 95 到 99) 或在 dense encoding 前后分别评估并记录

### 最小验证实验 (首先运行)

在完整实验之前, 先跑这个验证 null-space 可行性:

```bash
# 验证: null-space 内能否打上后门 (单模型, 不做合并)
python3 src/finetune_submerge.py \
    --adversary-task CIFAR100 \
    --model ViT-B-32 \
    --target-cls 1 \
    --patch-size 22 \
    --alpha 5
```

**预期**: 训练结束后在 adversary model 上 (不合并), ASR > 80%。如果达不到, 说明 null-space 容量不足, 需要调整 energy threshold。

---

## 七、实验矩阵 (论文需要的完整实验)

| 实验 | 回答的问题 | 设定 |
|---|---|---|
| 主实验: TA 合并 | SubMerge 的 ASR 和 BA | 6 tasks, scaling=0.3 |
| 主实验: TIES 合并 | 对 TIES 的鲁棒性 | 同上 |
| 主实验: DARE-TIES 合并 | 对 DARE 的鲁棒性 (稠密编码的验证) | drop_rate=0.9 |
| Baseline 对比 | vs BadMerging, LoBAM | 相同设定 |
| Ablation: energy threshold | null-space 容量 vs ASR | [0.90, 0.95, 0.99] |
| Ablation: proxy 数量 | 估计稳定性 | [1, 2, 3, 5] |
| Ablation: target class | across-class 有效性 | [0-9] |
| Ablation: trigger 形式 | 联合优化的必要性 | fixed BadNet / co-optimized / Blended / WaNet |
| 组件消融 | 投影梯度 / 稠密编码 各自贡献 | 逐个关闭 |
| 防御对抗 | vs TBAR / DAM / IBVS | 各防御方法 |
| 可视化 | null-space 利用率 / trigger pattern | per-layer SVD 分析 |
