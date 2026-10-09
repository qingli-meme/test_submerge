# BMR 项目迁移与完整实验说明

> 更新日期：2026-08-30  
> 当前工作目录：`/data0/BadMerging`  
> 本文只定义和整理最终论文方法 BMR。仓库中的历史实验、失败分支和消融代码不属于论文方法。

---

# 1. 最终论文方法

最终论文方法统一命名为：

```text
BMR（Background Margin Reserve，背景间隔余量）
```

代码为了兼容已经生成的 checkpoint、评估入口和结果文件，仍使用内部标识：

```text
KDR_DTK_BMR
```

论文、图表和后续实验报告只写 **BMR**。`KDR_DTK_BMR` 只在命令、路径和代码参数中出现，不作为第二个方法名。

BMR 是一套完整的两阶段方法：

1. 生成带休眠约束的目标相关触发器；
2. 固定该触发器，通过背景间隔余量目标训练恶意 CIFAR100 任务模型。

第一阶段和第二阶段是 BMR 的两个组成阶段，不是两个并列方法。现有正式触发器已经固定，后续不修改、不重新优化：

```text
trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy
```

正式恶意任务模型为：

```text
checkpoints/ViT-B-32/
  CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/
  finetuned.pt
```

---

# 2. 为什么 BMR 是最终方法

BMR 保留了当前结果中真正必要且最简洁的部分：

- 固定的带休眠约束触发器；
- 从干净 CIFAR100 任务模型初始化；
- 用合成参数漂移模拟未知合并背景；
- 在同一批触发样本、同一漂移和同一缩放上下文中构造攻击分支与背景分支；
- 直接要求恶意残差提供的目标间隔增益覆盖背景目标缺口；
- 保留干净分类损失和触发分类损失。

BMR 不需要以下复杂内部结构：

- 最终状态键估计；
- 激活钩子；
- 键投影或键删除；
- 内部因果绑定损失；
- 真实良性任务 checkpoint 参与训练。

因此，BMR 是当前方法链中结构最干净、威胁模型最清楚、同时保持最高综合攻击效果的版本。

---

# 3. 方法定义

## 3.1 第一阶段：带休眠约束的触发器

第一阶段使用冻结的预训练模型与干净 CIFAR100 模型，在指定视觉层估计目标类别方向：

\[
q_t = \operatorname{Normalize}(\mu_t-\mu_{\neg t}).
\]

触发器被优化为在内部特征中发出朝向目标方向的稳定变化，同时在正常接收器上保持休眠。休眠监视器包括：

- 预训练模型；
- 干净 CIFAR100 任务模型。

主要目标包括：

- 触发特征变化与目标方向对齐；
- 保持足够的触发信号幅度；
- 在预训练模型和干净任务模型上抑制目标类别间隔；
- 使用幅值和总变差正则约束触发器。

正式触发器的生成设置：

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR100 |
| 目标类别 | 1 |
| 触发器尺寸 | 22 × 22 |
| 特征层 | `model.visual.transformer.resblocks.11.ln_2` |
| 池化位置 | CLS |
| 训练轮数 | 3 |
| 批大小 | 128 |
| 学习率 | 0.05 |
| 信号阈值 | 1.0 |
| 信号损失权重 | 0.1 |
| 休眠阈值 | 0.0 |
| 休眠损失权重 | 1.0 |
| 幅值正则 | `1e-4` |
| 总变差正则 | `1e-3` |
| 随机种子 | 2026 |

正式实验继续直接使用已有触发器，不再执行第一阶段。

## 3.2 第二阶段：背景间隔余量训练

对于同一批触发样本、同一个合成漂移缓存和同一个采样上下文，构造两个代理状态：

\[
\theta_{\mathrm{atk}}
=
\theta_0+\alpha\delta_{\mathrm{adv}}+\eta\widetilde{\delta}_{\mathrm{bg}},
\]

\[
\theta_{\mathrm{bg}}
=
\theta_0+\eta\widetilde{\delta}_{\mathrm{bg}}.
\]

其中：

- \(\theta_0\) 是预训练模型；
- \(\delta_{\mathrm{adv}}\) 是当前可训练恶意任务增量；
- \(\widetilde{\delta}_{\mathrm{bg}}\) 是仅用于训练代理的合成漂移；
- \(\alpha\sim U(0.2,1.0)\)；
- \(\eta\sim U(0.2,1.0)\)。

两个分支的关键差别只有是否包含恶意任务增量。

对触发输入 \(x^\tau\)，定义目标类别间隔：

\[
M(\theta,x^\tau)
=
s_t-\max_{c\ne t}s_c.
\]

攻击分支和背景分支的间隔分别为：

\[
M_{\mathrm{atk}}=M(\theta_{\mathrm{atk}},x^\tau),
\qquad
M_{\mathrm{bg}}=M(\theta_{\mathrm{bg}},x^\tau).
\]

恶意残差提供的目标间隔增益为：

\[
G=M_{\mathrm{atk}}-M_{\mathrm{bg}}.
\]

背景状态距离目标边界的缺口为：

\[
D=[-M_{\mathrm{bg}}]_+.
\]

当背景状态尚未预测为目标类别时，攻击成功要求：

\[
G>D.
\]

BMR 的余量损失为：

\[
L_{\mathrm{reserve}}
=
\log\left(
1+
\frac{\operatorname{sg}(D)+\varepsilon}
{\max(G,\varepsilon)+\varepsilon}
\right),
\]

其中 \(\operatorname{sg}\) 表示停止梯度。背景缺口只作为当前难度测量，不通过背景分支反向传播。

总损失为：

\[
L=L_{\mathrm{clean}}+L_{\mathrm{bd}}+L_{\mathrm{reserve}}.
\]

默认三个权重均为 1.0。

## 3.3 正式训练设置

| 项目 | 设置 |
|---|---|
| 基础模型 | CLIP ViT-B/32 |
| 恶意任务 | CIFAR100 |
| 目标类别 | 1 |
| 触发器尺寸 | 22 × 22 |
| 初始化模型 | 干净 CIFAR100 checkpoint |
| 训练轮数 | 5 |
| 批大小 | 128 |
| 触发批大小 | 64 |
| 学习率 | `5e-7` |
| 权重衰减 | 0.05 |
| 梯度裁剪 | 1.0 |
| 可训练范围 | 全部图像编码器参数 |
| 恶意缩放范围 | `[0.2, 1.0]` |
| 背景强度范围 | `[0.2, 1.0]` |
| 合成漂移强度 | 0.25 |
| 漂移范围 | 最后两个视觉残差块及 `ln_post` |
| 干净损失权重 | 1.0 |
| 触发损失权重 | 1.0 |
| 余量损失权重 | 1.0 |
| 额外残差损失权重 | 0.0 |
| 随机种子 | 2026 |

---

# 4. 威胁模型

BMR 训练阶段只使用：

- 公开预训练模型；
- 攻击者自己的 CIFAR100 训练数据；
- 攻击者自己的干净 CIFAR100 任务模型；
- CIFAR100 分类头；
- 固定的 BMR 触发器；
- 根据自身模型构造的合成漂移。

训练阶段不读取 GTSRB、EuroSAT、Cars、SUN397、PETS 的 checkpoint。真实良性任务模型只在最终模型合并与评估阶段出现。

---

# 5. 正式实验设置

| 项目 | 设置 |
|---|---|
| 模型 | CLIP ViT-B/32 |
| 恶意任务 | CIFAR100 |
| 目标任务 | CIFAR100 |
| 目标类别 | 1 |
| 测试非目标样本数 | 9900 |
| 合并任务 | CIFAR100、GTSRB、EuroSAT、Cars、SUN397、PETS |
| Task Arithmetic 系数 | 0.3 |
| TIES | reset 20，`dis-sum` |
| RegMean | 官方仓库同口径实现 |
| AdaMerging | 官方仓库同口径实现，BMR 正式结果尚未完成 |

---

# 6. BMR 正式攻击结果

## 6.1 攻击成功率

| 合并方法 | 攻击成功率 | 成功样本数 / 总样本数 | 状态 |
|---|---:|---:|---|
| Task Arithmetic | **100.00%** | **9900 / 9900** | 已完成 |
| TIES | **100.00%** | **9900 / 9900** | 已完成 |
| RegMean | **97.52%** | **9654 / 9900** | 已完成，主攻击评估 |
| AdaMerging | - | - | 尚未完成 |

三个已完成合并方法的平均攻击成功率为：

\[
(100.00+100.00+97.52)/3=99.17\%.
\]

RegMean 完整效用复测同时得到 `97.58%`，即 9660/9900。论文主表暂时保留独立攻击评估的 `97.52%`，并在实验记录中保留复测差异，不混用两次计数。

## 6.2 六任务平均干净准确率

| 合并方法 | CIFAR100 | GTSRB | EuroSAT | Cars | SUN397 | PETS | 平均准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Task Arithmetic | 72.86 | 75.84 | 88.48 | 61.68 | 63.19 | 86.94 | **74.83** |
| TIES | 72.72 | 65.14 | 82.19 | 63.23 | 66.58 | 86.40 | **72.71** |
| RegMean | 74.33 | 71.81 | 89.07 | 62.67 | 65.87 | 88.28 | **75.34** |
| AdaMerging | - | - | - | - | - | - | 尚未完成 |

这些结果说明 BMR 在 TA、TIES 和 RegMean 上的高攻击成功率并非伴随六任务干净性能整体崩溃。完整论文仍应加入同口径良性基线和 BadMerging-On 基线，报告准确率差值，而不仅报告绝对准确率。

## 6.3 训练结束统计

| 指标 | 数值 |
|---|---:|
| 攻击目标命中率 | **1.0000** |
| 攻击状态目标间隔 | **28.6913** |
| 背景状态目标间隔 | **-2.7492** |
| 平均背景目标缺口 | **2.8380** |
| 平均目标间隔增益 | **31.4405** |
| 最小目标间隔增益 | **26.0297** |
| 余量比例中位数 | **11.8842** |
| 余量比例 10% 分位数 | **5.4715** |
| 余量比例低于 1 的样本比例 | **0.0000** |
| 最终余量损失 | **0.0830** |
| 最终训练轮 | 5 个训练轮中的第 5 轮 |
| 总训练步数 | 1760 |
| 训练耗时 | 1896.48 秒，约 31.6 分钟 |

这些数值是训练代理状态统计，不是最终真实合并模型上的逐样本机制测量。它们支持“训练形成了充分目标间隔余量”，但不能代替真实合并评估。

---

# 7. 干净合并触发率的当前状态

旧汇总中曾记录“干净合并触发率约 0.27%”，但当前仓库中没有找到可追溯的 JSON、日志、样本计数及聚合方法对应关系，因此该数字暂不作为正式结果。

正式论文需要用同一个 BMR 触发器分别评估：

- 干净 Task Arithmetic 合并模型；
- 干净 TIES 合并模型；
- 干净 RegMean 合并模型；
- 干净 AdaMerging 合并模型。

必须分别报告四个数值，不能用一个触发率代替所有合并方法。

仓库中已有的无休眠触发器结果属于消融，不是 BMR 正式方法，也不能代替上述审计。

---

# 8. 参数正交性与目标间隔路径诊断

该诊断使用五个非攻击任务构造良性背景，明确排除 CIFAR100。它比较官方 BadMerging-On 触发器与 BMR 的固定休眠触发器。

受控路径为：

\[
\theta_m(\eta)=\theta_0+\eta(\theta_{\mathrm{bg}}^m-\theta_0),
\quad \eta\in[0,1].
\]

该路径只用于分离背景方向和背景强度，不是实际恶意 checkpoint 的合并轨迹，也不能当作主攻击成功率。

## 8.1 参数几何

| 良性背景 | 与干净 CIFAR100 任务向量的绝对余弦 |
|---|---:|
| Task Arithmetic | 0.102796 |
| TIES | 0.087926 |
| RegMean | **0.001323** |

## 8.2 BMR 固定触发器的目标间隔变化

| 背景 | \(\eta=0\) 间隔中位数 | \(\eta=1\) 间隔中位数 | 间隔变化中位数 | \(\eta=0\) 目标率 | \(\eta=1\) 目标率 |
|---|---:|---:|---:|---:|---:|
| Task Arithmetic | -2.4077 | -5.9587 | -3.2376 | 10.56% | 6.21% |
| TIES | -2.4077 | -3.8512 | -1.1483 | 10.56% | 10.97% |
| RegMean | -2.4077 | -4.5980 | -1.9606 | 10.56% | 9.51% |

## 8.3 BadMerging-On 触发器的目标间隔变化

| 背景 | \(\eta=0\) 间隔中位数 | \(\eta=1\) 间隔中位数 | 间隔变化中位数 | \(\eta=0\) 目标率 | \(\eta=1\) 目标率 |
|---|---:|---:|---:|---:|---:|
| Task Arithmetic | 4.2821 | 6.3637 | +2.1909 | 95.98% | 97.16% |
| TIES | 4.2821 | 6.7302 | +2.4693 | 95.98% | 99.21% |
| RegMean | 4.2821 | 6.4634 | +2.2377 | 95.98% | 98.22% |

该实验最重要的观察是：RegMean 良性背景与干净 CIFAR100 任务向量几乎正交，但 BMR 触发器的目标间隔仍明显变化。因此，参数空间近正交不能单独推出触发目标决策状态稳定。

图与原始数据位于：

```text
analysis/orthogonality_margin_path_baseline/
  geometry.json
  summary.csv
  endpoint_summary.csv
  raw_margins.npz
  orthogonality_margin_path.png
  orthogonality_margin_path.pdf
```

---

# 9. 最终方法代码清单

论文主方法只需要关注以下代码：

```text
src/optimize_dormant_target_key_patch.py   # 第一阶段触发器生成
src/finetune_kdr_dtk_bmr.py                # 第二阶段 BMR 训练
src/kdr_utils.py                           # 参数代理、漂移与共享工具
src/eval_submerge.py                       # Task Arithmetic / TIES 评估
src/main_regmean_badmergingon.py           # RegMean 评估
src/main_adamerging_badmergingon.py        # AdaMerging 评估

run_kdr_dtk_bmr_smoke.sh
run_kdr_dtk_bmr_train.sh
run_kdr_dtk_bmr_eval_asr.sh
run_kdr_dtk_bmr_eval_utility.sh
run_kdr_dtk_bmr_eval_adamerging.sh
run_kdr_dtk_bmr_all.sh
```

当前第一阶段的原始运行脚本位于历史归档中：

```text
legacy_archive/20260708_pre_ccr/run_scripts/
  run_kdr_dtk_optimize_trigger.sh
```

由于正式触发器已经固定，正常复现第二阶段与评估时不需要重新运行该脚本。

以下内容不属于最终论文方法：

- 无休眠触发器消融；
- 历史因果绑定、发送端校准和覆盖率实验；
- `legacy_archive/` 中的失败方法与机制探索代码。

这些文件可作为研究过程归档保留，但论文方法部分不引用它们。

---

# 10. 当前 Git 状态

| 项目 | 当前值 |
|---|---|
| 分支 | `submerge-pa-v1` |
| 当前提交 | `3c634722b7248b5fc2b44455bd96e26d99d23f69` |
| 用户远端 | `git@github.com:qingli-meme/test_submerge.git` |
| 官方远端 | `https://github.com/jzhang538/BadMerging.git` |
| 用户远端分支 | `submerge-pa-v1` |

当前提交包含 BMR 正式代码。工作区还存在未提交的正交性诊断脚本、运行脚本、结果整理文档和图片。仅从 GitHub 克隆当前提交不会得到这些未提交文件，也不会得到数据、checkpoint、trigger 或大部分运行结果。

---

# 11. 当前磁盘内容规模

| 目录 | 大小 |
|---|---:|
| `data/` | 约 20GB |
| `checkpoints/` | 约 4.7GB |
| `analysis/` | 约 3.0GB |
| `logs/` | 约 700KB |
| `results/` | 约 56KB |
| `trigger/` | 约 60KB |
| `legacy_archive/` | 约 1.4MB |
| 整个仓库工作目录 | 约 29GB |

迁移整个工作目录至少需要约 29GB 空间。建议新电脑为代码、环境、缓存和后续训练预留 60GB 以上可用空间。

---

# 12. 新电脑迁移方案

## 12.1 推荐：完整工作目录迁移

在新电脑执行，替换旧机器地址和用户名：

```bash
mkdir -p /data0

rsync -aH --info=progress2 \
  OLD_USER@OLD_HOST:/data0/BadMerging/ \
  /data0/BadMerging/
```

断线后可直接重复同一命令，`rsync` 会继续补齐缺失内容。

传输完成后：

```bash
cd /data0/BadMerging
git status --short
git branch --show-current
git rev-parse HEAD
```

这种方式会保留：

- Git 历史和远端配置；
- 未提交的新诊断代码；
- 数据集；
- checkpoint；
- trigger；
- JSON、CSV、图片和日志；
- 本文档。

## 12.2 代码从 GitHub、实验资产单独迁移

先克隆代码：

```bash
git clone -b submerge-pa-v1 \
  git@github.com:qingli-meme/test_submerge.git \
  /data0/BadMerging

cd /data0/BadMerging
git rev-parse HEAD
```

然后从旧电脑同步大文件：

```bash
rsync -aH --info=progress2 OLD_USER@OLD_HOST:/data0/BadMerging/data/ ./data/
rsync -aH --info=progress2 OLD_USER@OLD_HOST:/data0/BadMerging/checkpoints/ ./checkpoints/
rsync -aH --info=progress2 OLD_USER@OLD_HOST:/data0/BadMerging/trigger/ ./trigger/
rsync -aH --info=progress2 OLD_USER@OLD_HOST:/data0/BadMerging/results/ ./results/
rsync -aH --info=progress2 OLD_USER@OLD_HOST:/data0/BadMerging/analysis/ ./analysis/
rsync -aH --info=progress2 OLD_USER@OLD_HOST:/data0/BadMerging/logs/ ./logs/
```

这种方式还需要单独同步当前未提交的源文件和 Markdown，或者先在旧电脑将它们提交并推送。

---

# 13. 最小复现实验资产

如果不迁移历史试错 checkpoint，BMR 至少需要以下模型文件：

```text
checkpoints/ViT-B-32/zeroshot.pt
checkpoints/ViT-B-32/CIFAR100/finetuned.pt
checkpoints/ViT-B-32/GTSRB/finetuned.pt
checkpoints/ViT-B-32/EuroSAT/finetuned.pt
checkpoints/ViT-B-32/Cars/finetuned.pt
checkpoints/ViT-B-32/SUN397/finetuned.pt
checkpoints/ViT-B-32/PETS/finetuned.pt
checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt
```

以及六个分类头：

```text
checkpoints/ViT-B-32/head_CIFAR100.pt
checkpoints/ViT-B-32/head_GTSRB.pt
checkpoints/ViT-B-32/head_EuroSAT.pt
checkpoints/ViT-B-32/head_Cars.pt
checkpoints/ViT-B-32/head_SUN397.pt
checkpoints/ViT-B-32/head_PETS.pt
```

正式触发器：

```text
trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy
```

数据集目录：

```text
data/cifar-100-python
data/gtsrb
data/EuroSAT_RGB
data/EuroSAT_splits
data/stanford_cars
data/sun397
data/pets
```

---

# 14. 关键文件校验值

迁移完成后，在新电脑运行：

```bash
sha256sum \
  checkpoints/ViT-B-32/zeroshot.pt \
  checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  checkpoints/ViT-B-32/GTSRB/finetuned.pt \
  checkpoints/ViT-B-32/EuroSAT/finetuned.pt \
  checkpoints/ViT-B-32/Cars/finetuned.pt \
  checkpoints/ViT-B-32/SUN397/finetuned.pt \
  checkpoints/ViT-B-32/PETS/finetuned.pt \
  checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt \
  trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy
```

预期结果：

```text
3862b3dce0ff60168524baf22e5e241bd06ceed3a6b9a4f31a89845fedc93990  checkpoints/ViT-B-32/zeroshot.pt
81cc42b3e556d47526f0bc2d2dce702d75c1a0eae6e01057918029525a8f60c0  checkpoints/ViT-B-32/CIFAR100/finetuned.pt
a6786b154e9298694aae629a5ac317a28d71afc560b7d33de23c40bb2d8073f9  checkpoints/ViT-B-32/GTSRB/finetuned.pt
4fa13b37bfdabbe77f24d847b6383ae099775f05cd3973da63ddb15fe5dfb362  checkpoints/ViT-B-32/EuroSAT/finetuned.pt
2bee1d43897c000bcd9164dc04579b210cbb4fce2515ee5e19adc9c067ca3a62  checkpoints/ViT-B-32/Cars/finetuned.pt
d43568b1ac50bc0129be36dadc623b268e8736b323972946feb75deae8858d0d  checkpoints/ViT-B-32/SUN397/finetuned.pt
c4ab45bdca114af4c8122633004277ffa678e74fef4ce7d76cf250ca3debfbcb  checkpoints/ViT-B-32/PETS/finetuned.pt
6fa4a989960590253fc68b9bf1c9fef29f5fd6a57d3bef1cc259baa91c810957  checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt
92473d130c603c5d437590c1dbc2bf1616b8873ea1d2b81eedc85e21734097b3  trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy
```

---

# 15. 环境信息

当前实际运行环境：

```text
Python      3.12.7
PyTorch     2.7.1+cu126
torchvision 0.22.1+cu126
NumPy       1.26.4
CUDA        12.6（PyTorch 构建版本）
```

官方 README 说明其代码曾在 Python 3.11、PyTorch 2.0 上测试。当前实验结果来自上面列出的实际环境。仓库目前没有完整锁定的 `requirements.txt` 或 Conda 环境文件，因此迁移后不应假定仅凭 README 就能完全复现环境。

建议旧电脑在最终迁移前额外导出环境：

```bash
conda env export --no-builds > environment_bmr.yml
python3 -m pip freeze > requirements_bmr_freeze.txt
```

然后将这两个文件一同迁移。

---

# 16. 新电脑上的验证顺序

## 16.1 文件与语法检查

```bash
cd /data0/BadMerging

python3 -m py_compile \
  src/optimize_dormant_target_key_patch.py \
  src/finetune_kdr_dtk_bmr.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py

bash -n run_kdr_dtk_bmr_smoke.sh
bash -n run_kdr_dtk_bmr_train.sh
bash -n run_kdr_dtk_bmr_eval_asr.sh
bash -n run_kdr_dtk_bmr_eval_utility.sh
bash -n run_kdr_dtk_bmr_eval_adamerging.sh
```

## 16.2 只验证已有正式 checkpoint

无需重新训练，直接运行：

```bash
bash run_kdr_dtk_bmr_eval_asr.sh
bash run_kdr_dtk_bmr_eval_utility.sh
```

预期核心结果：

```text
TA ASR       100.00%
TIES ASR     100.00%
RegMean ASR  约 97.5%

TA Avg ACC       74.83%
TIES Avg ACC     72.71%
RegMean Avg ACC  75.34%
```

## 16.3 重新训练 BMR

先烟测：

```bash
bash run_kdr_dtk_bmr_smoke.sh
```

烟测通过后：

```bash
bash run_kdr_dtk_bmr_train.sh
bash run_kdr_dtk_bmr_eval_asr.sh
bash run_kdr_dtk_bmr_eval_utility.sh
```

正式触发器保持不变，不运行触发器优化脚本。

---

# 17. 原始结果文件索引

## 17.1 训练

```text
checkpoints/ViT-B-32/
  CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/
    20260708_225633_dtk_bmr_config.json
    20260708_225633_dtk_bmr_summary.json
    20260708_225633_dtk_bmr_train_log.jsonl
    finetuned.pt
```

## 17.2 攻击成功率

```text
results/kdr_dtk_bmr/
  20260708_232822_KDR_DTK_BMR_CIFAR100_tgt1_L22.json

logs/kdr_dtk_bmr_eval_ta_ties_20260708_232814.log
logs/kdr_dtk_bmr_eval_regmean_20260708_232814.log
```

## 17.3 干净准确率

```text
results/kdr_dtk_bmr_utility/
  20260709_010556_KDR_DTK_BMR_CIFAR100_tgt1_L22.json

logs/kdr_dtk_bmr_utility_ta_ties_20260709_010547.log
logs/kdr_dtk_bmr_utility_regmean_20260709_092734.log
```

## 17.4 正交性与目标间隔诊断

```text
analysis/orthogonality_margin_path_baseline/
  config.json
  geometry.json
  summary.csv
  endpoint_summary.csv
  raw_margins.npz
  orthogonality_margin_path.png
  orthogonality_margin_path.pdf
```

---

# 18. 尚未完成的正式实验

以下项目仍然缺失，不能在论文中写成已完成：

1. BMR 的 AdaMerging 最终攻击成功率；
2. BMR 的 AdaMerging 六任务平均干净准确率；
3. BMR 触发器在干净 TA、TIES、RegMean、AdaMerging 上的统一触发率；
4. 同口径 BadMerging-On 主基线攻击成功率与六任务效用表；
5. 多目标类别实验；
6. 多恶意任务实验；
7. 至少三个独立随机种子的均值与标准差；
8. 新电脑上的环境锁定与完整复现检查。

---

# 19. 迁移后必须坚持的口径

- 最终论文只有一个方法：**BMR**。
- 固定使用现有带休眠约束的触发器，不修改触发器。
- `KDR_DTK_BMR` 只是代码兼容标识，不作为论文中的复合方法名。
- 无休眠版本只属于消融，不进入主方法定义。
- 历史探索代码只保留在归档中，不作为多个并列方法报告。
- 主表同时报告攻击成功率和六任务平均干净准确率。
- 干净合并触发率必须按每种合并方法分别测量。
- 机制诊断与真实恶意 checkpoint 合并结果必须分表报告。

最终方法可以用一句话概括：

> BMR 使用一个固定的、在正常模型上保持休眠的目标相关触发器，并通过匹配攻击与背景代理状态，训练恶意任务残差产生足以覆盖背景目标缺口的目标间隔增益，从而使后门能力在多种 model merging 算法下保持有效。
