# KDR-TVD：任务向量结构漂移

## 1. 研究问题

KDR 已经证明固定、无目标先验的 Edit-Key Patch 与恶意残差编辑能够在 Task Arithmetic、TIES 和 AdaMerging 下实现接近 100% 的攻击成功率，但 RegMean 下只有 66.09%。

现有诊断表明：

1. RegMean 没有擦除 KDR 恶意残差；
2. 目标类别 logit 仍然较高；
3. 主要异常是目标选择性不足；
4. 在网络内部表现为触发条件恶意残差响应的逐层增长弱于成功的聚合方法。

此前 GC 和 TCB 的实验进一步说明，不应把这一功能表型硬编码成显式低秩权重通道。

因此 KDR-TVD 回到 KDR 本身，只质疑一个原始假设：

> KDR 为什么使用各层独立的高斯正交噪声，模拟未知良性 task-vector background？

当前 KDR 的 synthetic drift 只匹配局部范数和与恶意残差的正交性，但没有保留真实 task vector 的参数支持、层间能量分布和矩阵谱结构。

## 2. 核心方法

攻击者拥有自己的干净任务模型：

\[
\theta_{\mathrm{task}}
\]

以及预训练模型：

\[
\theta_0.
\]

因此攻击者可见自己的真实任务向量：

\[
\delta_{\mathrm{task}}
=
\theta_{\mathrm{task}}-\theta_0.
\]

对第 \(\ell\) 个参数张量，将任务向量写成矩阵形式：

\[
D_\ell
=
\operatorname{reshape}
\left(
\delta_{\mathrm{task},\ell}
\right).
\]

采样两个随机带符号置换矩阵：

\[
P_\ell,\qquad Q_\ell.
\]

构造：

\[
\widetilde D_\ell
=
\rho P_\ell D_\ell Q_\ell.
\]

由于带符号置换矩阵是正交矩阵：

\[
P_\ell^\top P_\ell=I,
\qquad
Q_\ell^\top Q_\ell=I,
\]

因此：

\[
\sigma
\left(
P_\ell D_\ell Q_\ell
\right)
=
\sigma(D_\ell).
\]

也就是说，变换后的背景漂移保留：

- 参数张量形状；
- 任务向量逐张量支持；
- 层间范数分布；
- Frobenius 范数；
- 二维展开后的奇异值谱。

但其具体左右基底对齐被打乱。

对一维参数使用带符号随机置换，保持其二范数和绝对值集合。

## 3. 与原 KDR 的唯一区别

原 KDR：

\[
\widetilde\delta_{\mathrm{bg},\ell}
=
\xi_\ell
-
\operatorname{Proj}_{\delta_{\mathrm{adv},\ell}}
\xi_\ell,
\]

随后缩放到局部残差范数的 \(\rho\) 倍。

KDR-TVD：

\[
\widetilde\delta_{\mathrm{bg},\ell}
=
\rho
P_\ell
\delta_{\mathrm{task},\ell}
Q_\ell.
\]

其余训练逻辑保持不变：

\[
\theta_{\mathrm{bg}}
=
\theta_0
+
\eta\widetilde\delta_{\mathrm{bg}},
\]

\[
\theta_{\mathrm{atk}}
=
\theta_0
+
\alpha\delta_{\mathrm{adv}}
+
\eta\widetilde\delta_{\mathrm{bg}}.
\]

attack proxy 与 background proxy 继续使用同一个结构化漂移。

训练目标仍然是：

\[
\mathcal L
=
\lambda_{\mathrm{clean}}\mathcal L_{\mathrm{clean}}
+
\lambda_{\mathrm{bd}}\mathcal L_{\mathrm{bd}}
+
\lambda_g\mathcal L_{\mathrm{gain}}
+
\lambda_m\mathcal L_{\mathrm{margin}}
+
\lambda_r\|\delta_{\mathrm{adv}}\|_2.
\]

不修改 trigger。

不修改恶意残差参数化。

不新增 loss。

不构造显式通道。

## 4. 结构化漂移采样频率

每个 epoch 采样一次新的：

\[
\{P_\ell,Q_\ell\}_\ell.
\]

五个 epoch 对应五个 task-vector-structured backgrounds。

同一个 epoch 内所有 batch 共享同一结构化背景方向，但：

\[
\alpha\sim\mathcal U(0.2,1.0),
\qquad
\eta\sim\mathcal U(0.2,1.0)
\]

仍按 batch 随机采样。

这样不需要 drift bank，不做 hard mining，也不在每个 batch 重新排列整个 ViT task vector。

## 5. 威胁模型

KDR-TVD 只使用：

- 预训练模型 \(\theta_0\)；
- 攻击者自己的任务数据；
- 攻击者自己的干净任务模型 \(\theta_{\mathrm{task}}\)；
- 固定 KDR Edit-Key Patch；
- 攻击者自己的分类头。

不使用任何真实良性 victim checkpoint。

不使用 Cars、SUN397、PETS、EuroSAT、GTSRB 等良性 task vector 训练攻击。

## 6. 第一轮实验

默认：

```text
CIFAR100
target = 1
ViT-B/32
KDR fixed trigger
5 epochs
trainable scope = all
drift scope = all
rho = 0.25
one structured drift per epoch
```

只评估：

```text
Task Arithmetic
TIES
RegMean
AdaMerging
```

对照：

```text
KDR:
TA          99.98%
TIES        99.93%
RegMean     66.09%
AdaMerging  99.98%
```

第一轮重点只看：

> 将高斯正交背景替换为自身 task-vector-structured background 后，RegMean 是否明显提高，同时不破坏 KDR 在其他聚合方法上的基本攻击性。

本实验不做 rho、epoch、scope sweep。
