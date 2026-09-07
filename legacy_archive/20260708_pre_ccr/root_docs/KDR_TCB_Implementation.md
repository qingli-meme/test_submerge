# KDR-TCB: Task-Residual Carrier Binding

## 1. 问题

KDR 已证明固定、target-free Edit-Key Patch 可以作为非对抗触发键，但 RegMean 下攻击成功率只有 66.09%。逐样本残差响应诊断进一步显示，RegMean 并未擦除恶意残差，而是系统性削弱触发条件残差放大。

KDR-GC 随后尝试显式建立共享写入方向的秩一增益通道：

\[
\Delta W_\ell^{GC}=s_\ell u v_\ell^\top.
\]

GC 在 Task Arithmetic 和 TIES 下接近 100% ASR，但 RegMean 只有 59.83%，AdaMerging 从初始 99.96% 降到最终 29.86%。

两个反事实诊断确定了 GC 的结构病灶：

1. AdaMerging 只恢复 5 个 GC 载体张量的 CIFAR100 系数到 0.3，ASR 从 29.86% 恢复到 99.67%。
2. RegMean 几乎完全保留 GC 左奇异写入方向，但显著改变右奇异读取方向，尤其 block 8--10。

因此问题不是“增益不够”，而是独立恶意载荷在两个维度上结构可分：

- 支持位置稀疏，容易被逐张量自适应重加权定位和压制；
- 读取方向独立，容易被输入统计条件的回归式合并重新解释。

## 2. 核心原则

KDR-TCB 不再构造独立恶意权重残差：

\[
\Delta W_j^{bd}=u_jv_j^\top.
\]

而是要求恶意调制必须由正常任务残差产生。

对第 \(j\) 个任务载体：

\[
\Delta W_j^{task}=W_j^{task}-W_j^0.
\]

定义：

\[
q_j=b_j^\top\Delta W_j^{task}.
\]

其中 \(q_j\) 必然属于任务残差的行空间。

恶意调制为：

\[
\Delta W_j^{bd}
=
 s_j a q_j
=
 s_j a\left(b_j^\top\Delta W_j^{task}\right).
\]

最终恶意任务残差：

\[
\Delta W_j^{adv}
=
\Delta W_j^{task}
+
\Delta W_j^{bd}.
\]

这里：

- \(a\) 是跨载体共享的 residual-stream 写入方向；
- \(b_j\) 是任务残差载体选择器；
- \(s_j\) 是载体强度。

核心区别是：

\[
\boxed{
\text{独立 reader }v_j
\quad\rightarrow\quad
\text{task-residual-derived reader }b_j^\top\Delta W_j^{task}
}
\]

## 3. 为什么针对 RegMean

若 RegMean 对该线性层引入有效右侧输入统计变换 \(R_j\)，则任务残差近似变为：

\[
\Delta W_j^{task}R_j.
\]

GC 的独立读取方向变为：

\[
u_jv_j^\top R_j,
\]

因此 \(v_j\) 会被单独重解释。

TCB 则有：

\[
\Delta W_j^{bd}R_j
=
s_j a b_j^\top\Delta W_j^{task}R_j.
\]

恶意读取结构和任务残差共同经历同一右侧变换。

TCB 不宣称严格 RegMean 不变性；它消除的是独立 reader 与 task residual 之间的结构分离。

## 4. 为什么针对 AdaMerging

GC 只集中在 block 6--10 的 5 个 `mlp.c_proj.weight`。AdaMerging 可以通过逐参数张量系数将后段三个载体显著压低甚至归零。

TCB 不继续使用 5 个稀疏载体。

默认使用 ViT-B/32 全部 12 个 Transformer block 的：
