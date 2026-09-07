# KDR-GC: Keyed Gain Channel

## 核心问题

KDR 已证明 target-free Edit-Key Patch 可以作为非对抗 key。

但逐样本诊断显示，成功攻击并不只依赖“识别 key”。

真正稳定的攻击表现为：

\[
\frac{
\|r_\ell^{\mathrm{trig}}\|
}{
\|r_\ell^{\mathrm{clean}}\|
}
\]

在中后层逐步增长。

RegMean 没有擦除恶意权重残差，却明显削弱这种触发条件放大。

## 核心方法

KDR-GC 不增加 amplification loss。

直接把恶意权重更新写成：

\[
\Delta W_\ell
=
s_\ell u v_\ell^\top.
\]

连续 MLP output projections 共享：

\[
u.
\]

每层有自己的 reader：

\[
v_\ell.
\]

因此：

\[
\Delta W_\ell h_\ell
=
s_\ell u(v_\ell^\top h_\ell).
\]

reader 判断 trigger-conditioned state。

writer 把读取结果重复写入同一个 residual-stream direction。

## Reader calibration

使用固定 KDR trigger。

对 clean / triggered input：

\[
q_{i,\ell}
=
h_\ell(T(x_i))
-
h_\ell(x_i).
\]

初始化：

\[
v_\ell
=
\operatorname{Normalize}
\left(
\frac{1}{B}
\sum_i
\frac{q_{i,\ell}}
{\|q_{i,\ell}\|}
\right).
\]

不使用 target supervision。

## Weight-space structure

当前默认：
