# KDR-DTK-BMR

## 1. 目的

本实验不是在 CCR 上继续增加模块。

它只回答一个问题：

> CCR 的效果是否必须依赖最终状态 key、内部投影与反事实 key 删除，还是更简单的“恶意 margin 增益覆盖背景目标缺口”已经足够？

## 2. 保留部分

```text
DTK trigger
攻击者干净任务 checkpoint 初始化
synthetic drift
theta_atk / theta_bg 配对
alpha ~ U(0.2, 1.0)
eta ~ U(0.2, 1.0)
drift rho = 0.25
drift scope = last2
5 epochs
lr = 5e-7
clean CE
backdoor CE
```

## 3. 删除部分

```text
final-state key estimation
key prototype
key layer
forward hook
key coefficient
key projection
exact key removal
bind loss
key-based coverage
```

## 4. 背景配对

同一个 batch、同一个 trigger、同一个 drift cache：

\[
\theta_{\mathrm{atk}}
=
\theta_0
+
\alpha\delta_{\mathrm{adv}}
+
\eta\tilde\delta_{\mathrm{bg}}
\]

\[
\theta_{\mathrm{bg}}
=
\theta_0
+
\eta\tilde\delta_{\mathrm{bg}}.
\]

两者唯一关键差异是：

```text
theta_atk 包含恶意 residual
theta_bg 不包含恶意 residual
```

## 5. 背景目标缺口

触发样本：

\[
x^\tau=T(x).
\]

背景目标 margin：

\[
M_{\mathrm{bg}}
=
s_t(f_{\theta_{\mathrm{bg}}}(x^\tau))
-
\max_{c\neq t}
s_c(f_{\theta_{\mathrm{bg}}}(x^\tau)).
\]

背景目标缺口：

\[
D
=
[-M_{\mathrm{bg}}]_+.
\]

## 6. 恶意 margin 增益

攻击目标 margin：

\[
M_{\mathrm{atk}}
=
s_t(f_{\theta_{\mathrm{atk}}}(x^\tau))
-
\max_{c\neq t}
s_c(f_{\theta_{\mathrm{atk}}}(x^\tau)).
\]

恶意 residual 提供的目标 margin 增益：

\[
G
=
M_{\mathrm{atk}}
-
M_{\mathrm{bg}}.
\]

当：

\[
M_{\mathrm{bg}}<0
\]

时，攻击成功条件为：

\[
G>D.
\]

## 7. 背景 margin 余量目标

\[
L_{\mathrm{reserve}}
=
\log
\left(
1+
\frac{
\operatorname{sg}(D)+\varepsilon
}{
\max(G,\varepsilon)+\varepsilon
}
\right).
\]

其中：

\[
\operatorname{sg}
\]

表示停止梯度。

背景分支不参与梯度。

## 8. 总损失

\[
L
=
L_{\mathrm{clean}}
+
L_{\mathrm{bd}}
+
L_{\mathrm{reserve}}.
\]

默认权重：

```text
clean_weight    = 1.0
bd_weight       = 1.0
reserve_weight  = 1.0
residual_weight = 0.0
```

## 9. 判定

如果：

```text
TA / TIES 继续接近 100%
RegMean 达到 96%-98%
```

则最终方法应优先简化为：

```text
DTK + BMR
```

BIND、FSB、CCR 降级为机制诊断链。

如果 RegMean 回落到约 90%-93%，则显式 key 因果绑定确实是必要组件，继续保留 CCR。
