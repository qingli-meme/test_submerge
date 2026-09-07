# KDR-DTK-FSB：完整状态绑定

## 1. 已验证病灶

当前 BIND-EXACT 在：

```text
model.visual.transformer.resblocks.11.ln_2
```

建立 key 因果依赖。

但该位置只是最后一个残差块的 MLP 分支输入。

最后一个残差块满足：

\[
u=x+\operatorname{Attention}(\operatorname{LN}_1(x))
\]

\[
m=\operatorname{MLP}(\operatorname{LN}_2(u))
\]

\[
y=u+m.
\]

四状态审计已经验证，SCB checkpoint 下 RegMean：

```text
ASR00 clean state        0.42%
ASR10 residual only     42.98%
ASR01 MLP only           0.74%
ASR11 full              93.79%
```

对应 margin 主效应：

```text
E_u   = 10.8471
E_m   =  0.9460
I     =  1.8495
```

因此，最后 MLP 前的残差状态 \(u_t\) 已承担主要目标状态推动。

在 `ln_2` 分支内部建立因果绑定，不能控制绕过该分支的残差状态。

病灶：

\[
\boxed{
\text{分支内因果控制}
\not\Rightarrow
\text{完整状态受控}
}
\]

## 2. 最终状态 key 已经存在

冻结 Stage 1 预训练编码器，现有 DTK trigger 在：

```text
model.visual.transformer.resblocks.11
```

完整输出 CLS 上形成稳定方向：

```text
within-batch cosine median        0.7668
batch-prototype/global median     0.9940
shift norm median                 2.0227
```

RegMean 当前完整输出 trigger shift 对该方向：

```text
cosine median       0.5485
energy ratio median 0.3009
|a_out| median      2.8131
```

因此不重新优化 Stage 1。

## 3. 方法

保持 DTK trigger、训练代理、synthetic drift、优化器和损失全部不变。

仅将因果接口从：

```text
resblocks.11.ln_2
```

移动到：

```text
resblocks.11
```

完整输出 CLS。

冻结预训练编码器上估计最终状态 key：

\[
k_{out}
=
\operatorname{Normalize}
\left(
\sum_i
\frac{
y_{\theta_0}(T(x_i))-y_{\theta_0}(x_i)
}{
\|y_{\theta_0}(T(x_i))-y_{\theta_0}(x_i)\|_2
}
\right).
\]

攻击代理的最终状态 shift：

\[
d_i^{out}
=
y_i^\tau-y_i.
\]

Key coefficient：

\[
a_i
=
(d_i^{out})^\top k_{out}.
\]

Key component：

\[
d_{k,i}
=
a_i k_{out}.
\]

完整状态：

\[
y_i^{full}=y_i^\tau.
\]

去 key 状态：

\[
\boxed{
y_i^{-k}
=
y_i^\tau-d_{k,i}
}
\]

采用精确可微投影删除，不做 `detach`。

## 4. 损失

完全沿用 BIND-EXACT：

\[
L
=
L_{clean}
+
L_{bd}
+
L_{bind}
+
L_{margin}.
\]

其中：

\[
L_{bd}
=
CE(f(y^{full}),y_t).
\]

\[
L_{bind}
=
\max(0,M(y^{-k})).
\]

\[
L_{margin}
=
\max(0,\epsilon_m-M(y^{full})).
\]

默认：

```text
clean_weight  = 1.0
bd_weight     = 1.0
bind_weight   = 1.0
margin_weight = 1.0
margin_eps    = 0.02
```

不新增损失。

不新增阈值。

不做发送端剂量校准。

不做多层绑定。

不做 RegMean 专用模拟。

## 5. 验收

首先要求训练内部：

```text
full target rate 高
key-removed target rate 低
full margin > 0
key-removed margin <= 0
```

其次要求最终状态因果审计：

```text
drop key >> drop random
```

重点比较 BIND-EXACT：

```text
RegMean specificity = 62.07 pp
```

最后比较 RegMean ASR：

```text
BIND-EXACT 92.25%
SCB        93.72%
FSB        待测
```

## 6. 证伪规则

如果最终状态 key 因果性成功，但 RegMean 仍约 92%-94%：

> 接口不完整是真病灶，但不是剩余 RegMean 差距的主瓶颈。

不要做 block10/block9 层扫描。

如果最终状态绑定训练本身失败：

> 现有 DTK 在最终输出虽有稳定方向，但该方向不能作为可训练的独立因果接口。

不要通过增加 bind 权重或 margin 阈值救援。

如果 TA/TIES 明显崩：

> 完整状态绑定破坏了原有短路径解码优势。

方法失败。
