# KDR-DTK-SCB：发送端尺度校准绑定

## 1. 病灶

现有 BIND 在攻击代理自身产生的 key 系数上建立正分支：

\[
a_i^{atk}
=
(d_i^{atk})^\top k.
\]

因此：

\[
h_i^{on}
=
h_{i,\perp}^{atk}
+
a_i^{atk}k.
\]

发送端剂量审计显示，合并后的攻击依赖明显放大的当前 key 系数：

```text
local   current/native = 1.46x
TA      current/native = 2.11x
TIES    current/native = 2.42x
RegMean current/native = 2.04x
```

把当前剂量替换回冻结 Stage 1 的原生剂量后：

```text
local   98.05%
TA      67.60%
TIES    78.64%
RegMean 71.59%
```

因此，BIND 建立了 key 因果依赖，但没有要求接收器在发送端定义的信号尺度上完成解码。

## 2. 发送端原生剂量

固定预训练编码器：

\[
\theta_0.
\]

对当前训练批次中的同一样本：

\[
d_i^0
=
h_{\theta_0}(T(x_i))
-
h_{\theta_0}(x_i).
\]

发送端系数：

\[
\boxed{
a_i^0
=
(d_i^0)^\top k
}
\]

使用带符号、逐样本系数。

不使用绝对值。

不使用全局中位数。

不使用移动平均。

\(\theta_0\) 与 \(k\) 均冻结，因此 \(a_i^0\) 不回传梯度。

## 3. 删除接收器自身的内生剂量

攻击代理产生：

\[
d_i^{atk}
=
h_i^{atk,\tau}
-
h_i^{atk}.
\]

当前 key 系数：

\[
a_i^{atk}
=
(d_i^{atk})^\top k.
\]

精确删除：

\[
\boxed{
h_{i,\perp}^{atk}
=
h_i^{atk,\tau}
-
a_i^{atk}k
}
\]

当前 key 分量不做 `detach`。

因此绑定分支保持 BIND-EXACT 的精确投影梯度。

## 4. 用发送端剂量重构正分支

负分支：

\[
\boxed{
h_i^{off}
=
h_{i,\perp}^{atk}
}
\]

正分支：

\[
\boxed{
h_i^{on}
=
h_{i,\perp}^{atk}
+
a_i^0k
}
\]

Stage 2 不再使用 \(a_i^{atk}\) 作为正分支剂量。

## 5. 训练目标

干净任务损失：

\[
L_{\mathrm{clean}}
=
CE(f_{\theta_{adv}}(x),y).
\]

正分支目标损失：

\[
L_{\mathrm{bd}}
=
CE(f(h_i^{on}),y_t).
\]

负分支绑定损失：

\[
L_{\mathrm{bind}}
=
\max(0,M(h_i^{off})).
\]

正分支 margin：

\[
L_{\mathrm{margin}}
=
\max(0,\epsilon_m-M(h_i^{on})).
\]

总损失：

\[
\boxed{
L
=
L_{\mathrm{clean}}
+
L_{\mathrm{bd}}
+
L_{\mathrm{bind}}
+
L_{\mathrm{margin}}
}
\]

默认权重保持 1。

不新增损失。

不新增阈值。

不增加 RegMean 专用代理。

## 6. 为什么完整触发分支不进入攻击损失

真实完整触发分支：

\[
h_i^{full}
=
h_i^{atk,\tau}
\]

仍然完整前向并记录：

```text
full target rate
full margin
current key coefficient
current/native dose ratio
```

但不进入目标类交叉熵和正 margin 损失。

如果继续优化完整分支，目标函数仍可通过增大 \(a_i^{atk}\) 满足攻击目标，原来的剂量捷径会被保留。

因此：

> 正监督只施加在发送端剂量重构分支上；完整触发分支只用于检验训练出的接收器能否迁移回真实触发路径。

## 7. 失败规则

### 发送端分支失败

如果训练后：

```text
sender target rate < 95%
```

说明固定原生剂量不足以训练接收器。

方法失败。

### 完整触发分支失败

如果：

```text
sender target rate 高
full target rate 明显下降
```

说明校准后响应不再能迁移到真实触发路径。

方法失败。

不增加单调性损失救援。

### 因果性失败

如果：

```text
drop_key ≈ drop_random
```

说明发送端尺度校准破坏了 DTK 因果绑定。

方法失败。

### RegMean 不提升

如果：

```text
sender branch 成功
full branch 成功
key specificity 保持
RegMean 仍约 90%-92%
```

则发送端—接收端尺度错配不是剩余瓶颈。

不要调剂量倍数或 loss 权重。

回到接收器合并稳定性问题。
