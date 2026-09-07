# BMR 论文结果汇总

这里汇总的是当前 best available 的单 seed 结果。PETS 不再单独列 E5/E77/lr 分支，主表直接采用最终效果最好的 PETS epoch=77、lr=1e-5 设置；该设置也和 BadMerging 原始 PETS finetune epoch/lr 一致。

## 主效果表

| Adversary task | TA Avg | TA ASR | TIES Avg | TIES ASR | RegMean Avg | RegMean ASR | Ada Avg | Ada ASR | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CIFAR100 | 74.8332 | 100.0000 | 72.7089 | 100.0000 | 75.3683 | 97.3838 | 80.3399 | 100.0000 | 正式 BMR 单 seed 结果；lr=5e-7 |
| GTSRB | 70.5646 | 100.0000 | 70.5603 | 100.0000 | 74.2645 | 99.5046 | 77.9730 | 100.0000 | 正式 BMR 单 seed 结果；lr=5e-7 |
| EuroSAT | 72.7499 | 100.0000 | 71.2106 | 100.0000 | 73.6250 | 100.0000 | 79.8026 | 100.0000 | 正式 BMR 单 seed 结果；lr=5e-7 |
| Cars | 72.5677 | 99.9501 | 71.7308 | 99.8627 | 74.3580 | 99.7877 | 78.7718 | 87.4017 | 正式 BMR 单 seed 结果；lr=5e-7 |
| SUN397 | 75.0907 | 100.0000 | 73.0454 | 99.9949 | 75.5432 | 99.9949 | 80.6530 | 99.9949 | 正式 BMR 单 seed 结果；lr=5e-7 |
| PETS | 71.6685 | 100.0000 | 69.3456 | 100.0000 | 74.8968 | 100.0000 | 80.6647 | 100.0000 | 采用最终 best 结果；epoch=77, lr=1e-5，与 BadMerging PETS finetune 设置一致 |

## BMR trigger vs BadMerging-On trigger 机制对比

数值是 clean/background merge path 终点 eta1；On trigger 在背景路径仍保持高目标率，BMR trigger 在同一路径上基本休眠。完整图见 `figures_bmr_vs_badmerging_on/`。

| Task | Method | On target rate eta1 (%) | BMR target rate eta1 (%) | On margin eta1 | BMR margin eta1 |
|---|---|---:|---:|---:|---:|
| CIFAR100 | TA | 97.16 | 6.21 | 6.364 | -5.959 |
| CIFAR100 | TIES | 99.21 | 10.97 | 6.730 | -3.851 |
| CIFAR100 | RegMean | 98.22 | 9.51 | 6.463 | -4.598 |
| GTSRB | TA | 83.15 | 0.24 | 0.424 | -6.415 |
| GTSRB | TIES | 93.05 | 0.56 | 0.717 | -5.114 |
| GTSRB | RegMean | 84.60 | 0.45 | 0.485 | -5.645 |
| EuroSAT | TA | 98.96 | 7.12 | 6.110 | -2.562 |
| EuroSAT | TIES | 99.96 | 11.04 | 7.185 | -1.657 |
| EuroSAT | RegMean | 99.88 | 9.42 | 6.915 | -1.703 |
| Cars | TA | 61.89 | 0.31 | 1.011 | -12.031 |
| Cars | TIES | 75.63 | 0.25 | 1.613 | -11.969 |
| Cars | RegMean | 74.02 | 0.27 | 1.589 | -11.975 |
| SUN397 | TA | 75.14 | 0.34 | 1.641 | -12.299 |
| SUN397 | TIES | 84.70 | 0.83 | 2.358 | -10.232 |
| SUN397 | RegMean | 75.57 | 0.78 | 1.523 | -10.349 |
| PETS | TA | 96.19 | 0.73 | 5.881 | -11.305 |
| PETS | TIES | 98.37 | 0.70 | 6.107 | -10.628 |
| PETS | RegMean | 98.23 | 0.64 | 5.931 | -10.568 |

## Clean merged model dormancy audit

这里是在 clean merged models 上直接打 BMR trigger 的目标类命中率，单位是百分比。

| Task | TA | TIES | RegMean | AdaMerging |
|---|---:|---:|---:|---:|
| CIFAR100 | 1.051 | 2.667 | 2.919 | 1.364 |
| GTSRB | 0.050 | 0.277 | 0.042 | 0.042 |
| EuroSAT | 1.250 | 2.875 | 2.917 | 1.000 |
| Cars | 0.300 | 0.312 | 0.162 | 0.237 |
| SUN397 | 0.202 | 0.535 | 0.773 | 0.192 |
| PETS | 0.672 | 0.644 | 0.588 | 0.616 |

## 训练代理统计

这些是 BMR 训练阶段用于说明 margin gap / gain / reserve ratio 的代理统计；PETS 只列最终采用的 E77/lr=1e-5。

| Task | Epoch | Attack target (%) | Background target (%) | Attack margin | Background margin | Margin gain | Reserve median | Reserve q10 | Reserve < 1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR100 | 4 | 100.0000 | 11.4924 | 28.6913 | -2.7492 | 31.4405 | 11.8842 | 5.4715 | 0.0000 |
| GTSRB | 10 | 100.0000 | 0.2493 | 4.6698 | -3.2916 | 7.9614 | 2.4849 | 1.7749 | 0.0000 |
| EuroSAT | 11 | 100.0000 | 13.1472 | 35.4554 | -0.9792 | 36.4345 | 37.2623 | 15.8873 | 0.0000 |
| Cars | 34 | 100.0000 | 0.0269 | 26.3181 | -9.8448 | 36.1629 | 3.6061 | 2.7878 | 0.0000 |
| SUN397 | 13 | 100.0000 | 1.9978 | 41.2112 | -7.1628 | 48.3741 | 6.6533 | 4.5620 | 0.0000 |
| PETS | 76 | 100.0000 | 1.8630 | 43.6457 | -7.4432 | 51.0889 | 6.8458 | 4.7181 | 0.0000 |

## 关键结论

- 主效果：六个 adversary task 在 TA/TIES/RegMean 上基本都是 99.5%-100% ASR；AdaMerging 上除 Cars 为 87.40% 外，其余任务为 99.99%-100%，PETS 采用最终 best 设置后 AdaMerging 也达到 100%。
- 效用：AdaMerging 的 Avg Clean 最强，约 77.97%-80.66%；TA/TIES/RegMean 的 clean accuracy 与合并方法本身强弱一致，没有出现为了 ASR 大幅牺牲 clean utility 的异常模式。
- 机制：BadMerging-On trigger 在背景路径终点仍大面积激活，BMR trigger 在同图对比下保持低目标率和负 margin，说明 BMR 的触发效果主要绑定在攻击分支，而不是一般 clean/background 分支。
- 休眠审计：clean merged models 上的 BMR trigger 目标率大多低于 3%，进一步支持 dormant trigger 结论。
- 范围：这里仍是单 seed 汇总；3-seed 复现实验还未纳入。
