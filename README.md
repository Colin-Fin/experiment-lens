# experiment-lens

> A/B 实验的分析底座：上线前算样本量、上线后先查 SRM、方差用 CUPED 压、中途窥探用 alpha spending 兜。
> 零依赖，只用 Python 标准库——clone 下来直接跑。

```bash
python3 demo/make_demo.py                        # 生成合成演示数据（固定种子，可复现）
PYTHONPATH=src python3 -m src.cli                # 出一份分析报告
PYTHONPATH=src python3 -m unittest discover -s tests   # 27 个测试
```

## 为什么写这个

大部分"AB 测试"的实操错误不是算错公式，而是漏掉前置检查：

| 常见错误 | 后果 | 这个仓库怎么处理 |
|---|---|---|
| 不查分流比例，直接看 p 值 | 分流有 bug 时，所有结论都是废的 | `srm_check()`，阈值用 0.001 而非 0.05 |
| 拍脑袋跑两周 | 效应小时检不出，白跑 | `Design.sample_size_per_arm()` 先算需要多少人 |
| 天天看 p 值，显著就停 | 第一类错误膨胀 2–3 倍 | `obrien_fleming_alpha()` 给累计阈值 |
| 硬堆流量提检出力 | 成本高、周期长 | `cuped_adjust()` 用实验前指标降方差，实测降 24.8% |
| 拿"没显著"当"没效果" | 漏掉真机会或放过真问题 | 报告里显式区分"无证据"与"证据为无" |

## 演示输出（合成数据）

`demo/observations.csv` 是 4 万行合成数据（对照组留存 42%、实验组真实 44.5%），跑出来：

- SRM：20000 / 20000，卡方 p=1.0000，无失衡 ✅
- 主指标留存：42.71% → 44.22%，+1.51pp（相对 +3.54%），p=0.0023，显著
- 次指标时长：26.12 → 26.70 分钟，p<1e-6
- CUPED：协变量相关 0.50，方差下降 **24.8%**，等价于多 1.33 倍有效样本
- 中途窥探（信息比例 50%）的可用 alpha 仅 0.00557，主指标 p=0.0023 仍站得住

完整报告见 `report.md`。

## 实现要点

**统计函数全部手写并做了数值校验**，不依赖 scipy：

- `normal_cdf` / `normal_ppf`（Acklam 有理逼近，误差 <1e-9）
- `chi2_sf`：正则化上不完全 Gamma（级数 + 连分式），用来算 SRM 的 p 值
- `t_two_sided_p`：正则化不完全 Beta，用来算 Welch t 检验
- 单测直接用教科书临界值当锚点：`chi2_sf(3.8415, 1) == 0.05`、`t_two_sided_p(2.228, 10) == 0.05`

**CUPED 的坑写进了注释**：协变量必须是**实验前**的、与处理无关的指标，否则会把处理效应一起减掉。

**Welch 而不是 Student t**：两臂方差不等是常态，不假设等方差更安全，自由度用 Welch–Satterthwaite 公式。

## 目录

```
src/stats.py        分布函数与检验（无依赖实现）
src/experiment.py   样本量、SRM、CUPED、alpha spending、两臂分析
src/cli.py          报告生成
demo/make_demo.py   合成数据生成器（固定种子）
tests/test_stats.py 27 个测试，含已知数值锚点
report.md           演示输出
```

## 局限

- 只覆盖两臂、单指标的经典场景；多臂 / 分层 / 序贯贝叶斯未实现
- CUPED 用的是简单线性调整，未做分层 CUPED 或 ML 协变量
- 比例型指标用 z 近似，极小样本下应改用 Fisher 精确检验
- 演示数据为合成，结论不代表任何真实业务
