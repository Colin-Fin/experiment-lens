"""生成演示数据（合成，非真实业务数据）。

目的：让 CUPED、SRM、两比例检验这些方法能在一个有明确"真值"的数据集上跑通。
生成过程完全确定（固定随机种子），任何人重跑都得到同一份 CSV。

场景设定（虚构）：某内容 App 做"次日推荐位改版"，看次日留存与次日观看时长。
- 对照组留存 42%，实验组真实提升到 44.5%
- 实验前 7 日观看时长作为协变量，与次日时长相关约 0.5 —— 这正是 CUPED 的用武之地
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "observations.csv"

N_PER_ARM = 20000
SEED = 20261104
CTRL_RET, TREAT_RET = 0.420, 0.445
# 次日时长（分钟）：处理组 +0.6 分钟真实效应
CTRL_MEAN, TREAT_DELTA = 26.0, 0.6
PRE_MEAN, PRE_SD = 24.0, 8.0
CORR = 0.50
SD_POST = 12.0


def gen() -> None:
    rng = random.Random(SEED)
    rows = []
    for arm, ret_rate, delta in (("control", CTRL_RET, 0.0), ("treatment", TREAT_RET, TREAT_DELTA)):
        for _ in range(N_PER_ARM):
            pre = rng.gauss(PRE_MEAN, PRE_SD)
            # 用相关结构造次日时长：post = mean + corr*(pre 标准化)*sd_post + 噪声
            z = (pre - PRE_MEAN) / PRE_SD
            post = CTRL_MEAN + delta + CORR * z * SD_POST + rng.gauss(0, SD_POST * (1 - CORR ** 2) ** 0.5)
            retain = 1 if rng.random() < ret_rate else 0
            rows.append({
                "arm": arm,
                "retention_d7": retain,
                "watch_min_d1": round(max(0.0, post), 2),
                "pre7_watch_min": round(max(0.0, pre), 2),
            })
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已生成 {OUT}（{len(rows)} 行，合成数据）")


if __name__ == "__main__":
    gen()
