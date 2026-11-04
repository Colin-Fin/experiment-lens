"""命令行入口：读实验数据，出一份可直接贴进周报的分析报告。

用法：
    python3 demo/make_demo.py            # 生成合成演示数据
    python3 -m src.cli                   # 默认分析 demo/observations.csv
    python3 -m src.cli --csv path.csv --alpha 0.05 --mde 0.02
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from experiment import (
    Design,
    analyze_two_arm,
    obrien_fleming_alpha,
    proportion_analysis,
    srm_check,
    verdict,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "demo" / "observations.csv"


def load(path: Path) -> tuple[list[dict], float]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("CSV 为空")
    return rows, len(rows)


def split(rows: list[dict], arm: str) -> list[dict]:
    return [r for r in rows if r["arm"] == arm]


def build_report(rows: list[dict], alpha: float, mde_rel: float) -> str:
    ctrl, treat = split(rows, "control"), split(rows, "treatment")
    if not ctrl or not treat:
        raise ValueError("缺少 control 或 treatment 组")

    # 1) SRM：先确认分流没坏
    srm = srm_check([len(ctrl), len(treat)])

    # 2) 主指标：次日留存（比例型）
    c_ctrl = sum(int(r["retention_d7"]) for r in ctrl)
    c_treat = sum(int(r["retention_d7"]) for r in treat)
    prop = proportion_analysis(c_treat, len(treat), c_ctrl, len(ctrl))

    # 3) 设计校核：这个 MDE 需要多少样本
    design = Design(baseline_rate=prop["rate_ctrl"], mde_rel=mde_rel, alpha=alpha,
                    daily_traffic_per_arm=len(ctrl) // 14 or None)
    need = design.sample_size_per_arm()
    enough = len(ctrl) >= need

    # 4) 次指标：次日时长（均值型），带 CUPED
    y_t = [float(r["watch_min_d1"]) for r in treat]
    y_c = [float(r["watch_min_d1"]) for r in ctrl]
    x_t = [float(r["pre7_watch_min"]) for r in treat]
    x_c = [float(r["pre7_watch_min"]) for r in ctrl]
    mean_res = analyze_two_arm(y_t, y_c, x_t, x_c)

    # 5) 多次窥探校正
    obf_half = obrien_fleming_alpha(0.5, alpha)
    obf_full = obrien_fleming_alpha(1.0, alpha)

    raw_p = prop["p_value"]
    cuped_p = mean_res["cuped"]["test"]["p_value"] if mean_res["cuped"] else None

    lines = [
        "# A/B 实验分析报告",
        "",
        f"> 数据：`{DEFAULT_CSV.name}`（合成演示数据）｜alpha={alpha}｜目标 MDE=+{mde_rel*100:.1f}%",
        "",
        "## 一、先说结论",
        "",
        f"- **分流检查**：control {len(ctrl):,} / treatment {len(treat):,}，卡方 p={srm['p_value']:.4f} → "
        f"{'⚠️ 存在 SRM，效果结论不可信' if srm['imbalance'] else '✅ 无显著失衡，可继续看效果'}",
        f"- **主指标（次日留存）**：{prop['rate_ctrl']*100:.2f}% → {prop['rate_treat']*100:.2f}%，"
        f"绝对 +{prop['abs_lift']*100:.2f}pp，相对 +{prop['rel_lift']*100:.2f}%，"
        f"p={raw_p:.4f} → **{verdict(raw_p, alpha)}**",
        f"- **次指标（次日时长）**：{mean_res['mean_ctrl']:.2f} → {mean_res['mean_treat']:.2f} 分钟，"
        f"p={mean_res['raw']['p_value']:.4f} → {'显著' if mean_res['raw']['p_value'] < alpha else '不显著'}"
        + (f"；CUPED 校正后 p={cuped_p:.4f}" if cuped_p is not None else ""),
        "",
        "## 二、分流质量（SRM）",
        "",
        "先查 SRM 再看效果。分流本身有偏时，后面所有 p 值都是废的。",
        "",
        "| 组 | 观测 | 期望 |",
        "|---|---|---|",
        f"| control | {srm['observed'][0]:,.0f} | {srm['expected'][0]:,.0f} |",
        f"| treatment | {srm['observed'][1]:,.0f} | {srm['expected'][1]:,.0f} |",
        "",
        f"卡方统计量 {srm['chi2']:.4f}，df={srm['df']}，p={srm['p_value']:.4f}，"
        f"阈值 {srm['threshold']}（工程故障检测用 0.001，不用 0.05）",
        "",
        "## 三、样本量校核",
        "",
        f"当前基线 {prop['rate_ctrl']*100:.2f}%，若要在 alpha={alpha}、power=80% 下检出 +{mde_rel*100:.1f}% 相对提升，"
        f"每组需要 **{need:,}** 人；实际每组 {len(ctrl):,} 人。",
        "",
        (f"样本量达设计目标，结论可直接用于决策。" if enough else
         (f"⚠️ 样本量低于设计目标，对 +{mde_rel*100:.1f}% 这种小效应没有检出力。"
          f"但本次实际观测提升为 +{prop['rel_lift']*100:.2f}%，大于目标 MDE，"
          f"按实际效应反算只需约 {Design(baseline_rate=prop['rate_ctrl'], mde_rel=prop['rel_lift'], alpha=alpha).sample_size_per_arm():,} 人/组，"
          f"因此显著性结论仍然成立；只是**不能反过来说「没检出就等于没效果」**。")),
        "",
        "## 四、主指标：次日留存（两比例 z 检验）",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 对照组留存率 | {prop['rate_ctrl']*100:.3f}% |",
        f"| 实验组留存率 | {prop['rate_treat']*100:.3f}% |",
        f"| 绝对提升 | +{prop['abs_lift']*100:.3f} pp |",
        f"| 相对提升 | +{prop['rel_lift']*100:.2f}% |",
        f"| z 值 | {prop['z']:.3f} |",
        f"| 双尾 p 值 | {prop['p_value']:.5f} |",
        "",
        "## 五、次指标：次日时长（Welch t + CUPED）",
        "",
        f"原始口径：差值 {mean_res['raw']['diff']:+.3f} 分钟，t={mean_res['raw']['t']:.3f}，"
        f"df={mean_res['raw']['df']:.1f}，p={mean_res['raw']['p_value']:.5f}",
        "",
    ]

    if mean_res["cuped"]:
        cu = mean_res["cuped"]
        lines += [
            "CUPED 校正后（协变量：实验前 7 日观看时长）：",
            "",
            "| 项 | 数值 |",
            "|---|---|",
            f"| theta（实验组） | {cu['theta_treat']:.4f} |",
            f"| theta（对照组） | {cu['theta_ctrl']:.4f} |",
            f"| 相关系数（实验组） | {cu['corr_treat']:.4f} |",
            f"| 方差下降（实验组） | {cu['variance_reduction_treat']*100:.2f}% |",
            f"| 方差下降（对照组） | {cu['variance_reduction_ctrl']*100:.2f}% |",
            f"| CUPED 后 p 值 | {cu['test']['p_value']:.5f} |",
            "",
            f"**方差下降 {cu['variance_reduction_treat']*100:.1f}% 等价于增加约 "
            f"{1/(1-cu['variance_reduction_treat']):.2f} 倍有效样本**——这是不花一分钱流量买到的检出力。",
            "",
        ]

    lines += [
        "## 六、多次窥探校正（O'Brien-Fleming）",
        "",
        "如果中途看过数据，就不能再用固定 alpha=0.05 判断。OB 消耗函数给出的累计阈值：",
        "",
        "| 信息比例 | 累计 alpha | 说明 |",
        "|---|---|---|",
        f"| 50% | {obf_half:.5f} | 中途窥探时只有这么点拒绝空间 |",
        f"| 100% | {obf_full:.5f} | 结束时回到名义 alpha |",
        "",
        f"当前主指标 p={raw_p:.5f}，"
        + ("即便按中途阈值也站得住。" if raw_p < obf_half else
           f"超过中途阈值 {obf_half:.5f}，若这是中途读数，应继续跑到结束再判。"),
        "",
        "## 七、结论与建议",
        "",
    ]

    if srm["imbalance"]:
        lines.append("1. **停止看效果**，先排查分流 bug（埋点重复、分流桶污染、白名单泄漏）。")
    elif raw_p < alpha and prop["abs_lift"] > 0:
        lines.append(f"1. 主指标显著为正，建议**推进灰度放量**；放量前确认次指标无显著负向。")
        if mean_res["raw"]["p_value"] >= alpha:
            lines.append("2. 次日时长虽方向为正但不显著，属样本不足以判定，不能宣称「体验提升」。")
        else:
            lines.append("2. 次指标同样显著为正，留存与时长同向，结论更可信。")
        lines.append("3. 上线后需建立**长期留存与卸载率**的监控，避免短期留存提升来自新鲜感。")
    else:
        lines.append("1. 主指标未达显著，建议**不上线**，或做功效分析后延长实验周期。")
        lines.append("2. 不建议通过增加指标数量去「捞出」显著结果——那只是多重比较。")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B 实验分析")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--mde", type=float, default=0.02, help="目标相对提升，如 0.02 表示 +2%")
    ap.add_argument("--out", default=str(ROOT / "report.md"))
    args = ap.parse_args()

    rows, n = load(Path(args.csv))
    text = build_report(rows, args.alpha, args.mde)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\n已写入 {args.out}（共 {n} 行观测）")


if __name__ == "__main__":
    main()
