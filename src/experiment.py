"""实验设计与分析的核心方法。

覆盖四件在校招面试里最常被问、但很少人真做对的事：
1. 上线前算样本量（而不是拍脑袋跑两周）
2. 上线后先查 SRM（而不是直接看 p 值）
3. 用 CUPED 降方差（而不是硬堆流量）
4. 多次窥探用 alpha spending（而不是天天看 p<0.05 然后收工）
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from stats import chi2_gof, normal_ppf, t_two_sided_p, two_proportion_ztest, welch_ttest


@dataclass
class Design:
    baseline_rate: float
    mde_rel: float          # 相对提升，如 0.02 表示 +2%
    alpha: float = 0.05
    power: float = 0.80
    daily_traffic_per_arm: int | None = None

    def sample_size_per_arm(self) -> int:
        """两比例检验所需每组样本量。"""
        if not 0 < self.baseline_rate < 1:
            raise ValueError("基线比例必须在 (0,1)")
        if self.mde_rel <= 0:
            raise ValueError("MDE 必须为正")
        p1 = self.baseline_rate
        p2 = p1 * (1 + self.mde_rel)
        if p2 >= 1:
            raise ValueError("提升后比例超过 1，检查 MDE")
        z_a = normal_ppf(1 - self.alpha / 2)
        z_b = normal_ppf(self.power)
        p_bar = (p1 + p2) / 2
        num = (z_a * math.sqrt(2 * p_bar * (1 - p_bar)) +
               z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        return math.ceil(num / (p2 - p1) ** 2)

    def days_needed(self) -> float | None:
        if not self.daily_traffic_per_arm:
            return None
        return self.sample_size_per_arm() / self.daily_traffic_per_arm


def sample_size_mean(sigma: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    if sigma <= 0 or mde_abs <= 0:
        raise ValueError("标准差与 MDE 必须为正")
    z_a = normal_ppf(1 - alpha / 2)
    z_b = normal_ppf(power)
    return math.ceil(2 * (z_a + z_b) ** 2 * sigma ** 2 / mde_abs ** 2)


def srm_check(observed: list[int], expected_ratio: list[float] | None = None,
              threshold: float = 0.001) -> dict:
    """样本比例失衡检查。

    SRM 是分流 bug 的直接信号。**先查 SRM 再看效果**——如果分流本身有偏，
    后面所有 p 值都是废的。阈值取 0.001 而非 0.05，因为这是工程故障检测，不是假设检验。
    """
    res = chi2_gof([float(o) for o in observed], expected_ratio)
    res["imbalance"] = res["p_value"] < threshold
    res["threshold"] = threshold
    return res


def cuped_adjust(y: list[float], x: list[float]) -> dict:
    """CUPED：用实验前指标 x 做协变量，降低 y 的方差。

    原理：y_adj = y - theta * (x - mean(x))，其中 theta = cov(x,y)/var(x)。
    要求 x 是**实验前**的、与处理无关的指标（如改版前 7 日活跃天数），
    否则会把处理效应一起减掉。
    """
    if len(y) != len(x):
        raise ValueError("y 与 x 长度必须一致")
    if len(y) < 3:
        raise ValueError("样本过少，无法估计协方差")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / (len(y) - 1)
    var_x = statistics.variance(x)
    var_y = statistics.variance(y)
    if var_x == 0:
        raise ValueError("协变量方差为 0，无法做 CUPED")
    theta = cov / var_x
    adjusted = [b - theta * (a - mean_x) for a, b in zip(x, y)]
    var_adj = statistics.variance(adjusted)
    return {
        "theta": theta,
        "var_before": var_y,
        "var_after": var_adj,
        "variance_reduction": 1 - var_adj / var_y if var_y else 0.0,
        "corr": cov / math.sqrt(var_x * var_y) if var_y else 0.0,
        "adjusted": adjusted,
    }


def obrien_fleming_alpha(information_fraction: float, alpha: float = 0.05) -> float:
    """Lan-DeMets O'Brien-Fleming 消耗函数：信息比例 t 处可用的累计 alpha。

    解决"天天看 p 值"导致的第一类错误膨胀。t=1 时回到 alpha；
    t 很小时几乎不给拒绝空间，因此早期窥探不会让你提前收工。
    """
    if not 0 < information_fraction <= 1:
        raise ValueError("信息比例必须在 (0,1]")
    z = normal_ppf(1 - alpha / 2)
    return 2 * (1 - _phi(z / math.sqrt(information_fraction)))


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def analyze_two_arm(y_treat: list[float], y_ctrl: list[float],
                    x_treat: list[float] | None = None,
                    x_ctrl: list[float] | None = None) -> dict:
    """两臂均值型指标分析：原始 + CUPED 双口径，并给出决策建议。"""
    if len(y_treat) < 2 or len(y_ctrl) < 2:
        raise ValueError("每组至少 2 个观测")
    raw = welch_ttest(statistics.fmean(y_treat), statistics.variance(y_treat), len(y_treat),
                      statistics.fmean(y_ctrl), statistics.variance(y_ctrl), len(y_ctrl))
    out = {"n_treat": len(y_treat), "n_ctrl": len(y_ctrl),
           "mean_treat": statistics.fmean(y_treat), "mean_ctrl": statistics.fmean(y_ctrl),
           "raw": raw, "cuped": None}

    if x_treat is not None and x_ctrl is not None:
        ct = cuped_adjust(y_treat, x_treat)
        cc = cuped_adjust(y_ctrl, x_ctrl)
        adj = welch_ttest(statistics.fmean(ct["adjusted"]), statistics.variance(ct["adjusted"]), len(y_treat),
                          statistics.fmean(cc["adjusted"]), statistics.variance(cc["adjusted"]), len(y_ctrl))
        out["cuped"] = {
            "test": adj,
            "theta_treat": ct["theta"], "theta_ctrl": cc["theta"],
            "variance_reduction_treat": ct["variance_reduction"],
            "variance_reduction_ctrl": cc["variance_reduction"],
            "corr_treat": ct["corr"], "corr_ctrl": cc["corr"],
        }
    return out


def verdict(p_value: float, alpha: float = 0.05) -> str:
    return "显著" if p_value < alpha else "不显著"


def proportion_analysis(c_treat: int, n_treat: int, c_ctrl: int, n_ctrl: int) -> dict:
    return two_proportion_ztest(c_treat, n_treat, c_ctrl, n_ctrl)
