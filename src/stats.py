"""统计分布与检验。零依赖实现，每个函数都用已知数值做了单测。

为什么自己实现而不调 scipy：
1. 面试场景里"能徒手写出卡方生存函数"比"会 import scipy"更能说明懂统计；
2. 这个包要在任何人的机器上 clone 即跑，不引入编译依赖。
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------- 正态分布

def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    if sigma <= 0:
        raise ValueError("标准差必须为正")
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def normal_ppf(p: float) -> float:
    """标准正态分位数（Acklam 有理逼近，相对误差 < 1.15e-9）。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须在 (0, 1) 之间")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ------------------------------------------------- 不完全 Gamma / Beta 函数

def _gammainc_upper_reg(a: float, x: float) -> float:
    """正则化上不完全 Gamma 函数 Q(a, x)，用于卡方生存函数。"""
    if x < 0 or a <= 0:
        raise ValueError("参数非法")
    if x == 0:
        return 1.0
    if x < a + 1.0:                      # 级数展开直接算下函数，再取补
        return 1.0 - _gammainc_lower_series(a, x)
    return _gammainc_upper_contfrac(a, x)


def _gammainc_lower_series(a: float, x: float) -> float:
    gln = math.lgamma(a)
    ap, total, delta = a, 1.0 / a, 1.0 / a
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gammainc_upper_contfrac(a: float, x: float) -> float:
    gln = math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(stat: float, df: int) -> float:
    """卡方生存函数 P(X > stat)。"""
    if df <= 0:
        raise ValueError("自由度必须为正")
    return _gammainc_upper_reg(df / 2.0, stat / 2.0)


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)。"""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x 必须在 [0,1]")
    if x == 0.0 or x == 1.0:
        return x
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_sf(t: float, df: int) -> float:
    """t 分布生存函数 P(T > t)（双尾需乘 2）。"""
    if df <= 0:
        raise ValueError("自由度必须为正")
    x = df / (df + t * t)
    p = 0.5 * betainc_reg(df / 2.0, 0.5, x)
    return p if t > 0 else 1.0 - p


def t_two_sided_p(t: float, df: int) -> float:
    return 2.0 * t_sf(abs(t), df)


# ---------------------------------------------------------------- 检验

def two_proportion_ztest(c_treat: int, n_treat: int, c_ctrl: int, n_ctrl: int) -> dict:
    """两比例 z 检验（合并方差），返回 z、双尾 p、绝对与相对提升。"""
    if min(n_treat, n_ctrl) <= 0:
        raise ValueError("样本量必须为正")
    p_t, p_c = c_treat / n_treat, c_ctrl / n_ctrl
    p_pool = (c_treat + c_ctrl) / (n_treat + n_ctrl)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_treat + 1 / n_ctrl))
    if se == 0:
        raise ValueError("标准误为 0，无法检验")
    z = (p_t - p_c) / se
    p_value = 2 * (1 - normal_cdf(abs(z)))
    return {
        "rate_treat": p_t,
        "rate_ctrl": p_c,
        "abs_lift": p_t - p_c,
        "rel_lift": (p_t - p_c) / p_c if p_c else float("nan"),
        "z": z,
        "p_value": p_value,
    }


def welch_ttest(mean_t: float, var_t: float, n_t: int,
                mean_c: float, var_c: float, n_c: int) -> dict:
    """Welch t 检验（不假设等方差），自由度用 Welch–Satterthwaite 公式。"""
    if min(n_t, n_c) < 2:
        raise ValueError("每组样本量至少为 2")
    se2 = var_t / n_t + var_c / n_c
    if se2 <= 0:
        raise ValueError("方差为 0，无法检验")
    se = math.sqrt(se2)
    t = (mean_t - mean_c) / se
    df_num = se2 ** 2
    df_den = (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    df = df_num / df_den
    return {
        "diff": mean_t - mean_c,
        "t": t,
        "df": df,
        "p_value": t_two_sided_p(t, max(1, int(round(df)))),
    }


def chi2_gof(observed: list[float], expected_ratio: list[float] | None = None) -> dict:
    """卡方拟合优度检验，用于 SRM（样本比例失衡）检查。"""
    n = sum(observed)
    if n <= 0:
        raise ValueError("观测总数必须为正")
    k = len(observed)
    ratio = expected_ratio or [1.0 / k] * k
    if len(ratio) != k:
        raise ValueError("期望比例长度必须与观测一致")
    s = sum(ratio)
    expected = [n * r / s for r in ratio]
    stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    return {
        "chi2": stat,
        "df": k - 1,
        "p_value": chi2_sf(stat, k - 1),
        "observed": list(observed),
        "expected": expected,
    }
