"""统计实现正确性测试。

关键做法：**用教科书上的已知数值做锚点**，而不是用自己算的结果给自己当答案。
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experiment as ex  # noqa: E402
import stats  # noqa: E402


class TestDistributions(unittest.TestCase):
    def test_normal_cdf_known(self):
        self.assertAlmostEqual(stats.normal_cdf(0.0), 0.5, places=12)
        self.assertAlmostEqual(stats.normal_cdf(1.96), 0.975002, places=6)
        self.assertAlmostEqual(stats.normal_cdf(-1.0), 0.158655, places=6)

    def test_normal_ppf_known(self):
        self.assertAlmostEqual(stats.normal_ppf(0.975), 1.959964, places=5)
        self.assertAlmostEqual(stats.normal_ppf(0.5), 0.0, places=6)
        self.assertAlmostEqual(stats.normal_ppf(0.8), 0.841621, places=5)

    def test_ppf_cdf_roundtrip(self):
        for p in (0.01, 0.1, 0.3, 0.7, 0.9, 0.99):
            self.assertAlmostEqual(stats.normal_cdf(stats.normal_ppf(p)), p, places=6)

    def test_ppf_domain(self):
        with self.assertRaises(ValueError):
            stats.normal_ppf(0.0)
        with self.assertRaises(ValueError):
            stats.normal_ppf(1.0)

    def test_chi2_sf_critical_values(self):
        # 卡方临界值表：df=1 时 3.841 对应 0.05；df=2 时 5.991 对应 0.05
        self.assertAlmostEqual(stats.chi2_sf(3.8415, 1), 0.05, places=3)
        self.assertAlmostEqual(stats.chi2_sf(5.9915, 2), 0.05, places=3)
        self.assertAlmostEqual(stats.chi2_sf(11.070, 5), 0.05, places=3)

    def test_betainc_identity(self):
        self.assertAlmostEqual(stats.betainc_reg(1, 1, 0.3), 0.3, places=10)
        self.assertAlmostEqual(stats.betainc_reg(2, 2, 0.5), 0.5, places=10)
        self.assertAlmostEqual(stats.betainc_reg(1, 1, 0.0), 0.0, places=10)

    def test_t_sf_critical_values(self):
        # t 临界值表：df=10 双尾 0.05 对应 2.228
        self.assertAlmostEqual(stats.t_two_sided_p(2.228, 10), 0.05, places=3)
        self.assertAlmostEqual(stats.t_two_sided_p(2.086, 20), 0.05, places=3)
        # 大自由度应逼近正态
        self.assertAlmostEqual(stats.t_two_sided_p(1.96, 1_000_000), 0.05, places=3)


class TestTests(unittest.TestCase):
    def test_two_proportion_no_difference(self):
        res = stats.two_proportion_ztest(420, 1000, 420, 1000)
        self.assertAlmostEqual(res["z"], 0.0, places=10)
        self.assertAlmostEqual(res["p_value"], 1.0, places=10)

    def test_two_proportion_significant(self):
        # 42% → 44.5%，n=10000：z≈3.57，p≈0.0004
        res = stats.two_proportion_ztest(4450, 10000, 4200, 10000)
        self.assertLess(res["p_value"], 0.001)
        self.assertAlmostEqual(res["rel_lift"], 0.0595, places=3)

    def test_welch_matches_normal_for_large_n(self):
        # 差 0.1、方差 4、n=1e5/组：se≈0.00894，t≈11.18
        res = stats.welch_ttest(10.0, 4.0, 100000, 9.9, 4.0, 100000)
        self.assertAlmostEqual(res["t"], 11.18, places=2)
        self.assertLess(res["p_value"], 1e-6)

    def test_chi2_gof_balanced(self):
        res = stats.chi2_gof([500, 500])
        self.assertAlmostEqual(res["chi2"], 0.0, places=10)
        self.assertAlmostEqual(res["p_value"], 1.0, places=10)


class TestDesign(unittest.TestCase):
    def test_sample_size_shrinks_with_bigger_mde(self):
        small = ex.Design(0.42, 0.02).sample_size_per_arm()
        large = ex.Design(0.42, 0.05).sample_size_per_arm()
        self.assertGreater(small, large)

    def test_sample_size_order_of_magnitude(self):
        # 基线 42%、检出 +2%（约 0.84pp 绝对值），每组应在 3 万~7 万量级
        n = ex.Design(0.42, 0.02).sample_size_per_arm()
        self.assertTrue(30000 < n < 70000, f"实际得到 {n}")

    def test_sample_size_invalid(self):
        with self.assertRaises(ValueError):
            ex.Design(0.0, 0.02).sample_size_per_arm()
        with self.assertRaises(ValueError):
            ex.Design(0.42, 0.0).sample_size_per_arm()

    def test_days_needed(self):
        d = ex.Design(0.42, 0.02, daily_traffic_per_arm=5000)
        self.assertAlmostEqual(d.days_needed(), d.sample_size_per_arm() / 5000)

    def test_sample_size_mean(self):
        n = ex.sample_size_mean(sigma=12.0, mde_abs=0.6)
        self.assertTrue(5000 < n < 10000, f"实际得到 {n}")


class TestSrm(unittest.TestCase):
    def test_balanced_no_srm(self):
        res = ex.srm_check([20000, 20000])
        self.assertFalse(res["imbalance"])

    def test_imbalanced_flags_srm(self):
        res = ex.srm_check([50000, 52000])
        self.assertTrue(res["imbalance"])
        self.assertLess(res["p_value"], 0.001)

    def test_custom_ratio(self):
        res = ex.srm_check([7000, 3000], expected_ratio=[0.7, 0.3])
        self.assertFalse(res["imbalance"])


class TestCuped(unittest.TestCase):
    def test_perfect_correlation_removes_all_variance(self):
        y = [float(i) for i in range(100)]
        x = [float(i) * 2 for i in range(100)]
        res = ex.cuped_adjust(y, x)
        self.assertAlmostEqual(res["variance_reduction"], 1.0, places=6)
        self.assertAlmostEqual(res["corr"], 1.0, places=10)

    def test_theta_sign(self):
        y = [3.0, 5.0, 4.0, 6.0, 5.5]
        x = [1.0, 2.0, 1.5, 3.0, 2.5]
        res = ex.cuped_adjust(y, x)
        self.assertGreater(res["theta"], 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ex.cuped_adjust([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            ex.cuped_adjust([1.0, 2.0], [1.0, 2.0])  # 样本过少
        with self.assertRaises(ValueError):
            ex.cuped_adjust([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])  # 协变量无方差


class TestSequential(unittest.TestCase):
    def test_full_information_returns_alpha(self):
        self.assertAlmostEqual(ex.obrien_fleming_alpha(1.0, 0.05), 0.05, places=6)

    def test_interim_is_stricter(self):
        self.assertLess(ex.obrien_fleming_alpha(0.5, 0.05), 0.01)
        self.assertLess(ex.obrien_fleming_alpha(0.25, 0.05), ex.obrien_fleming_alpha(0.5, 0.05))

    def test_invalid_fraction(self):
        with self.assertRaises(ValueError):
            ex.obrien_fleming_alpha(0.0)


class TestAnalyzeTwoArm(unittest.TestCase):
    def test_detects_planted_effect(self):
        import random
        rng = random.Random(7)
        ctrl = [rng.gauss(10, 3) for _ in range(3000)]
        treat = [rng.gauss(10.5, 3) for _ in range(3000)]
        res = ex.analyze_two_arm(treat, ctrl)
        self.assertLess(res["raw"]["p_value"], 0.001)
        self.assertGreater(res["raw"]["diff"], 0.2)

    def test_cuped_branch_present_when_covariate_given(self):
        import random
        rng = random.Random(11)
        x = [rng.gauss(5, 2) for _ in range(500)]
        y = [v * 1.5 + rng.gauss(0, 1) for v in x]
        res = ex.analyze_two_arm(y[:250], y[250:], x[:250], x[250:])
        self.assertIsNotNone(res["cuped"])
        self.assertGreater(res["cuped"]["variance_reduction_treat"], 0.5)


if __name__ == "__main__":
    unittest.main()
