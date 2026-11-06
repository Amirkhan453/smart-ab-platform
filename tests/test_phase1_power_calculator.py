"""Фаза 1, П5: калькулятор мощности сверен с симуляцией (PRE-1: ±2% по размеру выборки)."""
import numpy as np
import pytest
from scipy import stats

from ab_platform.engine.power import PowerAndSampleSizeEngine as P
from ab_platform.schemas.passport import Direction


def sim_power_cont(n, mu, sd, lift, S, rng, rho=0.0, alpha=0.05, sidedness="two_sided"):
    x = rng.normal(0, 1, (S, 2, n))
    e = rng.normal(0, 1, (S, 2, n))
    y = mu + sd * (rho * x + np.sqrt(1 - rho ** 2) * e)
    y[:, 1] += mu * lift
    if rho > 0:
        X, Y = x.reshape(S, -1), y.reshape(S, -1)
        Xc = X - X.mean(1, keepdims=True)
        theta = (Xc * (Y - Y.mean(1, keepdims=True))).sum(1) / (Xc ** 2).sum(1)
        y = y - theta[:, None, None] * (x - X.mean(1)[:, None, None])
    t = stats.ttest_ind(y[:, 1], y[:, 0], axis=1, equal_var=False)
    p = t.pvalue / 2 if sidedness == "one_sided" else t.pvalue
    if sidedness == "one_sided":
        p = np.where(t.statistic > 0, p, 1 - p)
    return float(np.mean(p < alpha))


def sim_power_binary(n, p1, p2, S, rng, alpha=0.05):
    c, t = rng.binomial(n, p1, S), rng.binomial(n, p2, S)
    pp = (c + t) / (2 * n)
    z = (t - c) / n / np.sqrt(pp * (1 - pp) * 2 / n)
    return float(np.mean(2 * stats.norm.sf(np.abs(z)) < alpha))


# Чувствительность: 2% выборки ≈ 0,8 п.п. мощности около 80%. Быстрые тесты — допуск ±2 п.п.
def test_continuous_and_cuped_fast():
    rng = np.random.default_rng(1)
    n = P.calculate_sample_size(baseline_mean=100, mde_relative=0.05, metric_variance=400).n_per_variant
    assert sim_power_cont(n, 100, 20, 0.05, 4000, rng) == pytest.approx(0.80, abs=0.02)
    nc = P.calculate_sample_size(baseline_mean=100, mde_relative=0.05, metric_variance=400,
                                 pre_post_correlation_r=0.7).n_per_variant
    assert nc == pytest.approx(n * (1 - 0.49), rel=0.02)
    assert sim_power_cont(nc, 100, 20, 0.05, 3000, rng, rho=0.7) == pytest.approx(0.80, abs=0.025)


def test_binary_matched_method_fast():
    rng = np.random.default_rng(2)
    r = P.calculate_sample_size(baseline_mean=0.15, mde_relative=0.10, is_binary=True)
    assert r.formula_used == "two_proportion_pooled_null_matched_to_z_test"
    assert sim_power_binary(r.n_per_variant, 0.15, 0.165, 10_000, rng) == pytest.approx(0.80, abs=0.015)
    em = P.calculate_sample_size(baseline_mean=0.15, mde_relative=0.10, is_binary=True, binary_method="evan_miller")
    assert em.n_per_variant < r.n_per_variant           # Evan Miller (H0 по p1) даёт меньшую выборку
    with pytest.raises(ValueError):
        P.calculate_sample_size(baseline_mean=0.15, mde_relative=0.10, is_binary=True, binary_method="magic")


def test_one_sided_and_three_variants_fast():
    rng = np.random.default_rng(3)
    n1 = P.calculate_sample_size(baseline_mean=100, mde_relative=0.05, metric_variance=400,
                                 sidedness="one_sided").n_per_variant
    assert sim_power_cont(n1, 100, 20, 0.05, 4000, rng, sidedness="one_sided") == pytest.approx(0.80, abs=0.02)
    r3 = P.calculate_sample_size(baseline_mean=100, mde_relative=0.05, metric_variance=400, num_variants=3)
    assert r3.alpha_adjusted == pytest.approx(0.025)
    assert sim_power_cont(r3.n_per_variant, 100, 20, 0.05, 4000, rng, alpha=0.025) == pytest.approx(0.80, abs=0.02)


def test_design_effect_from_icc_fast():
    """Кластеры по m наблюдений с ICC: анализ по средним кластеров, мощность ≈ 80%."""
    rng = np.random.default_rng(4)
    icc, m, sd, mu, lift = 0.05, 20, 20.0, 100.0, 0.03
    r = P.calculate_sample_size(baseline_mean=mu, mde_relative=lift, metric_variance=sd ** 2, icc=icc,
                                mean_cluster_size=m)
    assert r.design_effect == pytest.approx(1 + (m - 1) * icc)
    k = int(np.ceil(r.n_per_variant / m))                 # кластеров на руку
    S = 3000
    b = rng.normal(0, sd * np.sqrt(icc), (S, 2, k))
    w = rng.normal(0, sd * np.sqrt(1 - icc), (S, 2, k, m)).mean(-1)
    y = mu + b + w
    y[:, 1] += mu * lift
    p = stats.ttest_ind(y[:, 1], y[:, 0], axis=1, equal_var=False).pvalue
    assert float(np.mean(p < 0.05)) == pytest.approx(0.80, abs=0.03)


# ---------------- полная приёмка PRE-1: ±2% по выборке ≈ ±0,8 п.п. мощности ----------------
@pytest.mark.slow
@pytest.mark.parametrize("p1,rel", [(0.05, 0.10), (0.15, 0.10), (0.50, 0.02)])
def test_binary_power_full(p1, rel):
    rng = np.random.default_rng(int(p1 * 1000) + 7)
    n = P.calculate_sample_size(baseline_mean=p1, mde_relative=rel, is_binary=True).n_per_variant
    assert sim_power_binary(n, p1, p1 * (1 + rel), 100_000, rng) == pytest.approx(0.80, abs=0.008)


@pytest.mark.slow
def test_continuous_power_full():
    rng = np.random.default_rng(8)
    n = P.calculate_sample_size(baseline_mean=100, mde_relative=0.05, metric_variance=400).n_per_variant
    power = np.mean([sim_power_cont(n, 100, 20, 0.05, 5000, rng) for _ in range(4)])
    assert power == pytest.approx(0.80, abs=0.008)


def test_direction_decrease_binary():
    inc = P.calculate_sample_size(baseline_mean=0.6, mde_absolute=0.05, is_binary=True)
    dec = P.calculate_sample_size(baseline_mean=0.6, mde_absolute=0.05, is_binary=True,
                                  expected_direction=Direction.DECREASE)
    assert inc.n_per_variant < dec.n_per_variant

