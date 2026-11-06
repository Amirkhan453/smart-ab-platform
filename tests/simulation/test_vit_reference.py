"""
SYN-5: генератор воспроизводит эталонные расчёты ТЗ v4 — 19.2 (рестораны), 19.3 (приложение), 7.4 (киоск).

Быстрые тесты (по умолчанию) — сотни симуляций и широкий допуск, чтобы ловить поломки
генератора. Полные (`pytest -m slow`) — тысячи симуляций и допуски ТЗ.
Анализ здесь написан независимо от `ab_platform.engine` (как в эталонных скриптах).
"""
import numpy as np
import pytest
from scipy import stats

from ab_platform.simulation.vit import (
    KioskParams, NetworkConfig, apply_guest_effect, apply_kiosk_ui, generate_kiosk_sessions,
    generate_network, simulate_reference_revenue,
)
from ab_platform.simulation.vit.rng import stream

Z = 2.8016  # z_0.975 + z_0.8
PRE_W = 8

# Таблица MDE CUPED из ТЗ 19.2: (ресторанов на руку, недель) -> (оптимистичный, пессимистичный), %
TZ_MDE_CUPED = {
    (25, 2): (4.3, 8.2), (25, 4): (3.7, 7.8), (50, 2): (3.0, 6.1), (50, 4): (2.5, 5.8),
    (100, 2): (2.1, 4.7), (100, 4): (1.8, 4.1), (200, 2): (1.5, 3.2), (200, 4): (1.3, 2.8),
    (400, 2): (1.1, 2.4), (400, 4): (0.9, 2.1),
}


# ---------------- рестораны (19.2) ----------------
def restaurant_sims(preset, n_per_arm, test_w, n_sims, seed, effect=0.0):
    p = NetworkConfig.preset(preset).panel
    rng = stream(seed, "syn5", preset, str(n_per_arm), str(test_w))
    n_days, pre_days = (PRE_W + test_w) * 7, PRE_W * 7
    p_out = {k: [] for k in ("naive", "welch", "cuped", "paired")}
    est = {k: [] for k in ("welch", "cuped", "paired")}
    bases = []
    for _ in range(n_sims):
        y = simulate_reference_revenue(2 * n_per_arm, n_days, p, rng)
        bases.append(y[:, pre_days:].mean())
        pre = y[:, :pre_days].mean(1)
        perm = rng.permutation(2 * n_per_arm)
        t, c = perm[:n_per_arm], perm[n_per_arm:]
        post = y[:, pre_days:].copy()
        post[t] *= 1 + effect
        pm = post.mean(1)
        p_out["naive"].append(stats.ttest_ind(post[t].ravel(), post[c].ravel(), equal_var=False).pvalue)
        p_out["welch"].append(stats.ttest_ind(pm[t], pm[c], equal_var=False).pvalue)
        est["welch"].append(pm[t].mean() - pm[c].mean())
        theta = np.cov(pm, pre)[0, 1] / pre.var(ddof=1)
        adj = pm - theta * (pre - pre.mean())
        p_out["cuped"].append(stats.ttest_ind(adj[t], adj[c], equal_var=False).pvalue)
        est["cuped"].append(adj[t].mean() - adj[c].mean())
        pairs = np.argsort(pre).reshape(-1, 2)
        flip = rng.integers(0, 2, len(pairs))
        tt, cc = pairs[np.arange(len(pairs)), flip], pairs[np.arange(len(pairs)), 1 - flip]
        post2 = y[:, pre_days:].mean(1)
        post2[tt] *= 1 + effect
        theta2 = np.cov(post2, pre)[0, 1] / pre.var(ddof=1)
        adj2 = post2 - theta2 * (pre - pre.mean())
        d = adj2[tt] - adj2[cc]
        p_out["paired"].append(stats.ttest_1samp(d, 0).pvalue)
        est["paired"].append(d.mean())
    base = float(np.mean(bases))
    fpr = {k: float(np.mean(np.array(v) < 0.05)) for k, v in p_out.items()}
    mde = {k: float(Z * np.std(v, ddof=1) / base * 100) for k, v in est.items()}
    return fpr, mde


def test_restaurant_aa_and_mde_fast():
    fpr, mde = restaurant_sims("reference", 50, 2, n_sims=400, seed=1)
    assert fpr["naive"] > 0.35                       # запрещённый анализ по ресторан-дням раздувает FPR
    for k in ("welch", "cuped", "paired"):
        assert 0.02 <= fpr[k] <= 0.085, (k, fpr[k])  # ±3 SE при 400 симуляциях
    assert mde["cuped"] == pytest.approx(TZ_MDE_CUPED[(50, 2)][0], rel=0.15)
    assert 24 <= mde["welch"] <= 30                  # без CUPED ≈ 26–27%
    _, mde_p = restaurant_sims("reference_pessimistic", 50, 2, n_sims=300, seed=2)
    assert mde_p["cuped"] == pytest.approx(TZ_MDE_CUPED[(50, 2)][1], rel=0.15)


def test_rerandomization_aa_on_generated_history():
    """A/A перерандомизациями на одной сгенерированной истории сети (как требует ТЗ 5.5)."""
    net = generate_network(NetworkConfig.preset("reference", n_restaurants=600, days=84, seed=5))
    w = net.wide("revenue").to_numpy()
    pre, post = w[:, :56].mean(1), w[:, 56:].mean(1)
    rng = stream(5, "rerand")
    theta = np.cov(post, pre)[0, 1] / pre.var(ddof=1)
    adj = post - theta * (pre - pre.mean())
    pv = []
    for _ in range(600):
        idx = rng.permutation(len(adj))[:200]
        t, c = idx[:100], idx[100:]
        pv.append(stats.ttest_ind(adj[t], adj[c], equal_var=False).pvalue)
    assert 0.02 <= np.mean(np.array(pv) < 0.05) <= 0.085


@pytest.mark.slow
@pytest.mark.parametrize("n,w", [(50, 2), (100, 4)])
def test_restaurant_aa_full(n, w):
    """ТЗ 19.2: A/A на 2000 симуляциях; допуск — 99% интервал биномиальной погрешности (±1,26 п.п.)."""
    fpr, _ = restaurant_sims("reference", n, w, n_sims=2000, seed=10 + n)
    assert fpr["naive"] > 0.5
    for k in ("welch", "cuped", "paired"):
        assert 0.0374 <= fpr[k] <= 0.0626, (k, fpr[k])


@pytest.mark.slow
def test_paired_aa_10000_sims():
    """Перепроверка сноски ТЗ 19.2: парный дизайн 100 × 4 нед. дал 5,9% на 2000 симуляциях."""
    fpr, _ = restaurant_sims("reference", 100, 4, n_sims=10_000, seed=99)
    assert 0.0444 <= fpr["paired"] <= 0.0556, fpr["paired"]   # 99%: ±0,56 п.п.


@pytest.mark.slow
@pytest.mark.parametrize("cell", sorted(TZ_MDE_CUPED))
def test_restaurant_mde_grid_full(cell):
    """ТЗ 19.2: сетка MDE CUPED (оптимистичный / пессимистичный) в пределах 10%."""
    n, w = cell
    for k, preset in enumerate(("reference", "reference_pessimistic")):
        _, mde = restaurant_sims(preset, n, w, n_sims=600, seed=1000 + n + w + k)
        assert mde["cuped"] == pytest.approx(TZ_MDE_CUPED[cell][k], rel=0.10), (preset, mde["cuped"])


# ---------------- приложение (19.3) ----------------
def guest_rerandomization(n_per_arm, n_sims, seed):
    net = generate_network(NetworkConfig.preset("reference", n_restaurants=50, days=14, seed=seed))
    g = net.guests(2 * n_per_arm, window=f"syn5_{seed}")
    y, x, b = g["revenue"].to_numpy(), g["pre_revenue"].to_numpy(), g["converted"].to_numpy().astype(float)
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    adj = y - theta * (x - x.mean())
    rng = stream(seed, "syn5_guests")
    d_raw, d_adj, d_bin, pv = [], [], [], []
    for _ in range(n_sims):
        t = np.zeros(len(y), dtype=bool)
        t[rng.permutation(len(y))[:n_per_arm]] = True
        d_raw.append(y[t].mean() - y[~t].mean())
        d_adj.append(adj[t].mean() - adj[~t].mean())
        d_bin.append(b[t].mean() - b[~t].mean())
        pv.append(stats.ttest_ind(adj[t], adj[~t], equal_var=False).pvalue)
    cv, rho, p = y.std() / y.mean(), np.corrcoef(y, x)[0, 1], b.mean()
    sim = {
        "raw": Z * np.std(d_raw, ddof=1) / y.mean(), "cuped": Z * np.std(d_adj, ddof=1) / y.mean(),
        "binary": Z * np.std(d_bin, ddof=1) / p,
    }
    formula = {
        "raw": Z * cv * np.sqrt(2 / n_per_arm), "cuped": Z * cv * np.sqrt(2 * (1 - rho ** 2) / n_per_arm),
        "binary": Z * np.sqrt(2 * p * (1 - p) / n_per_arm) / p,
    }
    return sim, formula, float(np.mean(np.array(pv) < 0.05)), g


def test_guest_mde_matches_formula_fast():
    sim, formula, fpr, _ = guest_rerandomization(50_000, n_sims=300, seed=3)
    for k in sim:
        assert sim[k] == pytest.approx(formula[k], rel=0.15), (k, sim[k], formula[k])
    assert sim["cuped"] * 100 == pytest.approx(3.8, rel=0.15)   # ТЗ 19.3: 50 000 на руку, ρ = 0,5
    assert 0.015 <= fpr <= 0.09


@pytest.mark.slow
def test_guest_mde_and_power_full():
    """ТЗ 19.3 при 200 000 на руку: MDE ≈ 2,2% / 1,9%; мощность при эффекте = MDE ≈ 80%."""
    sim, formula, fpr, g = guest_rerandomization(200_000, n_sims=2000, seed=4)
    for k in sim:
        assert sim[k] == pytest.approx(formula[k], rel=0.08), (k, sim[k], formula[k])
    assert sim["raw"] * 100 == pytest.approx(2.2, rel=0.10)
    assert sim["cuped"] * 100 == pytest.approx(1.9, rel=0.10)
    assert 0.0374 <= fpr <= 0.0626
    rng = stream(4, "syn5_power")
    x = g["pre_revenue"].to_numpy()
    rejections = []
    for i in range(400):
        t = np.zeros(len(g), dtype=bool)
        t[rng.permutation(len(g))[:200_000]] = True
        g2, _ = apply_guest_effect(g, t, lift=sim["cuped"], test_days=28, seed=10_000 + i)
        y = g2["revenue"].to_numpy()
        theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
        adj = y - theta * (x - x.mean())
        rejections.append(stats.ttest_ind(adj[t], adj[~t], equal_var=False).pvalue < 0.05)
    assert 0.73 <= np.mean(rejections) <= 0.87     # 80% ± ~3,5 SE при 400 симуляциях


# ---------------- киоск (7.4) ----------------
def kiosk_rerandomization(n_sims, seed, params=KioskParams(orders_per_day=214)):
    ids = [f"R{i:04d}" for i in range(150)]
    s = generate_kiosk_sessions(seed, ids, "2025-01-06", 14, params, window=f"syn5_{seed}")
    o = s[s["converted"] == 1]
    y = o["check"].to_numpy()
    cell = (o["restaurant_id"] + "|" + o["date"].astype(str)).to_numpy()
    _, inv = np.unique(cell, return_inverse=True)
    yc = y - (np.bincount(inv, y) / np.bincount(inv))[inv]
    rng = stream(seed, "syn5_kiosk")
    d, pv = [], []
    for _ in range(n_sims):
        t = rng.random(len(y)) < 0.5
        diff = yc[t].mean() - yc[~t].mean()
        se = np.sqrt(yc[t].var(ddof=1) / t.sum() + yc[~t].var(ddof=1) / (~t).sum())
        d.append(diff)
        pv.append(2 * stats.norm.sf(abs(diff / se)))
    n_arm = len(y) / 2
    sim = Z * np.std(d, ddof=1) / y.mean()
    formula = Z * params.check_cv * np.sqrt(2 / n_arm)
    return sim, formula, float(np.mean(np.array(pv) < 0.05))


def test_kiosk_mde_fast():
    sim, formula, fpr = kiosk_rerandomization(n_sims=200, seed=6)
    assert sim == pytest.approx(formula, rel=0.15)
    assert sim * 100 == pytest.approx(0.70, rel=0.15)     # ТЗ 7.4: 150 × 150 заказов × 2 нед.
    assert 0.01 <= fpr <= 0.11


@pytest.mark.slow
def test_kiosk_mde_full():
    sim, formula, fpr = kiosk_rerandomization(n_sims=2000, seed=7)
    assert sim == pytest.approx(formula, rel=0.08)
    assert sim * 100 == pytest.approx(0.70, rel=0.10)
    assert 0.0374 <= fpr <= 0.0626


def test_kiosk_queue_interference_is_invisible_within_restaurant():
    """ТЗ 7.1: вред через общую очередь бьёт по обеим версиям, и сравнение внутри ресторана его не видит."""
    ids = [f"R{i:04d}" for i in range(60)]
    s = generate_kiosk_sessions(8, ids, "2025-01-06", 7, KioskParams(orders_per_day=200))
    t = stream(8, "q").random(len(s)) < 0.5
    s2, truth = apply_kiosk_ui(s, t, check_lift=0.0, extra_time_s=30, queue_conv_penalty=0.10, seed=8)
    conv = s2["converted"].to_numpy().astype(float)
    diff = conv[t].mean() - conv[~t].mean()
    se = np.sqrt(conv[t].var() / t.sum() + conv[~t].var() / (~t).sum())
    assert abs(diff) < 3 * se                                          # внутри ресторана «вреда нет»
    assert conv.mean() < s["converted"].mean() - 0.02                  # а конверсия всей сети упала
    assert truth.params["total_effect_if_all_b_rel"] == pytest.approx(-0.10)
