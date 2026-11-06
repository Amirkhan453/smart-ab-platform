"""
Симуляция ресторанных тестов на синтетической панели, похожей на сеть ~1000 ресторанов.
Параметры — допущения порядка величины (выручка ~600 тыс. руб./день на ресторан),
НЕ реальные данные ВИТ. Цель: проверить математику (FPR, MDE, эффект CUPED и пар).
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261106)

N_REST = 1000
PRE_W = 8          # недель пре-периода
DOW = np.array([0.95, 0.93, 0.95, 0.98, 1.08, 1.12, 0.99])  # профиль дня недели


# Сценарии шума: optimistic / pessimistic (см. ТЗ v3.1, раздел 9.2)
SCENARIOS = {
    "optimistic": dict(drift_sd=0.004, eps_sd=0.08, shock_p=0.0, shock_sd=0.0),
    "pessimistic": dict(drift_sd=0.012, eps_sd=0.15, shock_p=0.003, shock_sd=0.3),
}
SCENARIO = SCENARIOS["optimistic"]


def make_panel(n_rest, n_days, rng):
    """log y = a_i + dow + общий шок дня + медленный дрейф ресторана + AR(1) шум + редкие локальные события."""
    sc = SCENARIO
    a = rng.normal(np.log(600_000), 0.45, n_rest)                         # CV между ресторанами ~0.47
    common = rng.normal(0, 0.05, n_days)                                    # погода/промо сети — общие для всех
    drift = np.cumsum(rng.normal(0, sc["drift_sd"], (n_rest, n_days)), 1)   # медленный индивидуальный дрейф
    shocks = rng.binomial(1, sc["shock_p"], (n_rest, n_days)) * rng.normal(0, max(sc["shock_sd"], 1e-12), (n_rest, n_days))
    u = np.zeros((n_rest, n_days))
    eps = rng.normal(0, sc["eps_sd"], (n_rest, n_days))
    for t in range(n_days):
        u[:, t] = (0.5 * u[:, t - 1] if t else 0) + eps[:, t]
    dow = np.log(DOW[np.arange(n_days) % 7])
    return np.exp(a[:, None] + dow[None, :] + common[None, :] + drift + u + shocks)


def welch_p(x, y):
    return stats.ttest_ind(x, y, equal_var=False).pvalue


def run(n_per_arm, test_w, n_sims=2000, effect=0.0):
    n_days = (PRE_W + test_w) * 7
    pre_days = PRE_W * 7
    res = {k: [] for k in ["naive_storeday", "welch", "cuped", "paired_cuped"]}
    est = {k: [] for k in ["welch", "cuped", "paired_cuped"]}
    bases = []
    for _ in range(n_sims):
        y = make_panel(2 * n_per_arm, n_days, rng)
        bases.append(y[:, pre_days:].mean())
        pre = y[:, :pre_days].mean(1)
        # --- простая рандомизация
        perm = rng.permutation(2 * n_per_arm)
        t_idx, c_idx = perm[:n_per_arm], perm[n_per_arm:]
        post = y[:, pre_days:].copy()
        post[t_idx] *= (1 + effect)
        post_mean = post.mean(1)
        # naive: ресторан-день как независимые наблюдения (только для FPR)
        res["naive_storeday"].append(welch_p(post[t_idx].ravel(), post[c_idx].ravel()))
        # Welch на ресторанных средних
        d = post_mean[t_idx].mean() - post_mean[c_idx].mean()
        res["welch"].append(welch_p(post_mean[t_idx], post_mean[c_idx])); est["welch"].append(d)
        # CUPED (theta по объединённой выборке)
        theta = np.cov(post_mean, pre)[0, 1] / pre.var(ddof=1)
        adj = post_mean - theta * (pre - pre.mean())
        res["cuped"].append(welch_p(adj[t_idx], adj[c_idx]))
        est["cuped"].append(adj[t_idx].mean() - adj[c_idx].mean())
        # Пары по пре-периоду + монетка внутри пары + CUPED, тест — парный t по разностям
        order = np.argsort(pre)
        pairs = order.reshape(-1, 2)
        flip = rng.integers(0, 2, len(pairs))
        tt = pairs[np.arange(len(pairs)), flip]
        cc = pairs[np.arange(len(pairs)), 1 - flip]
        post2 = y[:, pre_days:].mean(1).copy()
        post2[tt] *= (1 + effect)
        theta2 = np.cov(post2, pre)[0, 1] / pre.var(ddof=1)
        adj2 = post2 - theta2 * (pre - pre.mean())
        diff = adj2[tt] - adj2[cc]
        res["paired_cuped"].append(stats.ttest_1samp(diff, 0).pvalue)
        est["paired_cuped"].append(diff.mean())
    base = float(np.mean(bases))
    out = {}
    for k, p in res.items():
        p = np.array(p)
        row = {"reject_rate": float(np.mean(p < 0.05))}
        if k in est:
            row["mde80_rel_pct"] = float(2.8016 * np.std(est[k], ddof=1) / base * 100)  # (z_.975+z_.8)*SE
        out[k] = row
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        SCENARIO = SCENARIOS[sys.argv[1]]
    print("Сценарий:", SCENARIO)
    print("=== A/A (effect=0): доля p<0.05, должна быть ~5% (±1% при 2000 сим.) ===")
    for n, w in [(50, 2), (100, 4)]:
        print(n, "рест./руку,", w, "нед.:", run(n, w, n_sims=2000))
    print("\n=== MDE (80% мощности, alpha=0.05, двусторонний) по сетке ===")
    for n in [25, 50, 100, 200, 400]:
        for w in [2, 4]:
            r = run(n, w, n_sims=600)
            print(f"n={n:>3}/руку, {w} нед.: " + ", ".join(
                f"{k}={v['mde80_rel_pct']:.1f}%" for k, v in r.items() if "mde80_rel_pct" in v))
    print("\n=== Проверка мощности на заданном эффекте 3%, n=100/руку, 4 нед. ===")
    print(run(100, 4, n_sims=1000, effect=0.03))
