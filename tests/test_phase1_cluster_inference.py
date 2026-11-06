"""
Фаза 1, П3: кластерный инференс ресторанного RCT (STA-2, RST-6). Данные — эталонная панель
генератора фазы 0. Быстрые тесты — сотни A/A; полные (`pytest -m slow`) — 2000 и допуск ТЗ.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ab_platform.engine import cluster_inference as CI
from ab_platform.engine.criterion_router import StatCriterionRouter as R
from ab_platform.simulation.vit import NetworkConfig, generate_network, inject, simulate_reference_revenue
from ab_platform.simulation.vit.rng import stream

PANEL = NetworkConfig.preset("reference").panel
PESS = NetworkConfig.preset("reference_pessimistic").panel
LO99, HI99 = 0.0374, 0.0626            # 99% интервал FPR при 2000 A/A


def aa_panel(rng, n_arm, panel=PANEL, pre_days=56, post_days=28):
    y = simulate_reference_revenue(2 * n_arm, pre_days + post_days, panel, rng)
    t = rng.permutation(np.r_[np.ones(n_arm), np.zeros(n_arm)]).astype(int)
    return y[:, pre_days:].mean(1), y[:, :pre_days].mean(1), t


# ---------------- тождества и граничные случаи ----------------
def test_cr2_equals_welch_se_without_covariates():
    rng = np.random.default_rng(0)
    y, t = rng.lognormal(0, 0.5, 60), np.arange(60) % 2
    se_welch = np.sqrt(y[t == 0].var(ddof=1) / 30 + y[t == 1].var(ddof=1) / 30)
    r = CI.lin_cr2(y, t)
    assert r["se"] == pytest.approx(se_welch, rel=1e-10)
    assert r["estimate"] == pytest.approx(y[t == 1].mean() - y[t == 0].mean())


def test_exact_pairs_minimum_p():
    rng = np.random.default_rng(1)
    blocks, t = np.repeat(np.arange(6), 2), np.tile([0, 1], 6)
    y = rng.normal(0, 1, 12) + 10 * t
    r = CI.randomization_test(y, t, blocks=blocks)
    assert r["exact"] and r["n_permutations"] == 64
    assert r["p_value"] == pytest.approx(2 / 64) and r["min_attainable_p"] == pytest.approx(2 / 64)
    five = CI.randomization_test(y[:10], t[:10], blocks=blocks[:10])
    assert five["p_value"] > 0.05                     # 5 пар: значимость недостижима (ТЗ 5.5)


def test_permutations_keep_block_counts():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 5, 200)
    treat = np.zeros(200, dtype=int)
    for b in range(5):
        idx = np.flatnonzero(codes == b)
        treat[idx[: len(idx) // 3]] = 1
    T = CI._permutation_assignments(treat, codes, 50, rng)
    for b in range(5):
        assert (T[:, codes == b].sum(1) == treat[codes == b].sum()).all()


def test_randomization_ci_matches_lin_on_large_sample():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.lognormal(13, 0.45, n)
    y = x * rng.lognormal(0, 0.05, n)
    t = rng.permutation(np.r_[np.ones(n // 2), np.zeros(n // 2)]).astype(int)
    rt = CI.randomization_test(y, t, x, n_perm=4000)
    lc = CI.lin_cr2(y, t, x)
    assert rt["ci_lower"] == pytest.approx(lc["ci_lower"], rel=0.1)
    assert rt["ci_upper"] == pytest.approx(lc["ci_upper"], rel=0.1)


def test_input_validation():
    with pytest.raises(ValueError):
        CI.lin_cr2(np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        CI.randomization_test(np.ones(3), np.array([0, 1, 0]))
    with pytest.raises(ValueError, match="обе руки"):
        CI.randomization_test(np.ones(6), np.array([1, 1, 0, 0, 0, 0]), blocks=[0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError):
        CI.wild_cluster_bootstrap(np.ones(3), np.ones((3, 2)))
    with pytest.raises(ValueError):
        CI.network_total_ratio(np.ones(6), np.array([0, 1] * 3), np.zeros(6))
    with pytest.raises(ValueError):
        CI.analyze_cluster_rct(pd.DataFrame({"y": [1.0]}))


def test_aggregate_clusters_rules():
    df = pd.DataFrame({"r": ["a", "a", "b", "b"], "arm": ["c", "c", "t", "t"], "y": [1.0, 3.0, 5.0, 7.0],
                       "pre": [2.0, 2.0, 6.0, 6.0]})
    out = CI.aggregate_clusters(df, "r", "arm", "y", "t", covariate_col="pre")
    assert out["y"].tolist() == [2.0, 6.0] and out["treat"].tolist() == [0, 1] and out["x"].tolist() == [2.0, 6.0]
    assert CI.aggregate_clusters(df, "r", "arm", "y", "t", how="sum")["y"].tolist() == [4.0, 12.0]
    bad = df.assign(arm=["c", "t", "t", "t"])
    with pytest.raises(ValueError, match="Рука меняется"):
        CI.aggregate_clusters(bad, "r", "arm", "y", "t")
    with pytest.raises(ValueError, match="постоянной"):
        CI.aggregate_clusters(df.assign(pre=[1.0, 2.0, 6.0, 6.0]), "r", "arm", "y", "t", covariate_col="pre")


def test_d4_cluster_robust_unweighted_estimand_fast():
    rng = np.random.default_rng(4)
    rej = 0
    for _ in range(600):
        k = 40
        sizes = rng.integers(1, 60, 2 * k)
        mu = 0.05 * sizes + rng.normal(0, 1, 2 * k)
        vals = [rng.normal(mu[j], 1, sizes[j]) for j in range(2 * k)]
        perm = rng.permutation(2 * k)
        tc, cc = perm[:k], perm[k:]
        r = R.cluster_robust_se_and_icc(
            np.concatenate([vals[j] for j in cc]), np.concatenate([[j] * sizes[j] for j in cc]),
            np.concatenate([vals[j] for j in tc]), np.concatenate([[j] * sizes[j] for j in tc]))
        rej += r["p_value"] < 0.05
    assert 0.025 <= rej / 600 <= 0.08


# ---------------- A/A и мощность на панели генератора ----------------
def test_aa_fast_50_per_arm_all_methods():
    rng = stream(10, "p3_fast")
    rej = {"cr2": 0, "rand": 0, "net": 0}
    n = 300
    for _ in range(n):
        post, pre, t = aa_panel(rng, 50)
        rej["cr2"] += CI.lin_cr2(post, t, pre)["p_value"] < 0.05
        rej["rand"] += CI.randomization_test(post, t, pre, n_perm=999, ci=False)["p_value"] < 0.05
        rej["net"] += CI.network_total_ratio(post, t, pre)["p_value"] < 0.05
    for k, v in rej.items():
        assert 0.015 <= v / n <= 0.09, (k, v / n)


def test_small_clusters_naive_asymptotics_inflate_but_randomization_holds():
    rng = stream(11, "p3_small")
    naive = rand = 0
    n = 400
    for _ in range(n):
        post, pre, t = aa_panel(rng, 8, PESS)
        X = CI.lin_design(t, pre)
        b = np.linalg.lstsq(X, post, rcond=None)[0]
        e = post - X @ b
        XtXi = np.linalg.inv(X.T @ X)
        v = (XtXi @ (X.T * e ** 2) @ X @ XtXi)[1, 1]
        naive += 2 * stats.norm.sf(abs(b[1] / np.sqrt(v))) < 0.05
        rand += CI.randomization_test(post, t, pre, n_perm=999, ci=False)["p_value"] < 0.05
    assert naive / n > 0.09                    # HC0 + z при 8 на руку: ≈ 16% (замер)
    assert rand / n <= 0.08


def test_power_and_truth_on_generated_network():
    """Эффект 5% через inject: Lin + CR2 его находит, ДИ накрывает истинный эффект из Truth."""
    net = generate_network(NetworkConfig.preset("reference", n_restaurants=200, days=84, seed=21))
    ids = net.restaurants["restaurant_id"].to_numpy()
    rng = stream(21, "p3_power")
    treated = list(rng.permutation(ids)[:100])
    start = pd.Timestamp(net.config.start_date) + pd.Timedelta(days=56)
    net2, truth = inject(net, treated, start, start + pd.Timedelta(days=27), lift=0.05, lift_on="avg_check")
    d = net2.daily
    pre = d[d.date < start].groupby("restaurant_id")["revenue"].mean().rename("pre")
    post = d[d.date >= start].merge(pre, left_on="restaurant_id", right_index=True)
    post["arm"] = np.where(post.restaurant_id.isin(treated), "t", "c")
    cl = CI.aggregate_clusters(post, "restaurant_id", "arm", "revenue", "t", covariate_col="pre")
    r = CI.analyze_cluster_rct(cl, "per_restaurant_mean", "asymptotic")
    assert r["is_significant"] and r["ci_lower"] <= truth.per_restaurant_mean_abs <= r["ci_upper"]
    nt = CI.analyze_cluster_rct(cl, "network_total", "asymptotic")
    assert nt["ci_lower_rel"] <= truth.network_total_rel <= nt["ci_upper_rel"]
    rt = CI.analyze_cluster_rct(cl.assign(block=np.arange(len(cl)) // 20), "per_restaurant_mean", "randomization",
                                n_perm=2000)
    assert rt["is_significant"] and rt["ci_lower"] <= truth.per_restaurant_mean_abs <= rt["ci_upper"]
    wb = CI.analyze_cluster_rct(cl, "per_restaurant_mean", "wild_bootstrap", n_boot=999)
    assert wb["is_significant"] and "warning" not in wb


def test_wild_bootstrap_warns_with_few_clusters():
    rng = stream(12, "p3_warn")
    post, pre, t = aa_panel(rng, 6)
    r = CI.analyze_cluster_rct(pd.DataFrame({"y": post, "x": pre, "treat": t}), inference="wild_bootstrap", n_boot=499)
    assert "warning" in r
    with pytest.raises(ValueError, match="randomization"):
        CI.analyze_cluster_rct(pd.DataFrame({"y": post, "x": pre, "treat": t}), "network_total", "wild_bootstrap")


# ---------------- полные приёмки (slow) ----------------
@pytest.mark.slow
@pytest.mark.parametrize("panel_name", ["reference", "reference_pessimistic"])
def test_aa_full_50_per_arm(panel_name):
    panel = NetworkConfig.preset(panel_name).panel
    rng = stream(20, "p3_full", panel_name)
    rej = {"cr2": 0, "rand": 0, "net": 0}
    n = 2000
    for _ in range(n):
        post, pre, t = aa_panel(rng, 50, panel)
        rej["cr2"] += CI.lin_cr2(post, t, pre)["p_value"] < 0.05
        rej["rand"] += CI.randomization_test(post, t, pre, n_perm=999, ci=False)["p_value"] < 0.05
        rej["net"] += CI.network_total_ratio(post, t, pre)["p_value"] < 0.05
    for k, v in rej.items():
        assert LO99 <= v / n <= HI99, (panel_name, k, v / n)


@pytest.mark.slow
def test_aa_full_10_per_arm_pessimistic():
    """ТЗ (заметка фазы 1, приёмка): при 10 на руку рандомизационный тест и wild bootstrap держат FPR."""
    rng = stream(30, "p3_full10")
    rej = {"rand": 0, "wild": 0, "net_perm": 0}
    n = 2000
    for i in range(n):
        post, pre, t = aa_panel(rng, 10, PESS)
        rej["rand"] += CI.randomization_test(post, t, pre, n_perm=999, ci=False)["p_value"] < 0.05
        rej["wild"] += CI.wild_cluster_bootstrap(post, CI.lin_design(t, pre), n_boot=999, seed=i)["p_value"] < 0.05
        rej["net_perm"] += CI._ratio_permutation_p(post, t, pre, None, 999, "two_sided", "increase", i) < 0.05
    for k, v in rej.items():
        assert LO99 <= v / n <= HI99, (k, v / n)


@pytest.mark.slow
def test_d4_full():
    rng = np.random.default_rng(40)
    rej = 0
    for _ in range(2000):
        k = 40
        sizes = rng.integers(1, 60, 2 * k)
        mu = 0.05 * sizes + rng.normal(0, 1, 2 * k)
        vals = [rng.normal(mu[j], 1, sizes[j]) for j in range(2 * k)]
        perm = rng.permutation(2 * k)
        tc, cc = perm[:k], perm[k:]
        r = R.cluster_robust_se_and_icc(
            np.concatenate([vals[j] for j in cc]), np.concatenate([[j] * sizes[j] for j in cc]),
            np.concatenate([vals[j] for j in tc]), np.concatenate([[j] * sizes[j] for j in tc]))
        rej += r["p_value"] < 0.05
    assert LO99 <= rej / 2000 <= HI99


@pytest.mark.slow
def test_randomization_ci_coverage():
    rng = stream(50, "p3_cov")
    cover = 0
    n = 1000
    for _ in range(n):
        post, pre, t = aa_panel(rng, 25)
        post = post + 20_000 * t                          # постоянный эффект
        r = CI.randomization_test(post, t, pre, n_perm=999)
        cover += r["ci_lower"] <= 20_000 <= r["ci_upper"]
    assert 0.93 <= cover / n <= 0.97


# ---------------- оркестратор: ресторанный домен end-to-end ----------------
from ab_platform.orchestrator import DecisionStatus, SmartABPlatformOrchestrator as O  # noqa: E402
from ab_platform.schemas.passport import (  # noqa: E402
    AnalysisUnit, ExperimentPassport, ExperimentType, GuardrailDefinition, InferenceMethod, MetricDefinition,
    MetricType, RandomizationUnit, VariantAllocation,
)


def restaurant_passport(n_pool, **kw):
    rev = MetricDefinition(name="revenue", metric_type=MetricType.CONTINUOUS, numerator_col="revenue",
                           winsorize_percentile=None)
    chk = MetricDefinition(name="checks", metric_type=MetricType.CONTINUOUS, numerator_col="checks",
                           winsorize_percentile=None)
    data = dict(
        experiment_id="rst", test_key="rst", hypothesis_statement="Если мы поменяем меню-борд, выручка вырастет",
        experiment_type=ExperimentType.CLUSTER_AB, randomization_unit=RandomizationUnit.STORE_ID,
        analysis_unit=AnalysisUnit.STORE_DAY,
        variants=[VariantAllocation(variant_id="control", traffic_percentage=50, is_control=True),
                  VariantAllocation(variant_id="treatment", traffic_percentage=50)],
        primary_metric=rev,
        guardrail_metrics=[GuardrailDefinition(metric=chk, max_allowed_degradation_rel=0.30, tradeoff_rationale="трафик")],
        baseline_value=600_000, mde_relative=0.03, planned_duration_days=28, daily_eligible_units=n_pool,
        covariate_col="pre_revenue",
    )
    data.update(kw)
    return ExperimentPassport(**data)


def store_days(net, treated, start):
    d = net.daily
    pre = d[d.date < start].groupby("restaurant_id")["revenue"].mean().rename("pre_revenue")
    post = d[d.date >= start].merge(pre, left_on="restaurant_id", right_index=True)
    post["variant"] = np.where(post.restaurant_id.isin(treated), "treatment", "control")
    return post[["restaurant_id", "variant", "date", "revenue", "checks", "pre_revenue"]]


@pytest.fixture(scope="module")
def rnet():
    return generate_network(NetworkConfig.preset("reference", n_restaurants=200, days=84, seed=31))


def test_orchestrator_restaurant_effect_is_ship_candidate(rnet):
    ids = rnet.restaurants.restaurant_id.to_numpy()
    treated = list(stream(31, "orch").permutation(ids)[:100])
    start = pd.Timestamp(rnet.config.start_date) + pd.Timedelta(days=56)
    net2, truth = inject(rnet, treated, start, start + pd.Timedelta(days=27), lift=0.05, lift_on="avg_check")
    r = O.evaluate_experiment(restaurant_passport(200), store_days(net2, treated, start), user_col="restaurant_id")
    assert r.decision_status == DecisionStatus.SHIP_CANDIDATE
    assert r.primary_metric_result["criterion"] == "lin_cr2_bell_mccaffrey"
    assert r.estimand == "per_restaurant_mean" and r.clean_sample_size == 200
    pr = r.primary_metric_result
    assert pr["ci_lower"] <= truth.per_restaurant_mean_abs <= pr["ci_upper"]


def test_orchestrator_few_restaurants_uses_randomization(rnet):
    ids = rnet.restaurants.restaurant_id.to_numpy()[:20]
    treated = list(ids[::2])
    start = pd.Timestamp(rnet.config.start_date) + pd.Timedelta(days=56)
    df = store_days(rnet, treated, start)
    df = df[df.restaurant_id.isin(ids)]
    with pytest.raises(ValueError, match="асимптотический"):
        restaurant_passport(20)
    p = restaurant_passport(20, inference=InferenceMethod.RANDOMIZATION)
    r = O.evaluate_experiment(p, df, user_col="restaurant_id")
    assert r.primary_metric_result["criterion"].startswith("randomization_test")
    assert r.decision_status != DecisionStatus.SHIP_CANDIDATE


def test_orchestrator_restaurant_aa_fpr(rnet):
    """A/A через весь боевой путь оркестратора на ресторан-днях: FPR в норме (ресторан-дни агрегированы)."""
    ids = rnet.restaurants.restaurant_id.to_numpy()
    start = pd.Timestamp(rnet.config.start_date) + pd.Timedelta(days=56)
    rng = stream(32, "orch_aa")
    p = restaurant_passport(200)
    base = store_days(rnet, [], start)
    sig = 0
    n = 150
    for _ in range(n):
        treated = set(rng.permutation(ids)[:100])
        df = base.assign(variant=np.where(base.restaurant_id.isin(treated), "treatment", "control"))
        sig += O.evaluate_experiment(p, df, user_col="restaurant_id").primary_metric_result["is_significant"]
    assert sig / n <= 0.10
