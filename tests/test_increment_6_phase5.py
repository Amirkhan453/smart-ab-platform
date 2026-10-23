import numpy as np
import pandas as pd
from ab_platform.engine.switchback import SwitchbackAndClusterEngine
from ab_platform.engine.offline_geo import OfflineRetailGeoEngine
from ab_platform.engine.causal_quasi import CausalAndMLDebiasingEngine
from ab_platform.engine.bandits import MultiArmedBanditEngine


def test_switchback_pairing_and_power_optimal_cupac():
    """
    Проверяем:
    1) Зеркальное спаривание (Pairing & Mirroring) дает меньшую дисперсию, чем наивный Switchback
    2) Power-Optimal CUPAC сжимает межъячеечную дисперсию и точно восстанавливает ATE
    """
    var_comp = SwitchbackAndClusterEngine.compare_design_variances(
        total_variance=1000.0,
        num_clusters_j=10,
        num_windows_h=12,
        mean_cell_size_n=150.0,
        cell_size_cv=1.2,
    )
    assert var_comp["var_paired_mirrored"] < var_comp["var_stratified"] < var_comp["var_unstratified"]

    sched = SwitchbackAndClusterEngine.generate_paired_mirror_schedule(
        cluster_ids=["c1", "c2", "c3", "c4"], num_windows_h=6
    )
    assert len(sched) == 24

    # Генерируем синтетический Switchback-датасет с сильным макро-шумом ячеек
    rng = np.random.default_rng(42)
    rows = []
    for cell_idx in range(40):
        is_treat = (cell_idx % 2) == 1
        variant = "treatment" if is_treat else "control"
        cell_macro_shock = rng.normal(0.0, 25.0)
        cell_n = int(rng.integers(40, 120))
        for _ in range(cell_n):
            x_feat = cell_macro_shock + rng.normal(0.0, 5.0)
            y_val = 100.0 + 0.9 * x_feat + (8.0 if is_treat else 0.0) + rng.normal(0.0, 3.0)
            rows.append({
                "cell_id": f"cell_{cell_idx}",
                "variant": variant,
                "x1": x_feat,
                "y": y_val,
            })

    sb_df = pd.DataFrame(rows)
    res = SwitchbackAndClusterEngine.analyze_power_optimal_cupac(
        sb_df, cell_col="cell_id", variant_col="variant", outcome_col="y", feature_cols=["x1"]
    )
    assert res.variance_reduction_pct > 60.0
    assert res.is_significant is True
    assert 6.0 <= res.ate_power_cupac <= 10.0


def test_offline_geo_matched_markets_and_synthetic_control():
    """
    Проверяем:
    1) Greedy Search находит пары магазинов с высокой корреляцией
    2) Synthetic Control точно оценивает оффлайн-эффект (+50 ед. выручки) после старта акции
    """
    rng = np.random.default_rng(101)
    days = np.arange(1, 61)
    base_trend = 500.0 + 50.0 * np.sin(days / 4.0)

    records = []
    for t in days:
        trend_t = base_trend[t - 1]
        # s_treat до 40-го дня равен 0.7 * s_donor1 + 0.3 * s_donor2, а с 40-го дня получает +50.0 эффекта
        d1 = trend_t + rng.normal(0, 2.0)
        d2 = trend_t * 0.6 + 100.0 + rng.normal(0, 2.0)
        d3 = 300.0 - trend_t * 0.2 + rng.normal(0, 5.0)
        lift = 50.0 if t >= 40 else 0.0
        s_treat = 0.7 * d1 + 0.3 * d2 + lift + rng.normal(0, 1.5)

        for sid, val in [("store_T", s_treat), ("store_D1", d1), ("store_D2", d2), ("store_D3", d3)]:
            records.append({"date": int(t), "store_id": sid, "revenue": float(val)})

    panel_df = pd.DataFrame(records)
    pre_df = panel_df[panel_df["date"] < 40]

    pairs = OfflineRetailGeoEngine.match_stores_greedy(pre_df, n_pairs=2)
    assert len(pairs) == 2
    assert pairs[0].pre_correlation > 0.90

    sc_res = OfflineRetailGeoEngine.estimate_synthetic_control(
        panel_df=panel_df,
        unit_col="store_id",
        time_col="date",
        metric_col="revenue",
        treated_units=["store_T"],
        intervention_start_time=40,
    )
    assert 45.0 <= sc_res.ate_absolute <= 55.0
    assert sc_res.is_significant is True
    assert sc_res.donor_weights["store_D1"] > 0.5


def test_causal_quasi_psm_did_and_multivariate_debiasing():
    """
    Проверяем:
    1) Многомерный дебайасинг LinkedIn сжимает дисперсию по набору ковариат
    2) PSM и Doubly Robust (AIPW) очищают смещение самоотбора (confounding bias)
    3) Difference-in-Differences (DiD) восстанавливает истинный эффект (+15.0)
    """
    rng = np.random.default_rng(77)
    n = 1500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)

    # Нерандомизированное назначение (зависит от x1 и x2) + истинный эффект = +10.0
    prob_t = 1.0 / (1.0 + np.exp(-(1.2 * x1 + 0.8 * x2)))
    t_assign = (rng.random(n) < prob_t).astype(int)
    y_obs = 50.0 + 12.0 * x1 + 8.0 * x2 + 10.0 * t_assign + rng.normal(0, 2.0, n)

    quasi_df = pd.DataFrame({
        "treatment": t_assign,
        "variant": np.where(t_assign == 1, "treatment", "control"),
        "x1": x1,
        "x2": x2,
        "y": y_obs,
    })

    dr_res = CausalAndMLDebiasingEngine.propensity_score_matching_and_aipw(
        quasi_df, treatment_col="treatment", outcome_col="y", confounder_cols=["x1", "x2"]
    )
    # Наивная разница сильно завышена (>20), а Doubly Robust точно бьет около 10.0
    assert dr_res["naive_biased_diff"] > 18.0
    assert 8.5 <= dr_res["ate_doubly_robust"] <= 11.5

    deb_res = CausalAndMLDebiasingEngine.multivariate_debiasing_ate(
        quasi_df, outcome_col="y", feature_cols=["x1", "x2"]
    )
    assert deb_res["variance_reduction_pct"] > 70.0

    # Проверка DiD (истинный эффект = +15.0)
    g = rng.integers(0, 2, size=800)
    post = rng.integers(0, 2, size=800)
    y_did = 20.0 + 5.0 * g + 3.0 * post + 15.0 * (g * post) + rng.normal(0, 1.5, size=800)
    did_df = pd.DataFrame({"group": g, "post": post, "y": y_did})
    did_res = CausalAndMLDebiasingEngine.difference_in_differences(did_df, "y", "group", "post")
    assert 14.0 <= did_res["did_treatment_effect"] <= 16.0
    assert did_res["is_significant"] is True


def test_multi_armed_and_contextual_bandits():
    """Проверяем Thompson Sampling, UCB1, Epsilon-Greedy и Contextual LinUCB."""
    successes = np.array([20, 45, 95])
    trials = np.array([200, 200, 200])

    ts = MultiArmedBanditEngine.thompson_sampling_allocation(successes, trials)
    assert ts["recommended_arm"] == 2
    assert ts["optimal_traffic_shares"][2] > 0.95

    ucb = MultiArmedBanditEngine.select_arm_ucb1(successes, trials)
    assert ucb["chosen_arm"] == 2

    eg = MultiArmedBanditEngine.select_arm_epsilon_greedy(successes, trials, epsilon=0.0)
    assert eg["chosen_arm"] == 2

    # Контекстный бандит LinUCB: рука 1 лучше для контекста [1, 0], рука 0 лучше для [0, 1]
    A0 = np.eye(2)
    b0 = np.array([0.1, 5.0])
    A1 = np.eye(2)
    b1 = np.array([5.0, 0.1])
    ctx_res = MultiArmedBanditEngine.contextual_linucb_select(
        user_context=np.array([1.0, 0.0]),
        arm_gram_matrices=[A0, A1],
        arm_reward_vectors=[b0, b1],
        alpha_explore=0.1,
    )
    assert ctx_res["chosen_arm"] == 1
