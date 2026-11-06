"""Фаза 1, П2: правило решения ТЗ v4, 5.3 — регрессии Д1, Д2, Д3, Д5 и скрытие до горизонта (STA-4, EVD-1)."""
import numpy as np
import pandas as pd
import pytest

from ab_platform.engine.criterion_router import StatCriterionRouter as R
from ab_platform.orchestrator import DecisionStatus, SmartABPlatformOrchestrator as O
from ab_platform.schemas.passport import (
    Direction, ExperimentPassport, ExperimentType, GuardrailDefinition, MetricDefinition, MetricType,
    QuasiMethod, VariantAllocation,
)
from ab_platform.simulation.vit import NetworkConfig, apply_guest_effect, generate_network
from ab_platform.simulation.vit.rng import stream


def passport(**kw):
    m = MetricDefinition(name="revenue", metric_type=MetricType.CONTINUOUS, numerator_col="revenue")
    g = MetricDefinition(name="orders", metric_type=MetricType.CONTINUOUS, numerator_col="orders")
    data = dict(
        experiment_id="e", test_key="k", hypothesis_statement="Если мы X, то выручка вырастет на 5%",
        variants=[VariantAllocation(variant_id="control", traffic_percentage=50, is_control=True),
                  VariantAllocation(variant_id="treatment", traffic_percentage=50)],
        primary_metric=m,
        guardrail_metrics=[GuardrailDefinition(metric=g, max_allowed_degradation_rel=0.30, tradeoff_rationale="частота")],
        baseline_value=1000, mde_relative=0.04, planned_duration_days=28, daily_eligible_units=1000,
        covariate_col="pre_revenue",
    )
    data.update(kw)
    return ExperimentPassport(**data)


@pytest.fixture(scope="module")
def guests():
    net = generate_network(NetworkConfig.preset("reference", n_restaurants=20, days=14, seed=1))
    return net.guests(40_000, window="decision")


def frame(g, treated, lift=0.0, seed=0):
    d = g[["guest_id", "revenue", "orders", "pre_revenue"]].rename(columns={"guest_id": "user_id"}).copy()
    if lift:
        g2, _ = apply_guest_effect(g, treated, lift=lift, test_days=28, seed=seed)
        d["revenue"], d["orders"] = g2["revenue"].to_numpy(), g2["orders"].to_numpy()
    d["variant"] = np.where(treated, "treatment", "control")
    return d


@pytest.fixture(scope="module")
def big_effect(guests):
    t = stream(2, "decision").random(len(guests)) < 0.5
    return frame(guests, t, lift=0.12, seed=3)


def test_alias_ship_to_production_is_ship_candidate():
    assert DecisionStatus.SHIP_TO_PRODUCTION is DecisionStatus.SHIP_CANDIDATE
    assert DecisionStatus.SHIP_CANDIDATE == "SHIP_CANDIDATE"


def test_ship_candidate_for_large_significant_effect(big_effect):
    r = O.evaluate_experiment(passport(), big_effect)
    assert r.decision_status == DecisionStatus.SHIP_CANDIDATE and r.winning_variant == "treatment"
    assert r.evidence_level == "A" and r.estimand == "itt" and "приложения" in r.generalization_population


def test_d1_significant_but_below_practical_threshold_is_not_shipped(big_effect):
    r = O.evaluate_experiment(passport(practical_significance_threshold=0.30), big_effect)
    assert r.primary_metric_result["is_significant"]
    assert r.decision_status == DecisionStatus.NEUTRAL_BELOW_PRACTICAL_THRESHOLD
    assert r.winning_variant is None


def test_d2_quasi_experiment_never_auto_ships(big_effect):
    p = passport(experiment_type=ExperimentType.QUASI_EXPERIMENT, quasi_method=QuasiMethod.AIPW,
                 why_not_randomized="Гости сами включают фичу, назначить нельзя")
    r = O.evaluate_experiment(p, big_effect)
    assert r.decision_status == DecisionStatus.HOLD_HUMAN_REVIEW_EVIDENCE
    assert r.evidence_level == "C" and r.winning_variant is None


def test_d3_sidedness_reaches_criteria(guests, big_effect):
    r = O.evaluate_experiment(passport(sidedness="one_sided"), big_effect)
    assert r.primary_metric_result["sidedness"] == "one_sided"
    two = O.evaluate_experiment(passport(), big_effect).primary_metric_result["p_value"]
    assert r.primary_metric_result["p_value"] == pytest.approx(two / 2, rel=1e-6)


def test_blinded_before_horizon_hides_primary_and_secondary(big_effect):
    sec = MetricDefinition(name="orders_sec", metric_type=MetricType.CONTINUOUS, numerator_col="orders")
    p = passport(secondary_metrics=[sec])
    r = O.evaluate_experiment(p, big_effect, analysis_day=14)
    assert r.decision_status == DecisionStatus.RUNNING_BLINDED and r.blinded
    assert "p_value" not in r.primary_metric_result
    assert all("p_value" not in v for v in r.variant_comparisons.values())
    assert r.secondary_results == [] and r.winning_variant is None
    assert r.guardrail_results                                   # guardrails видны всегда
    done = O.evaluate_experiment(p, big_effect, analysis_day=28)
    assert done.decision_status == DecisionStatus.SHIP_CANDIDATE and not done.blinded
    seq = O.evaluate_experiment(passport(analysis_mode="sequential"), big_effect, analysis_day=7)
    assert not seq.blinded
    with pytest.raises(ValueError):
        O.evaluate_experiment(p, big_effect, analysis_day=-1)


def test_guardrail_breach_visible_while_blinded(guests):
    t = stream(4, "decision").random(len(guests)) < 0.5
    d = frame(guests, t)
    d["orders"] = d["orders"].astype(float)
    d.loc[t, "orders"] = d.loc[t, "orders"] * 0.5                 # явный пробой частоты
    p = passport(guardrail_metrics=[GuardrailDefinition(
        metric=MetricDefinition(name="orders", metric_type=MetricType.CONTINUOUS, numerator_col="orders"),
        max_allowed_degradation_rel=0.05, tradeoff_rationale="частота")])
    r = O.evaluate_experiment(p, d, analysis_day=7)
    assert r.decision_status == DecisionStatus.DO_NOT_SHIP_GUARDRAIL_BREACH


def test_d5_double_exposure_threshold_from_passport(guests):
    t = stream(5, "decision").random(len(guests)) < 0.5
    d = frame(guests, t)
    dup = d.sample(frac=0.03, random_state=1).copy()
    dup["variant"] = np.where(dup["variant"] == "control", "treatment", "control")
    d2 = pd.concat([d, dup], ignore_index=True)
    assert O.evaluate_experiment(passport(), d2).decision_status != DecisionStatus.INVALID_EXPERIMENT_SRM_OR_CONTAMINATION
    strict = O.evaluate_experiment(passport(max_double_exposure_share=0.01), d2)
    assert strict.decision_status == DecisionStatus.INVALID_EXPERIMENT_SRM_OR_CONTAMINATION


def test_d5_secondary_tier_is_a_label_not_alpha(big_effect):
    sec = MetricDefinition(name="orders_sec", metric_type=MetricType.CONTINUOUS, numerator_col="orders")
    r = O.evaluate_experiment(passport(secondary_metrics=[sec]), big_effect, secondary_tiers={"orders_sec": 3})
    s = r.secondary_results[0]
    assert s.tier_order == 3 and s.tier_alpha == 0.05 and s.is_significant_tiered == s.is_significant_fdr


# ---------------- критерии: односторонний тест в нужную сторону ----------------
@pytest.mark.parametrize("fn,args", [
    ("welchs_t_test", (np.r_[np.zeros(50), np.ones(50)] * 1.0 + 5, np.r_[np.zeros(50), np.ones(50)] * 1.0 + 4.6)),
    ("proportions_z_test", (np.r_[np.ones(300), np.zeros(700)], np.r_[np.ones(250), np.zeros(750)])),
])
def test_one_sided_direction(fn, args):
    c, t = args                                           # в тесте метрика упала
    f = getattr(R, fn)
    dec = f(c, t, sidedness="one_sided", direction="decrease")
    inc = f(c, t, sidedness="one_sided", direction="increase")
    two = f(c, t)
    assert dec["p_value"] < 0.05 < inc["p_value"]
    assert dec["p_value"] + inc["p_value"] == pytest.approx(1.0, abs=1e-9)
    assert dec["p_value"] == pytest.approx(two["p_value"] / 2, rel=1e-6)


def test_one_sided_ratio_bootstrap_and_fisher():
    rng = np.random.default_rng(0)
    nc, dc = rng.poisson(5, 400) + 1.0, rng.poisson(10, 400) + 1.0
    nt, dt = rng.poisson(4, 400) + 1.0, rng.poisson(10, 400) + 1.0
    r = R.ratio_delta_test(nc, dc, nt, dt, sidedness="one_sided", direction="decrease")
    assert r["p_value"] < 0.05 and r["direction"] == "decrease"
    q = R.bootstrap_quantile_test(rng.normal(10, 1, 500), rng.normal(9.6, 1, 500), quantile=0.5,
                                  sidedness="one_sided", direction="decrease")
    assert q["p_value"] < 0.05 and q["is_significant"]
    fz = R.proportions_z_test(np.r_[np.ones(8), np.zeros(42)], np.r_[np.ones(1), np.zeros(49)],
                              sidedness="one_sided", direction="decrease")
    assert fz["criterion"] == "fisher_exact_wilson_ci" and fz["p_value"] < 0.05
    with pytest.raises(ValueError):
        R.welchs_t_test(np.ones(5), np.ones(5) * 2, direction="up")
    deg = R.welchs_t_test(np.ones(5), np.ones(5) * 0.5, sidedness="one_sided", direction="decrease")
    assert deg["p_value"] == 0.0
