"""Фаза 1, П4: A/A на боевом пути (PRE-2) и отрицательный контроль «ресторан-дни без кластеров»."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ab_platform.engine.aa_validator import AAValidationEngine as AA
from ab_platform.engine.assignment import DeterministicAssignmentEngine as ASG
from ab_platform.schemas.passport import (
    AnalysisUnit, ExperimentPassport, ExperimentType, GuardrailDefinition, InferenceMethod, MetricDefinition,
    MetricType, RandomizationUnit, VariantAllocation,
)
from ab_platform.simulation.vit import NetworkConfig, generate_network


def test_aa_hash_split_equals_assign_user():
    ids = [f"u{i}" for i in range(500)]
    z = ASG.buckets_for("aa:1:0", ids)
    ctrl = ASG.control_from_bucket(z, 50.0)
    combat = [ASG.assign_user(u, "aa:1:0").variant_id == "control" for u in ids]
    assert ctrl.tolist() == combat


@pytest.fixture(scope="module")
def guests():
    net = generate_network(NetworkConfig.preset("reference", n_restaurants=20, days=14, seed=2))
    g = net.guests(4000, window="aa")
    return g.rename(columns={"guest_id": "user_id"})


def test_guest_aa_combat_fast(guests):
    res = AA.run_aa_simulation(guests, "revenue", covariate_col="pre_revenue", iterations=300, seed=1)
    assert res["assignment"] == "hash" and res["iterations_run"] == 300
    assert 0.02 <= res["false_positive_rate"] <= 0.09
    binary = AA.run_aa_simulation(guests, "converted", iterations=200, seed=2)
    assert binary["criterion"] == "binary" and 0.015 <= binary["false_positive_rate"] <= 0.095
    with pytest.raises(ValueError):
        AA.run_aa_simulation(guests, "revenue", assignment="coin")


def restaurant_passport(n_pool, **kw):
    rev = MetricDefinition(name="revenue", metric_type=MetricType.CONTINUOUS, numerator_col="revenue",
                           winsorize_percentile=None)
    chk = MetricDefinition(name="checks", metric_type=MetricType.CONTINUOUS, numerator_col="checks",
                           winsorize_percentile=None)
    data = dict(
        experiment_id="rst_aa", test_key="rst_aa", hypothesis_statement="Если мы изменим промо, выручка вырастет",
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


def restaurant_days(n_rest, seed, preset="reference"):
    net = generate_network(NetworkConfig.preset(preset, n_restaurants=n_rest, days=84, seed=seed))
    d = net.daily
    start = pd.Timestamp(net.config.start_date) + pd.Timedelta(days=56)
    pre = d[d.date < start].groupby("restaurant_id")["revenue"].mean().rename("pre_revenue")
    post = d[d.date >= start].merge(pre, left_on="restaurant_id", right_index=True)
    return post[["restaurant_id", "date", "revenue", "checks", "pre_revenue"]]


def test_restaurant_aa_and_negative_control_fast():
    df = restaurant_days(100, 3)
    res = AA.run_cluster_aa_simulation(df, restaurant_passport(100), "restaurant_id", iterations=150, seed=3)
    assert res["criterion"].startswith("randomization_test") and res["n_clusters"] == 100
    assert res["false_positive_rate"] <= 0.10
    assert res["naive_storeday_fpr"] > 0.30              # запрещённый анализ: FPR раздут


# ---------------- полные приёмки ----------------
@pytest.mark.slow
def test_pre2_guest_aa_10000_runs(guests):
    """
    PRE-2: A/A боевым хэшем и боевым анализом, 10 000 прогонов на каждый из трёх заранее заданных seed.
    Критерий: FPR ∈ [4,5%; 5,5%] на каждом seed; KS p-value трёх seed, объединённые по Стауфферу, > 0,01.

    Почему не один seed (фаза 2, переход на MurmurHash3): одиночный KS при пороге 0,01 даёт ложную тревогу
    в ~1% прогонов по построению. На seed 11 новый хэш дал KS p = 0,0019 при FPR 5,14%, при этом на 40 seed
    × 10 000 хэш неотличим от бернуллиевского разбиения (Манн — Уитни p = 0,55; см. «фаза 2/диагностика PRE-2»).
    Дефект хэша систематический — он сдвигает KS во всех seed, и объединение по Стауфферу его ловит;
    единичный выброс — нет. Seed 11 оставлен в наборе намеренно.
    """
    ks_p = []
    for seed in (11, 12, 13):
        res = AA.run_aa_simulation(guests, "revenue", covariate_col="pre_revenue", iterations=10_000, seed=seed)
        assert 0.045 <= res["false_positive_rate"] <= 0.055, (seed, res["false_positive_rate"])
        ks_p.append(res["ks_pvalue"])
    combined = stats.combine_pvalues(ks_p, method="stouffer").pvalue
    assert combined > 0.01, (ks_p, combined)


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["reference", "reference_pessimistic"])
def test_restaurant_aa_full(preset):
    df = restaurant_days(100, 4, preset)
    res = AA.run_cluster_aa_simulation(df, restaurant_passport(100), "restaurant_id", iterations=2000, seed=4)
    assert 0.0374 <= res["false_positive_rate"] <= 0.0626, res["false_positive_rate"]
    assert res["naive_storeday_fpr"] > 0.30


@pytest.mark.slow
def test_asymptotic_cr2_on_heavy_tailed_history_documented():
    """Почему по умолчанию randomization: на этой истории асимптотический CR2 (50 на руку) завышает FPR."""
    df = restaurant_days(100, 4, "reference_pessimistic")
    p = restaurant_passport(100, inference=InferenceMethod.ASYMPTOTIC)
    res = AA.run_cluster_aa_simulation(df, p, "restaurant_id", iterations=2000, seed=4)
    assert res["criterion"] == "lin_cr2_bell_mccaffrey"
    assert res["false_positive_rate"] > 0.0626        # замер фазы 1: 6,75%


@pytest.mark.slow
def test_restaurant_aa_few_clusters_randomization():
    df = restaurant_days(20, 5, "reference_pessimistic")
    p = restaurant_passport(20, inference=InferenceMethod.RANDOMIZATION)
    res = AA.run_cluster_aa_simulation(df, p, "restaurant_id", iterations=1000, seed=5)
    assert res["criterion"].startswith("randomization_test")
    assert 0.03 <= res["false_positive_rate"] <= 0.07, res["false_positive_rate"]
