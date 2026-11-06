"""Фаза 1, П1: Паспорт v4 — поля, валидаторы ТЗ v4 (раздел 9), неизменяемость, статусы (PAS-1, PAS-2, K7)."""
import pytest

from ab_platform.schemas.passport import (
    AnalysisUnit, ClusterLevel, Domain, Estimand, EvidenceLevel, ExperimentPassport, ExperimentType,
    GuardrailDefinition, InferenceMethod, MetricDefinition, MetricType, Pairing, PassportStatus, QuasiMethod,
    RandomizationUnit, VariantAllocation,
)


def base(**kw):
    m = MetricDefinition(name="revenue", metric_type=MetricType.CONTINUOUS, numerator_col="revenue")
    data = dict(
        experiment_id="exp_v4", test_key="k_v4", hypothesis_statement="Если мы X, то выручка вырастет на 3%",
        variants=[VariantAllocation(variant_id="control", traffic_percentage=50, is_control=True),
                  VariantAllocation(variant_id="treatment", traffic_percentage=50)],
        primary_metric=m,
        guardrail_metrics=[GuardrailDefinition(metric=m, max_allowed_degradation_rel=0.02, tradeoff_rationale="маржа")],
        baseline_value=600_000, mde_relative=0.03, planned_duration_days=28, daily_eligible_units=10_000,
    )
    data.update(kw)
    return data


def restaurant(**kw):
    return base(experiment_type=ExperimentType.CLUSTER_AB, randomization_unit=RandomizationUnit.STORE_ID,
                analysis_unit=AnalysisUnit.STORE_DAY, daily_eligible_units=200, **kw)


# ---------------- значения по умолчанию ----------------
def test_v2_passport_gets_safe_v4_defaults():
    p = ExperimentPassport(**base())
    assert p.domain == Domain.GUEST and p.estimand == Estimand.ITT and p.cluster_level == ClusterLevel.NONE
    assert "приложения" in p.generalization_population
    assert p.evidence_level == EvidenceLevel.A and p.auto_ship_allowed
    assert p.status == PassportStatus.DRAFT


def test_restaurant_defaults_are_clustered():
    p = ExperimentPassport(**restaurant())
    assert p.domain == Domain.RESTAURANT
    assert p.estimand == Estimand.PER_RESTAURANT_MEAN
    assert p.cluster_level == ClusterLevel.RESTAURANT


# ---------------- валидаторы раздела 9 ----------------
def test_k7_store_day_without_clusters_rejected():
    with pytest.raises(ValueError, match="кластерного инференса"):
        ExperimentPassport(**restaurant(cluster_level=ClusterLevel.NONE))
    with pytest.raises(ValueError, match="store_day без cluster_level"):
        ExperimentPassport(**base(analysis_unit=AnalysisUnit.STORE_DAY))


def test_restaurant_requires_cuped():
    with pytest.raises(ValueError, match="CUPED"):
        ExperimentPassport(**restaurant(use_cuped=False))


def test_few_clusters_require_randomization_or_wild_bootstrap():
    with pytest.raises(ValueError, match="асимптотический"):
        ExperimentPassport(**restaurant(n_clusters=60))
    for inf in (InferenceMethod.RANDOMIZATION, InferenceMethod.WILD_BOOTSTRAP):
        assert ExperimentPassport(**restaurant(n_clusters=60, inference=inf)).inference == inf


def test_minimum_six_pairs():
    with pytest.raises(ValueError, match="2/2\\^k"):
        ExperimentPassport(**restaurant(n_clusters=10, inference=InferenceMethod.RANDOMIZATION, pairing=Pairing.PAIRS))
    ok = ExperimentPassport(**restaurant(n_clusters=12, inference=InferenceMethod.RANDOMIZATION, pairing=Pairing.PAIRS))
    assert ok.pairing == Pairing.PAIRS


def test_domain_and_estimand_consistency():
    with pytest.raises(ValueError, match="не соответствует единице рандомизации"):
        ExperimentPassport(**base(domain=Domain.RESTAURANT))
    with pytest.raises(ValueError, match="только для ресторанного домена"):
        ExperimentPassport(**base(estimand=Estimand.NETWORK_TOTAL))
    with pytest.raises(ValueError, match="per_restaurant_mean или network_total"):
        ExperimentPassport(**restaurant(estimand=Estimand.ITT))


def test_quasi_requires_justification_and_method():
    q = dict(experiment_type=ExperimentType.QUASI_EXPERIMENT)
    with pytest.raises(ValueError, match="why_not_randomized"):
        ExperimentPassport(**base(**q, quasi_method=QuasiMethod.SDID))
    with pytest.raises(ValueError, match="quasi_method"):
        ExperimentPassport(**base(**q, why_not_randomized="Пилот в одном городе по решению операций"))
    with pytest.raises(ValueError, match="только при experiment_type"):
        ExperimentPassport(**base(quasi_method=QuasiMethod.DID))


@pytest.mark.parametrize("method,level", [
    (QuasiMethod.SDID, EvidenceLevel.B), (QuasiMethod.STAGGERED_DID, EvidenceLevel.B_MINUS),
    (QuasiMethod.AIPW, EvidenceLevel.C), (QuasiMethod.ITS, EvidenceLevel.D),
])
def test_evidence_level_from_design(method, level):
    p = ExperimentPassport(**base(experiment_type=ExperimentType.QUASI_EXPERIMENT, quasi_method=method,
                                  why_not_randomized="Изменение на всю сеть по решению операций"))
    assert p.evidence_level == level and not p.auto_ship_allowed


def test_switchback_is_a_minus():
    p = ExperimentPassport(**base(experiment_type=ExperimentType.SWITCHBACK,
                                  randomization_unit=RandomizationUnit.CLUSTER_TIME_CELL,
                                  analysis_unit=AnalysisUnit.CLUSTER_CELL, daily_eligible_units=500))
    assert p.domain == Domain.RESTAURANT_TIME and p.evidence_level == EvidenceLevel.A_MINUS and p.auto_ship_allowed


def test_mvp_boundaries_price_and_franchise():
    with pytest.raises(ValueError, match="Ценовые тесты"):
        ExperimentPassport(**restaurant(levers=["promo", "Цена"]))
    with pytest.raises(ValueError, match="франчайзи"):
        ExperimentPassport(**restaurant(includes_franchise=True))


# ---------------- неизменяемость, версии, статусы ----------------
def test_model_copy_revalidates():
    p = ExperimentPassport(**base())
    with pytest.raises(ValueError, match="полных недель"):
        p.model_copy(update={"planned_duration_days": 3})
    with pytest.raises(ValueError):
        p.model_copy(update={"status": "whatever"})
    with pytest.raises(ValueError, match="Неизвестные поля"):
        p.model_copy(update={"no_such_field": 1})


def test_status_transitions_and_hash_lock():
    p = ExperimentPassport(**base())
    with pytest.raises(ValueError, match="Недопустимый переход"):
        p.with_status(PassportStatus.RUNNING)
    approved = p.locked()
    assert approved.status == PassportStatus.APPROVED and approved.passport_hash == approved.compute_hash()
    running = approved.with_status("running")
    assert running.passport_hash == approved.passport_hash
    with pytest.raises(ValueError, match="Недопустимый переход"):
        running.with_status(PassportStatus.APPROVED)


def test_tampered_hash_blocks_transition():
    approved = ExperimentPassport(**base()).locked()
    forged = approved.model_copy(update={"mde_relative": 0.10})  # содержимое изменено, хэш старый
    with pytest.raises(ValueError, match="Хэш Паспорта не совпадает"):
        forged.with_status(PassportStatus.RUNNING)


def test_new_version_bumps_and_resets():
    approved = ExperimentPassport(**base()).locked()
    v2 = approved.new_version(mde_relative=0.05)
    assert v2.version == 2 and v2.status == PassportStatus.DRAFT and v2.passport_hash is None
    assert v2.locked().passport_hash != approved.passport_hash
    with pytest.raises(ValueError):
        approved.new_version()
    with pytest.raises(ValueError, match="нельзя менять"):
        approved.new_version(status="running")
