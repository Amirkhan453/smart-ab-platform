import numpy as np
import pandas as pd
import pytest

from ab_platform.schemas.passport import (
    Direction,
    ExperimentPassport,
    GuardrailDefinition,
    MetricDefinition,
    MetricType,
    VariantAllocation,
)
from ab_platform.connectors.sql_builder import WarehouseSQLBuilder
from ab_platform.engine.assignment import DeterministicAssignmentEngine
from ab_platform.engine.power import PowerAndSampleSizeEngine
from ab_platform.engine.safeguards import ExperimentSafeguardEngine
from ab_platform.engine.criterion_router import StatCriterionRouter
from ab_platform.simulation.synthetic import SyntheticDataGenerator
from ab_platform.orchestrator import SmartABPlatformOrchestrator


def test_sql_injection_is_blocked():
    with pytest.raises(ValueError, match="SQL Injection Guard"):
        WarehouseSQLBuilder.build_user_level_cuped_query(
            "x' OR 1=1 --", "revenue"
        )

    with pytest.raises(ValueError, match="SQL Injection Guard"):
        WarehouseSQLBuilder.build_user_level_cuped_query(
            "exp_01", "revenue'; DROP TABLE conversion_events; --"
        )


def test_missing_guardrail_or_primary_column_never_silently_skipped():
    df = SyntheticDataGenerator.generate_ab_dataset(n_users=1000, true_lift_rel=0.05)
    df["page_error"] = 0

    passport = ExperimentPassport(
        experiment_id="exp_guard_check",
        test_key="k1",
        hypothesis_statement="Проверка обязательности всех колонок паспорта перед расчетом",
        variants=[
            VariantAllocation(variant_id="control", traffic_percentage=50.0, is_control=True),
            VariantAllocation(variant_id="treatment", traffic_percentage=50.0, is_control=False),
        ],
        primary_metric=MetricDefinition(name="rev", metric_type=MetricType.CONTINUOUS, numerator_col="post_revenue"),
        guardrail_metrics=[
            GuardrailDefinition(
                metric=MetricDefinition(name="err", metric_type=MetricType.BINARY, numerator_col="page_error"),
                max_allowed_degradation_rel=0.05,
                tradeoff_rationale="Ошибки страницы",
            )
        ],
        baseline_value=1000.0,
        mde_relative=0.03,
        planned_duration_days=14,
        daily_eligible_units=1000,
    )

    with pytest.raises(ValueError, match="page_error"):
        SmartABPlatformOrchestrator.evaluate_experiment(passport, df.drop(columns=["page_error"]))

    with pytest.raises(ValueError, match="post_revenue"):
        SmartABPlatformOrchestrator.evaluate_experiment(passport, df.drop(columns=["post_revenue"]))


def test_multi_arm_experiment_evaluation_and_holm_correction():
    rng = np.random.default_rng(123)
    n = 3000
    variants = np.where(
        np.arange(n) % 3 == 0,
        "control",
        np.where(np.arange(n) % 3 == 1, "treatment_1", "treatment_2"),
    )
    pre_rev = rng.normal(1000.0, 200.0, n)
    lift = np.where(variants == "treatment_2", 80.0, 0.0)
    post_rev = pre_rev * 0.7 + 300.0 + lift + rng.normal(0.0, 80.0, n)

    df_multi = pd.DataFrame({
        "user_id": [f"u_{i}" for i in range(n)],
        "variant": variants,
        "pre_revenue": pre_rev,
        "post_revenue": post_rev,
        "page_error": np.zeros(n, dtype=int),
    })

    passport_multi = ExperimentPassport(
        experiment_id="exp_multi_arm",
        test_key="multi_test",
        hypothesis_statement="Тестируем два варианта дизайна против контроля с поправкой Холма-Бонферрони",
        variants=[
            VariantAllocation(variant_id="control", traffic_percentage=33.34, is_control=True),
            VariantAllocation(variant_id="treatment_1", traffic_percentage=33.33, is_control=False),
            VariantAllocation(variant_id="treatment_2", traffic_percentage=33.33, is_control=False),
        ],
        primary_metric=MetricDefinition(name="rev", metric_type=MetricType.CONTINUOUS, numerator_col="post_revenue"),
        guardrail_metrics=[
            GuardrailDefinition(
                metric=MetricDefinition(name="err", metric_type=MetricType.BINARY, numerator_col="page_error"),
                max_allowed_degradation_rel=0.05,
                tradeoff_rationale="Контроль ошибок",
            )
        ],
        baseline_value=1000.0,
        mde_relative=0.03,
        planned_duration_days=14,
        daily_eligible_units=2000,
    )

    report = SmartABPlatformOrchestrator.evaluate_experiment(
        passport_multi, df_multi, covariate_col="pre_revenue"
    )
    assert report.decision_status == "SHIP_TO_PRODUCTION"
    assert report.winning_variant == "treatment_2"
    assert report.variant_comparisons["treatment_1"]["is_significant"] is False
    assert report.variant_comparisons["treatment_2"]["is_significant"] is True


def test_user_level_aggregation_before_srm_and_stats():
    df = SyntheticDataGenerator.generate_ab_dataset(n_users=1000, true_lift_rel=0.06)
    df["page_error"] = 0
    raw_event_logs = pd.concat([df, df.iloc[:300]], ignore_index=True)

    passport = ExperimentPassport(
        experiment_id="exp_agg_check",
        test_key="k_agg",
        hypothesis_statement="Проверка агрегации посессионных логов до уникальных пользователей",
        variants=[
            VariantAllocation(variant_id="control", traffic_percentage=50.0, is_control=True),
            VariantAllocation(variant_id="treatment", traffic_percentage=50.0, is_control=False),
        ],
        primary_metric=MetricDefinition(name="rev", metric_type=MetricType.CONTINUOUS, numerator_col="post_revenue"),
        guardrail_metrics=[
            GuardrailDefinition(
                metric=MetricDefinition(name="err", metric_type=MetricType.BINARY, numerator_col="page_error"),
                max_allowed_degradation_rel=0.05,
                tradeoff_rationale="Контроль ошибок",
            )
        ],
        baseline_value=1000.0,
        mde_relative=0.03,
        planned_duration_days=14,
        daily_eligible_units=1000,
    )

    report = SmartABPlatformOrchestrator.evaluate_experiment(passport, raw_event_logs, covariate_col="pre_revenue")
    assert report.clean_sample_size == 1000
    assert report.can_trust_data is True


def test_reviewer_edge_cases_validation():
    with pytest.raises(ValueError):
        DeterministicAssignmentEngine.assign_user(
            "u", "e", exposure_rate_pct=-1, control_split_pct=150, global_holdout_pct=200
        )

    with pytest.raises(ValueError):
        ExperimentSafeguardEngine.calculate_psi(np.array([0, 0]), np.array([0, 0]))

    with pytest.raises(ValueError):
        StatCriterionRouter.ratio_delta_test(np.array([]), np.array([]), np.array([1.0]), np.array([1.0]))

    with pytest.raises(ValueError):
        PowerAndSampleSizeEngine.calculate_sample_size(
            baseline_mean=0.0, mde_relative=0.1, is_binary=False, metric_variance=1.0
        )

    res_inc = PowerAndSampleSizeEngine.calculate_sample_size(
        baseline_mean=0.60,
        mde_absolute=0.05,
        is_binary=True,
        expected_direction=Direction.INCREASE,
    )
    res_dec = PowerAndSampleSizeEngine.calculate_sample_size(
        baseline_mean=0.60,
        mde_absolute=0.05,
        is_binary=True,
        expected_direction=Direction.DECREASE,
    )
    assert res_inc.n_per_variant < res_dec.n_per_variant
