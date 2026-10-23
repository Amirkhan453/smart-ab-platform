import numpy as np
from ab_platform.schemas.passport import (
    AnalysisUnit,
    Direction,
    ExperimentPassport,
    ExperimentType,
    GuardrailDefinition,
    MetricDefinition,
    MetricType,
    RandomizationUnit,
    VariantAllocation,
)
from ab_platform.simulation.synthetic import SyntheticDataGenerator
from ab_platform.orchestrator import SmartABPlatformOrchestrator


def _build_sample_passport(primary_type: MetricType = MetricType.CONTINUOUS) -> ExperimentPassport:
    num_col = "post_revenue" if primary_type == MetricType.CONTINUOUS else "orders"
    den_col = None if primary_type == MetricType.CONTINUOUS else "sessions"
    analysis_unit = AnalysisUnit.USER if primary_type == MetricType.CONTINUOUS else AnalysisUnit.PAGEVIEW

    return ExperimentPassport(
        experiment_id="exp_orchestrator_01",
        test_key="smart_checkout_v2",
        hypothesis_statement="Если мы внедрим умный чекаут, то выручка на пользователя вырастет на 5%, так как ускорится оплата",
        experiment_type=ExperimentType.CLASSIC_AB,
        randomization_unit=RandomizationUnit.USER_ID,
        analysis_unit=analysis_unit,
        variants=[
            VariantAllocation(variant_id="control", traffic_percentage=50.0, is_control=True),
            VariantAllocation(variant_id="treatment", traffic_percentage=50.0, is_control=False),
        ],
        primary_metric=MetricDefinition(
            name="revenue_per_user" if primary_type == MetricType.CONTINUOUS else "orders_per_session",
            metric_type=primary_type,
            numerator_col=num_col,
            denominator_col=den_col,
            expected_direction=Direction.INCREASE,
        ),
        secondary_metrics=[
            MetricDefinition(
                name="conversion_rate",
                metric_type=MetricType.BINARY,
                numerator_col="converted",
                expected_direction=Direction.INCREASE,
            )
        ],
        guardrail_metrics=[
            GuardrailDefinition(
                metric=MetricDefinition(
                    name="page_error_rate",
                    metric_type=MetricType.BINARY,
                    numerator_col="page_error",
                    expected_direction=Direction.DECREASE,
                ),
                max_allowed_degradation_rel=0.10,
                tradeoff_rationale="Контроль технических ошибок чекаута",
            )
        ],
        baseline_value=1000.0,
        mde_relative=0.03,
        planned_duration_days=14,
        daily_eligible_units=5000,
    )


def test_orchestrator_end_to_end_winning_ship_decision():
    """Проверяем полный цикл успешного теста: очистка -> CUPED -> Welch -> Guardrails OK -> SHIP_TO_PRODUCTION."""
    passport = _build_sample_passport(MetricType.CONTINUOUS)
    df = SyntheticDataGenerator.generate_ab_dataset(
        n_users=6000, true_lift_rel=0.06, pre_post_corr=0.75, double_exposure_fraction=0.01
    )
    df["page_error"] = 0

    report = SmartABPlatformOrchestrator.evaluate_experiment(
        passport=passport,
        df=df,
        covariate_col="pre_revenue",
        triggered_share_alpha=0.50,
    )
    assert report.decision_status == "SHIP_TO_PRODUCTION"
    assert report.can_trust_data is True
    assert report.cuped_diagnostics["applied"] is True
    assert report.cuped_diagnostics["variance_reduction_pct"] > 45.0
    assert "РЕКОМЕНДУЕТСЯ К РАСКАТКЕ" in report.executive_summary_ru


def test_orchestrator_blocks_ship_when_guardrail_breached():
    """Если основная метрика выросла, но Guardrail пробит, платформа обязана заблокировать раскатку."""
    passport = _build_sample_passport(MetricType.CONTINUOUS)
    df = SyntheticDataGenerator.generate_ab_dataset(n_users=6000, true_lift_rel=0.08)
    rng = np.random.default_rng(42)
    df["page_error"] = np.where(
        df["variant"] == "treatment",
        (rng.random(len(df)) < 0.08).astype(int),
        (rng.random(len(df)) < 0.02).astype(int),
    )

    report = SmartABPlatformOrchestrator.evaluate_experiment(
        passport=passport,
        df=df,
        covariate_col="pre_revenue",
    )
    assert report.decision_status == "DO_NOT_SHIP_GUARDRAIL_BREACH"
    assert any(g.breached for g in report.guardrail_results)
    assert "ЗАПРЕТ РАСКАТКИ" in report.executive_summary_ru


def test_orchestrator_blocks_ship_when_srm_detected():
    """Если обнаружен Sample Ratio Mismatch (SRM), платформа помечает эксперимент как недостоверный."""
    passport = _build_sample_passport(MetricType.CONTINUOUS)
    df = SyntheticDataGenerator.generate_ab_dataset(
        n_users=6000, treatment_ratio=0.60, true_lift_rel=0.05
    )
    df["page_error"] = 0

    report = SmartABPlatformOrchestrator.evaluate_experiment(passport=passport, df=df)
    assert report.decision_status == "INVALID_EXPERIMENT_SRM_OR_CONTAMINATION"
    assert report.can_trust_data is False


def test_orchestrator_ratio_metric_delta_routing():
    """Проверяем автоматический вызов Дельта-метода при оценке Ratio-метрики."""
    passport = _build_sample_passport(MetricType.RATIO)
    df = SyntheticDataGenerator.generate_ab_dataset(n_users=6000, true_lift_rel=0.08)
    df["page_error"] = 0

    report = SmartABPlatformOrchestrator.evaluate_experiment(passport=passport, df=df)
    assert report.primary_metric_result["criterion"] == "ratio_delta_method"
    assert report.decision_status == "SHIP_TO_PRODUCTION"
