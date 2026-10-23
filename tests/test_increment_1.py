import pytest
from ab_platform.schemas.passport import (
    AnalysisUnit,
    ExperimentPassport,
    ExperimentType,
    GuardrailDefinition,
    MetricDefinition,
    MetricType,
    RandomizationUnit,
    RiskProfile,
    VariantAllocation,
)
from ab_platform.engine.power import PowerAndSampleSizeEngine
from ab_platform.simulation.synthetic import SyntheticDataGenerator


def get_valid_passport_dict():
    return {
        "experiment_id": "exp_001",
        "test_key": "checkout_progress_bar",
        "hypothesis_statement": "Если мы добавим прогресс-бар в чекаут, то конверсия в оплату вырастет на 3%, так как снизится неопределенность",
        "experiment_type": ExperimentType.CLASSIC_AB,
        "randomization_unit": RandomizationUnit.USER_ID,
        "analysis_unit": AnalysisUnit.USER,
        "variants": [
            VariantAllocation(variant_id="control", traffic_percentage=50.0, is_control=True),
            VariantAllocation(variant_id="treatment", traffic_percentage=50.0, is_control=False),
        ],
        "primary_metric": MetricDefinition(
            name="checkout_cr",
            metric_type=MetricType.BINARY,
            numerator_col="converted",
        ),
        "guardrail_metrics": [
            GuardrailDefinition(
                metric=MetricDefinition(
                    name="page_error_rate",
                    metric_type=MetricType.BINARY,
                    numerator_col="has_error",
                ),
                max_allowed_degradation_rel=0.05,
                tradeoff_rationale="Контроль отсутствия технических регрессий при новом UI",
            )
        ],
        "baseline_value": 0.10,
        "mde_relative": 0.05,
        "planned_duration_days": 14,
        "daily_eligible_units": 10000,
    }


def test_valid_passport_creation():
    data = get_valid_passport_dict()
    passport = ExperimentPassport(**data)
    assert passport.experiment_id == "exp_001"
    assert len(passport.guardrail_metrics) == 1


def test_sutva_network_effects_rejection():
    """Проверка, что платформа блокирует обычный поюзерный A/B при сетевых эффектах."""
    data = get_valid_passport_dict()
    data["risk_profile"] = RiskProfile(has_network_effects=True)
    with pytest.raises(ValueError, match="SUTVA Violation"):
        ExperimentPassport(**data)


def test_short_duration_seasonality_rejection():
    """Проверка, что платформа блокирует тесты короче 7 дней (защита от недельной сезонности)."""
    data = get_valid_passport_dict()
    data["planned_duration_days"] = 3
    with pytest.raises(ValueError, match="минимум 7 дней"):
        ExperimentPassport(**data)


def test_cuped_sample_size_reduction():
    """Проверка: при корреляции ковариаты r=0.7 требуемая выборка должна сократиться ровно на r^2 = 49%."""
    res_no_cuped = PowerAndSampleSizeEngine.calculate_sample_size(
        baseline_mean=1000.0,
        mde_relative=0.03,
        metric_variance=90000.0,
        is_binary=False,
        pre_post_correlation_r=0.0,
    )
    res_with_cuped = PowerAndSampleSizeEngine.calculate_sample_size(
        baseline_mean=1000.0,
        mde_relative=0.03,
        metric_variance=90000.0,
        is_binary=False,
        pre_post_correlation_r=0.7,
    )
    assert res_with_cuped.variance_reduction_pct == 49.0
    ratio = res_with_cuped.total_sample_size / res_no_cuped.total_sample_size
    assert 0.50 <= ratio <= 0.52


def test_delta_method_and_switchback_engine():
    """Проверка расчета дисперсии Дельта-методом и детекции дисбаланса кластеров в Switchback."""
    df = SyntheticDataGenerator.generate_ab_dataset(n_users=5000, true_lift_rel=0.05)
    delta_var = PowerAndSampleSizeEngine.compute_ratio_delta_variance(
        y_values=df["orders"].to_numpy(),
        x_values=df["sessions"].to_numpy(),
    )
    assert delta_var > 0.0

    # Проверка Switchback с высоким коэффициентом вариации кластеров (CV = 2.5 >= 2.0)
    sb_res = PowerAndSampleSizeEngine.calculate_switchback_mde(
        num_clusters_j=10,
        num_time_windows_h=14,
        mean_cell_size_n=200.0,
        cell_size_cv=2.5,
        baseline_mean=500.0,
        total_variance=25000.0,
    )
    assert sb_res.recommend_cluster_robust_cr2 is True
    assert sb_res.macro_inflation_factor_lambda == 200.0 * (1.0 + 2.5 ** 2)