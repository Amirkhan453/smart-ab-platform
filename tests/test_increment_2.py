import numpy as np
from ab_platform.engine.hygiene import DataHygieneProcessor
from ab_platform.engine.cuped import VarianceReductionEngine
from ab_platform.engine.criterion_router import StatCriterionRouter
from ab_platform.engine.aa_validator import HistoricalAAValidator
from ab_platform.simulation.synthetic import SyntheticDataGenerator


def test_hygiene_double_exposure_and_winsorization():
    df = SyntheticDataGenerator.generate_ab_dataset(
        n_users=2000, double_exposure_fraction=0.05, outlier_fraction=0.01
    )
    clean_df, fraction = DataHygieneProcessor.remove_double_exposure(df)

    assert fraction > 0.0
    assert len(clean_df) < len(df)
    assert clean_df.groupby("user_id")["variant"].nunique().max() == 1

    win_df = DataHygieneProcessor.apply_winsorization(clean_df, "post_revenue", upper_quantile=0.99)
    assert win_df["post_revenue"].max() < clean_df["post_revenue"].max()


def test_cuped_lins_and_power_optimal_weight():
    df = SyntheticDataGenerator.generate_ab_dataset(
        n_users=5000, true_lift_rel=0.05, pre_post_corr=0.8
    )

    cuped_df, diag = VarianceReductionEngine.apply_cuped(df, "post_revenue", "pre_revenue")
    assert diag["applied"] is True
    assert diag["variance_reduction_pct"] > 50.0
    assert diag["sensitivity_gain_factor"] > 1.4

    lin_df = VarianceReductionEngine.apply_lins_method(df, "post_revenue", "pre_revenue")
    assert "post_revenue_lin_cuped" in lin_df.columns
    assert lin_df["post_revenue_lin_cuped"].var() < df["post_revenue"].var()

    # Проверка веса lambda = n_bar * (1 + cv^2) для Switchback CUPAC
    cell_info = VarianceReductionEngine.compute_power_optimal_cell_weight(np.array([100, 300, 500]))
    assert cell_info["lambda_macro_weight"] > cell_info["mean_cell_size"]


def test_stat_router_all_criteria_and_fdr():
    df = SyntheticDataGenerator.generate_ab_dataset(n_users=4000, true_lift_rel=0.08)
    c_df = df[df["variant"] == "control"]
    t_df = df[df["variant"] == "treatment"]

    # 1. Welch's t-test
    res_t = StatCriterionRouter.welchs_t_test(c_df["post_revenue"].to_numpy(), t_df["post_revenue"].to_numpy())
    assert res_t["is_significant"] is True

    # 2. Proportions Z-test
    res_z = StatCriterionRouter.proportions_z_test(c_df["converted"].to_numpy(), t_df["converted"].to_numpy())
    assert 0.0 <= res_z["p_value"] <= 1.0

    # 3. Ratio Delta-method (orders per session)
    res_ratio = StatCriterionRouter.ratio_delta_test(
        c_df["orders"].to_numpy(),
        c_df["sessions"].to_numpy(),
        t_df["orders"].to_numpy(),
        t_df["sessions"].to_numpy(),
    )
    assert res_ratio["criterion"] == "ratio_delta_method"
    assert res_ratio["is_significant"] is True

    # 4. FDR Benjamini-Hochberg
    p_values = [0.001, 0.002, 0.045, 0.15, 0.8]
    rejects = StatCriterionRouter.apply_fdr_correction(p_values, alpha=0.05)
    assert rejects[0] is True
    assert rejects[2] is False


def test_historical_aa_validator():
    df = SyntheticDataGenerator.generate_ab_dataset(
        n_users=3000, true_lift_rel=0.0, outlier_fraction=0.0
    )
    res = HistoricalAAValidator.run_aa_simulation(df, "post_revenue", iterations=200)

    assert 0.01 <= res["false_positive_rate"] <= 0.10
    assert res["is_uniform_distribution"] is True
