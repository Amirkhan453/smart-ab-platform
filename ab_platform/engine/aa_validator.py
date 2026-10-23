import pandas as pd
import numpy as np
from scipy import stats
from ab_platform.engine.criterion_router import StatCriterionRouter


class HistoricalAAValidator:
    """
    Монте-Карло симулятор А/А-тестов на исторических данных пользователя.
    Проверяет, контролируется ли ошибка I рода (FPR ~ alpha) и равномерно ли распределены p-values.
    """

    @staticmethod
    def run_aa_simulation(
        df: pd.DataFrame, metric_col: str, iterations: int = 1000, alpha: float = 0.05
    ) -> dict:
        p_values = []
        n = len(df)
        metric_vals = df[metric_col].to_numpy(dtype=float)
        rng = np.random.default_rng(42)

        for _ in range(iterations):
            is_treatment = rng.random(n) < 0.5
            treatment_vals = metric_vals[is_treatment]
            control_vals = metric_vals[~is_treatment]

            if len(treatment_vals) < 2 or len(control_vals) < 2:
                continue

            res = StatCriterionRouter.welchs_t_test(control_vals, treatment_vals, alpha=alpha)
            if not np.isnan(res["p_value"]):
                p_values.append(res["p_value"])

        p_values_arr = np.array(p_values, dtype=float)
        fpr = float(np.mean(p_values_arr < alpha))

        # Критерий Колмогорова-Смирнова на соответствие равномерному распределению U(0, 1)
        ks_stat, ks_pval = stats.kstest(p_values_arr, "uniform")

        return {
            "iterations_run": len(p_values_arr),
            "false_positive_rate": fpr,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "is_uniform_distribution": bool(ks_pval > 0.01),
        }
