import math
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from scipy import stats
from ab_platform.engine.criterion_router import StatCriterionRouter
from ab_platform.engine.cuped import VarianceReductionEngine


class AAValidationEngine:
    """
    Пре-flight валидация эксперимента через симуляцию A/A тестов:
    - Сплит по единице рандомизации (user_col), опциональный CUPED-препроцессинг
    - Проверка попадания FPR в 95% биномиальный интервал вокруг alpha (нормальное приближение)
    - Проверка равномерности p-value (KS-тест, is_uniform_distribution при ks_pvalue > 0.01)
    - Проверка баланса ковариат пре-периода по |SMD| < 0.1 (check_covariate_balance_smd)
    """

    @staticmethod
    def run_aa_simulation(
        df: pd.DataFrame,
        metric_col: str,
        user_col: str = "user_id",
        iterations: int = 10000,
        alpha: float = 0.05,
        covariate_col: Optional[str] = None,
        cuped_min_r2: float = 0.05,
        seed: int = 42,
    ) -> dict:
        """
        Боевой A/A контур: сплит выполняется по уникальной единице рандомизации
        (user_col), а не по строкам, чтобы корректно работать при событийных
        (не агрегированных) входных данных. Если передан covariate_col, на
        каждой итерации применяется CUPED (VarianceReductionEngine.apply_cuped)
        с тем же порогом cuped_min_r2, что и в боевом анализе.
        """
        if metric_col not in df.columns:
            raise ValueError(f"Колонка '{metric_col}' отсутствует в датасете.")
        if user_col not in df.columns:
            raise ValueError(f"Колонка '{user_col}' отсутствует в датасете.")
        if covariate_col is not None and covariate_col not in df.columns:
            raise ValueError(f"Колонка ковариаты '{covariate_col}' отсутствует в датасете.")
        if iterations < 1:
            raise ValueError("iterations должен быть >= 1.")

        cols = [user_col, metric_col] + ([covariate_col] if covariate_col else [])
        unit_df = df[cols].drop_duplicates(subset=[user_col]).reset_index(drop=True)
        n = len(unit_df)
        if n < 10:
            raise ValueError("Недостаточно уникальных единиц рандомизации для A/A симуляции (минимум 10).")

        metric_vals = unit_df[metric_col].to_numpy(dtype=float)
        covariate_vals = unit_df[covariate_col].to_numpy(dtype=float) if covariate_col else None

        rng = np.random.default_rng(seed)
        p_values = []
        cuped_applied_count = 0

        for _ in range(iterations):
            is_treatment = rng.random(n) < 0.5

            y = metric_vals
            if covariate_vals is not None:
                tmp = pd.DataFrame({metric_col: metric_vals, covariate_col: covariate_vals})
                tmp, cuped_diag = VarianceReductionEngine.apply_cuped(
                    tmp,
                    metric_col=metric_col,
                    covariate_col=covariate_col,
                    min_r2_threshold=cuped_min_r2,
                    winsorize_quantile=None,
                )
                if cuped_diag["applied"]:
                    y = tmp[f"{metric_col}_cuped"].to_numpy(dtype=float)
                    cuped_applied_count += 1

            treatment_vals = y[is_treatment]
            control_vals = y[~is_treatment]
            if len(treatment_vals) < 2 or len(control_vals) < 2:
                continue
            res = StatCriterionRouter.welchs_t_test(control_vals, treatment_vals, alpha=alpha)
            if not np.isnan(res["p_value"]):
                p_values.append(res["p_value"])

        p_values_arr = np.asarray(p_values, dtype=float)
        n_runs = int(len(p_values_arr))
        if n_runs == 0:
            raise ValueError("Ни одна итерация A/A не дала валидного p-value; проверьте входные данные.")

        false_positive_rate = float(np.mean(p_values_arr < alpha))

        margin = 1.96 * math.sqrt(alpha * (1.0 - alpha) / n_runs)
        fpr_ci_lower = alpha - margin
        fpr_ci_upper = alpha + margin
        is_fpr_in_expected_range = bool(fpr_ci_lower <= false_positive_rate <= fpr_ci_upper)

        ks_stat, ks_pvalue = stats.kstest(p_values_arr, "uniform")
        is_uniform_distribution = bool(ks_pvalue > 0.01)

        return {
            "iterations_requested": iterations,
            "iterations_run": n_runs,
            "false_positive_rate": round(false_positive_rate, 4),
            "expected_alpha": alpha,
            "fpr_expected_ci_lower": round(fpr_ci_lower, 5),
            "fpr_expected_ci_upper": round(fpr_ci_upper, 5),
            "fpr_deviation_abs": round(abs(false_positive_rate - alpha), 4),
            "is_fpr_within_expected_range": is_fpr_in_expected_range,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "is_uniform_distribution": is_uniform_distribution,
            "cuped_applied_share": round(cuped_applied_count / iterations, 4) if covariate_col else None,
            "is_valid": bool(is_fpr_in_expected_range and is_uniform_distribution),
            "is_fpr_acceptable": is_fpr_in_expected_range,
        }

    @staticmethod
    def check_covariate_balance_smd(
        df: pd.DataFrame,
        covariate_cols: Sequence[str],
        variant_col: str = "variant",
        smd_threshold: float = 0.1,
    ) -> dict:
        """
        Проверка баланса ковариат пре-периода между вариантами по стандартизированной
        разности средних (SMD), попарно по всем парам вариантов. Безопасный дизайн
        считается сбалансированным при |SMD| < smd_threshold (по умолчанию 0.1).
        """
        for col in list(covariate_cols) + [variant_col]:
            if col not in df.columns:
                raise ValueError(f"Колонка '{col}' отсутствует в датасете для проверки баланса.")
        if len(covariate_cols) == 0:
            raise ValueError("covariate_cols не может быть пустым.")

        variants = sorted(df[variant_col].dropna().unique().tolist(), key=str)
        if len(variants) < 2:
            raise ValueError("Для проверки баланса SMD нужно минимум 2 варианта.")

        smd_by_covariate: dict = {}
        max_abs_smd = 0.0

        for cov in covariate_cols:
            pair_smd: dict = {}
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    v1, v2 = variants[i], variants[j]
                    x1 = df.loc[df[variant_col] == v1, cov].to_numpy(dtype=float)
                    x2 = df.loc[df[variant_col] == v2, cov].to_numpy(dtype=float)
                    x1 = x1[~np.isnan(x1)]
                    x2 = x2[~np.isnan(x2)]
                    if len(x1) < 2 or len(x2) < 2:
                        continue
                    mean1, mean2 = float(np.mean(x1)), float(np.mean(x2))
                    var1, var2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))
                    pooled_sd = math.sqrt((var1 + var2) / 2.0)
                    smd = (mean1 - mean2) / pooled_sd if pooled_sd > 1e-12 else 0.0
                    pair_smd[f"{v1}_vs_{v2}"] = round(float(smd), 4)
                    max_abs_smd = max(max_abs_smd, abs(smd))
            smd_by_covariate[cov] = pair_smd

        return {
            "smd_by_covariate": smd_by_covariate,
            "max_abs_smd": round(max_abs_smd, 4),
            "threshold": smd_threshold,
            "is_balanced": bool(max_abs_smd < smd_threshold),
        }


HistoricalAAValidator = AAValidationEngine
