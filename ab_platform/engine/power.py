import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
import numpy as np
from scipy import stats


@dataclass
class SampleSizeResult:
    n_per_variant: int
    total_sample_size: int
    estimated_days: int
    mde_relative: float
    mde_absolute: float
    effective_variance: float
    variance_reduction_pct: float
    is_feasible_under_5_weeks: bool
    formula_used: str
    recommendation: str
    tradeoff_table: List[Dict[str, float]]


@dataclass
class ConfidenceIntervalAuditResult:
    sample_size: int
    proportion: float
    confidence_level: float
    margin_of_error_abs: float
    margin_of_error_pct_points: float
    ci_lower: float
    ci_upper: float
    fpc_factor_applied: float
    achievable_ab_mde_rel_if_split_50_50: float


@dataclass
class SwitchbackPowerResult:
    required_total_cells: int
    required_time_windows_per_cluster: int
    mde_relative: float
    mde_absolute: float
    macro_inflation_factor_lambda: float
    recommend_cluster_robust_cr2: bool
    warning_message: Optional[str]


class PowerAndSampleSizeEngine:
    """
    Математическое ядро расчета выборки, MDE, доверительных интервалов и длительности:
    - Поддерживает точную формулу Эвана Миллера (evanmiller.org) с раздельной дисперсией H0 и H1
    - Поддерживает Pooled-формулу и классическую формулу для непрерывных/Ratio метрик с CUPED
    - Поддерживает расчет предельной ошибки (Margin of Error) и FPC (calculator.net)
    """

    @staticmethod
    def compute_ratio_delta_variance(
        y_values: np.ndarray,
        x_values: np.ndarray,
    ) -> float:
        """Вычисляет эффективную поюзерную дисперсию Ratio-метрики R = sum(Y)/sum(X) через Дельта-метод."""
        if len(y_values) != len(x_values):
            raise ValueError("Длины массивов числителя Y и знаменателя X должны совпадать.")

        mean_x = float(np.mean(x_values))
        if mean_x <= 0:
            raise ValueError("Среднее значение знаменателя X должно быть строго положительным.")

        mean_y = float(np.mean(y_values))
        ratio_est = mean_y / mean_x

        var_y = float(np.var(y_values, ddof=1))
        var_x = float(np.var(x_values, ddof=1))
        cov_yx = float(np.cov(y_values, x_values, ddof=1)[0, 1])

        delta_var = (var_y - 2.0 * ratio_est * cov_yx + (ratio_est ** 2) * var_x) / (mean_x ** 2)
        return max(delta_var, 1e-12)

    @classmethod
    def calculate_sample_size(
        cls,
        baseline_mean: float,
        mde_relative: Optional[float] = None,
        mde_absolute: Optional[float] = None,
        metric_variance: Optional[float] = None,
        is_binary: bool = False,
        binary_method: Literal["evan_miller", "pooled", "baseline"] = "evan_miller",
        alpha: float = 0.05,
        power: float = 0.80,
        daily_traffic: int = 10000,
        num_variants: int = 2,
        treatment_allocation_ratio: float = 0.5,
        pre_post_correlation_r: float = 0.0,
        finite_population_n: Optional[int] = None,
    ) -> SampleSizeResult:
        """
        Универсальный расчет выборки и длительности теста:
        - При is_binary=True и binary_method='evan_miller' считает 1-в-1 с evanmiller.org
        - Поддерживает как относительный (mde_relative), так и абсолютный (mde_absolute) эффект
        - Поддерживает поправку на конечную генеральную совокупность (finite_population_n из calculator.net)
        """
        if mde_relative is None and mde_absolute is None:
            raise ValueError("Необходимо передать либо mde_relative, либо mde_absolute.")

        if mde_absolute is not None:
            if mde_absolute <= 0:
                raise ValueError("mde_absolute должен быть строго положительным.")
            mde_abs = float(mde_absolute)
            mde_rel = mde_abs / baseline_mean
        else:
            if mde_relative is None or mde_relative <= 0:
                raise ValueError("mde_relative должен быть строго положительным.")
            mde_rel = float(mde_relative)
            mde_abs = baseline_mean * mde_rel

        if not (0.0 <= abs(pre_post_correlation_r) < 1.0):
            raise ValueError("Корреляция ковариаты pre_post_correlation_r должна быть в пределах [0, 1).")

        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_beta = float(stats.norm.ppf(power))
        cuped_factor = 1.0 - (pre_post_correlation_r ** 2)
        variance_reduction_pct = (pre_post_correlation_r ** 2) * 100.0

        p_t = treatment_allocation_ratio
        p_c = 1.0 - p_t if num_variants == 2 else (1.0 - p_t) / (num_variants - 1)
        allocation_multiplier = (1.0 / p_t) + (1.0 / p_c)

        if is_binary:
            if not (0.0 < baseline_mean < 1.0):
                raise ValueError("Для бинарной метрики baseline_mean должен быть в интервале (0, 1).")
            p1 = baseline_mean
            raw_variance = p1 * (1.0 - p1)
            effective_variance = raw_variance * cuped_factor

            if binary_method == "evan_miller":
                # Точный алгоритм Эвана Миллера: консервативный выбор p2 ближе к 0.5 и раздельные sd1, sd2
                p2 = (p1 + mde_abs) if p1 < 0.5 else (p1 - mde_abs)
                p2 = min(max(p2, 1e-6), 1.0 - 1e-6)
                sd1 = math.sqrt(2.0 * p1 * (1.0 - p1) * cuped_factor)
                sd2 = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) * cuped_factor)
                n_equal_per_var = ((z_alpha * sd1 + z_beta * sd2) ** 2) / (mde_abs ** 2)
                # Масштабируем при неравном сплите (при 50/50 allocation_multiplier / 4.0 == 1.0)
                total_n_float = n_equal_per_var * num_variants * (allocation_multiplier / 4.0)
                formula_used = "evan_miller_exact_two_proportion"
            elif binary_method == "pooled":
                p2 = min(max(p1 + mde_abs, 1e-6), 1.0 - 1e-6)
                p_pooled = (p1 + p2) / 2.0
                pooled_var = p_pooled * (1.0 - p_pooled) * cuped_factor
                total_n_float = ((z_alpha + z_beta) ** 2) * pooled_var * allocation_multiplier / (mde_abs ** 2)
                formula_used = "pooled_two_proportion"
            else:
                total_n_float = ((z_alpha + z_beta) ** 2) * effective_variance * allocation_multiplier / (mde_abs ** 2)
                formula_used = "baseline_variance_z_test"
        else:
            if metric_variance is None or metric_variance <= 0:
                raise ValueError("Для непрерывной или Ratio-метрики необходимо передать metric_variance > 0.")
            raw_variance = metric_variance
            effective_variance = raw_variance * cuped_factor
            total_n_float = ((z_alpha + z_beta) ** 2) * effective_variance * allocation_multiplier / (mde_abs ** 2)
            formula_used = "continuous_welch_or_delta"

        # Поправка на конечную генеральную совокупность (FPC из calculator.net), если база ограничена
        if finite_population_n is not None and finite_population_n > 1:
            total_n_float = (total_n_float * finite_population_n) / (total_n_float + finite_population_n - 1.0)
            formula_used += "_with_fpc"

        n_per_variant = int(math.ceil(total_n_float / num_variants))
        total_sample_size = n_per_variant * num_variants

        raw_days = math.ceil(total_sample_size / max(daily_traffic, 1))
        estimated_days = max(7, int(raw_days))
        is_feasible = estimated_days <= 35

        tradeoff_table = []
        for candidate_days in (7, 14, 21, 28, 35):
            available_n = candidate_days * daily_traffic
            achievable_mde_abs = (z_alpha + z_beta) * math.sqrt(
                (effective_variance * allocation_multiplier) / max(available_n, 2)
            )
            achievable_mde_rel = achievable_mde_abs / baseline_mean
            tradeoff_table.append({
                "duration_days": candidate_days,
                "total_users": available_n,
                "min_detectable_effect_rel_pct": round(achievable_mde_rel * 100.0, 2),
                "min_detectable_effect_abs": round(achievable_mde_abs, 5),
            })

        if is_feasible:
            recommendation = (
                f"Дизайн корректен ({formula_used}): тест наберет мощность {int(power*100)}% за {estimated_days} дн. "
                f"({n_per_variant:,} юзеров на группу, всего {total_sample_size:,}, сжатие CUPED: {variance_reduction_pct:.1f}%)."
            )
        else:
            mde_14d = tradeoff_table[1]["min_detectable_effect_rel_pct"]
            recommendation = (
                f"ВНИМАНИЕ: При трафике {daily_traffic:,}/день для MDE={mde_rel*100:.1f}% нужно {estimated_days} дн. (>5 недель). "
                f"Рекомендация ИИ: 1) Включите CUPED; 2) За 14 дней можно поймать эффект от {mde_14d}%; 3) Используйте бандитов."
            )

        return SampleSizeResult(
            n_per_variant=n_per_variant,
            total_sample_size=total_sample_size,
            estimated_days=estimated_days,
            mde_relative=mde_rel,
            mde_absolute=mde_abs,
            effective_variance=effective_variance,
            variance_reduction_pct=round(variance_reduction_pct, 2),
            is_feasible_under_5_weeks=is_feasible,
            formula_used=formula_used,
            recommendation=recommendation,
            tradeoff_table=tradeoff_table,
        )

    @staticmethod
    def calculate_survey_margin_of_error(
        sample_size: int,
        proportion: float = 0.5,
        confidence_level: float = 0.95,
        finite_population_n: Optional[int] = None,
        power_for_ab: float = 0.80,
    ) -> ConfidenceIntervalAuditResult:
        """
        Точная реализация калькулятора Calculator.net (режим Find Confidence Interval / Margin of Error):
        1) Считает предельную ошибку (MoE) одной выборки с учетом Finite Population Correction (FPC)
        2) Дополнительно показывает, какой относительный MDE можно прокрасить в A/B-тесте, если разделить эту выборку 50/50
        """
        if sample_size <= 1:
            raise ValueError("Размер выборки sample_size должен быть > 1.")
        if not (0.0 < proportion < 1.0):
            raise ValueError("Доля proportion должна лежать в интервале (0, 1).")
        if not (0.50 <= confidence_level < 1.0):
            raise ValueError("Уровень доверия confidence_level должен быть в интервале [0.5, 1.0).")

        alpha = 1.0 - confidence_level
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

        fpc_factor = 1.0
        if finite_population_n is not None and finite_population_n > sample_size:
            fpc_factor = math.sqrt((finite_population_n - sample_size) / (finite_population_n - 1.0))

        se = math.sqrt(proportion * (1.0 - proportion) / sample_size) * fpc_factor
        moe_abs = z_crit * se

        # Какой относительный MDE можно засечь в честном A/B-тесте, если эти sample_size юзеров разбить 50/50
        z_beta = float(stats.norm.ppf(power_for_ab))
        ab_mde_abs = (z_crit + z_beta) * math.sqrt(4.0 * proportion * (1.0 - proportion) / sample_size) * fpc_factor
        ab_mde_rel = ab_mde_abs / proportion

        return ConfidenceIntervalAuditResult(
            sample_size=sample_size,
            proportion=proportion,
            confidence_level=confidence_level,
            margin_of_error_abs=round(moe_abs, 6),
            margin_of_error_pct_points=round(moe_abs * 100.0, 2),
            ci_lower=round(max(0.0, proportion - moe_abs), 6),
            ci_upper=round(min(1.0, proportion + moe_abs), 6),
            fpc_factor_applied=round(fpc_factor, 6),
            achievable_ab_mde_rel_if_split_50_50=round(ab_mde_rel, 4),
        )

    @staticmethod
    def calculate_switchback_mde(
        num_clusters_j: int,
        num_time_windows_h: int,
        mean_cell_size_n: float,
        cell_size_cv: float,
        baseline_mean: float,
        total_variance: float,
        within_cell_variance_share: float = 0.85,
        macro_cell_variance_share: float = 0.15,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> SwitchbackPowerResult:
        """Расчет MDE для Switchback и кластерных тестов с учетом макро-дисперсии и CV размеров ячеек."""
        total_cells_b = num_clusters_j * num_time_windows_h
        if total_cells_b < 4:
            raise ValueError("Общее число ячеек (J кластеров * H окон времени) должно быть >= 4.")

        cv_sq = cell_size_cv ** 2
        macro_lambda = mean_cell_size_n * (1.0 + cv_sq)

        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_beta = float(stats.norm.ppf(power))

        effective_cell_var_factor = (
            (within_cell_variance_share / mean_cell_size_n)
            + macro_cell_variance_share * ((1.0 / mean_cell_size_n) + 1.0 + cv_sq)
        )

        mde_abs = (z_alpha + z_beta) * math.sqrt(
            (4.0 * total_variance / total_cells_b) * effective_cell_var_factor
        )
        mde_rel = mde_abs / baseline_mean

        recommend_cr2 = (total_cells_b <= 10) or (cell_size_cv >= 2.0)
        warning_msg = None
        if recommend_cr2:
            warning_msg = (
                f"Высокий дисбаланс размеров ячеек (CV={cell_size_cv:.2f}) или мало ячеек (B={total_cells_b}). "
                "Рекомендуется кластерно-робастный оцениватель CR2."
            )

        return SwitchbackPowerResult(
            required_total_cells=total_cells_b,
            required_time_windows_per_cluster=num_time_windows_h,
            mde_relative=round(mde_rel, 5),
            mde_absolute=round(mde_abs, 5),
            macro_inflation_factor_lambda=round(macro_lambda, 2),
            recommend_cluster_robust_cr2=recommend_cr2,
            warning_message=warning_msg,
        )
