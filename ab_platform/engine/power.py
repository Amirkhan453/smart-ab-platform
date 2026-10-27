import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
import numpy as np
from scipy import stats
from ab_platform.schemas.passport import Direction


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
    sidedness: str
    num_treatments: int
    alpha_adjusted: float
    design_effect: float
    traffic_exposure_share: float


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
    - Строгая валидация всех входных параметров (baseline, traffic, alpha, power, allocation)
    - Точная формула Эвана Миллера с учетом направления изменения (Direction.INCREASE / DECREASE)
    - Поправка на сторонность теста (sidedness) и число тестовых вариантов (Bonferroni по FWER)
    - design effect для кластерных/switchback-дизайнов (через ICC и средний размер кластера)
    - Учет фактической доли трафика эксперимента (слот слоя, ramp-up) через traffic_exposure_share

    При параметрах по умолчанию (sidedness="two_sided", num_variants=2, design_effect=1.0,
    traffic_exposure_share=1.0) численные результаты (n_per_variant, total_sample_size)
    совпадают с результатами версии без этих параметров.
    """

    @staticmethod
    def compute_ratio_delta_variance(
        y_values: np.ndarray,
        x_values: np.ndarray,
    ) -> float:
        y = np.asarray(y_values, dtype=float)
        x = np.asarray(x_values, dtype=float)
        if len(y) < 2 or len(x) < 2:
            raise ValueError("Для расчета Дельта-дисперсии необходимо минимум 2 наблюдения.")
        if len(y) != len(x):
            raise ValueError("Длины массивов числителя Y и знаменателя X должны совпадать.")

        mean_x = float(np.mean(x))
        if mean_x <= 0:
            raise ValueError("Среднее значение знаменателя X должно быть строго положительным.")

        mean_y = float(np.mean(y))
        ratio_est = mean_y / mean_x

        var_y = float(np.var(y, ddof=1))
        var_x = float(np.var(x, ddof=1))
        cov_yx = float(np.cov(y, x, ddof=1)[0, 1])

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
        expected_direction: Direction = Direction.INCREASE,
        alpha: float = 0.05,
        power: float = 0.80,
        sidedness: Literal["two_sided", "one_sided"] = "two_sided",
        daily_traffic: int = 10000,
        traffic_exposure_share: float = 1.0,
        num_variants: int = 2,
        treatment_allocation_ratio: float = 0.5,
        pre_post_correlation_r: float = 0.0,
        design_effect: float = 1.0,
        icc: Optional[float] = None,
        mean_cluster_size: Optional[float] = None,
        finite_population_n: Optional[int] = None,
    ) -> SampleSizeResult:
        if baseline_mean <= 0.0:
            raise ValueError(f"baseline_mean должен быть строго положительным (> 0), получено: {baseline_mean}")
        if daily_traffic <= 0:
            raise ValueError(f"daily_traffic должен быть > 0, получено: {daily_traffic}")
        if num_variants < 2:
            raise ValueError(f"num_variants должен быть >= 2, получено: {num_variants}")
        if not (0.0 < treatment_allocation_ratio < 1.0):
            raise ValueError("treatment_allocation_ratio должен быть строго в интервале (0, 1).")
        if not (0.0 < alpha < 0.5):
            raise ValueError("alpha должен быть в интервале (0, 0.5).")
        if not (0.5 <= power < 1.0):
            raise ValueError("power должен быть в интервале [0.5, 1.0).")
        if not (0.0 < traffic_exposure_share <= 1.0):
            raise ValueError(f"traffic_exposure_share должен быть в (0, 1], получено: {traffic_exposure_share}")
        if sidedness not in ("two_sided", "one_sided"):
            raise ValueError("sidedness должен быть 'two_sided' или 'one_sided'.")
        if design_effect <= 0.0:
            raise ValueError("design_effect должен быть строго положительным.")
        if icc is not None or mean_cluster_size is not None:
            if icc is None or mean_cluster_size is None:
                raise ValueError("icc и mean_cluster_size должны передаваться вместе.")
            if not (0.0 <= icc < 1.0):
                raise ValueError("icc должен быть в интервале [0, 1).")
            if mean_cluster_size < 1.0:
                raise ValueError("mean_cluster_size должен быть >= 1.")
            design_effect = 1.0 + (mean_cluster_size - 1.0) * icc

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

        num_treatments = max(num_variants - 1, 1)
        alpha_adjusted = alpha / num_treatments
        if sidedness == "two_sided":
            z_alpha = float(stats.norm.ppf(1.0 - alpha_adjusted / 2.0))
        else:
            z_alpha = float(stats.norm.ppf(1.0 - alpha_adjusted))
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
            p2 = (p1 + mde_abs) if expected_direction == Direction.INCREASE else (p1 - mde_abs)
            if not (0.0 < p2 < 1.0):
                raise ValueError(
                    f"Целевая конверсия p2={p2:.4f} выходит за пределы (0, 1) при baseline={p1} и mde_abs={mde_abs}."
                )
            raw_variance = p1 * (1.0 - p1)
            effective_variance = raw_variance * cuped_factor * design_effect

            if binary_method == "evan_miller":
                sd1 = math.sqrt(2.0 * p1 * (1.0 - p1) * cuped_factor * design_effect)
                sd2 = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) * cuped_factor * design_effect)
                n_equal_per_var = ((z_alpha * sd1 + z_beta * sd2) ** 2) / (mde_abs ** 2)
                total_n_float = n_equal_per_var * num_variants * (allocation_multiplier / 4.0)
                formula_used = "evan_miller_exact_two_proportion"
            elif binary_method == "pooled":
                p_pooled = (p1 + p2) / 2.0
                pooled_var = p_pooled * (1.0 - p_pooled) * cuped_factor * design_effect
                total_n_float = ((z_alpha + z_beta) ** 2) * pooled_var * allocation_multiplier / (mde_abs ** 2)
                formula_used = "pooled_two_proportion"
            else:
                total_n_float = ((z_alpha + z_beta) ** 2) * effective_variance * allocation_multiplier / (mde_abs ** 2)
                formula_used = "baseline_variance_z_test"
        else:
            if metric_variance is None or metric_variance <= 0:
                raise ValueError("Для непрерывной или Ratio-метрики необходимо передать metric_variance > 0.")
            raw_variance = metric_variance
            effective_variance = raw_variance * cuped_factor * design_effect
            total_n_float = ((z_alpha + z_beta) ** 2) * effective_variance * allocation_multiplier / (mde_abs ** 2)
            formula_used = "continuous_welch_or_delta"

        if finite_population_n is not None:
            if finite_population_n <= 1:
                raise ValueError("finite_population_n должен быть > 1.")
            total_n_float = (total_n_float * finite_population_n) / (total_n_float + finite_population_n - 1.0)
            formula_used += "_with_fpc"

        n_per_variant = int(math.ceil(total_n_float / num_variants))
        total_sample_size = n_per_variant * num_variants

        effective_daily_traffic = daily_traffic * traffic_exposure_share
        raw_days = math.ceil(total_sample_size / effective_daily_traffic)
        estimated_days = max(7, int(math.ceil(raw_days / 7.0) * 7))
        is_feasible = estimated_days <= 35

        tradeoff_table = []
        for candidate_days in (7, 14, 21, 28, 35):
            available_n = candidate_days * effective_daily_traffic
            achievable_mde_abs = (z_alpha + z_beta) * math.sqrt(
                (effective_variance * allocation_multiplier) / max(available_n, 2)
            )
            achievable_mde_rel = achievable_mde_abs / baseline_mean
            tradeoff_table.append({
                "duration_days": candidate_days,
                "total_users": round(available_n, 1),
                "min_detectable_effect_rel_pct": round(achievable_mde_rel * 100.0, 2),
                "min_detectable_effect_abs": round(achievable_mde_abs, 5),
            })

        if is_feasible:
            recommendation = (
                f"Дизайн корректен ({formula_used}): тест наберет мощность {int(power*100)}% за {estimated_days} дн. "
                f"({n_per_variant:,} юзеров на группу, всего {total_sample_size:,}, сжатие CUPED: {variance_reduction_pct:.1f}%, "
                f"design_effect: {design_effect:.2f}, traffic_exposure_share: {traffic_exposure_share:.2f})."
            )
        else:
            mde_14d = tradeoff_table[1]["min_detectable_effect_rel_pct"]
            recommendation = (
                f"ВНИМАНИЕ: При эффективном трафике {effective_daily_traffic:,.0f}/день для MDE={mde_rel*100:.1f}% "
                f"нужно {estimated_days} дн. (>5 недель, округлено до кратного 7). Варианты: "
                f"1) Включите CUPED, если ковариата доступна (снижает дисперсию и объём выборки); "
                f"2) За 14 дней можно поймать эффект от {mde_14d}%; "
                f"3) Увеличьте traffic_exposure_share (долю слота слоя/ramp-up) или дневной трафик; "
                f"4) Используйте стратификацию по ключевым ковариатам для снижения остаточной дисперсии."
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
            sidedness=sidedness,
            num_treatments=num_treatments,
            alpha_adjusted=alpha_adjusted,
            design_effect=round(design_effect, 4),
            traffic_exposure_share=traffic_exposure_share,
        )

    @staticmethod
    def calculate_survey_margin_of_error(
        sample_size: int,
        proportion: float = 0.5,
        confidence_level: float = 0.95,
        finite_population_n: Optional[int] = None,
        power_for_ab: float = 0.80,
    ) -> ConfidenceIntervalAuditResult:
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
        if baseline_mean <= 0 or total_variance <= 0 or mean_cell_size_n <= 0 or cell_size_cv < 0:
            raise ValueError("Параметры Switchback должны быть положительными (cell_size_cv >= 0).")
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
