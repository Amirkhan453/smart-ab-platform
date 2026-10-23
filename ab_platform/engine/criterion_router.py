import math
from typing import List
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from ab_platform.schemas.passport import MetricType


class StatCriterionRouter:
    """
    Детерминированный маршрутизатор статистических критериев:
    - BINARY -> Двухвыборочный Z-тест для пропорций (pooled SE для Z, unpooled SE для ДИ)
    - CONTINUOUS -> Двухвыборочный t-тест Уэлча (Welch's t-test)
    - RATIO -> Дельта-метод (разложение Тейлора 1-го порядка для отношения метрик)
    - QUANTILE -> Непараметрический бутстрап
    - Пакетная коррекция p-values -> Метод Бенджамини-Хохберга (FDR)
    """

    @staticmethod
    def welchs_t_test(
        control_vals: np.ndarray, treatment_vals: np.ndarray, alpha: float = 0.05
    ) -> dict:
        """Двухвыборочный t-тест Уэлча с поправкой на неравенство дисперсий и 95% ДИ."""
        c = np.asarray(control_vals, dtype=float)
        t = np.asarray(treatment_vals, dtype=float)

        t_stat, p_val = stats.ttest_ind(t, c, equal_var=False)
        mean_c, mean_t = float(np.mean(c)), float(np.mean(t))
        diff = mean_t - mean_c

        se_c = float(np.var(c, ddof=1)) / max(len(c), 1)
        se_t = float(np.var(t, ddof=1)) / max(len(t), 1)
        se_diff = math.sqrt(max(se_c + se_t, 1e-15))
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

        return {
            "criterion": "welchs_t_test",
            "p_value": float(p_val),
            "statistic": float(t_stat),
            "mean_control": mean_c,
            "mean_treatment": mean_t,
            "absolute_diff": diff,
            "relative_diff": (diff / mean_c) if mean_c != 0 else 0.0,
            "ci_lower": diff - z_crit * se_diff,
            "ci_upper": diff + z_crit * se_diff,
            "is_significant": bool(p_val < alpha),
        }

    @staticmethod
    def proportions_z_test(
        control_binary: np.ndarray, treatment_binary: np.ndarray, alpha: float = 0.05
    ) -> dict:
        """Z-тест для двух пропорций (конверсий) с пулированной стандартной ошибкой."""
        c = np.asarray(control_binary, dtype=float)
        t = np.asarray(treatment_binary, dtype=float)

        n_c, n_t = len(c), len(t)
        conv_c, conv_t = float(np.sum(c)), float(np.sum(t))
        cr_c, cr_t = conv_c / max(n_c, 1), conv_t / max(n_t, 1)
        diff = cr_t - cr_c

        pooled_p = (conv_c + conv_t) / max(n_c + n_t, 1)
        pooled_se = math.sqrt(max(pooled_p * (1.0 - pooled_p) * (1.0 / n_c + 1.0 / n_t), 1e-15))
        z_score = diff / pooled_se if pooled_se > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))

        # Непулированная стандартная ошибка для доверительного интервала
        se_unpooled = math.sqrt(
            max((cr_c * (1.0 - cr_c) / n_c) + (cr_t * (1.0 - cr_t) / n_t), 1e-15)
        )
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

        return {
            "criterion": "proportions_z_test",
            "p_value": p_val,
            "statistic": float(z_score),
            "mean_control": cr_c,
            "mean_treatment": cr_t,
            "absolute_diff": diff,
            "relative_diff": (diff / cr_c) if cr_c != 0 else 0.0,
            "ci_lower": diff - z_crit * se_unpooled,
            "ci_upper": diff + z_crit * se_unpooled,
            "is_significant": bool(p_val < alpha),
        }

    @staticmethod
    def ratio_delta_test(
        num_c: np.ndarray,
        den_c: np.ndarray,
        num_t: np.ndarray,
        den_t: np.ndarray,
        alpha: float = 0.05,
    ) -> dict:
        """
        Дельта-метод для сравнения двух Ratio-метрик R = sum(Y) / sum(X).
        Корректно учитывает внутрипользовательскую корреляцию при повторных визитах/сессиях.
        """
        def _ratio_stats(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
            n = len(y)
            mean_y, mean_x = float(np.mean(y)), float(np.mean(x))
            r = mean_y / mean_x
            var_y = float(np.var(y, ddof=1))
            var_x = float(np.var(x, ddof=1))
            cov_yx = float(np.cov(y, x, ddof=1)[0, 1])
            var_r = (var_y - 2.0 * r * cov_yx + (r ** 2) * var_x) / (n * (mean_x ** 2))
            return r, max(var_r, 1e-15)

        r_c, var_r_c = _ratio_stats(np.asarray(num_c, dtype=float), np.asarray(den_c, dtype=float))
        r_t, var_r_t = _ratio_stats(np.asarray(num_t, dtype=float), np.asarray(den_t, dtype=float))

        diff = r_t - r_c
        se_diff = math.sqrt(var_r_c + var_r_t)
        z_stat = diff / se_diff
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

        return {
            "criterion": "ratio_delta_method",
            "p_value": p_val,
            "statistic": float(z_stat),
            "mean_control": r_c,
            "mean_treatment": r_t,
            "absolute_diff": diff,
            "relative_diff": (diff / r_c) if r_c != 0 else 0.0,
            "ci_lower": diff - z_crit * se_diff,
            "ci_upper": diff + z_crit * se_diff,
            "is_significant": bool(p_val < alpha),
        }

    @staticmethod
    def bootstrap_quantile_test(
        control_vals: np.ndarray,
        treatment_vals: np.ndarray,
        quantile: float = 0.90,
        n_boot: int = 1000,
        alpha: float = 0.05,
        random_seed: int = 42,
    ) -> dict:
        """Непараметрический бутстрап для оценки разницы квантилей (медиана, p90, p95)."""
        rng = np.random.default_rng(random_seed)
        c = np.asarray(control_vals, dtype=float)
        t = np.asarray(treatment_vals, dtype=float)

        q_c = float(np.quantile(c, quantile))
        q_t = float(np.quantile(t, quantile))
        obs_diff = q_t - q_c

        boot_diffs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sample_c = rng.choice(c, size=len(c), replace=True)
            sample_t = rng.choice(t, size=len(t), replace=True)
            boot_diffs[i] = np.quantile(sample_t, quantile) - np.quantile(sample_c, quantile)

        ci_lower = float(np.quantile(boot_diffs, alpha / 2.0))
        ci_upper = float(np.quantile(boot_diffs, 1.0 - alpha / 2.0))
        p_left = float(np.mean(boot_diffs <= 0.0))
        p_right = float(np.mean(boot_diffs >= 0.0))
        p_val = min(1.0, 2.0 * min(p_left, p_right))

        return {
            "criterion": f"bootstrap_q{int(quantile*100)}",
            "p_value": p_val,
            "statistic": obs_diff,
            "mean_control": q_c,
            "mean_treatment": q_t,
            "absolute_diff": obs_diff,
            "relative_diff": (obs_diff / q_c) if q_c != 0 else 0.0,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "is_significant": bool(ci_lower > 0.0 or ci_upper < 0.0),
        }

    @staticmethod
    def apply_fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """Контроль False Discovery Rate (поправка Бенджамини-Хохберга) при множественных проверках."""
        if not p_values:
            return []
        reject, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        return reject.tolist()
