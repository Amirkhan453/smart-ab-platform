import math
from typing import List
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


class StatCriterionRouter:
    """
    Защищенный маршрутизатор статистических критериев:
    - Валидация минимального размера выборки (n >= 2 в каждой группе) и отсутствия NaN
    - Обработка вырожденных случаев с нулевой дисперсией (без возврата NaN)
    - Поправка Бенджамини-Хохберга (FDR) для вторичных метрик и Холма-Бонферрони (FWER) для Multi-Arm
    """

    @staticmethod
    def _validate_two_samples(c: np.ndarray, t: np.ndarray, label: str) -> None:
        if c.size < 2 or t.size < 2:
            raise ValueError(
                f"Недостаточно данных для критерия '{label}': в каждой группе должно быть минимум 2 наблюдения "
                f"(получено control={c.size}, treatment={t.size})."
            )
        if np.isnan(c).any() or np.isnan(t).any():
            raise ValueError(f"Входные массивы для '{label}' содержат значения NaN.")

    @classmethod
    def welchs_t_test(
        cls, control_vals: np.ndarray, treatment_vals: np.ndarray, alpha: float = 0.05
    ) -> dict:
        c = np.asarray(control_vals, dtype=float)
        t = np.asarray(treatment_vals, dtype=float)
        cls._validate_two_samples(c, t, "welchs_t_test")

        mean_c, mean_t = float(np.mean(c)), float(np.mean(t))
        diff = mean_t - mean_c
        var_c = float(np.var(c, ddof=1))
        var_t = float(np.var(t, ddof=1))

        if var_c < 1e-15 and var_t < 1e-15:
            p_val = 1.0 if abs(diff) < 1e-12 else 0.0
            t_stat = 0.0
            se_diff = 0.0
        else:
            t_stat_raw, p_val_raw = stats.ttest_ind(t, c, equal_var=False)
            t_stat = 0.0 if np.isnan(t_stat_raw) else float(t_stat_raw)
            p_val = 1.0 if np.isnan(p_val_raw) else float(p_val_raw)
            se_diff = math.sqrt((var_c / len(c)) + (var_t / len(t)))

        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        return {
            "criterion": "welchs_t_test",
            "p_value": p_val,
            "statistic": t_stat,
            "mean_control": mean_c,
            "mean_treatment": mean_t,
            "absolute_diff": diff,
            "relative_diff": (diff / mean_c) if abs(mean_c) > 1e-12 else 0.0,
            "ci_lower": diff - z_crit * se_diff,
            "ci_upper": diff + z_crit * se_diff,
            "is_significant": bool(p_val < alpha),
        }

    @classmethod
    def proportions_z_test(
        cls, control_binary: np.ndarray, treatment_binary: np.ndarray, alpha: float = 0.05
    ) -> dict:
        c = np.asarray(control_binary, dtype=float)
        t = np.asarray(treatment_binary, dtype=float)
        cls._validate_two_samples(c, t, "proportions_z_test")

        n_c, n_t = len(c), len(t)
        conv_c, conv_t = float(np.sum(c)), float(np.sum(t))
        cr_c, cr_t = conv_c / n_c, conv_t / n_t
        diff = cr_t - cr_c

        pooled_p = (conv_c + conv_t) / (n_c + n_t)
        if pooled_p <= 0.0 or pooled_p >= 1.0:
            z_score = 0.0
            p_val = 1.0
            se_unpooled = 0.0
        else:
            pooled_se = math.sqrt(pooled_p * (1.0 - pooled_p) * (1.0 / n_c + 1.0 / n_t))
            z_score = diff / pooled_se if pooled_se > 0 else 0.0
            p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
            se_unpooled = math.sqrt(
                max((cr_c * (1.0 - cr_c) / n_c) + (cr_t * (1.0 - cr_t) / n_t), 0.0)
            )

        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        return {
            "criterion": "proportions_z_test",
            "p_value": p_val,
            "statistic": float(z_score),
            "mean_control": cr_c,
            "mean_treatment": cr_t,
            "absolute_diff": diff,
            "relative_diff": (diff / cr_c) if abs(cr_c) > 1e-12 else 0.0,
            "ci_lower": diff - z_crit * se_unpooled,
            "ci_upper": diff + z_crit * se_unpooled,
            "is_significant": bool(p_val < alpha),
        }

    @classmethod
    def ratio_delta_test(
        cls,
        num_c: np.ndarray,
        den_c: np.ndarray,
        num_t: np.ndarray,
        den_t: np.ndarray,
        alpha: float = 0.05,
    ) -> dict:
        nc = np.asarray(num_c, dtype=float)
        dc = np.asarray(den_c, dtype=float)
        nt = np.asarray(num_t, dtype=float)
        dt = np.asarray(den_t, dtype=float)

        cls._validate_two_samples(nc, nt, "ratio_delta_test")
        if len(nc) != len(dc) or len(nt) != len(dt):
            raise ValueError("Длины числителя и знаменателя в ratio_delta_test должны совпадать.")

        def _ratio_stats(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
            n = len(y)
            mean_y, mean_x = float(np.mean(y)), float(np.mean(x))
            if mean_x <= 0.0:
                raise ValueError("Среднее значение знаменателя в Ratio-метрике должно быть строго > 0.")
            r = mean_y / mean_x
            var_y = float(np.var(y, ddof=1))
            var_x = float(np.var(x, ddof=1))
            cov_yx = float(np.cov(y, x, ddof=1)[0, 1])
            var_r = (var_y - 2.0 * r * cov_yx + (r ** 2) * var_x) / (n * (mean_x ** 2))
            return r, max(var_r, 0.0)

        r_c, var_r_c = _ratio_stats(nc, dc)
        r_t, var_r_t = _ratio_stats(nt, dt)

        diff = r_t - r_c
        se_diff = math.sqrt(var_r_c + var_r_t)
        if se_diff < 1e-15:
            z_stat = 0.0
            p_val = 1.0 if abs(diff) < 1e-12 else 0.0
        else:
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
            "relative_diff": (diff / r_c) if abs(r_c) > 1e-12 else 0.0,
            "ci_lower": diff - z_crit * se_diff,
            "ci_upper": diff + z_crit * se_diff,
            "is_significant": bool(p_val < alpha),
        }

    @classmethod
    def bootstrap_quantile_test(
        cls,
        control_vals: np.ndarray,
        treatment_vals: np.ndarray,
        quantile: float = 0.90,
        n_boot: int = 1000,
        alpha: float = 0.05,
        random_seed: int = 42,
    ) -> dict:
        c = np.asarray(control_vals, dtype=float)
        t = np.asarray(treatment_vals, dtype=float)
        cls._validate_two_samples(c, t, "bootstrap_quantile_test")
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile должен быть в интервале (0, 1).")

        rng = np.random.default_rng(random_seed)
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
            "relative_diff": (obs_diff / q_c) if abs(q_c) > 1e-12 else 0.0,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "is_significant": bool(ci_lower > 0.0 or ci_upper < 0.0),
        }

    @staticmethod
    def apply_fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        if not p_values:
            return []
        reject, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        return reject.tolist()

    @staticmethod
    def apply_holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> tuple[List[bool], List[float]]:
        if not p_values:
            return [], []
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method="holm")
        return reject.tolist(), [float(p) for p in pvals_corrected]
