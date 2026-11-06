import math
from typing import List, Literal, Optional
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


class StatCriterionRouter:
    """
    Защищенный маршрутизатор статистических критериев:
    - Валидация минимального размера выборки (n >= 2 в каждой группе) и отсутствия NaN
    - Обработка вырожденных случаев с нулевой дисперсией (без возврата NaN)
    - Точные хвостовые p-value через scipy sf(...) вместо 1 - cdf(...), для two_sided и one_sided
    - Бинарные метрики: при n*p < 10 или n*(1-p) < 10 хотя бы в одной группе — точный тест Фишера
      и доверительный интервал методом Вильсона (гибридный метод Ньюкомба для разности долей)
    - Квантили: векторизованный Poisson-bootstrap (без Python-цикла по n_boot), p-value никогда не 0
    - Кластерно-устойчивые SE, ICC и design_effect при randomization_unit != analysis_unit
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

    @staticmethod
    def _check_side(sidedness: str, direction: str) -> None:
        if sidedness not in ("two_sided", "one_sided"):
            raise ValueError(f"sidedness: 'two_sided' или 'one_sided', получено {sidedness!r}")
        if direction not in ("increase", "decrease"):
            raise ValueError(f"direction: 'increase' или 'decrease', получено {direction!r}")

    @staticmethod
    def _two_tailed_or_one_tailed_normal_p(z: float, sidedness: str, direction: str = "increase") -> float:
        """Односторонний тест проверяет ожидаемую сторону: рост (increase) или снижение (decrease)."""
        if sidedness == "one_sided":
            return float(stats.norm.sf(z if direction == "increase" else -z))
        return float(2.0 * stats.norm.sf(abs(z)))

    @staticmethod
    def _crit_quantile(alpha: float, sidedness: str) -> float:
        """Квантиль для ДИ: двусторонний (1 - alpha) или (1 - 2alpha), чья нужная граница = односторонняя."""
        return 1.0 - alpha / 2.0 if sidedness == "two_sided" else 1.0 - alpha

    @staticmethod
    def _degenerate_p(diff: float, sidedness: str, direction: str) -> float:
        """p-value при нулевой дисперсии в обеих группах."""
        if sidedness == "one_sided":
            signed = diff if direction == "increase" else -diff
            return 0.0 if signed > 1e-12 else (1.0 if signed < -1e-12 else 0.5)
        return 1.0 if abs(diff) < 1e-12 else 0.0

    @classmethod
    def welchs_t_test(
        cls,
        control_vals: np.ndarray,
        treatment_vals: np.ndarray,
        alpha: float = 0.05,
        sidedness: Literal["two_sided", "one_sided"] = "two_sided",
        direction: Literal["increase", "decrease"] = "increase",
    ) -> dict:
        cls._check_side(sidedness, direction)
        c = np.asarray(control_vals, dtype=float)
        t = np.asarray(treatment_vals, dtype=float)
        cls._validate_two_samples(c, t, "welchs_t_test")

        mean_c, mean_t = float(np.mean(c)), float(np.mean(t))
        diff = mean_t - mean_c
        var_c = float(np.var(c, ddof=1))
        var_t = float(np.var(t, ddof=1))
        n_c, n_t = len(c), len(t)

        if var_c < 1e-15 and var_t < 1e-15:
            se_diff = 0.0
            t_stat = 0.0
            p_val = cls._degenerate_p(diff, sidedness, direction)
            z_crit = float(stats.norm.ppf(cls._crit_quantile(alpha, sidedness)))
        else:
            se_diff = math.sqrt((var_c / n_c) + (var_t / n_t))
            t_stat = diff / se_diff if se_diff > 1e-15 else 0.0
            df_welch = ((var_c / n_c + var_t / n_t) ** 2) / (
                (var_c / n_c) ** 2 / max(n_c - 1, 1) + (var_t / n_t) ** 2 / max(n_t - 1, 1)
            )
            if sidedness == "one_sided":
                p_val = float(stats.t.sf(t_stat if direction == "increase" else -t_stat, df_welch))
            else:
                p_val = float(2.0 * stats.t.sf(abs(t_stat), df_welch))
            z_crit = float(stats.t.ppf(cls._crit_quantile(alpha, sidedness), df_welch))

        return {
            "criterion": "welchs_t_test",
            "sidedness": sidedness,
            "direction": direction,
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

    @staticmethod
    def _wilson_ci(k: float, n: float, conf_level: float) -> tuple[float, float]:
        if n <= 0:
            return 0.0, 0.0
        z = float(stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0))
        phat = k / n
        denom = 1.0 + (z ** 2) / n
        center = (phat + (z ** 2) / (2.0 * n)) / denom
        margin = (z * math.sqrt((phat * (1.0 - phat) / n) + (z ** 2) / (4.0 * n ** 2))) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    @classmethod
    def proportions_z_test(
        cls,
        control_binary: np.ndarray,
        treatment_binary: np.ndarray,
        alpha: float = 0.05,
        sidedness: Literal["two_sided", "one_sided"] = "two_sided",
        direction: Literal["increase", "decrease"] = "increase",
    ) -> dict:
        cls._check_side(sidedness, direction)
        c = np.asarray(control_binary, dtype=float)
        t = np.asarray(treatment_binary, dtype=float)
        cls._validate_two_samples(c, t, "proportions_z_test")

        n_c, n_t = len(c), len(t)
        conv_c, conv_t = float(np.sum(c)), float(np.sum(t))
        cr_c, cr_t = conv_c / n_c, conv_t / n_t
        diff = cr_t - cr_c

        small_n = (
            n_c * cr_c < 10 or n_c * (1.0 - cr_c) < 10
            or n_t * cr_t < 10 or n_t * (1.0 - cr_t) < 10
        )

        if small_n:
            table = [[int(round(conv_t)), int(round(n_t - conv_t))], [int(round(conv_c)), int(round(n_c - conv_c))]]
            fisher_alt = "two-sided" if sidedness == "two_sided" else ("greater" if direction == "increase" else "less")
            _, p_val = stats.fisher_exact(table, alternative=fisher_alt)
            p_val = float(p_val)

            conf_level = 1.0 - alpha if sidedness == "two_sided" else 1.0 - 2.0 * alpha
            l_c, u_c = cls._wilson_ci(conv_c, n_c, conf_level)
            l_t, u_t = cls._wilson_ci(conv_t, n_t, conf_level)
            ci_lower = diff - math.sqrt((cr_t - l_t) ** 2 + (u_c - cr_c) ** 2)
            ci_upper = diff + math.sqrt((u_t - cr_t) ** 2 + (cr_c - l_c) ** 2)

            return {
                "criterion": "fisher_exact_wilson_ci",
                "sidedness": sidedness,
                "direction": direction,
                "p_value": p_val,
                "statistic": float("nan"),
                "mean_control": cr_c,
                "mean_treatment": cr_t,
                "absolute_diff": diff,
                "relative_diff": (diff / cr_c) if abs(cr_c) > 1e-12 else 0.0,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "is_significant": bool(p_val < alpha),
            }

        pooled_p = (conv_c + conv_t) / (n_c + n_t)
        if pooled_p <= 0.0 or pooled_p >= 1.0:
            z_score = 0.0
            p_val = 1.0
            se_unpooled = 0.0
        else:
            pooled_se = math.sqrt(pooled_p * (1.0 - pooled_p) * (1.0 / n_c + 1.0 / n_t))
            z_score = diff / pooled_se if pooled_se > 0 else 0.0
            p_val = cls._two_tailed_or_one_tailed_normal_p(z_score, sidedness, direction)
            se_unpooled = math.sqrt(
                max((cr_c * (1.0 - cr_c) / n_c) + (cr_t * (1.0 - cr_t) / n_t), 0.0)
            )

        z_crit = float(stats.norm.ppf(cls._crit_quantile(alpha, sidedness)))
        return {
            "criterion": "proportions_z_test",
            "sidedness": sidedness,
            "direction": direction,
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
        sidedness: Literal["two_sided", "one_sided"] = "two_sided",
        direction: Literal["increase", "decrease"] = "increase",
    ) -> dict:
        cls._check_side(sidedness, direction)
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
            p_val = cls._degenerate_p(diff, sidedness, direction)
        else:
            z_stat = diff / se_diff
            p_val = cls._two_tailed_or_one_tailed_normal_p(z_stat, sidedness, direction)

        z_crit = float(stats.norm.ppf(cls._crit_quantile(alpha, sidedness)))
        return {
            "criterion": "ratio_delta_method",
            "sidedness": sidedness,
            "direction": direction,
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

    @staticmethod
    def _poisson_bootstrap_quantiles(values: np.ndarray, quantile: float, n_boot: int, rng: np.random.Generator) -> np.ndarray:
        n = len(values)
        order = np.argsort(values)
        v_sorted = values[order]
        weights = rng.poisson(lam=1.0, size=(n_boot, n)).astype(float)
        cumw = np.cumsum(weights, axis=1)
        totals = cumw[:, -1]
        totals_safe = np.where(totals > 0, totals, 1.0)
        target = quantile * totals_safe
        idx = (cumw < target[:, None]).sum(axis=1)
        idx = np.clip(idx, 0, n - 1)
        return v_sorted[idx]

    @classmethod
    def bootstrap_quantile_test(
        cls,
        control_vals: np.ndarray,
        treatment_vals: np.ndarray,
        quantile: float = 0.90,
        n_boot: int = 2000,
        alpha: float = 0.05,
        random_seed: int = 42,
        sidedness: Literal["two_sided", "one_sided"] = "two_sided",
        direction: Literal["increase", "decrease"] = "increase",
    ) -> dict:
        cls._check_side(sidedness, direction)
        c = np.asarray(control_vals, dtype=float)
        t = np.asarray(treatment_vals, dtype=float)
        cls._validate_two_samples(c, t, "bootstrap_quantile_test")
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile должен быть в интервале (0, 1).")
        if n_boot < 1:
            raise ValueError("n_boot должен быть >= 1.")

        rng = np.random.default_rng(random_seed)
        q_c = float(np.quantile(c, quantile))
        q_t = float(np.quantile(t, quantile))
        obs_diff = q_t - q_c

        boot_q_c = cls._poisson_bootstrap_quantiles(c, quantile, n_boot, rng)
        boot_q_t = cls._poisson_bootstrap_quantiles(t, quantile, n_boot, rng)
        boot_diffs = boot_q_t - boot_q_c

        q_hi = cls._crit_quantile(alpha, sidedness)
        ci_lower = float(np.quantile(boot_diffs, 1.0 - q_hi))
        ci_upper = float(np.quantile(boot_diffs, q_hi))

        k_left = float(np.sum(boot_diffs <= 0.0))
        k_right = float(np.sum(boot_diffs >= 0.0))
        if sidedness == "one_sided":
            k_against = k_left if direction == "increase" else k_right
            p_val = min(1.0, (k_against + 1.0) / (n_boot + 1.0))
        else:
            p_val = min(1.0, 2.0 * (min(k_left, k_right) + 1.0) / (n_boot + 1.0))

        return {
            "criterion": f"bootstrap_q{int(quantile * 100)}",
            "sidedness": sidedness,
            "direction": direction,
            "p_value": p_val,
            "statistic": obs_diff,
            "mean_control": q_c,
            "mean_treatment": q_t,
            "absolute_diff": obs_diff,
            "relative_diff": (obs_diff / q_c) if abs(q_c) > 1e-12 else 0.0,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "is_significant": bool(p_val < alpha),
        }

    @staticmethod
    def _cluster_summary(values: np.ndarray, cluster_ids: np.ndarray):
        unique_ids, inverse = np.unique(cluster_ids, return_inverse=True)
        k = len(unique_ids)
        sums = np.bincount(inverse, weights=values, minlength=k)
        counts = np.bincount(inverse, minlength=k).astype(float)
        means = sums / counts
        return unique_ids, counts, means

    @classmethod
    def cluster_robust_se_and_icc(
        cls,
        control_vals: np.ndarray,
        control_cluster_ids: np.ndarray,
        treatment_vals: np.ndarray,
        treatment_cluster_ids: np.ndarray,
        alpha: float = 0.05,
    ) -> dict:
        """
        Кластерно-устойчивая оценка разности средних, ICC и design_effect для случая
        randomization_unit != analysis_unit (например, рандомизация по user_id, а строки
        метрики — на уровне session/pageview). ICC считается методом one-way random effects
        ANOVA на объединенном пуле control+treatment; SE разности — методом коллапсирования
        до средних по кластеру (состоятельный и устойчивый к дисбалансу размеров кластеров
        способ получить корректные SE без сэндвич-оценки по сырым строкам). Примечание:
        при кластерах размера 1 (randomization_unit == analysis_unit на практике) ICC
        вырожден и формула возвращает значение около 1.0 — это ожидаемое поведение
        one-way ANOVA оценщика на вырожденных входных данных, а не баг.
        """
        c = np.asarray(control_vals, dtype=float)
        cc = np.asarray(control_cluster_ids)
        t = np.asarray(treatment_vals, dtype=float)
        tc = np.asarray(treatment_cluster_ids)
        if len(c) != len(cc) or len(t) != len(tc):
            raise ValueError("Длины значений и cluster_ids должны совпадать внутри каждой группы.")
        cls._validate_two_samples(c, t, "cluster_robust_se_and_icc")

        all_vals = np.concatenate([c, t])
        all_clusters = np.concatenate([cc, tc])
        unique_ids, counts, cluster_means = cls._cluster_summary(all_vals, all_clusters)
        k = len(unique_ids)
        n_total = len(all_vals)
        if k < 2:
            raise ValueError("Недостаточно уникальных кластеров для оценки ICC (минимум 2).")

        grand_mean = float(np.mean(all_vals))
        msb = float(np.sum(counts * (cluster_means - grand_mean) ** 2) / (k - 1))

        within_ss = 0.0
        for uid in unique_ids:
            vals_in_cluster = all_vals[all_clusters == uid]
            if len(vals_in_cluster) > 1:
                within_ss += float(np.sum((vals_in_cluster - np.mean(vals_in_cluster)) ** 2))
        df_within = max(n_total - k, 1)
        msw = within_ss / df_within

        sum_sq_counts = float(np.sum(counts ** 2))
        n0 = (n_total - sum_sq_counts / n_total) / (k - 1) if k > 1 else float(np.mean(counts))
        n0 = max(n0, 1e-9)

        denom_icc = msb + (n0 - 1.0) * msw
        icc = (msb - msw) / denom_icc if denom_icc > 1e-15 else 0.0
        icc = float(min(max(icc, 0.0), 0.999))

        mean_cluster_size = float(np.mean(counts))
        design_effect = 1.0 + (mean_cluster_size - 1.0) * icc

        c_ids, c_counts, c_means = cls._cluster_summary(c, cc)
        t_ids, t_counts, t_means = cls._cluster_summary(t, tc)
        if len(c_means) < 2 or len(t_means) < 2:
            raise ValueError(
                "Недостаточно кластеров в одной из групп для кластерно-устойчивой оценки (минимум 2)."
            )

        mean_c = float(np.average(c_means, weights=c_counts))
        mean_t = float(np.average(t_means, weights=t_counts))
        diff = mean_t - mean_c

        var_c_cluster = float(np.var(c_means, ddof=1))
        var_t_cluster = float(np.var(t_means, ddof=1))
        n_c_clust, n_t_clust = len(c_means), len(t_means)
        se_diff = math.sqrt(var_c_cluster / n_c_clust + var_t_cluster / n_t_clust)

        df_c, df_t = n_c_clust - 1, n_t_clust - 1
        if se_diff > 1e-15 and min(df_c, df_t) > 0:
            welch_df = ((var_c_cluster / n_c_clust + var_t_cluster / n_t_clust) ** 2) / (
                (var_c_cluster / n_c_clust) ** 2 / df_c + (var_t_cluster / n_t_clust) ** 2 / df_t
            )
            t_stat = diff / se_diff
            p_val = float(2.0 * stats.t.sf(abs(t_stat), welch_df))
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, welch_df))
        else:
            t_stat = 0.0
            p_val = 1.0 if abs(diff) < 1e-12 else 0.0
            t_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

        return {
            "criterion": "cluster_robust_collapsed_means",
            "icc": icc,
            "design_effect": round(design_effect, 4),
            "mean_cluster_size": mean_cluster_size,
            "n_clusters_control": int(n_c_clust),
            "n_clusters_treatment": int(n_t_clust),
            "mean_control": mean_c,
            "mean_treatment": mean_t,
            "absolute_diff": diff,
            "relative_diff": (diff / mean_c) if abs(mean_c) > 1e-12 else 0.0,
            "p_value": p_val,
            "statistic": t_stat,
            "ci_lower": diff - t_crit * se_diff,
            "ci_upper": diff + t_crit * se_diff,
            "is_significant": bool(p_val < alpha),
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
