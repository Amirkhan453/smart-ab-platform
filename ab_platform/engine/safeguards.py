import math
from enum import Enum
from typing import List, Sequence
import numpy as np
from scipy import stats


class GuardrailStatus(str, Enum):
    SAFE_PROVEN = "SAFE_PROVEN"
    BREACH_PROVEN = "BREACH_PROVEN"
    INCONCLUSIVE_LOW_POWER = "INCONCLUSIVE_LOW_POWER"


class ExperimentSafeguardEngine:
    """
    Система защиты эксперимента в реальном времени (eBay / Cloud Wizard methodology):
    - Строгая валидация счетчиков и ожидаемых долей трафика
    - Основной критерий SRM: Хи-квадрат (alpha=0.001 по умолчанию) раздельно для Assigned и Triggered
    - SPRT оставлен как опциональный потоковый метод непрерывного мониторинга
    - Мультиномиальный Chi2 для Multi-Arm (A/B/C/n)
    - Односторонний тест не-худшести для Guardrail (SAFE_PROVEN / BREACH_PROVEN / INCONCLUSIVE_LOW_POWER)
    - Защищенный от нулевых сумм расчет индекса стабильности популяции (PSI)
    """

    @staticmethod
    def min_detectable_srm_shift(total_n: float, expected_treatment_ratio: float = 0.5, alpha: float = 0.001) -> float:
        """
        Минимальный перекос доли treatment (в абсолютных долях, 0.01 = 1 п.п.), при котором
        хи-квадрат (df=1) при текущем N достигает критического значения для alpha.
        """
        if total_n <= 0:
            return float("inf")
        p0 = expected_treatment_ratio
        chi2_crit = float(stats.chi2.ppf(1.0 - alpha, df=1))
        return float(math.sqrt(chi2_crit * p0 * (1.0 - p0) / total_n))

    @classmethod
    def detect_srm_chi2(
        cls,
        n_control: int,
        n_treatment: int,
        expected_treatment_ratio: float = 0.5,
        alpha: float = 0.001,
    ) -> dict:
        if n_control < 0 or n_treatment < 0:
            raise ValueError("Количество пользователей в группах не может быть отрицательным.")
        if not (0.0 < expected_treatment_ratio < 1.0):
            raise ValueError("expected_treatment_ratio должен лежать строго в интервале (0, 1).")
        if not (0.0 < alpha < 0.5):
            raise ValueError("alpha должен лежать в интервале (0, 0.5).")

        total_n = n_control + n_treatment
        min_shift = cls.min_detectable_srm_shift(total_n, expected_treatment_ratio, alpha)
        if total_n == 0:
            return {
                "method": "chi2",
                "srm_detected": False,
                "observed_treatment_share": 0.0,
                "expected_treatment_share": expected_treatment_ratio,
                "chi2_statistic": 0.0,
                "chi2_pvalue": 1.0,
                "alpha": alpha,
                "min_detectable_srm_shift": min_shift,
            }

        p0 = expected_treatment_ratio
        expected_t = total_n * p0
        expected_c = total_n * (1.0 - p0)
        chi2_stat = ((n_treatment - expected_t) ** 2) / expected_t + ((n_control - expected_c) ** 2) / expected_c
        chi2_pval = float(stats.chi2.sf(chi2_stat, df=1))

        return {
            "method": "chi2",
            "srm_detected": bool(chi2_pval < alpha),
            "observed_treatment_share": round(n_treatment / total_n, 5),
            "expected_treatment_share": p0,
            "chi2_statistic": float(chi2_stat),
            "chi2_pvalue": chi2_pval,
            "alpha": alpha,
            "min_detectable_srm_shift": min_shift,
        }

    @staticmethod
    def detect_srm_sprt(
        n_control: int,
        n_treatment: int,
        expected_treatment_ratio: float = 0.5,
        alpha: float = 0.01,
        beta: float = 0.10,
        tolerance_delta: float = 0.01,
    ) -> dict:
        """Опциональный потоковый SRM-критерий (SPRT); основной критерий — detect_srm_chi2."""
        if n_control < 0 or n_treatment < 0:
            raise ValueError("Количество пользователей в группах не может быть отрицательным.")
        if not (0.0 < expected_treatment_ratio < 1.0):
            raise ValueError("expected_treatment_ratio должен лежать строго в интервале (0, 1).")
        if not (0.0 < alpha < 0.5) or not (0.0 < beta < 0.5) or tolerance_delta <= 0.0:
            raise ValueError("Некорректные параметры чувствительности SPRT.")

        total_n = n_control + n_treatment
        if total_n == 0:
            return {
                "status": "continue",
                "srm_detected": False,
                "observed_treatment_share": 0.0,
                "expected_treatment_share": expected_treatment_ratio,
                "log_likelihood_ratio": 0.0,
                "chi2_pvalue": 1.0,
            }

        p0 = expected_treatment_ratio
        obs_p = n_treatment / total_n
        p1 = min(0.999, p0 + tolerance_delta) if obs_p >= p0 else max(0.001, p0 - tolerance_delta)

        log_lambda = (
            n_treatment * math.log(p1 / p0)
            + n_control * math.log((1.0 - p1) / (1.0 - p0))
        )

        upper_bound_b = math.log((1.0 - beta) / alpha)
        lower_bound_a = math.log(beta / (1.0 - alpha))

        if log_lambda >= upper_bound_b:
            status = "srm_alert_reject_h0"
            srm_detected = True
        elif log_lambda <= lower_bound_a:
            status = "healthy_accept_h0"
            srm_detected = False
        else:
            status = "continue_monitoring"
            srm_detected = False

        expected_t = total_n * p0
        expected_c = total_n * (1.0 - p0)
        chi2_stat = ((n_treatment - expected_t) ** 2) / expected_t + ((n_control - expected_c) ** 2) / expected_c
        chi2_pval = float(1.0 - stats.chi2.cdf(chi2_stat, df=1))

        return {
            "status": status,
            "srm_detected": srm_detected,
            "observed_treatment_share": round(obs_p, 5),
            "expected_treatment_share": p0,
            "log_likelihood_ratio": round(log_lambda, 4),
            "upper_threshold_log_b": round(upper_bound_b, 4),
            "lower_threshold_log_a": round(lower_bound_a, 4),
            "chi2_pvalue": chi2_pval,
        }

    @staticmethod
    def detect_multinomial_srm(
        observed_counts: Sequence[int],
        expected_shares: Sequence[float],
        alpha_srm: float = 0.001,
    ) -> dict:
        obs = np.asarray(observed_counts, dtype=float)
        exp_p = np.asarray(expected_shares, dtype=float)
        if len(obs) < 2 or len(obs) != len(exp_p):
            raise ValueError("observed_counts и expected_shares должны иметь одинаковую длину >= 2.")
        if np.any(obs < 0):
            raise ValueError("observed_counts не может содержать отрицательные значения.")
        if np.any(exp_p <= 0) or abs(float(np.sum(exp_p)) - 1.0) > 1e-3:
            raise ValueError("Сумма expected_shares должна быть равна 1.0, а все доли > 0.")

        total_n = float(np.sum(obs))
        if total_n == 0:
            return {"srm_detected": False, "chi2_pvalue": 1.0, "chi2_statistic": 0.0}

        expected_counts = total_n * exp_p
        chi2_stat, p_val = stats.chisquare(f_obs=obs, f_exp=expected_counts)
        return {
            "srm_detected": bool(p_val < alpha_srm),
            "chi2_pvalue": float(p_val),
            "chi2_statistic": float(chi2_stat),
            "observed_shares": [round(float(x / total_n), 5) for x in obs],
            "expected_shares": [round(float(p), 5) for p in exp_p],
        }

    @classmethod
    def diagnose_srm_stage(
        cls,
        assigned_control: int,
        assigned_treatment: int,
        triggered_control: int,
        triggered_treatment: int,
        expected_ratio: float = 0.5,
        alpha_srm: float = 0.001,
    ) -> dict:
        assigned_check = cls.detect_srm_chi2(assigned_control, assigned_treatment, expected_ratio, alpha_srm)
        triggered_check = cls.detect_srm_chi2(triggered_control, triggered_treatment, expected_ratio, alpha_srm)

        if assigned_check["srm_detected"]:
            root_cause = (
                "CRITICAL [Assignment Stage SRM]: Перекос трафика возник уже при бакетировании. "
                "Проверьте хэш-функцию, конфигурацию слоев (Planes) и правила таргетинга."
            )
        elif triggered_check["srm_detected"]:
            root_cause = (
                "CRITICAL [Trigger/Execution Stage SRM]: Назначение в группы честное, но перекос возник "
                "при активации фичи. Вероятные причины: 1) Латентность (latency) нового варианта сломала "
                "фильтрацию ботов или вызвала отвал загрузки; 2) Асимметричный триггеринг или сбой логов SDK."
            )
        else:
            root_cause = "OK: Оба этапа (Assigned и Triggered) проходят проверку баланса трафика."

        return {
            "assigned_srm": assigned_check["srm_detected"],
            "triggered_srm": triggered_check["srm_detected"],
            "diagnosis": root_cause,
            "method": "chi2",
            "alpha_srm": alpha_srm,
            "assigned_details": assigned_check,
            "triggered_details": triggered_check,
            "min_detectable_srm_shift": {
                "assigned": assigned_check["min_detectable_srm_shift"],
                "triggered": triggered_check["min_detectable_srm_shift"],
            },
        }

    @staticmethod
    def calculate_psi(
        control_dist: np.ndarray, treatment_dist: np.ndarray, epsilon: float = 1e-6
    ) -> dict:
        c = np.asarray(control_dist, dtype=float)
        t = np.asarray(treatment_dist, dtype=float)
        if c.size == 0 or t.size == 0 or c.size != t.size:
            raise ValueError("Массивы распределений для PSI должны быть непустыми и одинаковой длины.")
        if np.any(c < 0) or np.any(t < 0):
            raise ValueError("Частоты сегментов в PSI не могут быть отрицательными.")

        sum_c = float(np.sum(c))
        sum_t = float(np.sum(t))
        if sum_c <= 0.0 or sum_t <= 0.0:
            raise ValueError("Сумма элементов в распределениях PSI должна быть строго положительной (> 0).")

        p_c = np.clip(c / sum_c, epsilon, 1.0)
        p_t = np.clip(t / sum_t, epsilon, 1.0)

        psi_val = float(np.sum((p_t - p_c) * np.log(p_t / p_c)))
        return {
            "psi_value": round(psi_val, 6),
            "is_balanced": bool(psi_val < 0.05),
            "severity": "ok" if psi_val < 0.05 else ("warning" if psi_val < 0.15 else "critical_mismatch"),
        }

    @staticmethod
    def evaluate_guardrail_breach(
        baseline_control: float,
        treatment_value: float,
        max_allowed_degradation_rel: float,
        higher_is_better: bool = True,
        alpha_guardrail: float = 0.05,
        ci_lower_diff: float | None = None,
        ci_upper_diff: float | None = None,
        p_value: float | None = None,
    ) -> dict:
        """
        Односторонний тест не-худшести относительно границы delta = max_allowed_degradation_rel.
        ci_lower_diff/ci_upper_diff — двусторонний (1 - alpha_guardrail) ДИ разности (treatment - control)
        в абсолютных единицах метрики, построенный с тем же alpha_guardrail.
        """
        if max_allowed_degradation_rel <= 0.0:
            raise ValueError("max_allowed_degradation_rel должен быть > 0.")
        if not (0.0 < alpha_guardrail < 0.5):
            raise ValueError("alpha_guardrail должен лежать в интервале (0, 0.5).")

        delta = max_allowed_degradation_rel
        denom = abs(baseline_control) if abs(baseline_control) >= 1e-12 else 1.0

        rel_change = float(treatment_value - baseline_control) / denom
        if abs(baseline_control) < 1e-12 and abs(treatment_value) < 1e-12:
            rel_change = 0.0
        degradation = -rel_change if higher_is_better else rel_change

        if ci_lower_diff is None or ci_upper_diff is None:
            if p_value is None:
                raise ValueError(
                    "Нужно передать ci_lower_diff и ci_upper_diff либо legacy p_value."
                )
            breached = bool(degradation > delta and float(p_value) < alpha_guardrail)
            return {
                "relative_change_pct": round(rel_change * 100.0, 3),
                "degradation_pct": round(degradation * 100.0, 3),
                "threshold_pct": round(delta * 100.0, 3),
                "degradation_upper_bound_pct": round(degradation * 100.0, 3),
                "degradation_lower_bound_pct": round(degradation * 100.0, 3),
                "p_value_breach": float(p_value),
                "status": GuardrailStatus.BREACH_PROVEN if breached else GuardrailStatus.INCONCLUSIVE_LOW_POWER,
                "guardrail_breached": breached,
                "action": "AUTO_STOP_AND_REVERT" if breached else "KEEP_RUNNING",
            }

        lo = 0.0 if np.isnan(ci_lower_diff) else float(ci_lower_diff) / denom
        hi = 0.0 if np.isnan(ci_upper_diff) else float(ci_upper_diff) / denom
        if higher_is_better:
            deg_ci_lower, deg_ci_upper = -hi, -lo
        else:
            deg_ci_lower, deg_ci_upper = lo, hi

        z_two = float(stats.norm.ppf(1.0 - alpha_guardrail / 2.0))
        z_one = float(stats.norm.ppf(1.0 - alpha_guardrail))
        se = max((deg_ci_upper - deg_ci_lower) / (2.0 * z_two), 0.0)

        if se > 1e-12:
            p_value_breach = float(stats.norm.sf((degradation - delta) / se))
        else:
            p_value_breach = 0.0 if degradation > delta else 1.0

        upper_one_sided = degradation + z_one * se
        lower_one_sided = degradation - z_one * se

        if upper_one_sided < delta:
            status = GuardrailStatus.SAFE_PROVEN
            action = "KEEP_RUNNING"
        elif lower_one_sided > delta:
            status = GuardrailStatus.BREACH_PROVEN
            action = "AUTO_STOP_AND_REVERT"
        else:
            status = GuardrailStatus.INCONCLUSIVE_LOW_POWER
            action = "HOLD_HUMAN_REVIEW"

        return {
            "relative_change_pct": round(rel_change * 100.0, 3),
            "degradation_pct": round(degradation * 100.0, 3),
            "threshold_pct": round(delta * 100.0, 3),
            "degradation_upper_bound_pct": round(upper_one_sided * 100.0, 3),
            "degradation_lower_bound_pct": round(lower_one_sided * 100.0, 3),
            "p_value_breach": p_value_breach,
            "status": status,
            "guardrail_breached": bool(status == GuardrailStatus.BREACH_PROVEN),
            "action": action,
        }

    @staticmethod
    def apply_holm_to_breach_signals(breach_pvalues: List[float], alpha: float = 0.05) -> List[bool]:
        """Поправка Холма для сигнала «пробой доказан» по семейству guardrail-метрик."""
        m = len(breach_pvalues)
        if m == 0:
            return []
        order = sorted(range(m), key=lambda i: breach_pvalues[i])
        rejects = [False] * m
        for rank, idx in enumerate(order):
            if breach_pvalues[idx] <= alpha / (m - rank):
                rejects[idx] = True
            else:
                break
        return rejects
