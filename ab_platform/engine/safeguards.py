import math
from typing import Sequence
import numpy as np
from scipy import stats


class ExperimentSafeguardEngine:
    """
    Система защиты эксперимента в реальном времени (eBay / Cloud Wizard methodology):
    - Строгая валидация счетчиков и ожидаемых долей трафика
    - Последовательный критерий SPRT для парного SRM и мультиномиальный Chi2 для Multi-Arm (A/B/C/n)
    - Защищенный от нулевых сумм расчет индекса стабильности популяции (PSI)
    """

    @staticmethod
    def detect_srm_sprt(
        n_control: int,
        n_treatment: int,
        expected_treatment_ratio: float = 0.5,
        alpha: float = 0.01,
        beta: float = 0.10,
        tolerance_delta: float = 0.01,
    ) -> dict:
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
    ) -> dict:
        assigned_check = cls.detect_srm_sprt(assigned_control, assigned_treatment, expected_ratio)
        triggered_check = cls.detect_srm_sprt(triggered_control, triggered_treatment, expected_ratio)

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
        p_value: float,
        max_allowed_degradation_rel: float,
        higher_is_better: bool = True,
        alpha_guardrail: float = 0.05,
    ) -> dict:
        if max_allowed_degradation_rel <= 0.0:
            raise ValueError("max_allowed_degradation_rel должен быть > 0.")

        if abs(baseline_control) < 1e-12:
            if abs(treatment_value) < 1e-12:
                rel_change = 0.0
                degradation = 0.0
            else:
                rel_change = float(treatment_value - baseline_control)
                degradation = -rel_change if higher_is_better else rel_change
        else:
            rel_change = (treatment_value - baseline_control) / abs(baseline_control)
            degradation = -rel_change if higher_is_better else rel_change

        clean_pval = 1.0 if np.isnan(p_value) else float(p_value)
        is_breached = bool((degradation > max_allowed_degradation_rel) and (clean_pval < alpha_guardrail))
        return {
            "relative_change_pct": round(rel_change * 100.0, 3),
            "degradation_pct": round(degradation * 100.0, 3),
            "threshold_pct": round(max_allowed_degradation_rel * 100.0, 3),
            "guardrail_breached": is_breached,
            "action": "AUTO_STOP_AND_REVERT" if is_breached else "KEEP_RUNNING",
        }
