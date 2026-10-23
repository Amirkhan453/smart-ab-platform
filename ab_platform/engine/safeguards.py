import math
from typing import Dict, List
import numpy as np
from scipy import stats


class ExperimentSafeguardEngine:
    """
    Система защиты эксперимента в реальном времени (eBay / Cloud Wizard methodology):
    - SPRT (Sequential Probability Ratio Test) для детекции SRM без проблемы подглядывания (peeking)
    - PSI (Population Stability Index) для проверки пред-экспериментального баланса сегментов
    - Двухэтапная диагностика SRM (Assigned stage vs Triggered stage)
    - Проверка Guardrail-метрик для автоматической остановки (Auto-Revert) с поддержкой нулевого бейзлайна
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
        """
        Последовательный критерий отношения правдоподобия (SPRT) для детекции Sample Ratio Mismatch.
        Проверяет H0: p = p0 против H1: |p - p0| >= delta без инфляции ошибки I рода при частом мониторинге.
        """
        total_n = n_control + n_treatment
        if total_n == 0:
            return {"status": "continue", "srm_detected": False, "log_likelihood_ratio": 0.0}

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

    @classmethod
    def diagnose_srm_stage(
        cls,
        assigned_control: int,
        assigned_treatment: int,
        triggered_control: int,
        triggered_treatment: int,
        expected_ratio: float = 0.5,
    ) -> dict:
        """
        Определяет точную стадию возникновения SRM (по архитектуре eBay):
        - Если SRM уже в Assigned -> баг в роутинге/хэшировании/конфиге слоев
        - Если Assigned чистый, а SRM в Triggered -> клиентская задержка (latency), баг фильтрации ботов или редирект
        """
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
        """Расчет Population Stability Index (PSI) для валидации качества рандомизации по когортам/сегментам."""
        c = np.asarray(control_dist, dtype=float)
        t = np.asarray(treatment_dist, dtype=float)

        p_c = np.clip(c / np.sum(c), epsilon, 1.0)
        p_t = np.clip(t / np.sum(t), epsilon, 1.0)

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
        """
        Проверяет, пробила ли защитная метрика (Guardrail) допустимый порог деградации.
        Корректно обрабатывает случай baseline_control == 0.0 (например, 0 ошибок в контроле).
        """
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
