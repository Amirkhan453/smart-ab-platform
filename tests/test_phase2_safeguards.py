"""
Фаза 2, П6: хвост фазы 1 — тесты SPRT и непокрытых веток safeguards.py.

SPRT в конвейере не используется (основной критерий SRM — χ², ТЗ v4 5.4). Тесты фиксируют его фактические свойства,
включая известные ограничения: слепота к перекосам меньше допуска и ошибка I рода выше номинальной
из-за выбора альтернативы по направлению данных.
"""
import math

import numpy as np
import pytest
from scipy import stats

from ab_platform.engine.safeguards import ExperimentSafeguardEngine as S, GuardrailStatus


# --- SPRT ---------------------------------------------------------------------------------------

def test_sprt_matches_wald_formulas():
    nc, nt, p0, d, a, b = 4_900, 5_100, 0.5, 0.01, 0.01, 0.10
    r = S.detect_srm_sprt(nc, nt, p0, a, b, d)
    p1 = p0 + d  # доля treatment выше ожидаемой → альтернатива сверху
    llr = nt * math.log(p1 / p0) + nc * math.log((1 - p1) / (1 - p0))
    assert r["log_likelihood_ratio"] == round(llr, 4)
    assert r["upper_threshold_log_b"] == round(math.log((1 - b) / a), 4)
    assert r["lower_threshold_log_a"] == round(math.log(b / (1 - a)), 4)
    assert r["chi2_pvalue"] == pytest.approx(stats.chisquare([nc, nt]).pvalue, rel=1e-9)
    # перекос вниз → альтернатива снизу, LLR симметричен
    assert S.detect_srm_sprt(nt, nc, p0, a, b, d)["log_likelihood_ratio"] == r["log_likelihood_ratio"]


def test_sprt_statuses_and_edges():
    assert S.detect_srm_sprt(48_000, 52_000)["status"] == "srm_alert_reject_h0"
    assert S.detect_srm_sprt(48_000, 52_000)["srm_detected"] is True
    healthy = S.detect_srm_sprt(50_000, 50_000)
    assert healthy["status"] == "healthy_accept_h0" and healthy["srm_detected"] is False
    assert S.detect_srm_sprt(500, 510)["status"] == "continue_monitoring"
    empty = S.detect_srm_sprt(0, 0)
    assert empty["status"] == "continue" and empty["chi2_pvalue"] == 1.0
    # альтернатива не выходит за (0, 1): p0 + delta > 1 обрезается до 0.999
    edge = S.detect_srm_sprt(10, 990, expected_treatment_ratio=0.995, tolerance_delta=0.01)
    assert math.isfinite(edge["log_likelihood_ratio"])
    low = S.detect_srm_sprt(990, 10, expected_treatment_ratio=0.005, tolerance_delta=0.01)
    assert math.isfinite(low["log_likelihood_ratio"])


@pytest.mark.parametrize("kw", [
    {"n_control": -1, "n_treatment": 5}, {"expected_treatment_ratio": 1.0}, {"alpha": 0.0}, {"beta": 0.5},
    {"tolerance_delta": 0.0},
])
def test_sprt_validation(kw):
    args = {"n_control": 5, "n_treatment": 5, **kw}
    with pytest.raises(ValueError):
        S.detect_srm_sprt(**args)


def test_sprt_blind_to_shift_below_tolerance_chi2_is_not():
    """Замечание аудита №19: перекос 0,3 п.п. при N = 1 млн. MON-1 требует его обнаружить — это делает χ²."""
    sprt = S.detect_srm_sprt(497_000, 503_000)
    chi2 = S.detect_srm_chi2(497_000, 503_000)
    assert sprt["status"] == "healthy_accept_h0" and sprt["log_likelihood_ratio"] < -79
    assert chi2["srm_detected"] and chi2["chi2_pvalue"] < 1e-8
    assert chi2["min_detectable_srm_shift"] < 0.003  # 0,165 п.п. при N = 1 млн


def _sequential(p_true, sims, seed, step=1000, max_n=200_000):
    """Непрерывный мониторинг: проверка после каждых step гостей; остановка на тревоге или принятии H0."""
    rng = np.random.default_rng(seed)
    alerts = 0
    for _ in range(sims):
        nt = nc = 0
        for _ in range(max_n // step):
            x = int(rng.binomial(step, p_true))
            nt, nc = nt + x, nc + step - x
            status = S.detect_srm_sprt(nc, nt)["status"]
            if status != "continue_monitoring":
                alerts += status == "srm_alert_reject_h0"
                break
    return alerts / sims


def test_sprt_sequential_type1_error_and_power():
    """
    H0 (50/50): номинал α = 1%, но альтернатива выбирается по знаку отклонения — это два односторонних SPRT,
    граница Вальда ≈ 2α/(1−β) = 2,2%. Замер: ~1,25% на 2000 прогонов. Мощность при перекосе = допуску (1 п.п.):
    номинал 1−β = 90%, замер ~95%.
    """
    sims = 2000
    fpr = _sequential(0.5, sims, seed=0)
    bound = 2 * 0.01 / (1 - 0.10)
    assert fpr <= bound + 3 * math.sqrt(bound * (1 - bound) / sims), fpr
    power = _sequential(0.51, 1000, seed=1)
    assert power >= 0.90 - 3 * math.sqrt(0.09 / 1000), power


# --- χ² SRM и остальные ветки ----------------------------------------------------------------------

def test_chi2_srm_against_scipy_and_min_shift_definition():
    r = S.detect_srm_chi2(10_300, 9_700)
    assert r["chi2_pvalue"] == pytest.approx(stats.chisquare([10_300, 9_700]).pvalue, rel=1e-9)
    # минимальный обнаружимый сдвиг — тот, при котором χ² ровно на критическом значении
    n, p0 = 1_000_000, 0.5
    shift = S.min_detectable_srm_shift(n, p0, 0.001)
    assert n * shift ** 2 / (p0 * (1 - p0)) == pytest.approx(stats.chi2.ppf(0.999, 1), rel=1e-12)
    assert S.min_detectable_srm_shift(0) == float("inf")
    empty = S.detect_srm_chi2(0, 0)
    assert empty["srm_detected"] is False and empty["chi2_pvalue"] == 1.0
    for kw in ({"n_control": -1}, {"expected_treatment_ratio": 0.0}, {"alpha": 0.5}):
        with pytest.raises(ValueError):
            S.detect_srm_chi2(**{"n_control": 1, "n_treatment": 1, **kw})


def test_multinomial_srm():
    r = S.detect_multinomial_srm([3_400, 3_300, 3_300], [1 / 3] * 3)
    assert r["chi2_pvalue"] == pytest.approx(stats.chisquare([3_400, 3_300, 3_300]).pvalue, rel=1e-9)
    assert S.detect_multinomial_srm([3_700, 3_300, 3_000], [1 / 3] * 3)["srm_detected"]
    assert S.detect_multinomial_srm([0, 0], [0.5, 0.5]) == {"srm_detected": False, "chi2_pvalue": 1.0,
                                                           "chi2_statistic": 0.0}
    for obs, exp in (([1], [1.0]), ([1, 2], [0.5]), ([-1, 2], [0.5, 0.5]), ([1, 2], [0.0, 1.0]), ([1, 2], [0.3, 0.3])):
        with pytest.raises(ValueError):
            S.detect_multinomial_srm(obs, exp)


def test_psi_validation_and_value():
    same = S.calculate_psi(np.array([10, 20, 30]), np.array([20, 40, 60]))
    assert same["psi_value"] == 0.0 and same["severity"] == "ok"
    shifted = S.calculate_psi(np.array([50, 30, 20]), np.array([20, 30, 50]))
    expected = sum((t - c) * math.log(t / c) for c, t in ((0.5, 0.2), (0.3, 0.3), (0.2, 0.5)))
    assert shifted["psi_value"] == round(expected, 6) and shifted["severity"] == "critical_mismatch"
    for c, t in (([], []), ([1, 2], [1]), ([-1, 2], [1, 2]), ([0, 0], [1, 2])):
        with pytest.raises(ValueError):
            S.calculate_psi(np.array(c), np.array(t))


def test_guardrail_branches():
    for kw in ({"max_allowed_degradation_rel": 0.0}, {"alpha_guardrail": 0.5}):
        with pytest.raises(ValueError):
            S.evaluate_guardrail_breach(100.0, 95.0, **kw)
    # без CI и p-value: неопределённости нет, решение по точечной оценке
    assert S.evaluate_guardrail_breach(100.0, 90.0)["status"] == GuardrailStatus.BREACH_PROVEN
    assert S.evaluate_guardrail_breach(100.0, 99.0)["status"] == GuardrailStatus.SAFE_PROVEN
    # NaN в p-value трактуется как p = 1 (нет информации) → широкий интервал → решение за человеком
    nan_p = S.evaluate_guardrail_breach(100.0, 96.0, p_value=float("nan"))
    assert nan_p["status"] == GuardrailStatus.INCONCLUSIVE_LOW_POWER
    # higher_is_better=False: рост метрики — ухудшение
    worse = S.evaluate_guardrail_breach(100.0, 120.0, ci_lower_diff=15.0, ci_upper_diff=25.0, higher_is_better=False)
    assert worse["guardrail_breached"] and worse["degradation_pct"] == 20.0
    assert S.apply_holm_to_breach_signals([]) == []
    assert S.apply_holm_to_breach_signals([0.01, 0.04, 0.03]) == [True, False, False]


def test_diagnose_srm_stage_points_to_assignment_or_trigger():
    assign = S.diagnose_srm_stage(52_000, 48_000, 26_000, 24_000)
    assert assign["assigned_srm"] and "Assignment Stage" in assign["diagnosis"]
    trigger = S.diagnose_srm_stage(50_000, 50_000, 27_000, 23_000)
    assert not trigger["assigned_srm"] and trigger["triggered_srm"] and "Trigger" in trigger["diagnosis"]
    ok = S.diagnose_srm_stage(50_000, 50_000, 25_000, 25_000)
    assert ok["diagnosis"].startswith("OK")


def test_sprt_decision_boundaries_are_exact():
    """Тревога — с первого nt, где LLR ≥ log B; принятие H0 — с первого N, где LLR ≤ log A."""
    total = 20_000
    rows = [S.detect_srm_sprt(total - nt, nt) for nt in range(total // 2, total)]
    k = next(i for i, r in enumerate(rows) if r["status"] == "srm_alert_reject_h0")
    b = rows[k]["upper_threshold_log_b"]
    assert rows[k]["log_likelihood_ratio"] >= b > rows[k - 1]["log_likelihood_ratio"]
    assert rows[k - 1]["status"] == "continue_monitoring"
    rows = [S.detect_srm_sprt(n, n) for n in range(1, 50_000, 50)]
    k = next(i for i, r in enumerate(rows) if r["status"] == "healthy_accept_h0")
    a = rows[k]["lower_threshold_log_a"]
    assert rows[k]["log_likelihood_ratio"] <= a < rows[k - 1]["log_likelihood_ratio"]
