"""
Логика формы Паспорта (UI-0) без Streamlit: сборка тела Паспорта из полей формы и калькулятор длительности.
Страница pages/1_passport_form.py только рисует поля и вызывает эти функции и HTTP API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ab_platform.engine.power import PowerAndSampleSizeEngine
from ab_platform.schemas.passport import Direction

DEFAULT_CV = 2.5  # CV выручки на гостя за 28 дней (ТЗ 19.3); для непрерывной метрики без истории


def parse_numbers(text: str, what: str) -> List[float]:
    """«10, 50, 100» → [10.0, 50.0, 100.0]; ошибка — понятным текстом."""
    parts = [p.strip() for p in (text or "").replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{what}: укажите хотя бы одно число")
    try:
        return [float(p.replace(" ", "")) for p in parts]
    except ValueError:
        raise ValueError(f"{what}: только числа через запятую, получено «{text}»") from None


def build_passport(values: Dict[str, Any], definitions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Тело POST /passports из полей формы. definitions — {ref: определение метрики из GET /metrics/{ref}}.
    Проверки содержания делает сервер (валидаторы Паспорта); здесь — только форма полей.
    """
    def metric(ref: str) -> Dict[str, Any]:
        if ref not in definitions:
            raise ValueError(f"Метрики {ref} нет в реестре")
        return definitions[ref]

    split = float(values["treatment_share_pct"])
    if not 0 < split < 100:
        raise ValueError("Доля тестовой группы: от 0 до 100% не включая края")
    platforms = list(values.get("platforms") or [])
    targeting = {k: v for k, v in {"platforms": platforms or None,
                                   "app_version_min": (values.get("app_version_min") or "").strip() or None}.items()
                 if v is not None}
    return {
        "experiment_id": values["experiment_id"].strip(),
        "test_key": values["test_key"].strip(),
        "layer_id": values["layer_id"].strip(),
        "hypothesis_statement": values["hypothesis_statement"].strip(),
        "randomization_unit": values.get("randomization_unit", "user_id"),
        "estimand": "triggered",  # тесты приложения оцениваются по показанным (решение 4 дизайн-заметки фазы 4)
        "variants": [
            {"variant_id": "control", "traffic_percentage": 100.0 - split, "is_control": True},
            {"variant_id": "treatment", "traffic_percentage": split},
        ],
        "primary_metric": metric(values["primary_ref"]),
        "secondary_metrics": [metric(r) for r in values.get("secondary_refs", [])],
        "guardrail_metrics": [{"metric": metric(r), "max_allowed_degradation_rel": float(values["max_degradation_pct"]) / 100,
                               "tradeoff_rationale": values.get("tradeoff_rationale") or "защитная метрика приложения"}
                              for r in values.get("guardrail_refs", [])],
        "baseline_value": float(values["baseline_value"]),
        "mde_relative": float(values["mde_pct"]) / 100,
        "alpha": float(values.get("alpha", 0.05)),
        "power": float(values.get("power", 0.8)),
        "planned_duration_days": int(values["planned_duration_days"]),
        "daily_eligible_units": int(values["daily_eligible_units"]),
        "targeting": targeting,
        "targeting_audience_share": float(values["audience_share_pct"]) / 100 if targeting else None,
        "ramp_plan": parse_numbers(values.get("ramp_plan", "100"), "План раскатки"),
        "attribution_window_days": int(values.get("attribution_window_days", 7)),
        "use_cuped": bool(values.get("use_cuped", True)),
    }


def values_from_passport(p: Dict[str, Any]) -> Dict[str, Any]:
    """Обратное к build_passport: поля формы из сохранённого Паспорта (GET /passports/{id}) — для правки черновика."""
    treat = next((v for v in p["variants"] if not v.get("is_control")), p["variants"][-1])
    tgt = p.get("targeting") or {}
    guards = p.get("guardrail_metrics") or []
    return {
        "experiment_id": p["experiment_id"], "test_key": p["test_key"], "layer_id": p["layer_id"],
        "hypothesis_statement": p["hypothesis_statement"], "randomization_unit": p["randomization_unit"],
        "treatment_share_pct": float(treat["traffic_percentage"]),
        "primary_ref": p["primary_metric"].get("registry_ref"),
        "secondary_refs": [m.get("registry_ref") for m in p.get("secondary_metrics") or []],
        "guardrail_refs": [g["metric"].get("registry_ref") for g in guards],
        "max_degradation_pct": guards[0]["max_allowed_degradation_rel"] * 100 if guards else 10.0,
        "tradeoff_rationale": guards[0]["tradeoff_rationale"] if guards else "",
        "baseline_value": p["baseline_value"], "mde_pct": p["mde_relative"] * 100, "alpha": p["alpha"],
        "power": p["power"], "planned_duration_days": p["planned_duration_days"],
        "daily_eligible_units": p["daily_eligible_units"], "platforms": tgt.get("platforms") or [],
        "app_version_min": tgt.get("app_version_min") or "",
        "audience_share_pct": (p.get("targeting_audience_share") or 1.0) * 100,
        "ramp_plan": ", ".join(f"{x:g}" for x in p.get("ramp_plan") or [100]),
        "attribution_window_days": p["attribution_window_days"], "use_cuped": p["use_cuped"],
    }


def estimate_duration(metric_type: str, baseline: float, mde_relative: float, daily_units: int,
                      alpha: float = 0.05, power: float = 0.8, cv: Optional[float] = None,
                      pre_post_correlation: float = 0.0, treatment_share: float = 0.5,
                      direction: str = "increase") -> Dict[str, Any]:
    """
    Длительность по целевой аудитории (трафик с учётом таргетинга): выборка на вариант и дни, округлённые
    до целых недель. cv — коэффициент вариации непрерывной или ratio-метрики (σ / среднее).
    """
    binary = metric_type == "binary"
    variance = None
    if not binary:
        cv = DEFAULT_CV if cv is None else cv
        if cv <= 0:
            raise ValueError("Коэффициент вариации должен быть > 0")
        variance = (cv * baseline) ** 2
    r = PowerAndSampleSizeEngine.calculate_sample_size(
        baseline_mean=baseline, mde_relative=mde_relative, metric_variance=variance, is_binary=binary,
        expected_direction=Direction(direction), alpha=alpha, power=power, daily_traffic=int(daily_units),
        treatment_allocation_ratio=treatment_share, pre_post_correlation_r=pre_post_correlation)
    return {"n_per_variant": r.n_per_variant, "total": r.total_sample_size, "days": r.estimated_days,
            "feasible_5_weeks": r.is_feasible_under_5_weeks, "recommendation": r.recommendation,
            "tradeoff": r.tradeoff_table}
