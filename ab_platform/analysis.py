"""
Запуски анализа тестов приложения (ТЗ v4.1: RPT-1, анализ по расписанию; фаза 4, решения 5–6).

- run_analysis — срез событий → таблица «гость × метрики» → правило решения (evaluate_experiment) → AnalysisRun.
  Промежуточный анализ (monitoring) до горизонта показывает только SRM, guardrails и данные; итоговый (final) —
  только когда тест остановлен и у всех гостей закрылось окно атрибуции.
- report_markdown — отчёт для людей из результата запуска; каждое число ссылается на run_id.
- reproduce — повторный расчёт по сохранённым входам; результат должен совпасть байт в байт.
- run_due_analyses — ежедневный проход по тестам: мониторинг, тревоги, автопауза по guardrail, итоговый отчёт.
  Автопауза — только при пробое, доказанном на уровне α / K (K — число плановых ежедневных проверок):
  ежедневная проверка — это подглядывание, и без поправки ложные паузы копились бы.
- preflight — предстартовая проверка: A/A на 500 солях по истории заказов и баланс настоящей соли.
"""
import dataclasses
import enum
import hashlib
import math
import os
import subprocess
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ab_platform.control import Actor, ControlError, ExperimentService, rule_from_passport, rule_id
from ab_platform.engine.assignment import (
    DeterministicAssignmentEngine as ASG, allocate_variant_buckets,
)
from ab_platform.engine.exposure_analysis import NO_VALUE, srm_by_segment
from ab_platform.engine.safeguards import ExperimentSafeguardEngine
from ab_platform.engine.unit_table import build_history_table, build_unit_table
from ab_platform.metrics import MetricVersion
from ab_platform.orchestrator import SmartABPlatformOrchestrator as ORCH
from ab_platform.schemas.analysis import Alert, AnalysisRun, RunInputs, canonical_json
from ab_platform.schemas.passport import ExperimentPassport, MetricType, PassportStatus

PREFLIGHT_SALTS = 500
PREFLIGHT_FPR_RANGE = (0.03, 0.07)   # 5% ± 2 SE при 500 прогонах
PREFLIGHT_MAX_SMD = 0.1
PREFLIGHT_MIN_UNITS = 200
SEGMENT_ATTRIBUTES = ("platform", "app_version", "city", "region")
SCHEDULER = Actor("scheduler", "admin")
_AFTER_STOP = (PassportStatus.STOPPED, PassportStatus.ANALYSIS, PassportStatus.DECISION)


@lru_cache(maxsize=1)
def code_version() -> str:
    """Версия кода в отчёте: ABP_CODE_VERSION, иначе коммит git, иначе 'unknown'."""
    env = os.environ.get("ABP_CODE_VERSION", "").strip()
    if env:
        return env
    try:
        # --dirty: незакоммиченные правки видны в версии, иначе отчёт ссылался бы на чужой код
        out = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], capture_output=True, text=True,
                             timeout=5, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _plain(x: Any) -> Any:
    """Результат для JSON: numpy → Python, NaN и бесконечности → None, перечисления → значения."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _plain(dataclasses.asdict(x))
    if isinstance(x, enum.Enum):
        return _plain(x.value)
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    return x


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _metrics_of(service: ExperimentService, passport: ExperimentPassport) -> List[Tuple[str, MetricVersion]]:
    """Метрики Паспорта из реестра с ролями: primary, secondary, guardrail."""
    defs = [("primary", passport.primary_metric)] + [("secondary", m) for m in passport.secondary_metrics] + [
        ("guardrail", g.metric) for g in passport.guardrail_metrics]
    out, seen = [], set()
    for role, d in defs:
        if d.registry_ref is None:
            raise ControlError("invalid", f"Метрика '{d.name}' не из реестра: анализ по событиям невозможен.")
        m = service.metric(d.registry_ref)
        if m.name not in seen:
            seen.add(m.name)
            out.append((role, m))
    return out


def _rule_and_passport(service: ExperimentService, experiment_id: str):
    ruleset = service.ruleset()
    rule = next((r for r in (ruleset.experiments if ruleset else ()) if r.passport_id == experiment_id), None)
    if rule is None:
        raise ControlError("conflict", f"Тест '{experiment_id}' не запущен: правила в ruleset нет.")
    return rule, service.get(experiment_id, rule.passport_version)


def _inputs(kind, passport, rid, metrics, cutoff, upto_seq, params=None) -> RunInputs:
    return RunInputs(kind=kind, experiment_id=passport.experiment_id, passport_version=passport.version,
                     passport_hash=passport.passport_hash, rule_id=rid,
                     metrics=[{"ref": m.ref, "definition_hash": m.definition_hash, "role": role} for role, m in metrics],
                     cutoff=_iso(cutoff), upto_seq=upto_seq, code_version=code_version(), params=params or {})


def _table_hash(table: pd.DataFrame) -> str:
    return hashlib.sha256(table.to_csv(index=False).encode("utf-8")).hexdigest()


def _segments(table: pd.DataFrame, passport: ExperimentPassport, alpha: float) -> Dict[str, Any]:
    spec = [(v.variant_id, v.traffic_percentage) for v in passport.variants]
    out = {}
    for attr in SEGMENT_ATTRIBUTES:
        col = f"attr_{attr}"
        if col in table.columns and table[col].nunique() >= 2:
            res = srm_by_segment(table, spec, col, alpha=alpha)
            out[attr] = {"segment_alpha": res["segment_alpha"], "segments": res["segments"],
                         "srm_detected": any(s["srm_detected"] for s in res["segments"].values())}
    return out


def _compute(passport: ExperimentPassport, metrics: List[Tuple[str, MetricVersion]], events, rule, kind: str,
             cutoff: datetime, upto_seq: int, started_at: datetime) -> Tuple[Dict[str, Any], str]:
    """Результат запуска (только из входов) и хэш входной таблицы."""
    W, pre = passport.attribution_window_days, passport.pre_period_window_days
    table = build_unit_table(events, rule.experiment_id, [m for _, m in metrics], W, pre, cutoff, rule.unit_type,
                             upto_seq)
    input_hash = _table_hash(table)
    primary = next(m for role, m in metrics if role == "primary")
    analysis_day = None if kind == "final" else int((cutoff - started_at).total_seconds() // 86_400)
    counts = {v.variant_id: int((table["variant"] == v.variant_id).sum()) for v in passport.variants}
    population = {"exposed": int(len(table)), "by_variant": counts,
                  "windows_complete": int(table["window_complete"].sum()) if len(table) else 0}
    caveats = [
        f"Эффект — на гостях, увидевших изменение (triggered), за {W} дн. после первого показа; "
        "заказы до показа идут только в ковариату.",
        f"Популяция обобщения: {passport.generalization_population}.",
        *passport.design_warnings,
    ]
    if passport.practical_significance_threshold == 0:
        caveats.append(f"Порог практической значимости командой не задан: использован MDE {passport.mde_relative:.1%}.")
    if len(table) and population["windows_complete"] < len(table):
        caveats.append(f"Окно атрибуции не закрыто у {len(table) - population['windows_complete']} гостей: "
                       "промежуточный срез.")
    base = {
        "kind": kind, "experiment_id": passport.experiment_id, "rule_id": rule.experiment_id,
        "passport": {"version": passport.version, "hash": passport.passport_hash},
        "snapshot": {"cutoff": _iso(cutoff), "upto_seq": upto_seq, "input_hash": input_hash},
        "metrics": [{"role": role, "ref": m.ref, "title": m.title, "definition_hash": m.definition_hash}
                    for role, m in metrics],
        "estimand": "triggered", "attribution_window_days": W, "analysis_day": analysis_day,
        "population": population, "code_version": code_version(),
    }
    shortage = None
    if min(counts.values()) < 2:
        shortage = "Мало показанных гостей: в каждом варианте нужно хотя бы 2."
    else:
        # у ratio-метрики (средний чек, доля отмен) без единого заказа в варианте знаменатель 0 — считать нечего
        empty = [f"{m.name} в варианте {v}" for _, m in metrics if m.metric_type == MetricType.RATIO
                 for v, g in table.groupby("variant")[m.columns()[1]] if g.sum() <= 0]
        if empty:
            shortage = "Мало данных: нет заказов для ratio-метрики — " + ", ".join(empty) + "."
    if shortage:
        return {**base, "decision": "INSUFFICIENT_DATA", "can_trust_data": False, "blinded": analysis_day is not None,
                "summary": shortage, "caveats": caveats, "covariate": None,
                "srm": {}, "segments_srm": {}, "guardrails": [], "auto_pause": {"breached": []}}, input_hash

    covariate = f"{primary.name}__pre" if primary.metric_type != MetricType.RATIO and passport.use_cuped else None
    report = ORCH.evaluate_experiment(passport, table, user_col="unit_id", variant_col="variant",
                                      covariate_col=covariate, analysis_day=analysis_day)
    # автопауза: пробой guardrail, доказанный на уровне α / K по Холму внутри варианта
    K = max(int(passport.planned_duration_days), 1)
    alpha_look = passport.alpha / K
    breached = []
    for variant in {g.variant_id for g in report.guardrail_results}:
        rows = [g for g in report.guardrail_results if g.variant_id == variant]
        rejects = ExperimentSafeguardEngine.apply_holm_to_breach_signals([g.p_value_breach for g in rows], alpha_look)
        breached += [{"variant": variant, "metric": g.metric_name, "p_value_breach": g.p_value_breach}
                     for g, r in zip(rows, rejects) if r]
    blinded = bool(report.blinded)
    if blinded:
        caveats.append(f"До горизонта ({passport.planned_duration_days} дн.) эффект primary и вторичных метрик скрыт.")
    segments = _segments(table, passport, passport.srm_alpha)
    result = {
        **base,
        "decision": report.decision_status, "can_trust_data": report.can_trust_data, "blinded": blinded,
        # до горизонта ядро само скрывает эффект: победителя, primary и вторичные метрики (слепой режим)
        "summary": report.executive_summary_ru, "winning_variant": report.winning_variant,
        "primary": report.primary_metric_result, "comparisons": report.variant_comparisons,
        "secondary": report.secondary_results,
        "guardrails": report.guardrail_results, "covariate": covariate, "cuped": report.cuped_diagnostics,
        "srm": report.srm_diagnosis, "segments_srm": segments,
        "srm_detected": bool(not report.can_trust_data or any(s["srm_detected"] for s in segments.values())),
        "hygiene": {"double_exposure_removed_pct": report.double_exposure_removed_pct,
                    "clean_sample_size": report.clean_sample_size},
        "evidence_level": report.evidence_level, "generalization_population": report.generalization_population,
        "auto_pause": {"alpha_per_look": alpha_look, "planned_looks": K, "breached": breached},
        "caveats": caveats,
    }
    return _plain(result), input_hash


def run_analysis(service: ExperimentService, events, experiment_id: str, kind: str = "monitoring",
                 cutoff: Optional[datetime] = None, actor: str = "scheduler", upto_seq: Optional[int] = None,
                 save: bool = True) -> AnalysisRun:
    """Запуск анализа идущего или остановленного теста. Повтор с теми же входами возвращает тот же запуск."""
    if kind not in ("monitoring", "final"):
        raise ControlError("invalid", "kind: monitoring или final.")
    rule, passport = _rule_and_passport(service, experiment_id)
    cutoff = (cutoff or service.clock()).astimezone(timezone.utc)
    times = service.lifecycle_times(experiment_id, passport.version)
    started = _utc(times["started_at"])
    if kind == "monitoring" and passport.status not in (PassportStatus.RUNNING, PassportStatus.PAUSED):
        raise ControlError("conflict", f"Мониторинг — только идущего теста; сейчас '{passport.status.value}'.")
    if kind == "final":
        if passport.status not in _AFTER_STOP:
            raise ControlError("conflict", "Итоговый анализ — только после остановки теста.")
        ready = _utc(times["stopped_at"]) + timedelta(days=passport.attribution_window_days + passport.maturation_days)
        if cutoff < ready:
            raise ControlError("conflict", f"Окна атрибуции закроются {_iso(ready)}: итоговый анализ раньше нельзя.")
    metrics = _metrics_of(service, passport)
    upto = events.last_seq() if upto_seq is None else upto_seq
    inputs = _inputs(kind, passport, rule.experiment_id, metrics, cutoff, upto)
    existing = service.storage.get_analysis_run(inputs.run_id)
    if existing is not None:
        return existing
    result, input_hash = _compute(passport, metrics, events, rule, kind, cutoff, upto, started)
    run = AnalysisRun(run_id=inputs.run_id, inputs=inputs, input_hash=input_hash, result=result,
                      created_at=_iso(service.clock()), actor=actor)
    if save:
        with service.storage.transaction():
            service.storage.save_analysis_run(run)
            service.storage.append_audit(actor, f"analysis.{kind}", experiment_id,
                                         {"run_id": run.run_id, "decision": result["decision"]})
    return run


def reproduce(service: ExperimentService, events, run_id: str) -> Dict[str, Any]:
    """Повтор запуска по сохранённым входам. identical — результат и входная таблица совпали байт в байт."""
    run = service.storage.get_analysis_run(run_id)
    if run is None:
        raise ControlError("not_found", f"Запуска {run_id} нет.")
    inp = run.inputs
    if inp.kind == "preflight":
        raise ControlError("invalid", "Предстартовую проверку повторяют новым запуском.")
    passport = service.get(inp.experiment_id, inp.passport_version)
    metrics = [(m["role"], service.metric(m["ref"])) for m in inp.metrics]
    # правило — из Паспорта: анализу нужны только его идентификатор и тип единицы, а в архиве правила в ruleset уже нет
    rule = rule_from_passport(passport)
    started = _utc(service.lifecycle_times(inp.experiment_id, inp.passport_version)["started_at"])
    result, input_hash = _compute(passport, metrics, events, rule, inp.kind, _utc(inp.cutoff), inp.upto_seq, started)
    same_code = inp.code_version == code_version()
    identical = canonical_json(result) == run.result_json() and input_hash == run.input_hash
    return {"run_id": run_id, "identical": identical, "same_code_version": same_code,
            "stored_code_version": inp.code_version, "current_code_version": code_version()}


# ---------------- отчёт ----------------
def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:+.2%}"


def _num(x: Optional[float], digits: int = 4) -> str:
    return "—" if x is None else f"{x:.{digits}g}"


def report_markdown(run: AnalysisRun) -> str:
    """Отчёт RPT-1 для людей. Все числа — из результата запуска run_id."""
    r = run.result
    lines = [f"# Отчёт: {r['experiment_id']} — {'итоговый' if r['kind'] == 'final' else 'промежуточный'}", "",
             f"Запуск `{run.run_id}` · срез {r['snapshot']['cutoff']} · событий до №{r['snapshot']['upto_seq']}", "",
             "## Итог", "", f"**Вердикт:** {r['decision']}", "", r.get("summary", ""), ""]
    pop = r["population"]
    lines += [f"Показанных гостей: {pop['exposed']} ({', '.join(f'{k}: {v}' for k, v in pop['by_variant'].items())}); "
              f"окно атрибуции закрыто у {pop['windows_complete']}.", ""]
    if r.get("blinded"):
        lines += ["## Эффект", "", "Скрыт до горизонта теста (фиксированный горизонт, ТЗ v4, 5.4).", ""]
    elif r.get("comparisons"):
        lines += ["## Эффект (primary)", "", "| Вариант | Эффект | ДИ эффекта (относительно контроля) | p-value | Значим |",
                  "| --- | --- | --- | --- | --- |"]
        for variant, c in r["comparisons"].items():
            base_mean = c.get("mean_control")
            lo, hi = ((c.get("ci_lower") / base_mean, c.get("ci_upper") / base_mean)
                      if base_mean and c.get("ci_lower") is not None and c.get("ci_upper") is not None else (None, None))
            lines.append(f"| {variant} | {_pct(c.get('relative_diff'))} | {_pct(lo)} … {_pct(hi)} | "
                         f"{_num(c.get('p_value'))} | {'да' if c.get('is_significant') else 'нет'} |")
        lines.append("")
    if r.get("guardrails"):
        lines += ["## Защитные метрики", "", "| Вариант | Метрика | Изменение | Допуск | Статус |",
                  "| --- | --- | --- | --- | --- |"]
        for g in r["guardrails"]:
            lines.append(f"| {g['variant_id']} | {g['metric_name']} | {g['relative_change_pct']:+.2f}% | "
                         f"{g['max_allowed_degradation_pct']:.1f}% | {g['status']} |")
        lines.append("")
    if r.get("secondary"):
        lines += ["## Вторичные метрики", "", "| Вариант | Метрика | Эффект | p-value | Значим (BH) |",
                  "| --- | --- | --- | --- | --- |"]
        for s in r["secondary"]:
            lines.append(f"| {s['variant_id']} | {s['metric_name']} | {s['relative_diff_pct']:+.2f}% | "
                         f"{_num(s['p_value'])} | {'да' if s['is_significant_fdr'] else 'нет'} |")
        lines.append("")
    lines += ["## SRM и данные", "", f"Данным можно доверять: {'да' if r.get('can_trust_data') else 'НЕТ'}.", ""]
    for attr, seg in (r.get("segments_srm") or {}).items():
        bad = [k for k, v in seg["segments"].items() if v["srm_detected"]]
        lines.append(f"- SRM по сегментам {attr}: {'перекос в ' + ', '.join(bad) if bad else 'нет'} "
                     f"(порог на сегмент {seg['segment_alpha']:.1e}).")
    paused = r.get("auto_pause", {}).get("breached", [])
    if paused:
        lines.append(f"- Пробой guardrail на уровне α/K: {', '.join(p['metric'] for p in paused)} — тест поставлен на паузу.")
    lines += ["", "## Оговорки", "", *[f"- {c}" for c in r.get("caveats", [])], "",
              "## Версии и воспроизводимость", "",
              f"- Паспорт v{r['passport']['version']}, хэш `{r['passport']['hash'][:16]}…`; правило `{r['rule_id']}`.",
              f"- Метрики: {', '.join(m['ref'] for m in r['metrics'])}.",
              f"- Код: `{r['code_version']}`; входная таблица: `{r['snapshot']['input_hash'][:16]}…`.",
              f"- Повторный расчёт: `POST /analysis-runs/{run.run_id}/reproduce`.", ""]
    return "\n".join(lines)


# ---------------- тревоги и расписание ----------------
def _alert(service: ExperimentService, experiment_id: str, kind: str, message: str, run_id: Optional[str]) -> Alert:
    alert = Alert(alert_id=f"{run_id or _iso(service.clock())}:{kind}", experiment_id=experiment_id, kind=kind,
                  message=message, run_id=run_id, created_at=_iso(service.clock()))
    if not any(a.alert_id == alert.alert_id for a in service.storage.list_alerts(experiment_id)):
        service.storage.save_alert(alert)
    return alert


def run_due_analyses(service: ExperimentService, events, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    Ежедневный проход (cron): идущие тесты — мониторинг раз в сутки, тревога при SRM, автопауза при пробое
    guardrail на уровне α / K; остановленные — итоговый отчёт, когда закрылись все окна атрибуции.
    """
    now = (now or service.clock()).astimezone(timezone.utc)
    ruleset = service.ruleset()
    done = []
    for rule in (ruleset.experiments if ruleset else ()):
        if rule.passport_id is None:
            continue
        passport = service.get(rule.passport_id, rule.passport_version)
        eid = passport.experiment_id
        try:
            if passport.status == PassportStatus.RUNNING:
                today = now.date().isoformat()
                if any(r.inputs.cutoff[:10] == today and r.inputs.passport_version == passport.version
                       for r in service.storage.list_analysis_runs(eid, "monitoring")):
                    continue
                run = run_analysis(service, events, eid, "monitoring", now)
                res = run.result
                if res.get("srm_detected"):
                    _alert(service, eid, "srm", "Перекос групп (SRM): анализ заблокирован, проверьте показы.", run.run_id)
                if res.get("auto_pause", {}).get("breached"):
                    service.pause(SCHEDULER, eid)
                    names = ", ".join(sorted({b["metric"] for b in res["auto_pause"]["breached"]}))
                    _alert(service, eid, "guardrail_auto_pause",
                           f"Пробой защитных метрик ({names}) доказан на уровне α/K: тест на паузе, решение за владельцем.",
                           run.run_id)
                done.append({"experiment_id": eid, "kind": "monitoring", "run_id": run.run_id,
                             "decision": res["decision"]})
            elif passport.status in _AFTER_STOP:
                if service.storage.list_analysis_runs(eid, "final"):
                    continue
                stopped = service.lifecycle_times(eid, passport.version)["stopped_at"]
                ready = _utc(stopped) + timedelta(days=passport.attribution_window_days + passport.maturation_days)
                if now < ready:
                    continue
                run = run_analysis(service, events, eid, "final", now)
                _alert(service, eid, "final_report_ready", f"Итоговый отчёт готов: {run.result['decision']}.",
                       run.run_id)
                done.append({"experiment_id": eid, "kind": "final", "run_id": run.run_id,
                             "decision": run.result["decision"]})
        except (ControlError, ValueError) as exc:
            _alert(service, eid, "analysis_failed", f"Анализ не выполнен: {exc}", None)
            done.append({"experiment_id": eid, "kind": "error", "error": str(exc)})
    return done


# ---------------- предстартовая проверка ----------------
def _smd(a: np.ndarray, b: np.ndarray) -> float:
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0) if len(a) > 1 and len(b) > 1 else 0.0
    return 0.0 if pooled == 0.0 else abs(float(np.mean(a) - np.mean(b))) / pooled


def preflight(service: ExperimentService, events, experiment_id: str, actor: Actor,
              now: Optional[datetime] = None, n_salts: int = PREFLIGHT_SALTS) -> AnalysisRun:
    """
    A/A на истории: n_salts случайных разбиений гостей, активных за последние W + pre дней, боевой функцией анализа
    primary с CUPED; доля p < α должна попасть в [3%; 7%]. Баланс: разбиение настоящей солью правила, SMD < 0,1
    по всем метрикам. Результат привязан к хэшу Паспорта: запуск проверяет именно его.
    """
    passport = service.get(experiment_id)
    service._require_owner_or(actor, passport, "analyst")
    if passport.status != PassportStatus.APPROVED:
        raise ControlError("conflict", f"Предстартовая проверка — для утверждённого Паспорта; сейчас "
                                       f"'{passport.status.value}'.")
    now = (now or service.clock()).astimezone(timezone.utc)
    metrics = _metrics_of(service, passport)
    W, pre = passport.attribution_window_days, passport.pre_period_window_days
    split, start = now - timedelta(days=W), now - timedelta(days=W + pre)
    upto = events.last_seq()
    inputs = _inputs("preflight", passport, rule_id(passport), metrics, now, upto, {"n_salts": n_salts})
    existing = service.storage.get_analysis_run(inputs.run_id)
    if existing is not None:
        return existing
    unit_type = "device_id" if passport.randomization_unit.value == "device_id" else "guest_id"
    table = build_history_table(events, [m for _, m in metrics], start, split, now, unit_type, upto)
    primary = next(m for role, m in metrics if role == "primary")
    control = next(v for v in passport.variants if v.is_control)
    problems: List[str] = []
    result: Dict[str, Any] = {"n_guests": int(len(table)), "history": {"start": _iso(start), "split": _iso(split),
                                                                     "end": _iso(now)}}
    if len(table) < PREFLIGHT_MIN_UNITS:
        problems.append(f"мало истории: {len(table)} активных гостей, нужно не меньше {PREFLIGHT_MIN_UNITS}")
    else:
        ids = table["unit_id"].tolist()
        covariate = f"{primary.name}__pre" if primary.metric_type != MetricType.RATIO and passport.use_cuped else None
        p_values = []
        for i in range(n_salts):
            is_control = ASG.control_from_bucket(ASG.buckets_for(f"preflight:{passport.passport_hash[:12]}:{i}", ids),
                                                 control.traffic_percentage)
            frame = table.assign(variant=np.where(is_control, "control", "treatment"))
            res, _ = ORCH._evaluate_pair_metric(frame, primary.definition(), "control", "treatment",
                                                variant_col="variant", covariate_col=covariate, alpha=passport.alpha)
            if res["p_value"] is not None and not np.isnan(res["p_value"]):
                p_values.append(res["p_value"])
        p = np.asarray(p_values)
        fpr = float(np.mean(p < passport.alpha)) if len(p) else float("nan")
        result["aa"] = {"runs": int(len(p)), "fpr": fpr, "ks_p": float(stats.kstest(p, "uniform").pvalue) if len(p) else None,
                        "range": list(PREFLIGHT_FPR_RANGE), "covariate": covariate}
        if not (PREFLIGHT_FPR_RANGE[0] <= fpr <= PREFLIGHT_FPR_RANGE[1]):
            problems.append(f"A/A на истории: доля ложных срабатываний {fpr:.1%} вне [3%; 7%]")
        # баланс настоящей соли: вариант гостя — по бакету соли правила, как в движке (шаг 2)
        z = ASG.buckets_for(rule_id(passport), ids)
        bounds = np.cumsum(allocate_variant_buckets([v.traffic_percentage for v in passport.variants]))
        variant = np.array([v.variant_id for v in passport.variants])[np.searchsorted(bounds, z, side="right")]
        smd = {}
        for _, m in metrics:
            for col in [c for c in (m.columns()[0], f"{m.name}__pre") if c in table.columns]:
                c_vals = table.loc[variant == control.variant_id, col].to_numpy(dtype=float)
                smd[col] = max(_smd(table.loc[variant == v.variant_id, col].to_numpy(dtype=float), c_vals)
                               for v in passport.variants if not v.is_control)
        result["balance"] = {"max_smd": max(smd.values()), "by_column": smd,  # primary в таблице всегда
                             "threshold": PREFLIGHT_MAX_SMD}
        bad = [c for c, v in smd.items() if v >= PREFLIGHT_MAX_SMD]
        if bad:
            problems.append(f"дисбаланс настоящей соли (SMD ≥ {PREFLIGHT_MAX_SMD}): {', '.join(bad)}")
    result.update({"passed": not problems, "problems": problems})
    run = AnalysisRun(run_id=inputs.run_id, inputs=inputs, input_hash=_table_hash(table), result=_plain(result),
                      created_at=_iso(service.clock()), actor=actor.subject)
    with service.storage.transaction():
        service.storage.save_analysis_run(run)
        service.storage.append_audit(actor.subject, "experiment.preflight", experiment_id,
                                     {"run_id": run.run_id, "passed": run.result["passed"],
                                      "passport_hash": passport.passport_hash})
    return run
