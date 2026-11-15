"""
Фаза 4, П4: запуски анализа, отчёт и воспроизводимость (RPT-1), мониторинг по расписанию с тревогами и автопаузой,
предстартовая проверка (A/A на истории и баланс).
"""
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import ab_platform.analysis as AN
from ab_platform.analysis import preflight, report_markdown, reproduce, run_analysis, run_due_analyses
from ab_platform.auth import Auth
from ab_platform.control import ControlError, ExperimentService
from ab_platform.engine.ruleset_engine import evaluate
from ab_platform.events import AppEventIn, ExposureIn, MemoryEventStore, OrderAdjustmentIn, OrderIn
from ab_platform.schemas.analysis import AnalysisRun
from ab_platform.server import create_app
from ab_platform.simulation.vit import simulate_app_world
from ab_platform.storage import MemoryStorage, SqliteStorage, StorageError
from tests.test_phase3_control import ADMIN, ANALYST, OWNER, OWNER2, draft
from tests.test_phase3_http import KEYS, SPEC, h

START = datetime(2026, 9, 1, tzinfo=timezone.utc)


@contextmanager
def denied(kind, text=None):
    with pytest.raises(ControlError, match=text) as e:
        yield
    assert e.value.kind == kind, (e.value.kind, str(e.value))


@pytest.fixture(scope="module")
def world():
    return simulate_app_world(7, 8000, START)


def load_events(world, ev):
    store = MemoryEventStore()
    store.add_orders([OrderIn(**o) for o in ev.payload("orders")])
    store.add_adjustments([OrderAdjustmentIn(**a) for a in ev.payload("adjustments")])
    store.add_app_events([AppEventIn(**e) for e in ev.payload("app_events")])
    return store


class Lab:
    """Тест приложения на синтетике: часы, контур, события; показы — через настоящий движок."""

    def __init__(self, world, storage=None, lift=0.0, crash_lift=0.0, drop_treatment_exposures=0.0,
                 require_preflight=False, **passport):
        self.world, self.now = world, world.test_start - timedelta(hours=1)
        self.svc = ExperimentService(storage or MemoryStorage(), clock=lambda: self.now,
                                     require_preflight=require_preflight)
        self.svc.ensure_default_metrics()
        self.svc.add_layer(ADMIN, "ui")
        fields = dict(targeting={}, targeting_audience_share=None, ramp_plan=[100], planned_duration_days=28)
        fields.update(passport)
        self.svc.save_draft(OWNER, draft(**fields))
        self.svc.submit(OWNER, "exp_menu")
        self.svc.approve(ANALYST, "exp_menu")
        self.lift, self.crash_lift, self.drop = lift, crash_lift, drop_treatment_exposures
        self.events = None

    def history(self):
        """События пре-периода (до старта) — для предстартовой проверки."""
        self.events = load_events(self.world, self.world.realize())
        return self.events

    def start(self):
        rs = self.svc.start(OWNER, "exp_menu")
        rule = rs.by_flag("flag_exp_menu")
        ex = self.world.exposure_times()
        decided = {u: evaluate(rs, rule, u) for u, t in ex.items() if not pd.isna(t)}
        treated = np.array([u in decided and decided[u].variant_id == "treatment" for u in ex.index])
        ev = self.world.realize(treated, lift=self.lift, crash_lift=self.crash_lift)
        self.events = load_events(self.world, ev)
        rng = np.random.default_rng(1)
        self.events.add([ExposureIn(event_id=f"x{i}", event_time=ex[u].isoformat(), unit_id=u,
                                    experiment_id=rule.experiment_id, variant_id=d.variant_id,
                                    ruleset_version=rs.version, unit_type="guest_id",
                                    attributes={"platform": self.world.guests.set_index("unit_id").at[u, "platform"]})
                         for i, (u, d) in enumerate(decided.items())
                         if d.is_exposed and not (d.variant_id == "treatment" and rng.random() < self.drop)])
        self.truth = ev.truth
        return rs

    def at(self, days):
        self.now = self.world.test_start + timedelta(days=days)
        return self.now


@pytest.fixture(scope="module")
def finished(world):
    """Тест с эффектом +30%: мониторинг на 10-й день, остановка на 28-й, итог на 36-й."""
    lab = Lab(world, lift=0.30)
    lab.start()
    lab.at(10)
    monitoring = run_due_analyses(lab.svc, lab.events)
    lab.at(28)
    lab.svc.stop(OWNER, "exp_menu")
    lab.at(36)
    final = run_due_analyses(lab.svc, lab.events)
    return lab, monitoring, final


# ---------------- запуски и отчёт ----------------
def test_monitoring_is_blinded_and_final_shows_effect(finished):
    lab, monitoring, final = finished
    assert [m["kind"] for m in monitoring] == ["monitoring"] and [f["kind"] for f in final] == ["final"]
    mon = lab.svc.storage.get_analysis_run(monitoring[0]["run_id"]).result
    assert mon["decision"] == "RUNNING_BLINDED" and mon["blinded"] and mon["analysis_day"] == 10
    assert all(c == {"blinded": True} for c in mon["comparisons"].values()) and mon["secondary"] == []
    assert mon["primary"] == {"blinded": True, "criterion": mon["primary"]["criterion"]}
    assert len(mon["guardrails"]) == 3 and mon["population"]["windows_complete"] < mon["population"]["exposed"]
    assert any("промежуточный срез" in c for c in mon["caveats"])
    fin = lab.svc.storage.get_analysis_run(final[0]["run_id"]).result
    assert not fin["blinded"] and fin["analysis_day"] is None and fin["population"]["windows_complete"] == fin["population"]["exposed"]
    t = fin["comparisons"]["treatment"]
    assert t["relative_diff"] > 0.1 and t["is_significant"]  # эффект 30% виден
    assert {m["ref"] for m in fin["metrics"]} == {"revenue_per_guest@1", "crash_share@1", "payment_error_share@1",
                                                  "cancel_rate@1"}
    assert fin["estimand"] == "triggered" and "platform" in fin["segments_srm"]
    assert [a.kind for a in lab.svc.storage.list_alerts("exp_menu")] == ["final_report_ready"]


def test_report_markdown_and_json(finished):
    lab, _, final = finished
    run = lab.svc.storage.get_analysis_run(final[0]["run_id"])
    md = report_markdown(run)
    assert run.run_id in md and "## Эффект (primary)" in md and "## Защитные метрики" in md
    assert "revenue_per_guest@1" in md and "triggered" in md and "Повторный расчёт" in md
    json.loads(run.result_json())  # строгий JSON: без NaN и бесконечностей
    mon = lab.svc.storage.list_analysis_runs("exp_menu", "monitoring")[0]
    assert "Скрыт до горизонта" in report_markdown(mon) and "Эффект (primary)" not in report_markdown(mon)


def test_reproduce_is_byte_identical_even_after_late_events(finished):
    lab, _, final = finished
    run_id = final[0]["run_id"]
    assert reproduce(lab.svc, lab.events, run_id)["identical"]
    # события, пришедшие после запуска, в срез не попадают — повтор тот же
    lab.events.add_orders([OrderIn(order_id="late", unit_id=lab.world.guests["unit_id"][0], unit_type="guest_id",
                                   order_time=(lab.world.test_start + timedelta(days=1)).isoformat(), amount=99999.0)])
    res = reproduce(lab.svc, lab.events, run_id)
    assert res["identical"] and res["same_code_version"]
    # MET-1: новая версия метрики не меняет сохранённый отчёт — он ссылается на метрика@1
    saved = lab.svc.storage.get_analysis_run(run_id).result
    v2 = lab.svc.define_metric(ANALYST, {"name": "revenue_per_guest", "title": "Выручка, винзоризация 95%",
                                         "formula": "revenue", "winsorize_percentile": 0.95})
    assert v2.ref == "revenue_per_guest@2"
    assert lab.svc.storage.get_analysis_run(run_id).result == saved
    assert "revenue_per_guest@1" in {m["ref"] for m in saved["metrics"]}
    assert reproduce(lab.svc, lab.events, run_id)["identical"]
    # тот же запуск ещё раз — тот же run_id, без нового сохранения
    again = run_analysis(lab.svc, lab.events, "exp_menu", "final", lab.now,
                         upto_seq=lab.svc.storage.get_analysis_run(run_id).inputs.upto_seq)
    assert again.run_id == run_id and len(lab.svc.storage.list_analysis_runs("exp_menu", "final")) == 1
    with denied("not_found"):
        reproduce(lab.svc, lab.events, "нет")


def test_final_only_after_stop_and_closed_windows(world):
    lab = Lab(world)
    lab.start()
    lab.at(5)
    with denied("conflict", "только после остановки"):
        run_analysis(lab.svc, lab.events, "exp_menu", "final")
    lab.at(28)
    lab.svc.stop(OWNER, "exp_menu")
    lab.at(33)
    with denied("conflict", "Окна атрибуции закроются"):
        run_analysis(lab.svc, lab.events, "exp_menu", "final")
    with denied("conflict", "Мониторинг — только идущего"):
        run_analysis(lab.svc, lab.events, "exp_menu", "monitoring")
    assert run_due_analyses(lab.svc, lab.events) == []  # итог ещё не готов
    lab.at(35)
    assert [d["kind"] for d in run_due_analyses(lab.svc, lab.events)] == ["final"]
    assert run_due_analyses(lab.svc, lab.events) == []  # итог — один раз
    with denied("invalid"):
        run_analysis(lab.svc, lab.events, "exp_menu", "weekly")


def test_monitoring_once_per_day_and_failures_alert(world):
    lab = Lab(world)
    lab.start()
    lab.at(3)
    assert len(run_due_analyses(lab.svc, lab.events)) == 1 and run_due_analyses(lab.svc, lab.events) == []
    lab.at(3.5)
    assert run_due_analyses(lab.svc, lab.events) == []  # тот же день
    lab.at(4)
    assert len(run_due_analyses(lab.svc, lab.events)) == 1

    class Broken(MemoryEventStore):
        def first_exposures(self, *a, **k):
            raise ValueError("хранилище событий недоступно")
    lab.at(5)
    out = run_due_analyses(lab.svc, Broken())
    assert out[0]["kind"] == "error" and "недоступно" in out[0]["error"]
    assert lab.svc.storage.list_alerts("exp_menu")[-1].kind == "analysis_failed"
    assert lab.svc.get("exp_menu").status.value == "running"  # сбой анализа тест не трогает


def test_insufficient_data_before_exposures(world):
    lab = Lab(world)
    lab.svc.start(OWNER, "exp_menu")
    lab.at(1)
    run = run_analysis(lab.svc, MemoryEventStore(), "exp_menu")
    assert run.result["decision"] == "INSUFFICIENT_DATA" and run.result["population"]["exposed"] == 0


# ---------------- тревоги и автопауза ----------------
def test_auto_pause_on_proven_guardrail_breach(world):
    lab = Lab(world, crash_lift=6.0)  # падений в варианте в 7 раз больше
    lab.start()
    paused_on = None
    for day in range(1, 28):
        lab.at(day)
        run_due_analyses(lab.svc, lab.events)
        if lab.svc.get("exp_menu").status.value == "paused":
            paused_on = day
            break
    assert paused_on is not None and paused_on < 20, paused_on
    alert = lab.svc.storage.list_alerts("exp_menu")[-1]
    assert alert.kind == "guardrail_auto_pause" and "crash_share" in alert.message
    pause = [r for r in lab.svc.storage.audit("exp_menu") if r.action == "experiment.pause"]
    assert len(pause) == 1 and pause[0].actor == "scheduler"
    assert lab.svc.ruleset().by_flag("flag_exp_menu").kill_switch  # гости получают контроль
    run = lab.svc.storage.get_analysis_run(alert.run_id).result
    assert run["auto_pause"]["alpha_per_look"] == pytest.approx(0.05 / 28)
    lab.at(paused_on + 1)
    assert run_due_analyses(lab.svc, lab.events) == []  # на паузе мониторинг не идёт и не снимает паузу


def test_no_pause_and_no_srm_alert_in_aa(world):
    lab = Lab(world)
    lab.start()
    for day in range(1, 29):
        lab.at(day)
        run_due_analyses(lab.svc, lab.events)
    assert lab.svc.get("exp_menu").status.value == "running" and lab.svc.storage.list_alerts("exp_menu") == []
    assert len(lab.svc.storage.list_analysis_runs("exp_menu", "monitoring")) == 28


def test_srm_alert_when_treatment_exposures_are_lost(world):
    lab = Lab(world, drop_treatment_exposures=0.15)
    lab.start()
    lab.at(10)
    out = run_due_analyses(lab.svc, lab.events)
    run = lab.svc.storage.get_analysis_run(out[0]["run_id"]).result
    assert run["srm_detected"] and not run["can_trust_data"]
    assert run["decision"] == "INVALID_EXPERIMENT_SRM_OR_CONTAMINATION"
    assert [a.kind for a in lab.svc.storage.list_alerts("exp_menu")] == ["srm"]
    assert lab.svc.get("exp_menu").status.value == "running"  # SRM — тревога и блок анализа, не пауза


# ---------------- предстартовая проверка ----------------
@pytest.fixture(scope="module")
def checked(world):
    lab = Lab(world, require_preflight=True)
    lab.history()
    return lab, preflight(lab.svc, lab.events, "exp_menu", OWNER)


def test_preflight_passes_on_history_and_unblocks_start(checked):
    lab, run = checked
    r = run.result
    assert r["passed"] and r["problems"] == [] and r["n_guests"] > 1000
    assert r["aa"]["runs"] == 500 and 0.03 <= r["aa"]["fpr"] <= 0.07 and r["balance"]["max_smd"] < 0.1
    assert run.inputs.passport_hash == lab.svc.get("exp_menu").passport_hash and run.inputs.kind == "preflight"
    assert preflight(lab.svc, lab.events, "exp_menu", ANALYST).run_id == run.run_id  # те же входы — тот же запуск
    audit = [x for x in lab.svc.storage.audit("exp_menu") if x.action == "experiment.preflight"]
    assert len(audit) == 1 and audit[0].details["passed"]
    assert lab.svc.start(OWNER, "exp_menu").by_flag("flag_exp_menu") is not None


def test_start_requires_passed_preflight_of_this_passport(world, monkeypatch):
    lab = Lab(world, require_preflight=True)
    with denied("conflict", "Нет предстартовой проверки"):
        lab.svc.start(OWNER, "exp_menu")
    with denied("forbidden"):
        preflight(lab.svc, MemoryEventStore(), "exp_menu", OWNER2)
    empty = preflight(lab.svc, MemoryEventStore(), "exp_menu", OWNER)
    assert not empty.result["passed"] and "мало истории" in empty.result["problems"][0]
    with denied("conflict", "не пройдена: мало истории"):
        lab.svc.start(OWNER, "exp_menu")
    monkeypatch.setattr(AN, "PREFLIGHT_MAX_SMD", 1e-9)  # порог, который не пройдёт ни одна соль
    lab.history()
    bad = preflight(lab.svc, lab.events, "exp_menu", OWNER, n_salts=20)
    assert not bad.result["passed"] and any("дисбаланс" in p for p in bad.result["problems"])
    monkeypatch.setattr(AN, "PREFLIGHT_MAX_SMD", 0.1)
    assert preflight(lab.svc, lab.events, "exp_menu", OWNER).result["passed"]
    # новая версия Паспорта — новый хэш: проверка прежней версии к ней не относится
    lab.svc.new_version(OWNER, "exp_menu", {"mde_relative": 0.05})
    lab.svc.submit(OWNER, "exp_menu")
    lab.svc.approve(ANALYST, "exp_menu")
    with denied("conflict", "Нет предстартовой проверки"):
        lab.svc.start(OWNER, "exp_menu")
    assert preflight(lab.svc, lab.events, "exp_menu", OWNER).inputs.passport_version == 2
    assert lab.svc.start(OWNER, "exp_menu").by_flag("flag_exp_menu").experiment_id == "exp_menu@v2"


def test_preflight_only_for_approved_passport(world):
    lab = Lab(world)
    lab.svc.start(OWNER, "exp_menu")
    with denied("conflict", "для утверждённого Паспорта"):
        preflight(lab.svc, MemoryEventStore(), "exp_menu", OWNER)


# ---------------- хранилище запусков ----------------
@pytest.fixture(params=["memory", "sqlite"])
def storage(request, tmp_path):
    if request.param == "memory":
        yield MemoryStorage()
    else:
        s = SqliteStorage(str(tmp_path / "ab.db"))
        yield s
        s.close()


def test_run_storage_contract(storage, finished):
    lab, _, final = finished
    run = lab.svc.storage.get_analysis_run(final[0]["run_id"])
    storage.save_analysis_run(run)
    storage.save_analysis_run(run)  # тот же запуск — не дубль
    assert storage.get_analysis_run(run.run_id) == run and len(storage.list_analysis_runs()) == 1
    other = run.model_copy(update={"result": {**run.result, "decision": "SHIP_CANDIDATE"}})
    with pytest.raises(StorageError, match="другим результатом"):
        storage.save_analysis_run(other)
    with pytest.raises(StorageError, match="не совпадает с хэшем входов"):
        storage.save_analysis_run(run.model_copy(update={"run_id": "0" * 24}))
    assert storage.list_analysis_runs("exp_menu", "final") == [run] and storage.list_analysis_runs("x") == []
    assert storage.list_analysis_runs(kind="monitoring") == [] and storage.get_analysis_run("нет") is None
    alert = lab.svc.storage.list_alerts("exp_menu")[0]
    storage.save_alert(alert)
    assert storage.list_alerts() == [alert] and storage.list_alerts("другой") == []


def test_sqlite_runs_survive_restart(tmp_path, finished):
    lab, _, final = finished
    run = lab.svc.storage.get_analysis_run(final[0]["run_id"])
    s = SqliteStorage(str(tmp_path / "ab.db"))
    s.save_analysis_run(run)
    s.close()
    again = SqliteStorage(str(tmp_path / "ab.db"))
    back = again.get_analysis_run(run.run_id)
    assert back == run and back.result_json() == run.result_json()
    again.close()


# ---------------- HTTP ----------------
def test_http_runs_reports_alerts_and_roles(finished):
    lab, _, final = finished
    app = create_app(exposure_store=lab.events, service=lab.svc, auth=Auth.parse(SPEC))
    http = TestClient(app)
    run_id = final[0]["run_id"]
    runs = http.get("/analysis-runs", params={"experiment_id": "exp_menu"}, headers=h("guest")).json()
    assert {r["kind"] for r in runs} == {"monitoring", "final"} and all("decision" in r for r in runs)
    full = http.get(f"/analysis-runs/{run_id}", headers=h("guest")).json()
    assert AnalysisRun.model_validate(full).result_json() == lab.svc.storage.get_analysis_run(run_id).result_json()
    md = http.get(f"/analysis-runs/{run_id}/report.md", headers=h("guest"))
    assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown") and run_id in md.text
    assert http.get("/analysis-runs/нет", headers=h("guest")).status_code == 404
    assert http.get(f"/analysis-runs/{run_id}", headers=h("sdk")).status_code == 403
    for who in ("guest", "ivanova"):
        assert http.post(f"/analysis-runs/{run_id}/reproduce", headers=h(who)).status_code == 403
        assert http.post("/scheduler/run", headers=h(who)).status_code == 403
        assert http.post("/analysis-runs", json={"experiment_id": "exp_menu", "kind": "final"},
                         headers=h(who)).status_code == 403
    assert http.post(f"/analysis-runs/{run_id}/reproduce", headers=h("petrov")).json()["identical"]
    assert http.post("/scheduler/run", headers=h("root")).json() == []  # итог уже есть
    assert http.post("/analysis-runs", json={"experiment_id": "exp_menu", "kind": "final"},
                     headers=h("petrov")).json()["kind"] == "final"
    assert [a["kind"] for a in http.get("/alerts", headers=h("guest")).json()] == ["final_report_ready"]
    assert http.post("/analysis-runs/нет/reproduce", headers=h("petrov")).status_code == 404
    for path in ("/analysis-runs", "/alerts", "/metrics"):  # списки — только людям, не ключу SDK
        assert http.get(path, headers=h("sdk")).status_code == 403 and http.get(path).status_code == 401
    assert http.get("/health").json()["control_plane"] is True and http.get("/health").json()["preflight"] == (
        "required" if lab.svc.require_preflight else "disabled")
    bare = TestClient(create_app(exposure_store=lab.events, auth=Auth.parse(SPEC))).get("/health").json()
    assert bare["control_plane"] is False and bare["preflight"] is None


def test_http_preflight_command(world):
    lab = Lab(world, require_preflight=True)
    lab.history()
    http = TestClient(create_app(exposure_store=lab.events, service=lab.svc, auth=Auth.parse(SPEC)))
    assert http.post("/passports/exp_menu/start", headers=h("ivanova")).status_code == 409
    assert http.post("/passports/exp_menu/preflight", headers=h("sidorov")).status_code == 403
    r = http.post("/passports/exp_menu/preflight", headers=h("ivanova")).json()
    assert r["kind"] == "preflight" and r["passed"] is True and r["problems"] == []
    assert http.get(f"/analysis-runs/{r['run_id']}/report.md", headers=h("guest")).status_code == 404
    assert http.post("/passports/exp_menu/start", headers=h("ivanova")).json()["status"] == "running"


# ---------------- закрытие выживших мутантов ----------------
def test_secondary_hidden_while_blinded_and_shown_in_final(world):
    from ab_platform.metrics import DEFAULT_METRICS
    conv = next(m for m in DEFAULT_METRICS if m.name == "conversion")
    lab = Lab(world, lift=0.30, secondary_metrics=[conv.definition().model_dump()])
    lab.start()
    lab.at(10)
    mon = lab.svc.storage.get_analysis_run(run_due_analyses(lab.svc, lab.events)[0]["run_id"])
    assert mon.result["blinded"] and mon.result["secondary"] == []
    assert "Вторичные метрики" not in report_markdown(mon)
    lab.at(28)
    lab.svc.stop(OWNER, "exp_menu")
    lab.at(36)
    fin = lab.svc.storage.get_analysis_run(run_due_analyses(lab.svc, lab.events)[0]["run_id"])
    assert {s["metric_name"] for s in fin.result["secondary"]} == {"conversion"}
    s = fin.result["secondary"][0]
    row = (f"| {s['variant_id']} | conversion | {s['relative_diff_pct']:+.2f}% | {AN._num(s['p_value'])} | "
           f"{'да' if s['is_significant_fdr'] else 'нет'} |\n\n## SRM и данные")
    assert "## Вторичные метрики" in report_markdown(fin) and row in report_markdown(fin)


def test_same_inputs_are_not_recomputed(finished, monkeypatch):
    lab, _, final = finished
    run = lab.svc.storage.get_analysis_run(final[0]["run_id"])

    def boom(*a, **k):
        raise AssertionError("повтор с теми же входами не должен пересчитываться")
    monkeypatch.setattr(AN, "_compute", boom)
    assert run_analysis(lab.svc, lab.events, "exp_menu", "final", lab.now, upto_seq=run.inputs.upto_seq) == run


def test_reproduce_detects_changed_result(finished, monkeypatch):
    lab, _, final = finished
    run_id = final[0]["run_id"]
    original = AN._compute

    def tampered(*a, **k):
        result, input_hash = original(*a, **k)
        return {**result, "decision": "SHIP_CANDIDATE"}, input_hash
    monkeypatch.setattr(AN, "_compute", tampered)
    assert not reproduce(lab.svc, lab.events, run_id)["identical"]

    def other_table(*a, **k):
        result, _ = original(*a, **k)
        return result, "0" * 64
    monkeypatch.setattr(AN, "_compute", other_table)
    assert not reproduce(lab.svc, lab.events, run_id)["identical"]


def test_alert_is_not_duplicated(world):
    lab = Lab(world)
    AN._alert(lab.svc, "exp_menu", "srm", "перекос", "run1")
    AN._alert(lab.svc, "exp_menu", "srm", "перекос", "run1")
    AN._alert(lab.svc, "exp_menu", "analysis_failed", "сбой", None)
    AN._alert(lab.svc, "exp_menu", "analysis_failed", "сбой", None)  # те же часы — та же тревога
    assert [a.kind for a in lab.svc.storage.list_alerts("exp_menu")] == ["srm", "analysis_failed"]


def test_preflight_fails_when_aa_fpr_out_of_range(world, monkeypatch):
    lab = Lab(world, require_preflight=True)
    lab.history()
    monkeypatch.setattr(AN, "PREFLIGHT_FPR_RANGE", (0.5, 0.6))  # недостижимый диапазон
    bad = preflight(lab.svc, lab.events, "exp_menu", OWNER, n_salts=20)
    assert not bad.result["passed"] and any("A/A на истории" in p for p in bad.result["problems"])
    with denied("conflict", "A/A на истории"):
        lab.svc.start(OWNER, "exp_menu")


# ---------------- граничные случаи (по итогам сплошной проверки порчей) ----------------
def test_preflight_thresholds_match_spec():
    """Пороги предстартовой проверки — из дизайн-заметки фазы 4: менять только вместе с ней."""
    assert AN.PREFLIGHT_SALTS == 500 and AN.PREFLIGHT_FPR_RANGE == (0.03, 0.07)
    assert AN.PREFLIGHT_MAX_SMD == 0.1 and AN.PREFLIGHT_MIN_UNITS == 200


def test_code_version_sources(monkeypatch):
    import subprocess

    class Out:
        def __init__(self, returncode, stdout):
            self.returncode, self.stdout = returncode, stdout
    calls = []

    def fake(result):
        def run(cmd, **kw):
            calls.append((cmd, kw))
            if isinstance(result, Exception):
                raise result
            return result
        return run
    monkeypatch.delenv("ABP_CODE_VERSION", raising=False)
    try:
        for result, expected in ((Out(0, "abc123-dirty\n"), "abc123-dirty"), (Out(128, "abc"), "unknown"),
                                 (Out(0, "  \n"), "unknown"), (OSError("нет git"), "unknown"),
                                 (subprocess.TimeoutExpired("git", 5), "unknown")):
            AN.code_version.cache_clear()
            monkeypatch.setattr(AN.subprocess, "run", fake(result))
            assert AN.code_version() == expected, result
        cmd, kw = calls[0]
        assert "--dirty" in cmd and kw["capture_output"] and kw["text"] and kw["timeout"] > 0
        monkeypatch.setenv("ABP_CODE_VERSION", " build-42 ")
        AN.code_version.cache_clear()
        assert AN.code_version() == "build-42"
        monkeypatch.setenv("ABP_CODE_VERSION", "  ")  # пустая переменная — берётся git
        monkeypatch.setattr(AN.subprocess, "run", fake(Out(0, "def456\n")))
        AN.code_version.cache_clear()
        assert AN.code_version() == "def456"
    finally:
        AN.code_version.cache_clear()


def test_metrics_of_requires_registry_and_dedupes(world):
    lab = Lab(world)
    p = lab.svc.get("exp_menu")
    plain = AN._metrics_of(lab.svc, p)
    assert [r for r, _ in plain] == ["primary", "guardrail", "guardrail", "guardrail"]
    dup = p.model_copy(update={"secondary_metrics": [p.primary_metric]})  # та же метрика второй раз — одна колонка
    assert [m.name for _, m in AN._metrics_of(lab.svc, dup)] == [m.name for _, m in plain]
    bare = p.model_copy(update={"primary_metric": p.primary_metric.model_copy(update={"registry_ref": None})})
    with denied("invalid", "не из реестра"):
        AN._metrics_of(lab.svc, bare)


def test_analysis_of_not_started_test_is_refused(world):
    lab = Lab(world)
    with denied("conflict", "не запущен"):
        run_analysis(lab.svc, MemoryEventStore(), "exp_menu")


def test_segment_srm_only_for_attributes_with_two_values(world):
    lab = Lab(world)
    n = 400
    table = pd.DataFrame({"unit_id": [f"g{i}" for i in range(n)], "variant": ["control", "treatment"] * (n // 2),
                          "attr_platform": ["ios", "ios", "android", "android"] * (n // 4), "attr_city": ["msk"] * n})
    out = AN._segments(table, lab.svc.get("exp_menu"), 0.001)
    assert set(out) == {"platform"} and set(out["platform"]["segments"]) == {"ios", "android"}
    assert not out["platform"]["srm_detected"]


def test_analysis_day_counts_full_days_from_start(world):
    lab = Lab(world)
    lab.start()
    started = AN._utc(lab.svc.lifecycle_times("exp_menu", 1)["started_at"])
    for delta, day in ((timedelta(days=3), 3), (timedelta(days=3, microseconds=-1), 2)):
        run = run_analysis(lab.svc, lab.events, "exp_menu", cutoff=started + delta, save=False)
        assert run.result["analysis_day"] == day


def test_caveats_and_covariate(finished, world):
    lab, monitoring, final = finished
    mon = lab.svc.storage.get_analysis_run(monitoring[0]["run_id"]).result
    fin = lab.svc.storage.get_analysis_run(final[0]["run_id"]).result
    assert fin["covariate"] == mon["covariate"] == "revenue_per_guest__pre"
    assert any("скрыт" in c for c in mon["caveats"]) and not any("скрыт" in c for c in fin["caveats"])
    assert not any("промежуточный срез" in c for c in fin["caveats"])
    assert any("Порог практической значимости" in c for c in fin["caveats"])  # порог 0 — используется MDE
    assert mon["winning_variant"] is None  # до горизонта победитель скрыт
    # без CUPED и с заданным порогом
    other = Lab(world, use_cuped=False, practical_significance_threshold=0.02)
    other.start()
    other.at(3)
    res = run_analysis(other.svc, other.events, "exp_menu", save=False).result
    assert res["covariate"] is None and not any("Порог практической значимости" in c for c in res["caveats"])


def test_two_guests_per_variant_are_enough(world):
    lab = Lab(world)
    rs = lab.svc.start(OWNER, "exp_menu")
    rule = rs.by_flag("flag_exp_menu")
    lab.at(1)

    def events(variants, orders=True):
        store = MemoryEventStore()
        store.add([ExposureIn(event_id=f"e{i}", event_time=(lab.world.test_start + timedelta(hours=i)).isoformat(),
                              unit_id=f"u{i}", experiment_id=rule.experiment_id, variant_id=v,
                              ruleset_version=rs.version, unit_type="guest_id") for i, v in enumerate(variants)])
        if orders:
            store.add_orders([OrderIn(order_id=f"o{i}", unit_id=f"u{i}", unit_type="guest_id", amount=300.0 + 50 * i,
                                      order_time=(lab.world.test_start + timedelta(hours=i + 2)).isoformat())
                              for i in range(len(variants))])
        return store
    few = run_analysis(lab.svc, events(["control", "treatment", "treatment"]), "exp_menu", save=False).result
    assert few["decision"] == "INSUFFICIENT_DATA" and few["can_trust_data"] is False and few["blinded"] is True
    assert few["population"]["by_variant"] == {"control": 1, "treatment": 2}
    enough = run_analysis(lab.svc, events(["control", "treatment"] * 2), "exp_menu", save=False).result
    assert enough["decision"] != "INSUFFICIENT_DATA" and enough["population"]["by_variant"] == {"control": 2,
                                                                                               "treatment": 2}
    # показы есть, заказов нет: у ratio-метрики (доля отмен) знаменатель 0 — мало данных, а не сбой анализа
    no_orders = run_analysis(lab.svc, events(["control", "treatment"] * 2, orders=False), "exp_menu", save=False)
    assert no_orders.result["decision"] == "INSUFFICIENT_DATA" and "cancel_rate" in no_orders.result["summary"]


def test_runs_are_audited(finished):
    lab, monitoring, final = finished
    audit = {(r.action, r.details.get("run_id")) for r in lab.svc.storage.audit("exp_menu")}
    assert ("analysis.monitoring", monitoring[0]["run_id"]) in audit and ("analysis.final", final[0]["run_id"]) in audit


def test_reproduce_refuses_preflight(checked):
    lab, run = checked
    with denied("invalid", "новым запуском"):
        reproduce(lab.svc, lab.events, run.run_id)


def test_reproduce_after_archive(world):
    """В архиве правило убрано из ruleset — повтор строит его из Паспорта и совпадает байт в байт."""
    lab = Lab(world)
    lab.start()
    lab.at(28)
    lab.svc.stop(OWNER, "exp_menu")
    lab.at(35)
    run_id = run_due_analyses(lab.svc, lab.events)[0]["run_id"]
    for to in ("analysis", "decision", "archived"):
        lab.svc.advance(ANALYST, "exp_menu", to)
    assert lab.svc.ruleset().by_flag("flag_exp_menu") is None
    assert reproduce(lab.svc, lab.events, run_id)["identical"]


def test_report_markdown_details(finished):
    lab, monitoring, final = finished
    run = lab.svc.storage.get_analysis_run(final[0]["run_id"])
    md, r = report_markdown(run), run.result
    mon_md = report_markdown(lab.svc.storage.get_analysis_run(monitoring[0]["run_id"]))
    assert md.startswith("# Отчёт: exp_menu — итоговый") and mon_md.startswith("# Отчёт: exp_menu — промежуточный")
    c = r["comparisons"]["treatment"]
    lo, hi = c["ci_lower"] / c["mean_control"], c["ci_upper"] / c["mean_control"]
    assert f"| treatment | {AN._pct(c['relative_diff'])} | {AN._pct(lo)} … {AN._pct(hi)} | {AN._num(c['p_value'])} | да |" in md
    for g in r["guardrails"]:
        assert f"| {g['variant_id']} | {g['metric_name']} | {g['relative_change_pct']:+.2f}% |" in md
    assert "|\n\n## Защитные метрики" in md and "|\n\n## SRM и данные" in md
    assert "Данным можно доверять: да." in md and "- SRM по сегментам platform: нет" in md
    assert "Пробой guardrail" not in md
    assert f"хэш `{r['passport']['hash'][:16]}…`" in md and f"входная таблица: `{r['snapshot']['input_hash'][:16]}…`" in md
    # нет ДИ — прочерки; перекос в сегменте — назван
    bare = {**r, "comparisons": {"treatment": {**c, "ci_lower": None}},
            "segments_srm": {"platform": {**r["segments_srm"]["platform"], "segments": {
                "ios": {"srm_detected": True}, "android": {"srm_detected": False}}}}}
    md2 = report_markdown(run.model_copy(update={"result": bare}))
    assert "| — … — |" in md2 and "- SRM по сегментам platform: перекос в ios" in md2
    mon = lab.svc.storage.get_analysis_run(monitoring[0]["run_id"]).result
    open_n = mon["population"]["exposed"] - mon["population"]["windows_complete"]
    assert f"Окно атрибуции не закрыто у {open_n} гостей: промежуточный срез." in mon["caveats"]


def test_report_number_format():
    assert AN._pct(None) == AN._num(None) == "—"
    assert AN._pct(0.1234) == "+12.34%" and AN._pct(-0.05) == "-5.00%"
    assert AN._num(1.234567) == "1.235" and AN._num(1.234567, 2) == "1.2"


def test_report_shows_pause_and_untrusted_data(world):
    lab = Lab(world, crash_lift=6.0)
    lab.start()
    for day in range(1, 28):
        lab.at(day)
        run_due_analyses(lab.svc, lab.events)
        if lab.svc.get("exp_menu").status.value == "paused":
            break
    run = lab.svc.storage.get_analysis_run(lab.svc.storage.list_alerts("exp_menu")[-1].run_id)
    assert "- Пробой guardrail на уровне α/K: crash_share" in report_markdown(run)
    srm = Lab(world, drop_treatment_exposures=0.15)
    srm.start()
    srm.at(10)
    out = run_due_analyses(srm.svc, srm.events)
    assert "Данным можно доверять: НЕТ." in report_markdown(srm.svc.storage.get_analysis_run(out[0]["run_id"]))


def test_smd_edge_cases():
    import math
    one, two = np.array([1.0]), np.array([1.0, 3.0])
    assert AN._smd(one, two) == 0.0 and AN._smd(two, one) == 0.0  # меньше двух значений — дисперсии нет
    assert AN._smd(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
    assert AN._smd(np.array([0.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(2 / math.sqrt(2))


def test_preflight_skips_missing_p_values(world, monkeypatch):
    lab = Lab(world, require_preflight=True)
    lab.history()
    p_values = iter([None, float("nan"), 0.05, 0.01] * 10)  # p = α не считается срабатыванием
    monkeypatch.setattr(AN.ORCH, "_evaluate_pair_metric", staticmethod(lambda *a, **k: ({"p_value": next(p_values)}, {})))
    aa = preflight(lab.svc, lab.events, "exp_menu", OWNER, n_salts=8).result["aa"]
    assert aa["runs"] == 4 and aa["fpr"] == 0.5 and 0.0 <= aa["ks_p"] <= 1.0
    p_values = iter([None, float("nan")] * 10)
    r = preflight(lab.svc, lab.events, "exp_menu", OWNER, n_salts=6).result
    assert r["aa"]["runs"] == 0 and r["aa"]["fpr"] is None and r["aa"]["ks_p"] is None and not r["passed"]


def test_preflight_min_units_boundary_and_covariate(world, monkeypatch):
    lab = Lab(world, require_preflight=True)
    lab.history()
    real = AN.build_history_table
    for n, enough in ((200, True), (199, False)):
        monkeypatch.setattr(AN, "build_history_table", lambda *a, n=n, **k: real(*a, **k).head(n))
        r = preflight(lab.svc, lab.events, "exp_menu", OWNER, n_salts=n - 190).result  # n_salts разный — разный запуск
        assert (not any("мало истории" in p for p in r["problems"])) == enough, (n, r["problems"])
    monkeypatch.setattr(AN, "build_history_table", real)
    assert preflight(lab.svc, lab.events, "exp_menu", OWNER, n_salts=5).result["aa"]["covariate"] == \
        "revenue_per_guest__pre"
    plain = Lab(world, require_preflight=True, use_cuped=False)
    plain.history()
    assert preflight(plain.svc, plain.events, "exp_menu", OWNER, n_salts=5).result["aa"]["covariate"] is None


def test_scheduler_skips_rules_without_passport(world):
    """Правило без Паспорта (наследие фазы 2) не анализируется и не ломает проход."""
    lab = Lab(world)
    lab.start()
    lab.svc.add_layer(ADMIN, "algo")
    rs = lab.svc.ruleset()
    lab.at(2)
    legacy = rs.by_flag("flag_exp_menu").model_copy(update={
        "experiment_id": "legacy_x", "flag_key": "flag_legacy", "salt": "legacy_x", "layer_id": "algo",
        "passport_id": None, "passport_version": None})
    lab.svc.storage.save_ruleset(rs.next_version(experiments=[*(e.model_dump() for e in rs.experiments),
                                                              legacy.model_dump()]))
    assert [d["experiment_id"] for d in run_due_analyses(lab.svc, lab.events)] == ["exp_menu"]


def test_final_waits_for_maturation(world):
    lab = Lab(world, maturation_days=2, attribution_window_days=7)
    lab.start()
    lab.at(28)
    lab.svc.stop(OWNER, "exp_menu")
    stopped = AN._utc(lab.svc.lifecycle_times("exp_menu", 1)["stopped_at"])
    lab.now = stopped + timedelta(days=7 + 2) - timedelta(microseconds=1)
    with denied("conflict", "Окна атрибуции закроются"):
        run_analysis(lab.svc, lab.events, "exp_menu", "final")
    assert run_due_analyses(lab.svc, lab.events) == []
    lab.now = stopped + timedelta(days=7 + 2)
    assert [d["kind"] for d in run_due_analyses(lab.svc, lab.events)] == ["final"]


def test_failure_alerts_without_run_are_kept_apart(world):
    lab = Lab(world)
    AN._alert(lab.svc, "exp_menu", "analysis_failed", "сбой", None)
    lab.now += timedelta(days=1)
    AN._alert(lab.svc, "exp_menu", "analysis_failed", "сбой", None)
    assert len(lab.svc.storage.list_alerts("exp_menu")) == 2


def test_preflight_smd_threshold_is_inclusive(world, monkeypatch):
    lab = Lab(world, require_preflight=True)
    lab.history()
    smd = preflight(lab.svc, lab.events, "exp_menu", OWNER, n_salts=3).result["balance"]["max_smd"]
    monkeypatch.setattr(AN, "PREFLIGHT_MAX_SMD", smd)  # SMD, равный порогу, — уже дисбаланс
    r = preflight(lab.svc, lab.events, "exp_menu", OWNER, n_salts=4).result
    assert any("дисбаланс" in p for p in r["problems"])


def test_run_schemas_are_canonical_and_frozen(finished):
    from ab_platform.schemas.analysis import Alert, canonical_json
    from pydantic import ValidationError
    assert canonical_json({"б": 1, "a": [1.5, None]}) == '{"a":[1.5,null],"б":1}'
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    lab, _, final = finished
    run = lab.svc.storage.get_analysis_run(final[0]["run_id"])
    assert len(run.run_id) == 24 and run.run_id == run.inputs.run_id
    alert = lab.svc.storage.list_alerts("exp_menu")[0]
    assert isinstance(alert, Alert)
    for obj, field in ((run, "actor"), (run.inputs, "kind"), (alert, "message")):
        with pytest.raises(ValidationError):
            setattr(obj, field, "x")
