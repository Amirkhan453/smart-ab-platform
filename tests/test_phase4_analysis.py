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
    assert "## Вторичные метрики" in report_markdown(fin)


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
