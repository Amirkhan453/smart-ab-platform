"""
Фаза 4, П3: реестр метрик (MET-1, MET-2) и таблица «гость × метрики» с привязкой к первому показу (CNV-2).
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ab_platform.control import Actor, ControlError, ExperimentService
from ab_platform.engine.unit_table import build_unit_table
from ab_platform.events import (
    AppEventIn, ExposureIn, MemoryEventStore, OrderAdjustmentIn, OrderIn, SqliteEventStore,
)
from ab_platform.metrics import APP_GUARDRAILS, DEFAULT_METRICS, MetricVersion, parse_ref
from ab_platform.schemas.passport import ExperimentPassport, MetricType
from ab_platform.simulation.vit import simulate_app_world
from ab_platform.storage import MemoryStorage, SqliteStorage, StorageError
from tests.test_phase1_passport_v4 import base
from tests.test_phase3_control import ADMIN, ANALYST, OWNER, approved, draft

REG = {m.name: m for m in DEFAULT_METRICS}
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


@contextmanager
def denied(kind, text=None):
    with pytest.raises(ControlError, match=text) as e:
        yield
    assert e.value.kind == kind, (e.value.kind, str(e.value))


# ---------------- версии метрик ----------------
def test_metric_version_rules():
    m = REG["avg_check"]
    assert m.ref == "avg_check@1" and m.metric_type == MetricType.RATIO
    assert m.columns() == ("avg_check__num", "avg_check__den")
    d = m.definition()
    assert (d.numerator_col, d.denominator_col, d.registry_ref) == ("avg_check__num", "avg_check__den", "avg_check@1")
    assert REG["crash_share"].definition().expected_direction.value == "decrease"
    # хэш определения не зависит от владельца, заголовка и описания
    other = m.model_copy(update={"owner": "petrov", "title": "Средний чек (другое имя)", "description": "x"})
    assert other.definition_hash == m.definition_hash
    assert m.model_copy(update={"formula": "revenue"}).definition_hash != m.definition_hash
    assert REG["crash_share"].definition_hash != REG["payment_error_share"].definition_hash  # отличаются event_type
    base_kw = dict(name="x_metric", version=1, title="Метрика", owner="petrov")
    for bad in (dict(formula="has_app_event"), dict(formula="has_order", event_type="app_crash"),
                dict(formula="has_order", winsorize_percentile=0.99), dict(formula="sql"), dict(formula="revenue",
                name="Bad-Name"), dict(formula="revenue", version=0)):
        with pytest.raises(ValidationError):
            MetricVersion(**{**base_kw, **bad})
    assert parse_ref("revenue_per_guest@12") == ("revenue_per_guest", 12)
    for bad in ("revenue", "revenue@0", "Revenue@1", "revenue@1@2", ""):
        with pytest.raises(ValueError):
            parse_ref(bad)
    assert set(APP_GUARDRAILS) == {"crash_share", "payment_error_share", "cancel_rate"}


@pytest.fixture(params=["memory", "sqlite"])
def storage(request, tmp_path):
    if request.param == "memory":
        yield MemoryStorage()
    else:
        s = SqliteStorage(str(tmp_path / "ab.db"))
        yield s
        s.close()


def test_metric_storage_contract(storage):
    m1 = REG["revenue_per_guest"]
    storage.save_metric(m1)
    storage.save_metric(m1)  # то же содержимое — без ошибки и без дубля
    with pytest.raises(StorageError, match="неизменяема"):
        storage.save_metric(m1.model_copy(update={"winsorize_percentile": 0.99}))
    with pytest.raises(StorageError, match="ожидается версия 2"):
        storage.save_metric(m1.model_copy(update={"version": 3}))
    m2 = m1.model_copy(update={"version": 2, "winsorize_percentile": 0.99})
    storage.save_metric(m2)
    storage.save_metric(REG["conversion"])
    assert storage.get_metric("revenue_per_guest") == m2 and storage.get_metric("revenue_per_guest", 1) == m1
    assert storage.get_metric("нет") is None and storage.get_metric("conversion", 2) is None
    assert [m.ref for m in storage.list_metrics()] == ["conversion@1", "revenue_per_guest@1", "revenue_per_guest@2"]


def test_sqlite_metrics_survive_restart(tmp_path):
    s = SqliteStorage(str(tmp_path / "ab.db"))
    ExperimentService(s).ensure_default_metrics()
    s.close()
    again = SqliteStorage(str(tmp_path / "ab.db"))
    assert [m.ref for m in again.list_metrics()] == sorted(m.ref for m in DEFAULT_METRICS)
    assert ExperimentService(again).ensure_default_metrics() == 0
    again.close()


def test_service_seeds_and_defines_metrics(storage):
    svc = ExperimentService(storage)
    assert svc.ensure_default_metrics() == 7 and svc.ensure_default_metrics() == 0
    assert [r.action for r in storage.audit()] == ["metric.seed"] * 7
    with denied("not_found"):
        svc.metric("revenue_per_guest@2")
    with denied("invalid"):
        svc.metric("revenue_per_guest")
    for actor in (OWNER, Actor("app", "sdk"), Actor("guest", "reader")):
        with denied("forbidden"):
            svc.define_metric(actor, {"name": "revenue_per_guest", "title": "Выручка", "formula": "revenue"})
    with denied("conflict", "не отличается"):
        svc.define_metric(ANALYST, {"name": "revenue_per_guest", "title": "Другое название", "formula": "revenue",
                                    "winsorize_percentile": 0.995})
    v2 = svc.define_metric(ANALYST, {"name": "revenue_per_guest", "title": "Выручка, винзоризация 99%",
                                     "formula": "revenue", "winsorize_percentile": 0.99, "version": 77, "owner": "x"})
    assert (v2.ref, v2.owner) == ("revenue_per_guest@2", "petrov")  # версию и владельца задаёт платформа
    new = svc.define_metric(ADMIN, {"name": "refund_free_revenue", "title": "Новая метрика", "formula": "revenue"})
    assert new.ref == "refund_free_revenue@1"
    with denied("invalid"):
        svc.define_metric(ANALYST, {"name": "bad", "title": "Плохая", "formula": "has_app_event"})
    # старая версия не изменилась: отчёты по revenue_per_guest@1 считают по прежнему определению
    assert svc.metric("revenue_per_guest@1") == REG["revenue_per_guest"]
    defined = [r for r in storage.audit() if r.action == "metric.define"]
    assert [r.details["ref"] for r in defined] == ["revenue_per_guest@2", "refund_free_revenue@1"]
    storage.save_metric(REG["avg_check"])  # стартовая версия повторно — без ошибки
    assert svc.ensure_default_metrics() == 0
    # в чужой базе стартовая версия с другим определением — тревога, а не молчаливая подмена
    other = MemoryStorage()
    other.save_metric(REG["cancel_rate"].model_copy(update={"formula": "order_count"}))
    with denied("conflict", "cancel_rate@1 отличается"):
        ExperimentService(other).ensure_default_metrics()


# ---------------- запуск теста приложения ----------------
@pytest.fixture
def svc():
    service = ExperimentService(MemoryStorage(), require_preflight=False)  # предстартовая проверка — в П4
    service.ensure_default_metrics()
    service.add_layer(ADMIN, "ui")
    return service


def test_app_launch_requires_triggered_and_registry_metrics(svc):
    approved(svc, "exp_itt", estimand="itt")
    with denied("invalid", "estimand = triggered"):
        svc.start(OWNER, "exp_itt")
    approved(svc, "exp_plain", primary_metric=base()["primary_metric"].model_dump())
    with denied("invalid", "primary 'revenue' не ссылается на реестр"):
        svc.start(OWNER, "exp_plain")
    tampered = REG["revenue_per_guest"].definition().model_dump() | {"winsorize_percentile": 0.9}
    approved(svc, "exp_tampered", primary_metric=tampered)
    with denied("invalid", "расходится с определением revenue_per_guest@1"):
        svc.start(OWNER, "exp_tampered")
    missing = REG["revenue_per_guest"].definition().model_dump() | {"registry_ref": "revenue_per_guest@9"}
    approved(svc, "exp_missing", primary_metric=missing)
    with denied("invalid", "revenue_per_guest@9 нет в реестре"):
        svc.start(OWNER, "exp_missing")
    sec = [REG["conversion"].definition().model_dump(), base()["primary_metric"].model_dump()]
    approved(svc, "exp_sec", secondary_metrics=sec)
    with denied("invalid", "secondary 'revenue'"):
        svc.start(OWNER, "exp_sec")
    # отказ ничего не записал: статус Approved, ruleset прежний
    assert svc.get("exp_itt").status.value == "approved" and svc.ruleset().experiments == []
    approved(svc, "exp_ok", secondary_metrics=[REG["avg_check"].definition().model_dump(),
                                               REG["conversion"].definition().model_dump()])
    assert svc.start(OWNER, "exp_ok").by_flag("flag_exp_ok") is not None


def test_met2_warning_for_guest_passport_without_app_guardrails():
    plain = ExperimentPassport(**base())
    assert any("MET-2" in w and "crash_share" in w for w in plain.design_warnings)
    app = ExperimentPassport(**{**draft(), "owner": "ivanova"})
    assert not any("MET-2" in w for w in app.design_warnings)
    partial = draft()
    partial["guardrail_metrics"] = partial["guardrail_metrics"][:1]
    w = [x for x in ExperimentPassport(**{**partial, "owner": "ivanova"}).design_warnings if "MET-2" in x]
    assert len(w) == 1 and "payment_error_share" in w[0] and "crash_share" not in w[0]


# ---------------- таблица «гость × метрики» (CNV-2) ----------------
def at(days=0.0, hours=0.0):
    return (T0 + timedelta(days=days, hours=hours)).isoformat()


def handmade(store):
    """g1 показан в T0, g2 — в T0 + 1 день, g3 — после среза, g4 — без показа."""
    store.add([
        ExposureIn(event_id="x1", event_time=at(), unit_id="g1", experiment_id="e@v1", variant_id="treatment",
                   ruleset_version=1, attributes={"platform": "ios"}),
        ExposureIn(event_id="x1b", event_time=at(2), unit_id="g1", experiment_id="e@v1", variant_id="treatment",
                   ruleset_version=1, attributes={"platform": "android"}),  # повторный показ не сдвигает окно
        ExposureIn(event_id="x2", event_time=at(1), unit_id="g2", experiment_id="e@v1", variant_id="control",
                   ruleset_version=1),
        ExposureIn(event_id="x3", event_time=at(30), unit_id="g3", experiment_id="e@v1", variant_id="control",
                   ruleset_version=1),
        ExposureIn(event_id="xo", event_time=at(), unit_id="g4", experiment_id="другой", variant_id="control",
                   ruleset_version=1),
        ExposureIn(event_id="x5", event_time=at(), unit_id="g5", experiment_id="e@v1", variant_id="control",
                   ruleset_version=1),
    ])

    def o(oid, unit, day, amount, unit_type="guest_id"):
        return OrderIn(order_id=oid, unit_id=unit, unit_type=unit_type, order_time=at(day), amount=amount)
    store.add_orders([
        o("pre1", "g1", -1, 100.0),           # до показа → только ковариата
        o("pre_old", "g1", -20, 1000.0),      # раньше пре-периода 14 дней → никуда
        o("w0", "g1", 0, 200.0),              # ровно в момент показа → в окне
        o("w1", "g1", 6.99, 300.0),           # в окне
        o("w_cancel", "g1", 3, 999.0),        # отменён после окна → не выручка, но в доле отмен
        o("w_refund", "g1", 4, 500.0),        # возврат 120 после окна → выручка 380
        o("edge", "g1", 7, 700.0),            # ровно конец окна → вне окна
        o("dev", "g1", 1, 50.0, "device_id"), # другой тип единицы → не учитывается
        o("g2a", "g2", 2, 400.0),
        o("g4a", "g4", 2, 400.0),             # гость без показа в этом тесте
        o("g5a", "g5", 1, 300.0),             # единственный заказ g5 отменён → конверсии нет
    ])
    store.add_adjustments([
        OrderAdjustmentIn(adjustment_id="a1", order_id="w_cancel", kind="cancel", event_time=at(9)),
        OrderAdjustmentIn(adjustment_id="a2", order_id="w_refund", kind="refund", amount=120.0, event_time=at(10)),
        OrderAdjustmentIn(adjustment_id="a3", order_id="pre1", kind="refund", amount=30.0, event_time=at(1)),
        OrderAdjustmentIn(adjustment_id="a5", order_id="g5a", kind="cancel", event_time=at(1.1)),
    ])
    store.add_app_events([
        AppEventIn(event_id="c1", unit_id="g1", unit_type="guest_id", event_type="app_crash", event_time=at(1)),
        AppEventIn(event_id="c2", unit_id="g2", unit_type="guest_id", event_type="app_crash", event_time=at(0)),  # до показа g2
        AppEventIn(event_id="p1", unit_id="g2", unit_type="guest_id", event_type="payment_error", event_time=at(3)),
        AppEventIn(event_id="c3", unit_id="g2", unit_type="device_id", event_type="app_crash", event_time=at(3)),
    ])


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield MemoryEventStore()
    else:
        s = SqliteEventStore(str(tmp_path / "events.db"))
        yield s
        s.close()


def test_unit_table_attribution_rules(store):
    handmade(store)
    tab = build_unit_table(store, "e@v1", DEFAULT_METRICS, 7, 14, T0 + timedelta(days=20)).set_index("unit_id")
    assert list(tab.index) == ["g1", "g2", "g5"]  # g3 показан после среза, g4 — не в тесте
    g1, g2 = tab.loc["g1"], tab.loc["g2"]
    assert g1["variant"] == "treatment" and g1["attr_platform"] == "ios"  # признаки — первого показа
    assert g1["revenue_per_guest"] == 200 + 300 + 380 and g1["order_frequency"] == 3 and g1["conversion"] == 1
    assert (g1["avg_check__num"], g1["avg_check__den"]) == (880, 3)
    assert (g1["cancel_rate__num"], g1["cancel_rate__den"]) == (1, 4)
    assert g1["revenue_per_guest__pre"] == 70 and g1["order_frequency__pre"] == 1  # возврат — по дате заказа
    assert g1["crash_share"] == 1 and g1["payment_error_share"] == 0
    assert g2["revenue_per_guest"] == 400 and g2["crash_share"] == 0 and g2["crash_share__pre"] == 1
    assert g2["payment_error_share"] == 1 and bool(g1["window_complete"]) and bool(g2["window_complete"])
    g5 = tab.loc["g5"]
    assert (g5["conversion"], g5["revenue_per_guest"], g5["cancel_rate__num"], g5["cancel_rate__den"]) == (0, 0, 1, 1)
    assert "avg_check__pre" not in tab.columns  # у ratio ковариаты нет


def test_unit_table_cutoff_snapshot_and_determinism(store, tmp_path):
    handmade(store)
    cut = store.last_seq()
    store.add_orders([OrderIn(order_id="late", unit_id="g1", unit_type="guest_id", order_time=at(5), amount=9999.0)])
    store.add_adjustments([OrderAdjustmentIn(adjustment_id="a9", order_id="w1", kind="cancel", event_time=at(30))])
    early = T0 + timedelta(days=3, hours=12)
    full = build_unit_table(store, "e@v1", DEFAULT_METRICS, 7, 14, T0 + timedelta(days=20), upto_seq=cut)
    again = build_unit_table(store, "e@v1", DEFAULT_METRICS, 7, 14, T0 + timedelta(days=20), upto_seq=cut)
    assert full.equals(again)
    assert full.set_index("unit_id").loc["g1", "revenue_per_guest"] == 880  # поздние события в срез не попали
    later = build_unit_table(store, "e@v1", DEFAULT_METRICS, 7, 14, T0 + timedelta(days=20)).set_index("unit_id")
    assert later.loc["g1", "revenue_per_guest"] == 880 - 300 + 9999
    # промежуточный срез: окно обрезано моментом анализа и помечено незакрытым
    part = build_unit_table(store, "e@v1", DEFAULT_METRICS, 7, 14, early, upto_seq=cut).set_index("unit_id")
    assert part.loc["g1", "revenue_per_guest"] == 200 and part.loc["g1", "window_end"].startswith("2026-10-05T00:00")
    assert not part["window_complete"].any() and part.loc["g1", "cancel_rate__den"] == 2
    # одна и та же таблица в памяти и в SQLite
    other = SqliteEventStore(str(tmp_path / "copy.db")) if isinstance(store, MemoryEventStore) else MemoryEventStore()
    handmade(other)
    assert build_unit_table(other, "e@v1", DEFAULT_METRICS, 7, 14, T0 + timedelta(days=20)).equals(full)
    with pytest.raises(ValueError, match="повторяться"):
        build_unit_table(store, "e@v1", [REG["conversion"], REG["conversion"]], 7, 14, early)
    with pytest.raises(ValueError):
        build_unit_table(store, "e@v1", DEFAULT_METRICS, 0, 14, early)
    with pytest.raises(ValueError):
        build_unit_table(store, "e@v1", DEFAULT_METRICS, 7, 14, early.replace(tzinfo=None))
    empty = build_unit_table(store, "нет", DEFAULT_METRICS, 7, 14, early)
    assert empty.empty and "revenue_per_guest" in empty.columns


# ---------------- CNV-2 на синтетике ----------------
@pytest.fixture(scope="module")
def world():
    return simulate_app_world(21, 30_000, datetime(2026, 9, 1, tzinfo=timezone.utc))


def load(world, events, treated):
    store = MemoryEventStore()
    store.add_orders([OrderIn(**o) for o in events.payload("orders")])
    store.add_adjustments([OrderAdjustmentIn(**a) for a in events.payload("adjustments")])
    store.add_app_events([AppEventIn(**e) for e in events.payload("app_events")])
    ex = world.exposure_times()
    store.add([ExposureIn(event_id=f"x{i}", event_time=t.isoformat(), unit_id=u, experiment_id="e@v1",
                          variant_id="treatment" if tr else "control", ruleset_version=1)
               for i, (u, t, tr) in enumerate(zip(ex.index, ex, treated)) if not pd.isna(t)])
    return store


def test_cnv2_effect_after_exposure_is_measured_exactly(world):
    """Эффект внедрён только после показа: в окне он виден целиком, контроль и ковариата не меняются."""
    treated = np.random.default_rng(1).random(len(world.guests)) < 0.5
    cut = world.end + timedelta(days=8)
    metrics = [REG["revenue_per_guest"], REG["order_frequency"]]
    base_tab = build_unit_table(load(world, world.realize(treated), treated), "e@v1", metrics, 7, 14, cut)
    ev = world.realize(treated, lift=0.10)
    eff_tab = build_unit_table(load(world, ev, treated), "e@v1", metrics, 7, 14, cut)
    b, e = base_tab.set_index("unit_id"), eff_tab.set_index("unit_id")
    assert list(b.index) == list(e.index) and len(b) > 20_000
    control = b["variant"] == "control"
    assert b.loc[control].equals(e.loc[control])  # у контроля ничего не изменилось
    pre_cols = [c for c in b.columns if c.endswith("__pre")]
    assert b[pre_cols].equals(e[pre_cols])  # до показа — тоже
    t = ~control
    realized = e.loc[t, "revenue_per_guest"].sum() / b.loc[t, "revenue_per_guest"].sum() - 1
    # истина генератора — по выручке до отмен; здесь — за вычетом отмен и возвратов (доли одинаковы)
    assert abs(realized - ev.truth.network_total_rel) < 0.01, (realized, ev.truth.network_total_rel)


def test_cnv2_orders_before_exposure_do_not_enter_effect(world):
    """Тестовые гости заказали больше ДО показа (например, их выбрали по активности): в эффект это не идёт."""
    treated = np.random.default_rng(2).random(len(world.guests)) < 0.5
    ev = world.realize(treated)
    store = load(world, ev, treated)
    cut = world.end + timedelta(days=8)
    before = build_unit_table(store, "e@v1", [REG["revenue_per_guest"]], 7, 14, cut).set_index("unit_id")
    ex = world.exposure_times()
    extra = [OrderIn(order_id=f"pre-{u}", unit_id=u, unit_type="guest_id", amount=5000.0,
                     order_time=(t - timedelta(hours=1)).isoformat())
             for u, t, tr in zip(ex.index, ex, treated) if tr and not pd.isna(t)]
    store.add_orders(extra)
    after = build_unit_table(store, "e@v1", [REG["revenue_per_guest"]], 7, 14, cut).set_index("unit_id")
    assert after["revenue_per_guest"].equals(before["revenue_per_guest"])
    t = after["variant"] == "treatment"
    assert np.allclose(after.loc[t, "revenue_per_guest__pre"] - before.loc[t, "revenue_per_guest__pre"], 5000.0)
