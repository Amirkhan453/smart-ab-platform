"""
Фаза 4, П1: контрактные тесты хранилища событий (CNV-1) и HTTP-приёма заказов, корректировок, событий приложения.
Один набор для всех реализаций: память, SQLite; ClickHouse (фаза 6) добавляется в STORES и проходит его же.
"""
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ab_platform.auth import Auth
from ab_platform.control import ExperimentService
from ab_platform.events import (
    AppEventIn, ExposureIn, ExposureStore, MemoryEventStore, OrderAdjustmentIn, OrderIn, SqliteEventStore, utc_text,
)
from ab_platform.server import create_app
from ab_platform.storage import MemoryStorage
from tests.test_phase3_http import KEYS, SPEC

STORES = ["memory", "sqlite", "legacy_jsonl"]
INGEST = f"ingest:orders-backend:{'i' * 30}"
H_INGEST = {"Authorization": f"Bearer {'i' * 30}"}
H_SDK = {"Authorization": f"Bearer {KEYS['sdk']}"}


@pytest.fixture(params=STORES)
def store(request, tmp_path):
    if request.param == "memory":
        yield MemoryEventStore()
    elif request.param == "legacy_jsonl":
        yield ExposureStore(str(tmp_path / "exposures.jsonl"))
    else:
        s = SqliteEventStore(str(tmp_path / "events.db"))
        yield s
        s.close()


def order(i, unit="g1", amount=500.0, t="2026-11-10T12:00:00+03:00", **kw):
    return OrderIn(order_id=f"o{i}", unit_id=unit, unit_type="guest_id", order_time=t, amount=amount, **kw)


def exposure(i, unit="g1", exp="exp@v1", t="2026-11-10T10:00:00+00:00", variant="control", **kw):
    return ExposureIn(event_id=f"x{i}", event_time=t, unit_id=unit, experiment_id=exp, variant_id=variant,
                      ruleset_version=1, **kw)


# ---------------- модели ----------------
def test_models_validate_and_normalize_time():
    o = order(1, items=[{"sku": "burger", "qty": 2, "price": 250.0}], discount=50.0)
    assert o.row()["order_time"] == "2026-11-10T09:00:00.000000+00:00" == utc_text(o.order_time)
    assert o.row()["items"] == [{"sku": "burger", "qty": 2, "price": 250.0}]
    bad_orders = [
        dict(amount=-1), dict(amount=float("nan")), dict(amount=float("inf")), dict(discount=-5),
        dict(order_time="2026-11-10T12:00:00"),  # без часового пояса
        dict(unit_type="session_id"), dict(order_id=""), dict(extra=1),
        dict(items=[{"sku": "a", "qty": 0, "price": 1}]), dict(items=[{"sku": "a", "qty": 1, "price": 1}] * 201),
    ]
    base = dict(order_id="o1", unit_id="g1", unit_type="guest_id", order_time="2026-11-10T12:00:00Z", amount=1.0)
    for bad in bad_orders:
        with pytest.raises(ValidationError):
            OrderIn(**{**base, **bad})
    adj = dict(adjustment_id="a1", order_id="o1", event_time="2026-11-11T10:00:00Z")
    assert OrderAdjustmentIn(**adj, kind="cancel").amount is None
    assert OrderAdjustmentIn(**adj, kind="refund", amount=100.0).amount == 100.0
    for bad in (dict(kind="cancel", amount=10.0), dict(kind="refund"), dict(kind="refund", amount=0),
                dict(kind="void")):
        with pytest.raises(ValidationError):
            OrderAdjustmentIn(**adj, **bad)
    ev = dict(event_id="e1", unit_id="g1", unit_type="device_id", event_time="2026-11-10T10:00:00Z")
    assert AppEventIn(**ev, event_type="app_crash").row()["event_type"] == "app_crash"
    with pytest.raises(ValidationError):
        AppEventIn(**ev, event_type="screen_view")  # список типов закрытый


# ---------------- хранилище ----------------
def test_orders_idempotent_and_conflicts(store):
    r = store.add_orders([order(1), order(2, amount=700.0), order(1)])  # дубль внутри пачки
    assert (r.accepted, r.duplicates, r.conflicts) == (2, 1, [])
    # тот же момент в другом поясе — то же содержимое, не конфликт
    same = order(1, t="2026-11-10T09:00:00+00:00")
    assert store.add_orders([same]).duplicates == 1
    changed = store.add_orders([order(3), order(1, amount=501.0), order(2, unit="g2")])
    assert changed.accepted == 1 and changed.duplicates == 0
    assert [(c["index"], c["order_id"]) for c in changed.conflicts] == [(1, "o1"), (2, "o2")]
    assert "другим содержимым" in changed.conflicts[0]["error"]
    frame = store.frame("orders")
    assert frame["order_id"].tolist() == ["o1", "o2", "o3"] and frame["amount"].tolist() == [500.0, 700.0, 500.0]
    assert frame["seq"].is_monotonic_increasing and store.counts_by_kind()["orders"] == 3


def test_resending_1000_orders_and_adjustments_does_not_change_sums(store):
    orders = [order(i, unit=f"g{i % 37}", amount=100.0 + i) for i in range(1000)]
    adjs = [OrderAdjustmentIn(adjustment_id=f"a{i}", order_id=f"o{i}", kind="refund", amount=10.0,
                              event_time="2026-11-12T10:00:00Z") for i in range(0, 1000, 10)]
    store.add_orders(orders)
    store.add_adjustments(adjs)
    before = (store.frame("orders")["amount"].sum(), store.frame("adjustments")["amount"].sum(), store.last_seq())
    for chunk in range(0, 1000, 250):  # повтор пачками и целиком
        r = store.add_orders(orders[chunk:chunk + 250])
        assert (r.accepted, r.duplicates) == (0, 250)
    assert store.add_adjustments(adjs).duplicates == 100
    after = (store.frame("orders")["amount"].sum(), store.frame("adjustments")["amount"].sum(), store.last_seq())
    assert before == after == (sum(100.0 + i for i in range(1000)), 1000.0, 1100)


def test_adjustment_before_order_is_kept_and_reported(store):
    store.add_adjustments([OrderAdjustmentIn(adjustment_id="a1", order_id="o9", kind="cancel",
                                             event_time="2026-11-10T10:00:00Z")])
    assert store.orphan_adjustments() == 1
    store.add_orders([order(9)])
    assert store.orphan_adjustments() == 0 and len(store.frame("adjustments")) == 1


def test_snapshot_by_seq_is_reproducible(store):
    store.add([exposure(1)])
    store.add_orders([order(1)])
    cut = store.last_seq()
    store.add_orders([order(2)])  # пришёл после среза
    store.add_app_events([AppEventIn(event_id="e1", unit_id="g1", unit_type="guest_id", event_type="app_crash",
                                     event_time="2026-11-09T10:00:00Z")])  # событие раньше, но принято позже
    assert store.frame("orders", upto_seq=cut)["order_id"].tolist() == ["o1"]
    assert store.frame("app_events", upto_seq=cut).empty and len(store.frame("app_events")) == 1
    assert store.frame("exposures", upto_seq=cut)["event_id"].tolist() == ["x1"]
    assert list(store.frame("orders").columns) == ["seq", "order_id", "unit_id", "unit_type", "order_time", "amount",
                                                    "discount", "items"]
    with pytest.raises(ValueError, match="Неизвестный вид"):
        store.frame("payments")


def test_exposures_keep_phase3_behaviour(store):
    assert store.add([exposure(1, t="2026-11-10T12:00:00+00:00", attributes={"platform": "ios"}),
                      exposure(2, t="2026-11-10T13:30:00+03:00", attributes={"platform": "web"}),  # 10:30 UTC
                      exposure(3, unit="g2"), exposure(1)]) == (3, 1)
    # показ с тем же event_id и другим содержимым — дубль, как в фазах 2–3
    assert store.add([exposure(1, variant="treatment")]) == (0, 1)
    first = store.first_exposures("exp@v1")
    assert first["g1"]["event_id"] == "x2" and first["g1"]["event_time"] == "2026-11-10T10:30:00.000000+00:00"
    assert first["g1"]["attributes"] == {"platform": "web"} and set(first) == {"g1", "g2"}
    assert store.enrolled("g1") == {"exp@v1"} and store.enrolled("нет") == set() and len(store) == 3
    assert store.counts() == {"exp@v1:control": 3}
    # при равном времени первым считается принятый раньше
    store.add([exposure(4, unit="g3", variant="treatment"), exposure(5, unit="g3")])
    assert store.first_exposures("exp@v1")["g3"]["event_id"] == "x4"
    cut = store.last_seq()
    store.add([exposure(6, unit="g4")])
    assert set(store.first_exposures("exp@v1", upto_seq=cut)) == {"g1", "g2", "g3"}


def test_concurrent_writers_keep_sequence_without_gaps(store):
    def work(w):
        for j in range(20):
            store.add_orders([order(f"{w}-{j}-{k}", unit=f"g{w}") for k in range(5)])
            store.add([exposure(f"{w}-{j}", unit=f"g{w}")])
    threads = [threading.Thread(target=work, args=(w,)) for w in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    seqs = sorted([*store.frame("orders")["seq"], *store.frame("exposures")["seq"]])
    assert seqs == list(range(1, 721)) and store.last_seq() == 720


# ---------------- только SQLite ----------------
def test_sqlite_survives_restart_and_two_connections(tmp_path):
    path = str(tmp_path / "events.db")
    a = SqliteEventStore(path)
    items = [{"sku": "x", "qty": 1, "price": 500.0}]
    a.add_orders([order(1, items=items)])
    a.add([exposure(1, attributes={"platform": "ios", "is_new_guest": False})])
    b = SqliteEventStore(path)  # второй процесс с тем же файлом
    assert b.add_orders([order(1, items=items)]).duplicates == 1
    assert b.add_orders([order(1)]).conflicts  # без позиций — другое содержимое
    b.add_orders([order(2)])
    a.close(), b.close()
    again = SqliteEventStore(path)
    assert again.frame("orders")["order_id"].tolist() == ["o1", "o2"] and again.last_seq() == 3
    assert again.frame("orders")["items"][0] == [{"sku": "x", "qty": 1, "price": 500.0}]
    assert again.first_exposures("exp@v1")["g1"]["attributes"] == {"platform": "ios", "is_new_guest": False}
    assert again.enrolled("g1") == {"exp@v1"}
    again.close()
    raw = sqlite3.connect(path)
    raw.execute("UPDATE meta SET value = '9' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()
    with pytest.raises(ValueError, match="нужна миграция"):
        SqliteEventStore(path)


def test_sqlite_failed_batch_leaves_nothing(tmp_path):
    s = SqliteEventStore(str(tmp_path / "events.db"))
    s.add_orders([order(1)])

    class Broken(OrderIn):
        def row(self):
            if self.order_id == "o3":
                raise RuntimeError("сбой посреди пачки")
            return super().row()
    batch = [Broken(**order(i).model_dump()) for i in (2, 3)]
    with pytest.raises(RuntimeError):
        s.add_orders(batch)
    assert s.frame("orders")["order_id"].tolist() == ["o1"] and s.last_seq() == 1
    assert s.add_orders([order(2)]).accepted == 1 and s.last_seq() == 2
    s.close()


# ---------------- HTTP ----------------
@pytest.fixture
def http():
    service = ExperimentService(MemoryStorage())
    app = create_app(exposure_store=MemoryEventStore(), service=service, auth=Auth.parse(f"{SPEC},{INGEST}"))
    return TestClient(app), app, service


def test_only_ingest_writes_orders(http):
    client, app, service = http
    body = [order(1).model_dump(mode="json")]
    for name in ("sdk", "ivanova", "petrov", "root", "guest"):
        r = client.post("/orders", json=body, headers={"Authorization": f"Bearer {KEYS[name]}"})
        assert r.status_code == 403, name
    assert client.post("/orders", json=body).status_code == 401
    assert len(app.state.exposures.frame("orders")) == 0
    denied = [r for r in service.storage.audit() if r.action == "auth.denied"]
    assert len(denied) == 5 and all(r.details["path"] == "/orders" for r in denied)
    r = client.post("/orders", json=body, headers=H_INGEST).json()
    assert r == {"accepted": 1, "duplicates": 0, "conflicts": [], "rejected": []}
    adj = [{"adjustment_id": "a1", "order_id": "o1", "kind": "cancel", "event_time": "2026-11-11T10:00:00Z"}]
    assert client.post("/order-adjustments", json=adj, headers=H_SDK).status_code == 403
    assert client.post("/order-adjustments", json=adj, headers=H_INGEST).json()["accepted"] == 1
    # ключ ingest не даёт управления
    assert client.get("/passports", headers=H_INGEST).status_code == 403


def test_order_batch_reports_rejected_conflicts_with_original_index(http):
    client, app, _ = http
    client.post("/orders", json=[order(1).model_dump(mode="json")], headers=H_INGEST)
    batch = [order(2).model_dump(mode="json"), {"order_id": "bad"}, order(1, amount=9.0).model_dump(mode="json"),
             {**order(3).model_dump(mode="json"), "amount": -1}, order(1).model_dump(mode="json")]
    r = client.post("/orders", json={"events": batch}, headers=H_INGEST).json()
    assert (r["accepted"], r["duplicates"]) == (1, 1)
    assert [x["index"] for x in r["rejected"]] == [1, 3] and "amount" in r["rejected"][1]["error"]
    assert [(c["index"], c["order_id"]) for c in r["conflicts"]] == [(2, "o1")]
    assert app.state.exposures.frame("orders")["amount"].tolist() == [500.0, 500.0]
    for bad, code in ((b"{", 400), (b'{"x": 1}', 400), (b"[" + b"{}," * 10_000 + b"{}]", 413)):
        assert client.post("/orders", content=bad, headers={**H_INGEST, "content-type": "application/json"}).status_code == code


def test_app_events_and_exposures_by_app_or_ingest_only(http):
    client, app, _ = http
    ev = [{"event_id": "e1", "unit_id": "g1", "unit_type": "guest_id", "event_type": "payment_error",
           "event_time": "2026-11-10T10:00:00Z"}]
    assert client.post("/app-events", json=ev, headers=H_SDK).json()["accepted"] == 1
    assert client.post("/app-events", json=ev, headers=H_INGEST).json()["duplicates"] == 1
    for name in ("ivanova", "guest"):
        h = {"Authorization": f"Bearer {KEYS[name]}"}
        assert client.post("/app-events", json=ev, headers=h).status_code == 403
        assert client.post("/exposures", json=[exposure(1).model_dump(mode="json")], headers=h).status_code == 403
    health = client.get("/events/health", headers=H_SDK).json()
    assert health["counts"] == {"exposures": 0, "orders": 0, "adjustments": 0, "app_events": 1}
    assert health["orphan_adjustments"] == 0 and health["last_seq"] == 1
    assert client.get("/events/health").status_code == 401


def test_stand_mode_does_not_open_orders():
    client = TestClient(create_app(auth=Auth.disabled()))
    assert client.post("/orders", json=[order(1).model_dump(mode="json")]).status_code == 403
    assert client.post("/exposures", json=[exposure(1).model_dump(mode="json")]).json()["accepted"] == 1


def test_sqlite_two_processes_write_concurrently(tmp_path):
    """Два соединения пишут одновременно: номера seq без дыр и повторов, ни одной ошибки блокировки."""
    path = str(tmp_path / "events.db")
    stores, errors = [SqliteEventStore(path), SqliteEventStore(path)], []

    def work(s, w):
        try:
            for j in range(80):
                s.add_orders([order(f"{w}-{j}", unit=f"g{w}")])
        except Exception as exc:  # noqa: BLE001 — ошибку потока надо донести до assert
            errors.append(exc)
    threads = [threading.Thread(target=work, args=(s, w)) for w, s in enumerate(stores)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and stores[0].frame("orders")["seq"].tolist() == list(range(1, 161))
    [s.close() for s in stores]


# ---------------- граничные случаи (по итогам сплошной проверки порчей) ----------------
@pytest.mark.parametrize("model,field,limit,valid", [
    (ExposureIn, "event_id", 128, dict(event_time="2026-11-10T10:00:00Z", unit_id="g1", experiment_id="e",
                                       variant_id="c", ruleset_version=1)),
    (ExposureIn, "unit_id", 1024, dict(event_id="x", event_time="2026-11-10T10:00:00Z", experiment_id="e",
                                       variant_id="c", ruleset_version=1)),
    (OrderIn, "order_id", 128, dict(unit_id="g1", unit_type="guest_id", order_time="2026-11-10T10:00:00Z", amount=1.0)),
    (OrderIn, "unit_id", 1024, dict(order_id="o", unit_type="guest_id", order_time="2026-11-10T10:00:00Z", amount=1.0)),
    (OrderAdjustmentIn, "adjustment_id", 128, dict(order_id="o", kind="cancel", event_time="2026-11-10T10:00:00Z")),
    (OrderAdjustmentIn, "order_id", 128, dict(adjustment_id="a", kind="cancel", event_time="2026-11-10T10:00:00Z")),
    (AppEventIn, "event_id", 128, dict(unit_id="g1", unit_type="guest_id", event_type="app_crash",
                                       event_time="2026-11-10T10:00:00Z")),
    (AppEventIn, "unit_id", 1024, dict(event_id="e", unit_type="guest_id", event_type="app_crash",
                                       event_time="2026-11-10T10:00:00Z")),
])
def test_identifier_length_limits(model, field, limit, valid):
    model(**valid, **{field: "a" * limit})
    with pytest.raises(ValidationError):
        model(**valid, **{field: "a" * (limit + 1)})


def test_amount_and_item_limits():
    base = dict(order_id="o1", unit_id="g1", unit_type="guest_id", order_time="2026-11-10T12:00:00Z")
    OrderIn(**base, amount=10_000_000.0, discount=10_000_000.0,
            items=[{"sku": "s" * 64, "qty": 1000, "price": 0.0}])
    for bad in (dict(amount=10_000_000.01), dict(amount=1.0, discount=10_000_000.01),
                dict(amount=1.0, items=[{"sku": "s" * 65, "qty": 1, "price": 1.0}]),
                dict(amount=1.0, items=[{"sku": "s", "qty": 1001, "price": 1.0}])):
        with pytest.raises(ValidationError):
            OrderIn(**base, **bad)
    adj = dict(adjustment_id="a", order_id="o", kind="refund", event_time="2026-11-10T12:00:00Z")
    OrderAdjustmentIn(**adj, amount=10_000_000.0)
    with pytest.raises(ValidationError):
        OrderAdjustmentIn(**adj, amount=10_000_000.01)


def test_adjustments_and_app_events_report_conflicts(store):
    a = OrderAdjustmentIn(adjustment_id="a1", order_id="o1", kind="refund", amount=10.0, event_time="2026-11-10T12:00:00Z")
    assert store.add_adjustments([a]).accepted == 1
    r = store.add_adjustments([a, a.model_copy(update={"amount": 20.0})])
    assert (r.accepted, r.duplicates, len(r.conflicts)) == (0, 1, 1) and r.conflicts[0]["adjustment_id"] == "a1"
    e = AppEventIn(event_id="c1", unit_id="g1", unit_type="guest_id", event_type="app_crash",
                   event_time="2026-11-10T12:00:00Z")
    assert store.add_app_events([e]).accepted == 1
    r = store.add_app_events([e, e.model_copy(update={"event_type": "payment_error"})])
    assert (r.accepted, r.duplicates, len(r.conflicts)) == (0, 1, 1) and r.conflicts[0]["event_id"] == "c1"


def test_first_exposure_tie_goes_to_earlier_accepted(store):
    store.add([exposure(1, variant="treatment"), exposure(2, variant="control")])  # одно и то же время
    assert store.first_exposures("exp@v1")["g1"]["variant_id"] == "treatment"


def test_sqlite_close_releases_connection(tmp_path):
    s = SqliteEventStore(str(tmp_path / "events.db"))
    s.close()
    with pytest.raises(sqlite3.ProgrammingError):
        s.last_seq()


def test_attribute_value_length_limit():
    exposure(1, attributes={"platform": "ж" * 64})  # ровно 64 знака — допустимо
    with pytest.raises(ValidationError, match="до 64 знаков"):
        exposure(1, attributes={"platform": "ж" * 65})


def test_legacy_jsonl_line_format_and_corrupt_first_line(tmp_path):
    path = tmp_path / "exposures.jsonl"
    s = ExposureStore(str(path))
    s.add([exposure(1, unit="гость", attributes={"platform": "ios"})])
    line = path.read_text(encoding="utf-8").strip()
    # канонический вид: ключи по алфавиту, без пробелов, кириллица как есть
    assert line.startswith('{"attributes":{"platform":"ios"},"event_id":"x1","event_time":') and '"unit_id":"гость"' in line
    path.write_text("{недописано\n" + line + "\n", encoding="utf-8")  # битая — самая первая строка
    again = ExposureStore(str(path))
    assert again.corrupt_lines == 1 and len(again) == 1


def test_canonical_rejects_nan():
    from ab_platform.events import _canonical
    with pytest.raises(ValueError):
        _canonical({"x": float("nan")})
