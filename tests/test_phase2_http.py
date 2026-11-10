"""Фаза 2, П4: HTTP — ruleset с ETag/304, идемпотентный приём exposure, kill-switch на стенде (ASG-7, DAT-1)."""
import socket
import threading
import time
from collections import Counter

import pytest
import uvicorn
from fastapi.testclient import TestClient

from ab_platform.engine.ruleset_engine import update_rule
from ab_platform.schemas.ruleset import ExperimentRule, Layer, Ruleset
from ab_platform.sdk import ExperimentClient
from ab_platform.sdk_http import HttpExposureSink, HttpRulesetFetcher
from ab_platform.auth import Auth
from ab_platform.server import MAX_BATCH, ExposureStore, RulesetStore, create_app

OPEN = Auth.disabled()  # стенд SDK без ключей; ключи и роли — tests/test_phase3_http.py


def variants(*ids):
    return [{"variant_id": v, "traffic_percentage": 100.0 / len(ids), "is_control": i == 0} for i, v in enumerate(ids)]


def make_ruleset(version=1):
    return Ruleset(version=version, holdout={"pct": 0.0}, layers=[Layer(layer_id="L0")],
                   experiments=[ExperimentRule(experiment_id="exp_0", layer_id="L0", variants=variants("A", "B"))])


def event(i=0, **kw):
    ev = {"event_id": f"ev_{i}", "event_time": "2026-11-07T10:00:00+00:00", "unit_id": f"g{i}",
          "experiment_id": "exp_0", "variant_id": "A" if i % 2 else "B", "ruleset_version": 1, "session_id": None}
    ev.update(kw)
    return ev


@pytest.fixture()
def stand():
    rs = make_ruleset()
    store, exp = RulesetStore(rs), ExposureStore()
    return TestClient(create_app(store, exp, auth=OPEN)), store, exp


# --- GET /ruleset --------------------------------------------------------------------------------

def test_ruleset_etag_and_304(stand):
    http, store, _ = stand
    r = http.get("/ruleset")
    rs = store.current()[0]
    assert r.status_code == 200 and r.headers["etag"] == f'"{rs.etag}"' and r.headers["cache-control"] == "no-cache"
    assert Ruleset.from_json(r.text) == rs
    for inm in (f'"{rs.etag}"', f'W/"{rs.etag}"', f'"x", "{rs.etag}"', "*", rs.etag):
        r304 = http.get("/ruleset", headers={"If-None-Match": inm})
        assert r304.status_code == 304 and r304.content == b"" and r304.headers["etag"] == f'"{rs.etag}"', inm
    assert http.get("/ruleset", headers={"If-None-Match": '"other"'}).status_code == 200
    store.publish(update_rule(rs, "exp_0", kill_switch=True))
    r2 = http.get("/ruleset", headers={"If-None-Match": f'"{rs.etag}"'})
    assert r2.status_code == 200 and Ruleset.from_json(r2.text).version == 2


def test_ruleset_unpublished_and_no_rollback():
    http = TestClient(create_app(auth=OPEN))
    assert http.get("/ruleset").status_code == 503
    store = RulesetStore(make_ruleset(version=3))
    with pytest.raises(ValueError, match="не новее"):
        store.publish(make_ruleset(version=3))
    with pytest.raises(ValueError, match="не новее"):
        store.publish(make_ruleset(version=2))
    assert store.current()[0].version == 3


# --- POST /exposures -----------------------------------------------------------------------------

def test_dat1_resend_1000_events_does_not_change_counts(stand):
    http, _, exp = stand
    batch = [event(i) for i in range(1000)]
    r = http.post("/exposures", json=batch)
    assert r.json() == {"accepted": 1000, "duplicates": 0, "rejected": []}
    before = exp.counts()
    assert sum(before.values()) == 1000
    # повтор целиком, повтор пачками и в обёртке {"events": ...}
    assert http.post("/exposures", json=batch).json() == {"accepted": 0, "duplicates": 1000, "rejected": []}
    for k in range(0, 1000, 100):
        assert http.post("/exposures", json={"events": batch[k:k + 100]}).json()["duplicates"] == 100
    assert exp.counts() == before and len(exp) == 1000
    # дубликат внутри одной пачки и новый event_id рядом с повтором
    r = http.post("/exposures", json=[event(5000), event(5000), event(3)])
    assert r.json() == {"accepted": 1, "duplicates": 2, "rejected": []} and len(exp) == 1001


@pytest.mark.parametrize("bad, field", [
    ({"event_id": ""}, "event_id"),
    ({"event_time": "2026-11-07T10:00:00"}, "event_time"),  # без часового пояса
    ({"event_time": "вчера"}, "event_time"),
    ({"ruleset_version": 0}, "ruleset_version"),
    ({"lost_field": 1}, "lost_field"),
])
def test_invalid_event_rejected_others_accepted(stand, bad, field):
    http, _, exp = stand
    r = http.post("/exposures", json=[event(1), event(2, **bad), event(3)]).json()
    assert r["accepted"] == 2 and len(r["rejected"]) == 1
    assert r["rejected"][0]["index"] == 1 and field in r["rejected"][0]["error"]
    missing = event(4)
    del missing["ruleset_version"]
    assert http.post("/exposures", json=[missing]).json()["rejected"][0]["error"].startswith("ruleset_version")
    assert len(exp) == 2


def test_bad_bodies(stand):
    http, _, _ = stand
    assert http.post("/exposures", content=b"{oops", headers={"content-type": "application/json"}).status_code == 400
    assert http.post("/exposures", json={"no_events": []}).status_code == 400
    assert http.post("/exposures", json=[event(i) for i in range(MAX_BATCH + 1)]).status_code == 413
    assert http.post("/exposures", json=[]).json() == {"accepted": 0, "duplicates": 0, "rejected": []}


def test_jsonl_persists_and_dedup_survives_restart(tmp_path):
    path = str(tmp_path / "exposures.jsonl")
    http = TestClient(create_app(RulesetStore(make_ruleset()), ExposureStore(path), auth=OPEN))
    http.post("/exposures", json=[event(i) for i in range(300)])
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"event_id": "недопис')  # оборванная строка при аварийной остановке
    restarted = ExposureStore(path)
    assert len(restarted) == 300 and restarted.corrupt_lines == 1
    http2 = TestClient(create_app(RulesetStore(make_ruleset()), restarted, auth=OPEN))
    assert http2.post("/exposures", json=[event(i) for i in range(310)]).json()["accepted"] == 10
    health = http2.get("/health").json()
    assert (health["ruleset_version"], health["exposures"]) == (1, 310)


def test_concurrent_resend_counts_each_event_once():
    exp = ExposureStore()
    http = TestClient(create_app(RulesetStore(make_ruleset()), exp, auth=OPEN))
    batch = [event(i) for i in range(500)]
    results = []

    def send():
        results.append(http.post("/exposures", json=batch).json()["accepted"])

    ts = [threading.Thread(target=send) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(results) == 500 and len(exp) == 500


# --- SDK через HTTP ------------------------------------------------------------------------------

def test_sdk_over_http_end_to_end(stand):
    http, store, exp = stand
    fetcher, sink = HttpRulesetFetcher(client=http), HttpExposureSink(client=http)
    c = ExperimentClient(fetcher, sink, batch_size=50)
    assert c.refresh() is True and c.refresh() is False and c.counters["not_modified"] == 1
    decisions = [c.get_variant("exp_0", f"g{i}", session_id="s") for i in range(1000)]
    c.flush()
    assert c.counters["sent"] == 1000 and len(exp) == 1000 and sink.rejected == []
    by_variant = Counter(d.variant_id for d in decisions)
    assert exp.counts() == {f"exp_0:{v}": n for v, n in by_variant.items()}
    # повторная отправка тех же событий SDK (например, после таймаута ответа) не меняет счётчики
    sink([e for e in exp.events()[:200]])
    assert len(exp) == 1000
    # kill-switch через HTTP
    store.publish(update_rule(store.current()[0], "exp_0", kill_switch=True))
    c.refresh()
    assert c.get_variant("exp_0", "g1").reason == "kill_switch"


def test_sdk_server_down_fail_open_and_retry():
    exp = ExposureStore()
    app = create_app(RulesetStore(make_ruleset()), exp, auth=OPEN)
    http = TestClient(app)
    c = ExperimentClient(HttpRulesetFetcher(client=http), HttpExposureSink(client=http), batch_size=10_000)
    c.refresh()
    for i in range(30):
        c.get_variant("exp_0", f"g{i}")
    # «падение» сервера: приём отвечает 500, а ruleset — 500
    down = TestClient(create_app(_Failing(), exp, auth=OPEN), raise_server_exceptions=False)
    c._fetcher, c._sink = HttpRulesetFetcher(client=down), HttpExposureSink(client=down)
    assert c.refresh() is False and c.counters["refresh_error"] == 1
    assert c.get_variant("exp_0", "g1").variant_id in {"A", "B"}  # ruleset ещё свежий — решение локальное
    exp_add, exp.add = exp.add, _raise
    assert c.flush() == 0 and len(c.pending_exposures()) == 30
    exp.add = exp_add
    assert c.flush() == 30 and len(exp) == 30


class _Failing(RulesetStore):
    def current(self):
        raise RuntimeError("БД недоступна")


def _raise(_):
    raise RuntimeError("хранилище недоступно")


def test_http_fetcher_owns_client_and_network_error():
    f = HttpRulesetFetcher("http://127.0.0.1:9", timeout_s=0.5)
    c = ExperimentClient(f)
    assert c.refresh() is False and "Connect" in c.last_refresh_error
    assert c.get_variant("exp_0", "g1").reason == "fallback:no_ruleset"
    f.close()


# --- Стенд на uvicorn: kill-switch доходит до всех SDK ---------------------------------------------

class Stand:
    """Настоящий HTTP-сервер uvicorn на свободном порту в фоновом потоке."""

    def __init__(self, ruleset=None, app=None):
        """ruleset — стенд SDK без ключей; app — готовое приложение (контур управления, ключи)."""
        if app is None:
            self.store, self.exposures = RulesetStore(ruleset), ExposureStore()
            app = create_app(self.store, self.exposures, auth=OPEN)
        else:
            self.store, self.exposures = app.state.rulesets, app.state.exposures
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert self.server.started
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.sock.close()


def measure_kill_switch(n_clients, interval_s, timeout_s):
    """Время от публикации kill-switch до момента, когда каждый SDK отдаёт контроль. Клиенты стартуют вразнобой."""
    rs = make_ruleset()
    with Stand(rs) as st:
        clients = []
        for k in range(n_clients):
            c = ExperimentClient(HttpRulesetFetcher(st.url), HttpExposureSink(st.url), refresh_interval_s=interval_s,
                                 max_stale_s=max(300.0, interval_s))
            assert c.refresh() is True
            c.start()
            clients.append(c)
            time.sleep(interval_s / n_clients)  # разные фазы опроса
        t0 = time.monotonic()
        st.store.publish(update_rule(rs, "exp_0", kill_switch=True))
        reached = {}
        while len(reached) < n_clients and time.monotonic() - t0 < timeout_s:
            for k, c in enumerate(clients):
                if k not in reached and c.get_variant("exp_0", "g1").reason == "kill_switch":
                    reached[k] = time.monotonic() - t0
            time.sleep(0.01)
        for c in clients:
            c.close()
        return reached


def test_kill_switch_on_uvicorn_stand_fast():
    reached = measure_kill_switch(n_clients=5, interval_s=0.3, timeout_s=5)
    assert len(reached) == 5 and max(reached.values()) < 0.3 + 1.0, reached


@pytest.mark.slow
def test_kill_switch_on_uvicorn_stand_default_interval():
    """ASG-7 на настоящем интервале SDK по умолчанию (30 с): все клиенты получают kill-switch < 60 с."""
    reached = measure_kill_switch(n_clients=10, interval_s=30.0, timeout_s=65)
    print(f"\nkill-switch: клиентов {len(reached)}, max {max(reached.values()):.2f} с, "
          f"min {min(reached.values()):.2f} с")
    assert len(reached) == 10 and max(reached.values()) < 60, reached
