"""Фаза 2, П3: Python SDK — локальное решение, fail-open, ETag, kill-switch, exposure (ASG-5, ASG-7, 10.2)."""
import time

import pytest

from ab_platform.engine.ruleset_engine import evaluate, update_rule
from ab_platform.schemas.ruleset import ExperimentRule, Layer, Ruleset
from ab_platform.sdk import (
    FALLBACK_INVALID_UNIT, FALLBACK_NO_RULESET, FALLBACK_STALE, FALLBACK_UNKNOWN_FLAG, ExperimentClient,
)


def variants(*ids):
    return [{"variant_id": v, "traffic_percentage": 100.0 / len(ids), "is_control": i == 0} for i, v in enumerate(ids)]


def make_ruleset(n_layers=1, version=1, **rule_kw):
    layers = [Layer(layer_id=f"L{i}") for i in range(n_layers)]
    exps = [ExperimentRule(experiment_id=f"exp_{i}", layer_id=f"L{i}", variants=variants("A", "B"), **rule_kw)
            for i in range(n_layers)]
    return Ruleset(version=version, holdout={"pct": 0.0}, layers=layers, experiments=exps)


class FakeServer:
    """Источник ruleset в памяти: отвечает 304 при совпадении ETag; можно «уронить»."""

    def __init__(self, rs):
        self.rs, self.down, self.calls, self.bodies = rs, False, 0, 0

    def fetch(self, if_none_match):
        self.calls += 1
        if self.down:
            raise ConnectionError("сервер недоступен")
        if if_none_match == self.rs.etag:
            return 304, None, self.rs.etag
        self.bodies += 1
        return 200, self.rs.to_json(), self.rs.etag


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


IDS = [f"guest_{i}" for i in range(3000)]


def test_decisions_match_engine_and_control_naming():
    rs = make_ruleset(ramp_pct=50.0)
    c = ExperimentClient(initial_ruleset=rs)
    for u in IDS:
        d = c.get_variant("exp_0", u)
        e = evaluate(rs, rs.by_flag("exp_0"), u)
        assert (d.variant_id, d.in_experiment, d.reason) == (e.variant_id, e.is_exposed, e.reason)
        assert d.variant_id in {"A", "B"} and d.ruleset_version == 1
    assert c.fallback_rate == 0.0


def test_fail_open_without_ruleset_and_unknown_flag():
    srv = FakeServer(make_ruleset())
    srv.down = True
    c = ExperimentClient(fetcher=srv)
    assert c.refresh() is False and c.counters["refresh_error"] == 1 and "недоступен" in c.last_refresh_error
    d = c.get_variant("exp_0", "g1", default="A")
    assert (d.variant_id, d.in_experiment, d.reason) == ("A", False, FALLBACK_NO_RULESET)
    srv.down = False
    assert c.refresh() is True
    assert c.get_variant("нет_флага", "g1").reason == FALLBACK_UNKNOWN_FLAG
    assert c.get_variant("exp_0", "").reason == FALLBACK_INVALID_UNIT
    assert c.get_variant("exp_0", None).variant_id == "control"
    assert c.get_variant("exp_0", "g1").reason == "assigned"
    m = c.metrics()
    assert m["decisions"][FALLBACK_NO_RULESET] == 1 and abs(m["fallback_rate"] - 4 / 5) < 1e-9
    assert m["ruleset_version"] == 1 and m["refresh_ok"] == 1


def test_fail_open_on_evaluation_exception(monkeypatch):
    c = ExperimentClient(initial_ruleset=make_ruleset())
    import ab_platform.sdk as sdk_mod

    def boom(*a, **k):
        raise RuntimeError("сбой")

    monkeypatch.setattr(sdk_mod, "evaluate", boom)
    d = c.get_variant("exp_0", "g1")
    assert (d.variant_id, d.reason) == ("A", "fallback:exception:RuntimeError")
    assert c.pending_exposures() == []


def test_etag_refresh_and_bad_payloads():
    rs = make_ruleset()
    srv = FakeServer(rs)
    c = ExperimentClient(fetcher=srv)
    assert c.refresh() is True and srv.bodies == 1
    assert c.refresh() is False and c.counters["not_modified"] == 1 and srv.bodies == 1  # 304, тело не качали

    class Broken:
        def __init__(self, resp):
            self.resp = resp

        def fetch(self, _):
            return self.resp

    for resp in [(200, "{не json", None), (500, None, None), (200, make_ruleset(version=1).to_json().replace(
            '"version":1', '"version":0'), None)]:
        c2 = ExperimentClient(fetcher=Broken(resp), initial_ruleset=rs)
        assert c2.refresh() is False and c2.ruleset is rs  # остаёмся на прежнем
    # откат на более старую версию не принимается
    c3 = ExperimentClient(fetcher=FakeServer(make_ruleset(version=1)), initial_ruleset=make_ruleset(version=5))
    assert c3.refresh() is False and c3.ruleset.version == 5


def test_kill_switch_reaches_sdk_on_next_refresh():
    rs = make_ruleset()
    srv = FakeServer(rs)
    c = ExperimentClient(fetcher=srv)
    c.refresh()
    assert {c.get_variant("exp_0", u).variant_id for u in IDS[:200]} == {"A", "B"}
    srv.rs = update_rule(rs, "exp_0", kill_switch=True)
    c.refresh()
    ds = [c.get_variant("exp_0", u) for u in IDS[:200]]
    assert {(d.variant_id, d.reason, d.in_experiment) for d in ds} == {("A", "kill_switch", False)}


def test_stale_ruleset_falls_back_to_control():
    clock = Clock()
    srv = FakeServer(make_ruleset())
    c = ExperimentClient(fetcher=srv, clock=clock, refresh_interval_s=30, max_stale_s=300)
    c.refresh()
    srv.down = True
    clock.t += 299
    c.refresh()
    assert c.get_variant("exp_0", "g1").reason != FALLBACK_STALE
    clock.t += 2
    d = c.get_variant("exp_0", "g1")
    assert (d.variant_id, d.reason) == ("A", FALLBACK_STALE)
    srv.down = False
    c.refresh()  # 304 тоже подтверждает свежесть
    assert c.get_variant("exp_0", "g1").reason != FALLBACK_STALE


def test_config_validation():
    with pytest.raises(ValueError):
        ExperimentClient(refresh_interval_s=61)
    with pytest.raises(ValueError):
        ExperimentClient(refresh_interval_s=30, max_stale_s=10)


def test_exposure_only_in_experiment_and_deduplicated():
    rs = Ruleset(version=3, holdout={"pct": 5.0}, layers=[Layer(layer_id="L0")],
                 experiments=[ExperimentRule(experiment_id="exp_0", layer_id="L0", ramp_pct=40.0,
                                             variants=variants("A", "B"))])
    c = ExperimentClient(initial_ruleset=rs)
    exposed = set()
    for u in IDS:
        for _ in range(3):  # повторные вызовы в той же сессии
            d = c.get_variant("exp_0", u, session_id="s1")
        if d.in_experiment:
            exposed.add(u)
    ev = c.pending_exposures()
    assert {e.unit_id for e in ev} == exposed and len(ev) == len(exposed)
    assert all(e.ruleset_version == 3 and e.variant_id in {"A", "B"} and e.session_id == "s1" for e in ev)
    assert len({e.event_id for e in ev}) == len(ev)
    # новая сессия — новое exposure
    u = next(iter(exposed))
    c.get_variant("exp_0", u, session_id="s2")
    assert len(c.pending_exposures()) == len(ev) + 1


def test_flush_batches_and_retry_on_sink_error():
    sent, fail = [], {"on": True}

    def sink(batch):
        if fail["on"]:
            raise IOError("приём недоступен")
        sent.append(batch)

    c = ExperimentClient(initial_ruleset=make_ruleset(), exposure_sink=sink, batch_size=50)
    for u in IDS[:120]:
        c.get_variant("exp_0", u)
    # попытки отправки только на границах пачки (50, 100), а не на каждом событии
    assert c.counters["sink_error"] == 2 and len(c.pending_exposures()) == 120
    fail["on"] = False
    assert c.flush() == 120 and sum(map(len, sent)) == 120
    assert {"event_id", "event_time", "ruleset_version", "unit_id", "experiment_id", "variant_id"} <= set(sent[0][0])
    for u in IDS[120:230]:
        c.get_variant("exp_0", u)
    assert sum(map(len, sent)) == 220 and len(c.pending_exposures()) == 10  # две автоматические пачки по 50


def test_buffer_bounded_drops_counted():
    c = ExperimentClient(initial_ruleset=make_ruleset(), max_buffer=100, batch_size=10_000)
    for u in IDS[:150]:
        c.get_variant("exp_0", u)
    assert len(c.pending_exposures()) == 100 and c.counters["dropped"] == 50


def test_background_refresh_thread_delivers_kill_switch():
    rs = make_ruleset()
    srv = FakeServer(rs)
    c = ExperimentClient(fetcher=srv, refresh_interval_s=0.05, max_stale_s=1)
    c.refresh()
    c.start()
    try:
        srv.rs = update_rule(rs, "exp_0", kill_switch=True)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and c.ruleset.version == 1:
            time.sleep(0.01)
        assert c.get_variant("exp_0", "g1").reason == "kill_switch"
    finally:
        c.close()
    assert c._thread is None


def test_asg5_latency_p99_under_1ms_with_100_experiments():
    c = ExperimentClient(initial_ruleset=make_ruleset(n_layers=100, ramp_pct=50.0))
    lat = []
    for i in range(20000):
        t = time.perf_counter_ns()
        c.get_variant(f"exp_{i % 100}", f"g{i}", session_id="s")
        lat.append(time.perf_counter_ns() - t)
    lat.sort()
    assert lat[int(0.99 * len(lat))] < 1_000_000, lat[int(0.99 * len(lat))]


def test_thread_safe_counters_and_dedup():
    import threading
    c = ExperimentClient(initial_ruleset=make_ruleset())
    n_threads, per = 8, 2000

    def work(k):
        for i in range(per):
            c.get_variant("exp_0", f"g{i}", session_id="s")  # одни и те же гости во всех потоках

    ts = [threading.Thread(target=work, args=(k,)) for k in range(n_threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(c.decisions.values()) == n_threads * per
    assert len(c.pending_exposures()) == per  # каждое exposure ровно один раз
