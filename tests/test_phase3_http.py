"""
Фаза 3, П5: HTTP — ключи и роли (AUTH-1), команды жизненного цикла, POST /decide (DEC-1…3),
перезапуск без потерь (LNK-3), тонкие клиенты API решений на Python и JS.
"""
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ab_platform.auth import Auth, AuthConfigError
from ab_platform.control import ExperimentService
from ab_platform.schemas.ruleset import Ruleset
from ab_platform.sdk import ExperimentClient
from ab_platform.sdk_http import (
    FALLBACK_DECIDE_UNAVAILABLE, FALLBACK_STALE_DECISION, FALLBACK_UNKNOWN_FLAG, HttpExposureSink, HttpRulesetFetcher,
    RemoteDecisionClient,
)
from ab_platform.server import ExposureStore, RulesetStore, create_app, create_app_from_env
from ab_platform.storage import MemoryStorage, SqliteStorage
from tests.test_phase2_http import OPEN, Stand
from tests.test_phase3_control import IOS, draft

ROOT = Path(__file__).resolve().parents[1]
V = json.loads((ROOT / "tests" / "vectors" / "assignment_vectors.json").read_text(encoding="utf-8"))
NODE = shutil.which("node")

# тестовые ключи (не секреты): роль → ключ
KEYS = {name: f"test-key-{name}-{'0' * 24}" for name in ("sdk", "ivanova", "sidorov", "petrov", "root", "guest")}
SPEC = (f"sdk:mobile-app:{KEYS['sdk']},owner:ivanova:{KEYS['ivanova']},owner:sidorov:{KEYS['sidorov']},"
        f"analyst:petrov:{KEYS['petrov']},admin:root:{KEYS['root']},reader:guest:{KEYS['guest']}")


def h(name):
    return {"Authorization": f"Bearer {KEYS[name]}"}


def make_app(storage=None, exposures=None):
    service = ExperimentService(storage or MemoryStorage())
    return create_app(exposure_store=exposures, service=service, auth=Auth.parse(SPEC)), service


def running(http, exp="exp_menu", **kw):
    """Провести тест по HTTP до Running: слой, черновик, рецензия, утверждение, запуск."""
    http.post("/layers", json={"layer_id": "ui"}, headers=h("root"))
    assert http.post("/passports", json=draft(exp, **kw), headers=h("ivanova")).status_code == 200
    assert http.post(f"/passports/{exp}/submit", headers=h("ivanova")).status_code == 200
    assert http.post(f"/passports/{exp}/approve", headers=h("petrov")).status_code == 200
    r = http.post(f"/passports/{exp}/start", headers=h("ivanova"))
    assert r.status_code == 200, r.text
    return r.json()


# ---------------- ключи (AUTH-1) ----------------
def test_auth_spec_parsing_and_errors():
    auth = Auth.parse(SPEC)
    assert auth.authenticate(h("petrov")["Authorization"]).role == "analyst"
    assert auth.authenticate(h("sdk")["Authorization"]).subject == "mobile-app"
    for bad in (None, "", KEYS["sdk"], "Basic " + KEYS["sdk"], "Bearer ", "Bearer " + KEYS["sdk"] + "x", "Bearer x",
                "Basic  " + KEYS["sdk"], "bearer " + KEYS["sdk"]):
        assert auth.authenticate(bad) is None
    long = "k" * 24
    for spec, text in ((f"sdk:app:{'k' * 23}", "короче 24"), (f"boss:app:{long}", "неизвестная роль"),
                       (f"sdk:app", "роль:имя:ключ"), (f"sdk::{long}", "роль:имя:ключ"), (" , ", "ни одного ключа"),
                       (f"sdk:a:{long},admin:b:{long}", "ключ записи №2 повторяется"),
                       (f"owner:a:{long},admin:a:{long}x", "имя 'a' повторяется")):
        with pytest.raises(AuthConfigError, match=text):
            Auth.parse(spec)
    # двоеточия в ключе допустимы: делится только по первым двум
    assert Auth.parse(f"sdk:app:{long}:with:colons").authenticate(f"Bearer {long}:with:colons").subject == "app"


def test_server_does_not_start_without_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("ABP_API_KEYS", raising=False)
    monkeypatch.delenv("ABP_AUTH_DISABLED", raising=False)
    with pytest.raises(AuthConfigError, match="Не заданы ключи доступа"):
        create_app()
    with pytest.raises(AuthConfigError):
        Auth.from_env({"ABP_AUTH_DISABLED": "true"})  # только явное «1»
    monkeypatch.setenv("ABP_AUTH_DISABLED", "1")
    assert create_app().state.auth.is_disabled
    monkeypatch.setenv("ABP_API_KEYS", SPEC)  # ключи заданы — режим стенда не включается
    assert not create_app().state.auth.is_disabled
    monkeypatch.delenv("ABP_DB_PATH", raising=False)
    with pytest.raises(RuntimeError, match="ABP_DB_PATH"):
        create_app_from_env()


def test_sdk_routes_require_a_key_and_management_requires_a_person():
    app, service = make_app()
    http = TestClient(app)
    running(http)
    event = {"event_id": "e1", "event_time": "2026-11-10T10:00:00+00:00", "unit_id": "g1",
             "experiment_id": "exp_menu@v1", "variant_id": "control", "ruleset_version": 2}
    decide = {"unit_id": "g1", "unit_type": "guest_id", "attributes": IOS}
    for headers in ({}, {"Authorization": "Bearer wrong-key-000000000000000000"}, {"Authorization": KEYS["sdk"]}):
        r = [http.get("/ruleset", headers=headers), http.post("/decide", json=decide, headers=headers),
             http.post("/exposures", json=[event], headers=headers), http.get("/passports", headers=headers),
             http.post("/passports/exp_menu/stop", headers=headers), http.get("/audit", headers=headers)]
        assert [x.status_code for x in r] == [401] * 6 and r[0].headers["www-authenticate"] == "Bearer"
    assert len(app.state.exposures) == 0 and service.get("exp_menu").status.value == "running"
    assert http.get("/health").status_code == 200  # без ключа: только версия и счётчик
    # ключ SDK: чтение ruleset, решение, запись событий — и ничего больше
    assert http.get("/ruleset", headers=h("sdk")).status_code == 200
    assert http.post("/decide", json=decide, headers=h("sdk")).status_code == 200
    assert http.post("/exposures", json=[event], headers=h("sdk")).json()["accepted"] == 1
    n = len(service.storage.audit())
    for method, path, body in (("get", "/passports", None), ("get", "/passports/exp_menu", None), ("get", "/audit", None),
                               ("post", "/passports", draft("x")), ("post", "/passports/exp_menu/stop", None),
                               ("post", "/layers", {"layer_id": "z"}), ("put", "/holdout", {"pct": 1}),
                               ("get", "/audit/verify", None)):
        r = http.request(method, path, json=body, headers=h("sdk"))
        assert r.status_code == 403, (path, r.text)
    denied = service.storage.audit()[n:]
    assert len(denied) == 8 and {(r.action, r.actor) for r in denied} == {("auth.denied", "mobile-app")}
    assert service.get("exp_menu").status.value == "running" and [l.layer_id for l in service.ruleset().layers] == ["ui"]
    # ключи людей читают ruleset (любая роль), читатель — только смотрит
    assert http.get("/ruleset", headers=h("guest")).status_code == 200
    assert http.get("/passports", headers=h("guest")).json()[0]["status"] == "running"
    assert http.post("/passports/exp_menu/stop", headers=h("guest")).status_code == 403


def test_stand_mode_opens_sdk_routes_but_not_management():
    service = ExperimentService(MemoryStorage())
    http = TestClient(create_app(service=service, auth=OPEN))
    assert http.get("/ruleset").status_code == 503 and http.get("/health").json()["auth"] == "disabled"
    for r in (http.get("/passports"), http.post("/layers", json={"layer_id": "ui"}),
              http.post("/passports", json=draft())):
        assert r.status_code == 403 and "режиме стенда" in r.json()["detail"]
    assert service.ruleset() is None
    # без контура управления маршрутов управления нет
    assert TestClient(create_app(auth=Auth.parse(SPEC))).get("/passports", headers=h("root")).status_code == 404


# ---------------- команды по HTTP ----------------
def test_lifecycle_over_http_with_roles_and_audit():
    app, service = make_app()
    http = TestClient(app)
    assert http.post("/layers", json={"layer_id": "ui"}, headers=h("ivanova")).status_code == 403
    assert http.post("/layers", json={"layer_id": "ui"}, headers=h("root")).json() == {"ruleset_version": 1, "layers": ["ui"]}
    assert http.post("/layers", json={"layer_id": "ui"}, headers=h("root")).status_code == 409
    assert http.post("/layers", json={}, headers=h("root")).status_code == 422
    assert http.put("/holdout", json={"pct": 2.5}, headers=h("root")).json() == {"ruleset_version": 2, "holdout_pct": 2.5}
    assert http.put("/holdout", json={"pct": "5"}, headers=h("root")).status_code == 422
    assert http.put("/holdout", json={"pct": 100}, headers=h("root")).status_code == 422

    p = http.post("/passports", json=draft(), headers=h("ivanova")).json()
    assert (p["status"], p["version"], p["owner"], p["target_daily_units"]) == ("draft", 1, "ivanova", 4000)
    assert http.post("/passports", json={**draft(), "exposure_surface": "pre_login"}, headers=h("ivanova")).status_code == 422
    assert http.post("/passports", json={**draft(), "targetting": {}}, headers=h("ivanova")).status_code == 422
    assert http.post("/passports", json=draft(), headers=h("sidorov")).status_code == 403  # чужой черновик
    assert http.post("/passports/exp_menu/start", headers=h("ivanova")).status_code == 409  # не утверждён
    assert http.post("/passports/exp_menu/submit", headers=h("ivanova")).json()["status"] == "review"
    for who in ("ivanova", "sidorov", "root", "guest"):
        assert http.post("/passports/exp_menu/approve", headers=h(who)).status_code == 403
    a = http.post("/passports/exp_menu/approve", headers=h("petrov")).json()
    assert a["status"] == "approved" and a["approved_by"] == "petrov" and len(a["passport_hash"]) == 64
    assert http.post("/passports/exp_menu/start", headers=h("petrov")).status_code == 403
    assert http.post("/passports/exp_menu/start", headers=h("ivanova")).json() == {
        "ruleset_version": 3, "status": "running", "version": 1}

    rs = Ruleset.from_json(http.get("/ruleset", headers=h("sdk")).text)
    rule = rs.by_flag("flag_exp_menu")
    assert (rule.experiment_id, rule.ramp_pct, rule.passport_hash) == ("exp_menu@v1", 10.0, a["passport_hash"])
    assert http.post("/passports/exp_menu/ramp", json={"ramp_pct": 30}, headers=h("ivanova")).status_code == 409
    assert http.post("/passports/exp_menu/ramp", json={}, headers=h("ivanova")).status_code == 422
    assert http.post("/passports/exp_menu/ramp", json={"ramp_pct": 50}, headers=h("ivanova")).json()["ruleset_version"] == 4
    assert http.post("/passports/exp_menu/pause", headers=h("root")).json()["status"] == "paused"
    assert http.post("/passports/exp_menu/resume", headers=h("ivanova")).json()["status"] == "running"
    assert http.post("/passports/exp_menu/stop", headers=h("ivanova")).json()["status"] == "stopped"
    assert http.post("/passports/exp_menu/pause", headers=h("ivanova")).status_code == 409
    assert http.post("/passports/exp_menu/fly", headers=h("ivanova")).status_code == 404
    assert http.post("/passports/нет/submit", headers=h("ivanova")).status_code == 404
    assert http.get("/passports/нет", headers=h("guest")).status_code == 404
    v2 = http.post("/passports/exp_menu/versions", json={"mde_relative": 0.05}, headers=h("ivanova")).json()
    assert (v2["version"], v2["status"]) == (2, "draft")
    assert http.get("/passports/exp_menu", params={"version": 1}, headers=h("guest")).json()["status"] == "stopped"
    for to in ("analysis", "decision", "archived"):
        assert http.post("/passports/exp_menu/advance", json={"to": to}, headers=h("ivanova")).status_code == 403
    # попытки без прав записаны в аудит; цепочка цела
    audit = http.get("/audit", headers=h("guest")).json()
    denied = [r for r in audit if r["action"] == "auth.denied"]
    assert {r["actor"] for r in denied} == {"ivanova", "sidorov", "root", "guest", "petrov"}
    assert any(r["actor"] == "ivanova" and "второй человек" not in r["details"]["reason"] for r in denied)
    actions = [r["action"] for r in http.get("/audit", params={"experiment_id": "exp_menu"}, headers=h("guest")).json()
               if r["action"] != "auth.denied"]
    assert actions == ["passport.create", "passport.submit", "passport.approve", "passport.sealed", "experiment.start",
                       "experiment.ramp", "experiment.pause", "experiment.resume", "experiment.stop",
                       "passport.new_version"]
    check = http.get("/audit/verify", headers=h("guest")).json()
    assert check == {"records": len(audit), "ruleset_problems": []}


def test_no_route_publishes_a_ruleset():
    """LNK-1: в API нет способа записать ruleset или правило мимо Паспорта."""
    app, _ = make_app()
    writes = {(m, r.path) for r in app.routes for m in getattr(r, "methods", ()) if m in ("POST", "PUT", "PATCH", "DELETE")}
    assert writes == {("POST", "/decide"), ("POST", "/exposures"), ("POST", "/passports"),
                      ("POST", "/passports/{experiment_id}/versions"), ("POST", "/passports/{experiment_id}/{action}"),
                      ("POST", "/layers"), ("PUT", "/holdout")}
    http = TestClient(app)
    for method in ("post", "put", "patch", "delete"):
        assert http.request(method, "/ruleset", json={}, headers=h("root")).status_code == 405


# ---------------- API решений (DEC-1, DEC-2, TGT-3, TGT-4, PRM-1) ----------------
def test_decide_matches_all_vectors():
    """DEC-1: POST /decide даёт те же решения, что SDK, на 100% общих векторов — включая таргетинг и параметры."""
    rss = [Ruleset.model_validate(r) for r in V["rulesets"]]
    plain = [TestClient(create_app(RulesetStore(r), ExposureStore(), auth=OPEN)) for r in rss]
    # второй сервер на ruleset: у него в хранилище уже есть первый показ каждого гостя в каждом тесте
    enrolled = []
    for r in rss:
        store = ExposureStore()
        http = TestClient(create_app(RulesetStore(r), store, auth=OPEN))
        units = sorted({c["unit"] for c in V["cases"] if c.get("enrolled")})
        events = [{"event_id": f"{u}|{e.experiment_id}", "event_time": "2026-11-10T10:00:00+00:00", "unit_id": u,
                   "experiment_id": e.experiment_id, "variant_id": e.control_variant_id, "ruleset_version": r.version}
                  for u in units for e in r.experiments]
        assert http.post("/exposures", json=events).json()["accepted"] == len(events)
        enrolled.append(http)
    n = 0
    for c in V["cases"]:
        rule = rss[c["rs"]].by_flag(c["flag"])
        http = (enrolled if c.get("enrolled") else plain)[c["rs"]]
        body = http.post("/decide", json={"unit_id": c["unit"], "unit_type": c.get("unit_type", rule.unit_type),
                                          "attributes": c.get("attrs"), "flags": [c["flag"]]}).json()
        (d,) = body["decisions"]
        got = (d["variant_id"], d["in_experiment"], d["reason"], d["params"], d["experiment_id"])
        assert got == (c["variant"], c["in"], c["reason"], c.get("params", {}), rule.experiment_id), c
        assert body["ruleset_version"] == rss[c["rs"]].version
        n += 1
    assert n == len(V["cases"]) > 5000
    assert all(len(http.app.state.exposures) == 0 for http in plain)  # DEC-2: решение показ не пишет


def test_decide_returns_all_flags_validates_input_and_writes_nothing():
    rs = Ruleset.model_validate(V["rulesets"][0])
    store = ExposureStore()
    http = TestClient(create_app(RulesetStore(rs), store, auth=OPEN))
    body = http.post("/decide", json={"unit_id": "guest_1", "unit_type": "guest_id"}).json()
    assert [d["flag_key"] for d in body["decisions"]] == [e.flag_key for e in rs.experiments]
    assert (body["unit_id"], body["unit_type"], body["ruleset_version"]) == ("guest_1", "guest_id", 7)
    two = http.post("/decide", json={"unit_id": "guest_1", "unit_type": "guest_id", "flags": ["ops_flag", "нет"]}).json()
    assert [d["flag_key"] for d in two["decisions"]] == ["ops_flag"]
    for bad in ({"unit_id": "", "unit_type": "guest_id"}, {"unit_id": "g", "unit_type": "session_id"}, {"unit_id": "g"},
                {"unit_type": "guest_id"}, {"unit_id": "g" * 1025, "unit_type": "guest_id"},
                {"unit_id": "g", "unit_type": "guest_id", "extra": 1}, {"unit_id": 5, "unit_type": "guest_id"},
                {"unit_id": "g", "unit_type": "guest_id", "attributes": "ios"}):
        assert http.post("/decide", json=bad).status_code == 422, bad
    assert http.post("/decide", content=b'{"unit_id": "\\ud800", "unit_type": "guest_id"}',
                     headers={"content-type": "application/json"}).status_code == 422  # одиночный суррогат
    assert len(store) == 0
    assert TestClient(create_app(auth=OPEN)).post("/decide", json={"unit_id": "g", "unit_type": "guest_id"}).status_code == 503


def test_sticky_eligibility_over_http_tgt3():
    """Гость с первым показом остаётся в тесте после смены признака; гость без показа — нет."""
    app, _ = make_app()
    http = TestClient(app)
    running(http, ramp_plan=[100])
    updated = {"platform": "ios", "app_version": "4.0"}  # ниже нижней границы 5.2

    def decide(unit, attrs):
        return http.post("/decide", json={"unit_id": unit, "unit_type": "guest_id", "attributes": attrs},
                         headers=h("sdk")).json()["decisions"][0]
    units = [u for u in (f"g{i}" for i in range(200)) if decide(u, IOS)["in_experiment"]][:20]
    assert len(units) == 20 and all(decide(u, updated)["reason"] == "targeting:app_version" for u in units)
    shown, hidden = units[:10], units[10:]
    events = [{"event_id": f"ev-{u}", "event_time": "2026-11-10T10:00:00+00:00", "unit_id": u, "unit_type": "guest_id",
               "experiment_id": "exp_menu@v1", "variant_id": decide(u, IOS)["variant_id"], "ruleset_version": 2,
               "attributes": IOS} for u in shown]
    assert http.post("/exposures", json=events, headers=h("sdk")).json()["accepted"] == 10
    for u in shown:
        after = decide(u, updated)
        assert after["in_experiment"] and after["variant_id"] == decide(u, IOS)["variant_id"]
        assert after["params"] in ({"title": "Меню"}, {"title": "Новое меню"})
    assert all(decide(u, updated)["reason"] == "targeting:app_version" for u in hidden)
    # закрепление — только в своём тесте и не сильнее kill-switch
    assert http.post("/passports/exp_menu/pause", headers=h("root")).status_code == 200
    assert {decide(u, IOS)["reason"] for u in units} == {"kill_switch"}
    # состав анализа: первый показ каждого гостя со снимком признаков
    first = app.state.exposures.first_exposures("exp_menu@v1")
    assert set(first) == set(shown) and all(e["attributes"] == IOS for e in first.values())


def test_first_exposure_is_the_earliest_by_event_time():
    store = ExposureStore()
    http = TestClient(create_app(RulesetStore(Ruleset.model_validate(V["rulesets"][0])), store, auth=OPEN))
    def ev(i, t, variant, attrs):
        return {"event_id": f"e{i}", "event_time": t, "unit_id": "g1", "experiment_id": "exp_menu",
                "variant_id": variant, "ruleset_version": 7, "attributes": attrs}
    # события приходят не по порядку и в разных часовых поясах
    http.post("/exposures", json=[ev(1, "2026-11-10T12:00:00+00:00", "treatment", {"app_version": "6.0"}),
                                  ev(2, "2026-11-10T13:30:00+03:00", "treatment", {"app_version": "5.4"}),
                                  ev(3, "2026-11-10T11:00:00Z", "treatment", {"app_version": "5.9"})])
    assert store.first_exposures("exp_menu")["g1"]["attributes"] == {"app_version": "5.4"}  # 10:30 UTC
    assert store.first_exposures("другой") == {} and store.enrolled("g1") == {"exp_menu"} and store.enrolled("g2") == set()


# ---------------- тонкий клиент Python (DEC-3) ----------------
class FlakyTransport(httpx.BaseTransport):
    """Прокладка перед приложением: по флагу down имитирует недоступный сервер."""

    def __init__(self, app):
        self.inner, self.down, self.calls = TestClient(app), False, 0

    def handle_request(self, request):
        self.calls += 1
        if self.down:
            raise httpx.ConnectTimeout("сервер недоступен", request=request)
        r = self.inner.request(request.method, str(request.url), content=request.content, headers=request.headers)
        return httpx.Response(r.status_code, headers=r.headers, content=r.content)


def remote(app, **kw):
    transport = FlakyTransport(app)
    client = RemoteDecisionClient(client=httpx.Client(transport=transport, base_url="http://test"), **kw)
    return client, transport


def test_remote_client_decides_exposes_and_fails_open():
    app, _ = make_app()
    running(TestClient(app), ramp_plan=[100])
    now = [1000.0]
    client, net = remote(app, api_key=KEYS["sdk"], clock=lambda: now[0], max_stale_s=600)
    unit = next(u for u in (f"g{i}" for i in range(100))
                if client.decide(u, "guest_id", IOS)["flag_exp_menu"].in_experiment)
    d = client.decide(unit, "guest_id", {**IOS, "phone": "+7"})
    menu = d["flag_exp_menu"]
    assert not d.from_cache and d.fallback is None and d.flags() == ["flag_exp_menu"]
    assert menu.in_experiment and menu.params in ({"title": "Меню"}, {"title": "Новое меню"})
    assert (menu.experiment_id, menu.ruleset_version, menu.unit_type) == ("exp_menu@v1", 2, "guest_id")
    unknown = d.get("нет_флага", default="old")
    assert (unknown.variant_id, unknown.reason, unknown.in_experiment) == ("old", FALLBACK_UNKNOWN_FLAG, False)
    # DEC-2: решение без показа ничего не отправляет; показ — один раз на сессию, со снимком признаков без лишнего
    assert client.pending_exposures() == [] and len(app.state.exposures) == 0
    assert client.expose(menu, "s1") and not client.expose(menu, "s1") and client.expose(menu, "s2")
    assert not client.expose(unknown, "s1") and client.flush() == 2 and client.flush() == 0
    stored = app.state.exposures.events()
    assert [(e["unit_id"], e["variant_id"], e["attributes"], e["unit_type"]) for e in stored] == \
        [(unit, menu.variant_id, IOS, "guest_id")] * 2 and client.rejected == []

    # DEC-3: сервер недоступен → последние решения этого гостя из кэша
    net.down = True
    cached = client.decide(unit, "guest_id", IOS)
    assert cached.from_cache and cached["flag_exp_menu"].variant_id == menu.variant_id
    assert cached["flag_exp_menu"].reason == "assigned" and client.counters["cache_hit"] == 1
    # другого гостя в кэше нет → контроль с причиной
    stranger = client.decide("никогда_не_был", "guest_id", IOS)
    got = stranger.get("flag_exp_menu", default="control")
    assert (got.variant_id, got.in_experiment, got.reason, got.params) == ("control", False, FALLBACK_DECIDE_UNAVAILABLE, {})
    assert stranger.fallback == FALLBACK_DECIDE_UNAVAILABLE and not client.expose(got, "s1")
    # кэш старше max_stale_s → контроль
    now[0] += 601
    stale = client.decide(unit, "guest_id", IOS)["flag_exp_menu"]
    assert (stale.variant_id, stale.reason) == ("control", FALLBACK_STALE_DECISION)
    # показ при недоступном сервере не теряется и не дублируется после восстановления
    now[0] -= 601
    assert client.expose(client.decide(unit, "guest_id", IOS)["flag_exp_menu"], "s3") and client.flush() == 0
    assert client.counters["sink_error"] == 1 and len(client.pending_exposures()) == 1
    net.down = False
    assert client.flush() == 1 and len(app.state.exposures) == 3
    assert client.decisions[FALLBACK_DECIDE_UNAVAILABLE] == 1 and client.decisions[FALLBACK_STALE_DECISION] == 1
    assert 0 < client.fallback_rate < 1 and "ConnectTimeout" in client.last_error


def test_remote_client_without_key_or_with_broken_cache_gives_control():
    app, _ = make_app()
    running(TestClient(app), ramp_plan=[100])
    nokey, _ = remote(app)
    d = nokey.decide("g1", "guest_id", IOS)["flag_exp_menu"]
    assert (d.variant_id, d.reason) == ("control", FALLBACK_DECIDE_UNAVAILABLE) and "401" in nokey.last_error

    class BrokenCache(dict):
        def get(self, *a):
            raise OSError("хранилище устройства недоступно")
    broken, net = remote(app, api_key=KEYS["sdk"], cache=BrokenCache())
    net.down = True
    assert broken.decide("g1", "guest_id", IOS)["flag_exp_menu"].reason == FALLBACK_DECIDE_UNAVAILABLE
    garbage, net = remote(app, api_key=KEYS["sdk"], cache={"guest_id:g1": "мусор"})
    net.down = True
    assert garbage.decide("g1", "guest_id", IOS)["flag_exp_menu"].reason == FALLBACK_DECIDE_UNAVAILABLE


def test_local_sdk_with_api_key_against_keyed_server():
    app, _ = make_app()
    http = TestClient(app)
    running(http, ramp_plan=[100])
    ok = ExperimentClient(HttpRulesetFetcher(client=http, api_key=KEYS["sdk"]),
                          HttpExposureSink(client=http, api_key=KEYS["sdk"]))
    assert ok.refresh() is True and ok.refresh() is False
    unit = next(u for u in (f"g{i}" for i in range(100)) if ok.get_variant("flag_exp_menu", u, attributes=IOS).in_experiment)
    assert ok.flush() >= 1 and unit in {e["unit_id"] for e in app.state.exposures.events()}
    anonymous = ExperimentClient(HttpRulesetFetcher(client=http), HttpExposureSink(client=http))
    assert anonymous.refresh() is False and "401" in anonymous.last_refresh_error
    assert anonymous.get_variant("flag_exp_menu", unit, attributes=IOS).reason == "fallback:no_ruleset"


# ---------------- перезапуск (LNK-3) ----------------
def test_restart_keeps_passports_rulesets_and_audit(tmp_path, monkeypatch):
    db, jsonl = str(tmp_path / "ab.db"), str(tmp_path / "exposures.jsonl")
    monkeypatch.setenv("ABP_DB_PATH", db)
    monkeypatch.setenv("ABP_API_KEYS", SPEC)
    monkeypatch.setenv("ABP_EXPOSURES_PATH", jsonl)
    first = create_app_from_env()
    http = TestClient(first)
    running(http, ramp_plan=[10, 100])
    http.post("/passports/exp_menu/ramp", json={"ramp_pct": 100}, headers=h("ivanova"))
    before = http.get("/ruleset", headers=h("sdk"))
    unit = next(u for u in (f"g{i}" for i in range(100)) if http.post(
        "/decide", json={"unit_id": u, "unit_type": "guest_id", "attributes": IOS}, headers=h("sdk")
    ).json()["decisions"][0]["in_experiment"])
    http.post("/exposures", headers=h("sdk"), json=[{
        "event_id": "e1", "event_time": "2026-11-10T10:00:00+00:00", "unit_id": unit, "experiment_id": "exp_menu@v1",
        "variant_id": "control", "ruleset_version": 3, "attributes": IOS}])
    audit_before = http.get("/audit", headers=h("guest")).json()
    first.state.service.storage.close()

    second = create_app_from_env()  # «перезапуск»: новый процесс на той же базе
    http2 = TestClient(second)
    after = http2.get("/ruleset", headers=h("sdk"))
    assert after.text == before.text and after.headers["etag"] == before.headers["etag"]
    assert http2.get("/ruleset", headers={**h("sdk"), "If-None-Match": before.headers["etag"]}).status_code == 304
    assert http2.get("/passports/exp_menu", headers=h("guest")).json()["status"] == "running"
    assert http2.get("/audit", headers=h("guest")).json() == audit_before
    assert http2.get("/audit/verify", headers=h("guest")).json()["ruleset_problems"] == []
    # история версий на месте; работа продолжается с того же места
    storage = second.state.service.storage
    assert [storage.get_ruleset(v).version for v in (1, 2, 3)] == [1, 2, 3] and storage.get_ruleset(4) is None
    assert http2.post("/passports/exp_menu/pause", headers=h("root")).json() == {
        "ruleset_version": 4, "status": "paused", "version": 1}
    # закрепление гостя после первого показа тоже пережило перезапуск (показы — в jsonl)
    assert second.state.exposures.enrolled(unit) == {"exp_menu@v1"}
    storage.close()


@pytest.mark.parametrize("sql, text", [
    ("UPDATE audit SET actor = 'mallory' WHERE seq = 3", "Аудит: запись №3"),
    ("UPDATE passports SET body = replace(body, '\"mde_relative\":0.03', '\"mde_relative\":0.5')", "Ruleset расходится с Паспортами"),
    ("UPDATE rulesets SET body = replace(body, '\"ramp_pct\":10.0', '\"ramp_pct\":77.0') "
     "WHERE version = (SELECT MAX(version) FROM rulesets)", "вне плана"),
])
def test_server_refuses_to_start_on_tampered_database(tmp_path, monkeypatch, sql, text):
    db = str(tmp_path / "ab.db")
    monkeypatch.setenv("ABP_DB_PATH", db)
    monkeypatch.setenv("ABP_API_KEYS", SPEC)
    app = create_app_from_env()
    running(TestClient(app))
    app.state.service.storage.close()
    raw = sqlite3.connect(db)
    assert raw.execute(sql).rowcount >= 1
    raw.commit()
    raw.close()
    with pytest.raises(RuntimeError, match=text):
        create_app_from_env()


# ---------------- тонкий клиент JS против настоящего сервера ----------------
JS_REMOTE = r"""
import { createRemoteClient, ExperimentClient, createHttpFetcher, createHttpSink } from "%s";
const [url, key] = [process.argv[2], process.argv[3]];
const IOS = { platform: "ios", app_version: "5.4" };
const saved = new Map();
const storage = { getItem: (k) => saved.get(k) ?? null, setItem: (k, v) => void saved.set(k, v) };
const client = createRemoteClient(url, { apiKey: key, storage });
const out = [];
let shown = null;
for (let i = 0; i < 60; i++) {
  const d = await client.decide(`g${i}`, "guest_id", { ...IOS, phone: "+7" });
  const m = d.get("flag_exp_menu");
  out.push([m.unitId, m.variantId, m.inExperiment, m.reason, m.params]);
  if (m.inExperiment && !shown) shown = m;
}
const exposed = [client.expose(shown, "s1"), client.expose(shown, "s1"), client.expose(shown, "s2")];
const sent = await client.flush();
const noKey = (await createRemoteClient(url).decide("g1", "guest_id", IOS)).get("flag_exp_menu");
const outside = [client.expose(noKey, "s1"), client.expose(null, "s1"), client.pendingExposures().length];
// сервер «пропал»: тот же клиент с адресом, где никто не слушает, и тем же хранилищем
const dead = createRemoteClient("http://127.0.0.1:9", { apiKey: key, storage, timeoutMs: 300 });
const cachedAll = await dead.decide(shown.unitId, "guest_id", IOS);
const cached = cachedAll.get("flag_exp_menu");
const stranger = (await dead.decide("никогда_не_был", "guest_id", IOS)).get("flag_exp_menu");
const stale = createRemoteClient("http://127.0.0.1:9", { apiKey: key, storage, timeoutMs: 300, maxStaleMs: -1 });
const old = (await stale.decide(shown.unitId, "guest_id", IOS)).get("flag_exp_menu");
// локальный SDK с ключом
const local = new ExperimentClient({ fetcher: createHttpFetcher(url, { apiKey: key }), sink: createHttpSink(url, { apiKey: key }) });
const refreshed = await local.refresh();
const anon = new ExperimentClient({ fetcher: createHttpFetcher(url) });
const anonRefreshed = await anon.refresh();
console.log(JSON.stringify({
  out, exposed, sent, rejected: client.rejected, counters: client.counters, shown: shown.unitId,
  noKey: [noKey.variantId, noKey.reason], outside,
  cached: [cachedAll.fromCache, cached.variantId, cached.reason],
  stranger: [stranger.variantId, stranger.inExperiment, stranger.reason],
  old: [old.variantId, old.reason], deadCounters: dead.counters, fallbackRate: dead.fallbackRate,
  local: [refreshed, local.getVariant("flag_exp_menu", shown.unitId, { attributes: IOS, logExposure: false }).variantId],
  anon: [anonRefreshed, anon.lastRefreshError],
}));
"""


@pytest.mark.skipif(NODE is None, reason="node не установлен")
def test_js_remote_client_against_real_server(tmp_path):
    app, _ = make_app()
    running(TestClient(app), ramp_plan=[100])
    script = tmp_path / "remote.mjs"
    script.write_text(JS_REMOTE % (ROOT / "sdk" / "js" / "ab-sdk.mjs").as_uri(), encoding="utf-8")
    with Stand(app=app) as st:
        r = subprocess.run([NODE, str(script), st.url, KEYS["sdk"]], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-3000:]
        res = json.loads(r.stdout)
        # те же решения считает Python-клиент через тот же сервер
        py = RemoteDecisionClient(st.url, api_key=KEYS["sdk"])
        for unit, variant, inexp, reason, params in res["out"]:
            d = py.decide(unit, "guest_id", IOS)["flag_exp_menu"]
            assert (d.variant_id, d.in_experiment, d.reason, d.params) == (variant, inexp, reason, params), unit
        py.close()
    assert len(res["out"]) == 60 and {x[1] for x in res["out"] if x[2]} == {"control", "treatment"}
    assert res["exposed"] == [True, False, True] and res["sent"] == 2 and res["rejected"] == []
    stored = [e for e in app.state.exposures.events()]
    assert [(e["unit_id"], e["attributes"], e["unit_type"], e["session_id"]) for e in stored] == \
        [(res["shown"], IOS, "guest_id", "s1"), (res["shown"], IOS, "guest_id", "s2")]
    shown_variant = next(x[1] for x in res["out"] if x[0] == res["shown"])
    assert res["noKey"] == ["control", "fallback:decide_unavailable"] and res["outside"] == [False, False, 0]
    assert res["cached"] == [True, shown_variant, "assigned"]
    assert res["stranger"] == ["control", False, "fallback:decide_unavailable"]
    assert res["old"] == ["control", "fallback:stale_decision"]
    assert res["deadCounters"]["decide_error"] == 2 and res["deadCounters"]["cache_hit"] == 1
    assert res["fallbackRate"] == 0.5 and res["local"] == [True, shown_variant]
    assert res["anon"][0] is False and "401" in res["anon"][1]
