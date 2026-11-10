"""
Фаза 3, П2: таргетинг, тип единицы и параметры в движке и SDK; «все флаги за один проход»; линейная проверка ruleset.
Совпадение Python, JS и векторов — в tests/test_phase2_vectors.py.
"""
import random
import time

import pytest

from ab_platform.engine.ruleset_engine import REASON_UNIT_TYPE, evaluate, evaluate_all, update_rule
from ab_platform.schemas.ruleset import ExperimentRule, Layer, Ruleset, lane_conflicts
from ab_platform.sdk import ExperimentClient, known_attributes
from ab_platform.server import ExposureIn

V2 = [{"variant_id": "c", "traffic_percentage": 50.0, "is_control": True, "params": {"n": 0}},
      {"variant_id": "t", "traffic_percentage": 50.0, "params": {"n": 1, "text": "новое"}}]
IOS = {"platform": "ios", "app_version": "5.4"}


def targeted(**kw):
    rule = dict(experiment_id="exp", layer_id="L", variants=V2,
                targeting={"platforms": ["ios"], "app_version_min": "5.2", "app_version_max": "5.9"})
    rule.update(kw)
    return Ruleset(version=1, holdout={"pct": 0.0}, layers=[Layer(layer_id="L")], experiments=[rule])


def unit_in(rs, attrs=IOS):
    return next(u for u in (f"g{i}" for i in range(1000)) if evaluate(rs, rs.experiments[0], u, attrs).is_exposed)


# ---------------- движок ----------------
def random_ruleset(rnd: random.Random) -> Ruleset:
    layers, rules = [], []
    for li in range(rnd.randint(1, 5)):
        layers.append({"layer_id": f"L{li}", "marketing": rnd.random() < 0.7})
        cuts = sorted(rnd.sample(range(1, 10000), rnd.randint(0, 4)))
        for start, end in zip([0, *cuts], [*cuts, 10000]):
            if rnd.random() < 0.2:
                continue  # пустое место в слое
            targeting = {}
            if rnd.random() < 0.5:
                targeting["platforms"] = rnd.sample(["ios", "android", "web"], rnd.randint(1, 2))
            if rnd.random() < 0.3:
                targeting["app_version_min"] = rnd.choice(["5.2", "5.10", "6"])
            if rnd.random() < 0.3:
                targeting["has_delivery"] = rnd.random() < 0.5
            rules.append({"experiment_id": f"e{len(rules)}", "layer_id": f"L{li}", "slot_start": start, "slot_end": end,
                          "ramp_pct": rnd.choice([0.0, 12.5, 50.0, 100.0]), "kill_switch": rnd.random() < 0.15,
                          "unit_type": rnd.choice(["guest_id", "guest_id", "device_id"]), "targeting": targeting,
                          "variants": V2})
    rnd.shuffle(rules)  # порядок правил в ruleset не совпадает с порядком слоёв
    return Ruleset(version=1, holdout={"pct": rnd.choice([0.0, 5.0, 30.0])}, layers=layers, experiments=rules)


def test_evaluate_all_equals_evaluate_on_random_rulesets():
    rnd = random.Random(7)
    seen = set()
    for _ in range(60):
        rs = random_ruleset(rnd)
        for i in range(40):
            unit = f"u{rnd.randrange(10**6)}"
            attrs = rnd.choice([None, {}, {"platform": "ios", "app_version": "5.10", "has_delivery": True},
                                {"platform": "android", "app_version": "5.3", "has_delivery": False}, {"platform": "web"}])
            unit_type = rnd.choice([None, "guest_id", "device_id"])
            enrolled = {e.experiment_id for e in rs.experiments if rnd.random() < 0.3}
            got = evaluate_all(rs, unit, attrs, unit_type, enrolled)
            want = [evaluate(rs, e, unit, attrs, unit_type, e.experiment_id in enrolled) for e in rs.experiments]
            assert got == want
            seen |= {r.reason.split(":")[0] for r in got}
            # в слое гость участвует не больше чем в одном тесте
            for layer in rs.layers:
                assert sum(r.is_exposed for r in got if r.layer_id == layer.layer_id) <= 1
    assert seen == {"assigned", "holdout", "not_in_lane", "ramp", "kill_switch", REASON_UNIT_TYPE, "targeting",
                    "targeting_missing"}


def test_evaluate_all_rejects_bad_unit_and_handles_empty_ruleset():
    rs = targeted()
    for bad in ("", None, 5):
        with pytest.raises(ValueError):
            evaluate_all(rs, bad)
    assert evaluate_all(Ruleset(version=1), "u") == []


def test_targeting_is_checked_before_variant_and_same_for_all_variants():
    """TGT-2: условия не зависят от варианта — доля вариантов среди подходящих остаётся 50/50, вне условий теста нет."""
    rs = targeted()
    rule = rs.experiments[0]
    units = [f"g{i}" for i in range(20000)]
    ios = [evaluate(rs, rule, u, IOS) for u in units]
    android = [evaluate(rs, rule, u, {**IOS, "platform": "android"}) for u in units]
    assert all(r.is_exposed for r in ios) and not any(r.is_exposed for r in android)
    assert {r.reason for r in android} == {"targeting:platform"} and {r.variant_id for r in android} == {"c"}
    share = sum(r.variant_id == "t" for r in ios) / len(ios)
    assert 0.485 < share < 0.515
    # признаки не участвуют в хэше: вариант гостя не зависит от того, какие признаки переданы
    other = [evaluate(rs, rule, u, {"platform": "IOS", "app_version": "5.9.0", "city": "msk"}) for u in units[:2000]]
    assert [r.variant_id for r in other] == [r.variant_id for r in ios[:2000]]


def test_reason_order_lane_then_targeting_then_ramp():
    rs = Ruleset(version=1, layers=[Layer(layer_id="L")], experiments=[
        dict(experiment_id="exp", layer_id="L", slot_end=5000, ramp_pct=50.0, variants=V2,
             targeting={"platforms": ["ios"]})])
    rule = rs.experiments[0]
    bad, units = {"platform": "web"}, [f"g{i}" for i in range(4000)]
    good_reason = {u: evaluate(rs, rule, u, {"platform": "ios"}).reason for u in units}
    bad_reason = {u: evaluate(rs, rule, u, bad).reason for u in units}
    assert set(good_reason.values()) == {"assigned", "ramp", "holdout", "not_in_lane"}
    for u in units:
        want = "targeting:platform" if good_reason[u] in ("assigned", "ramp") else good_reason[u]
        assert bad_reason[u] == want
    # «первый показ уже был» снимает только таргетинг
    assert {u: evaluate(rs, rule, u, bad, enrolled=True).reason for u in units} == good_reason


def test_unit_type_and_kill_switch_precede_everything():
    rs = targeted(unit_type="device_id")
    rule, u = rs.experiments[0], unit_in(targeted(unit_type="device_id"))
    assert evaluate(rs, rule, u, IOS, "device_id").is_exposed and evaluate(rs, rule, u, IOS).is_exposed
    wrong = evaluate(rs, rule, u, IOS, "guest_id", enrolled=True)
    assert (wrong.reason, wrong.is_exposed, wrong.variant_id) == (REASON_UNIT_TYPE, False, "c")
    killed = update_rule(rs, "exp", kill_switch=True)
    assert evaluate(killed, killed.experiments[0], u, IOS, "guest_id", enrolled=True).reason == "kill_switch"
    assert evaluate_all(killed, u, IOS, "guest_id", {"exp"})[0].reason == "kill_switch"


def test_rule_carries_targeting_params_and_passport_reference():
    rs = targeted(passport_id="p1", passport_version=2, passport_hash="ab" * 32)
    rule = Ruleset.from_json(rs.to_json()).experiments[0]
    assert rule.targeting.platforms == ["ios"] and rule.params_of("t") == {"n": 1, "text": "новое"}
    assert rule.params_of("нет такого") == {} and (rule.passport_id, rule.passport_version) == ("p1", 2)
    assert rs.etag != targeted().etag  # ссылка на Паспорт и условия входят в ETag
    assert targeted(targeting={"platforms": ["android"]}).etag != targeted().etag
    with pytest.raises(ValueError):
        targeted(unit_type="session_id")
    with pytest.raises(ValueError, match="Менять можно только"):
        update_rule(rs, "exp", targeting={})  # условия идущего теста не меняются


# ---------------- проверка ruleset за линейное время ----------------
def many_rules(n):
    return ([Layer(layer_id=f"L{i}") for i in range(n // 10)],
            [ExperimentRule(experiment_id=f"e{i}", layer_id=f"L{i // 10}", slot_start=(i % 10) * 1000,
                            slot_end=(i % 10 + 1) * 1000, variants=V2) for i in range(n)])


def test_ruleset_validation_is_not_quadratic():
    """10 000 правил: попарная проверка занимала 5,7 с (проба фазы 3); запас по порогу — больше чем в 10 раз."""
    layers, rules = many_rules(10000)
    random.Random(1).shuffle(rules)  # порядок правил произвольный: быстрая проверка не должна от него зависеть
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        Ruleset(version=1, layers=layers, experiments=rules)
        best = min(best, time.perf_counter() - t)
    assert best < 0.5, best


def test_fast_conflict_check_agrees_with_pairwise():
    rnd = random.Random(3)
    outcomes = set()
    for _ in range(400):
        rules = []
        for i in range(rnd.randint(1, 6)):
            start = rnd.randrange(0, 9000, 1000)
            rules.append(ExperimentRule(
                experiment_id=rnd.choice([f"e{i}", "e0"]) if rnd.random() < 0.1 else f"e{i}",
                flag_key=rnd.choice([f"f{i}", "f0"]) if rnd.random() < 0.1 else f"f{i}",
                salt=rnd.choice([f"s{i}", "s0"]) if rnd.random() < 0.1 else f"s{i}",
                layer_id=rnd.choice(["A", "B"]), slot_start=start, slot_end=start + rnd.choice([1000, 2000]),
                variants=V2))
        pairwise = any(lane_conflicts(e, rules[:i]) for i, e in enumerate(rules))
        try:
            Ruleset(version=1, layers=[Layer(layer_id="A"), Layer(layer_id="B")], experiments=rules)
            accepted = True
        except ValueError as exc:
            accepted = False
            assert "Некорректный ruleset" in str(exc) and any(w in str(exc) for w in ("пересекается", "уже"))
        assert accepted == (not pairwise)
        outcomes.add(accepted)
    assert outcomes == {True, False}


# ---------------- SDK ----------------
def test_sdk_decision_carries_params_and_targeting_reason():
    rs = targeted()
    c, u = ExperimentClient(initial_ruleset=rs), unit_in(rs)
    d = c.get_variant("exp", u, attributes=IOS)
    assert d.in_experiment and d.params == rs.experiments[0].params_of(d.variant_id) and d.unit_type == "guest_id"
    out = ExperimentClient(initial_ruleset=rs).get_variant("exp", u, attributes={"platform": "web"})
    assert (out.variant_id, out.in_experiment, out.reason, out.params) == ("c", False, "targeting:platform", {"n": 0})
    assert ExperimentClient(initial_ruleset=rs).get_variant("exp", u).reason == "targeting_missing:platform"
    wrong_type = c.get_variant("exp", u, attributes=IOS, unit_type="device_id")
    assert wrong_type.reason == REASON_UNIT_TYPE and not wrong_type.in_experiment
    # fail-open: неизвестный флаг — без параметров; известный, но ruleset устарел — параметры контроля
    assert c.get_variant("нет", u).params == {}
    assert c._fallback("exp", u, "fallback:test", "x", rs.experiments[0], 1).params == {"n": 0}
    assert c.decisions["targeting:platform"] == 0 and c.fallback_rate > 0


def test_sdk_split_decision_and_exposure_dec2():
    rs = targeted()
    c, u = ExperimentClient(initial_ruleset=rs), unit_in(rs)
    d = c.get_variant("exp", u, attributes={**IOS, "phone": "+7900", "city": 5}, log_exposure=False)
    assert d.in_experiment and c.pending_exposures() == []  # решение без показа в анализ не попадает
    assert c.log_exposure(d, "s1") is True and c.log_exposure(d, "s1") is False and c.log_exposure(d, "s2") is True
    events = c.pending_exposures()
    assert [e.session_id for e in events] == ["s1", "s2"]
    # снимок признаков: только признаки таргетинга простых типов — лишнее и ПДн не уходят
    assert events[0].attributes == IOS and events[0].unit_type == "guest_id" and events[0].variant_id == d.variant_id
    outside = c.get_variant("exp", u, attributes={"platform": "web"}, log_exposure=False)
    fresh = ExperimentClient(initial_ruleset=rs)
    assert fresh.log_exposure(fresh.get_variant("exp", u, attributes={"platform": "web"}, log_exposure=False)) is False
    assert fresh.log_exposure(fresh.get_variant("нет", u)) is False and fresh.pending_exposures() == []
    assert outside.in_experiment  # у клиента c первый показ уже был — см. следующий тест


def test_sdk_sticky_eligibility_after_first_exposure_tgt3():
    rs = targeted()
    u = unit_in(rs)
    updated = {"platform": "ios", "app_version": "6.0"}  # вариант ускорил обновление: версия выше границы
    c = ExperimentClient(initial_ruleset=rs)
    assert c.get_variant("exp", u, attributes=updated).reason == "targeting:app_version"  # показа не было — вне теста
    first = c.get_variant("exp", u, attributes=IOS, session_id="s1")
    after = c.get_variant("exp", u, attributes=updated, session_id="s2")
    assert first.in_experiment and after.in_experiment and after.variant_id == first.variant_id
    assert len(c.pending_exposures()) == 2
    # решение без показа гостя не закрепляет
    d = ExperimentClient(initial_ruleset=rs)
    d.get_variant("exp", u, attributes=IOS, log_exposure=False)
    assert d.get_variant("exp", u, attributes=updated).reason == "targeting:app_version"
    # закрепление не переживает kill-switch и не переносится на другого гостя
    killed = ExperimentClient(initial_ruleset=rs)
    killed.get_variant("exp", u, attributes=IOS)
    killed._ruleset = update_rule(rs, "exp", kill_switch=True)
    assert killed.get_variant("exp", u, attributes=updated).reason == "kill_switch"
    other = next(x for x in (f"h{i}" for i in range(100)) if x != u)
    assert c.get_variant("exp", other, attributes=updated).reason == "targeting:app_version"


def test_known_attributes_snapshot():
    assert known_attributes(None) is None and known_attributes({}) is None and known_attributes({"x": 1}) is None
    assert known_attributes({"platform": "ios", "is_new_guest": False, "has_delivery": 1, "city": None, "email": "a@b"}) \
        == {"platform": "ios", "is_new_guest": False}


def test_exposure_schema_accepts_snapshot_and_rejects_foreign_fields():
    base = dict(event_id="e1", event_time="2026-11-10T10:00:00+00:00", unit_id="u", experiment_id="exp",
                variant_id="c", ruleset_version=1)
    ev = ExposureIn(**base, unit_type="device_id", attributes={"platform": "ios", "is_new_guest": True})
    assert ev.attributes == {"platform": "ios", "is_new_guest": True} and ev.unit_type == "device_id"
    assert ExposureIn(**base).attributes is None
    for bad in (dict(attributes={"email": "a@b.ru"}), dict(attributes={"platform": 5}), dict(unit_type="session"),
                dict(attributes={"city": "x" * 65})):
        with pytest.raises(ValueError):
            ExposureIn(**base, **bad)
