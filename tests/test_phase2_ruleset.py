"""Фаза 2, П2: ruleset, назначение по нему и проверка запуска (ТЗ v4 4.1; ASG-2, ASG-3, ASG-4)."""
import json

import numpy as np
import pytest
from pydantic import ValidationError

from ab_platform.engine.assignment import DeterministicAssignmentEngine as ASG, REASON_HOLDOUT
from ab_platform.engine.ruleset_engine import (
    REASON_KILL_SWITCH, LaunchConflictError, evaluate, launch, update_rule, validate_launch,
)
from ab_platform.schemas.ruleset import ExperimentRule, Layer, Ruleset


def variants(*ids, control=None):
    control = control or ids[0]
    share = 100.0 / len(ids)
    return [{"variant_id": v, "traffic_percentage": share, "is_control": v == control} for v in ids]


def rule(eid, layer="ui", s=0, e=10000, **kw):
    kw.setdefault("variants", variants("control", "treatment"))
    return ExperimentRule(experiment_id=eid, layer_id=layer, slot_start=s, slot_end=e, **kw)


@pytest.fixture()
def rs():
    return Ruleset(
        version=1,
        layers=[Layer(layer_id="ui"), Layer(layer_id="algo"), Layer(layer_id="ops", marketing=False)],
        experiments=[
            rule("exp_menu", "ui", 0, 5000),
            rule("exp_promo", "ui", 5000, 10000, variants=variants("A", "B", "C", control="B")),
            rule("exp_rank", "algo"),
            rule("exp_ops", "ops"),
        ],
    )


IDS = [f"guest_{i}" for i in range(20000)]


# --- Схема ---------------------------------------------------------------------------------------

def test_defaults_and_lookup(rs):
    r = rs.by_flag("exp_promo")
    assert r.flag_key == r.salt == "exp_promo" and r.control_variant_id == "B"
    assert rs.layer("ui").salt == "layer:ui"
    assert rs.by_flag("нет_такого") is None
    assert rs.holdout.pct == 5.0


def test_etag_depends_on_content_only(rs):
    same = Ruleset.from_json(rs.to_json())
    assert same == rs and same.etag == rs.etag
    assert json.loads(rs.to_json())["etag"] == rs.etag
    killed = update_rule(rs, "exp_menu", kill_switch=True)
    assert killed.etag != rs.etag and killed.version == 2


def test_frozen(rs):
    with pytest.raises(ValidationError):
        rs.version = 7
    with pytest.raises(ValidationError):
        rs.experiments[0].ramp_pct = 1.0


@pytest.mark.parametrize("bad, msg", [
    ({"variants": variants("a", "b", control="zzz")}, "контрольным"),
    ({"variants": [{"variant_id": "a", "traffic_percentage": 60, "is_control": True},
                   {"variant_id": "b", "traffic_percentage": 30}]}, "100%"),
    ({"variants": [{"variant_id": "a", "traffic_percentage": 100, "is_control": True}]}, "двух"),
    ({"variants": [{"variant_id": "a", "traffic_percentage": 50, "is_control": True},
                   {"variant_id": "a", "traffic_percentage": 50}]}, "Повторяющиеся"),
    ({"slot_start": 6000, "slot_end": 6000}, "Пустая"),
    ({"unknown_field": 1}, "Extra"),
])
def test_rule_validation(bad, msg):
    kw = {"experiment_id": "e", "layer_id": "ui", "variants": variants("c", "t")}
    kw.update(bad)
    with pytest.raises(ValidationError, match=msg):
        ExperimentRule(**kw)


def test_ruleset_validation(rs):
    with pytest.raises(ValidationError, match="пересекается"):
        rs.next_version(experiments=[*rs.experiments, rule("exp_x", "ui", 4000, 6000)])
    with pytest.raises(ValidationError, match="неизвестный слой"):
        rs.next_version(experiments=[*rs.experiments, rule("exp_x", "нет")])
    with pytest.raises(ValidationError, match="соль"):
        rs.next_version(experiments=[*rs.experiments, rule("exp_x", "ops", salt="exp_rank")])
    with pytest.raises(ValidationError, match="Соли слоёв"):
        rs.next_version(layers=[*rs.layers, Layer(layer_id="ui2", salt="layer:ui")])
    with pytest.raises(ValidationError, match="не должны совпадать"):
        rs.next_version(holdout={"salt": "layer:ui", "pct": 5.0})
    with pytest.raises(ValidationError, match="layer_id"):
        rs.next_version(layers=[*rs.layers, Layer(layer_id="ui")])


# --- Запуск (ASG-3) ------------------------------------------------------------------------------

def test_launch_conflicts_named(rs):
    with pytest.raises(LaunchConflictError) as ei:
        validate_launch(rule("exp_new", "ui", 4000, 6000), rs)
    msg = str(ei.value)
    assert "exp_menu" in msg and "exp_promo" in msg and len(ei.value.conflicts) == 2

    with pytest.raises(LaunchConflictError, match="матрица.*exp_rank"):
        validate_launch(rule("exp_new", "ops2"), rs.next_version(layers=[*rs.layers, Layer(layer_id="ops2")]),
                        intersection_matrix={"exp_rank": ["exp_new"]})
    with pytest.raises(LaunchConflictError, match="не существует"):
        validate_launch(rule("exp_new", "нет"), rs)
    with pytest.raises(LaunchConflictError, match="соль"):
        validate_launch(rule("exp_new", "ops", salt="layer:ui"), rs)
    with pytest.raises(LaunchConflictError, match="уже есть"):
        validate_launch(rule("exp_menu", "algo"), rs)


def test_launch_ok_creates_new_version(rs):
    base = rs.next_version(experiments=[e for e in rs.experiments if e.experiment_id != "exp_promo"])
    new = launch(rule("exp_new", "ui", 5000, 10000), base, intersection_matrix={"exp_rank": ["exp_other"]})
    assert new.version == base.version + 1 and new.by_flag("exp_new") is not None
    assert base.by_flag("exp_new") is None  # старая версия не изменилась


def test_update_rule_only_grows(rs):
    r = rs.next_version(experiments=[rule("exp_menu", "ui", 0, 4000, ramp_pct=10.0)])
    grown = update_rule(r, "exp_menu", ramp_pct=50.0, slot_end=5000)
    assert grown.by_flag("exp_menu").ramp_pct == 50.0
    for bad in ({"ramp_pct": 5.0}, {"slot_end": 3000}, {"slot_start": 100}):
        with pytest.raises(ValueError, match="только расти"):
            update_rule(r, "exp_menu", **bad)
    with pytest.raises(ValueError, match="Сдвиг начала"):
        update_rule(rs.next_version(experiments=[rule("exp_menu", "ui", 1000, 4000, ramp_pct=10.0)]),
                    "exp_menu", slot_start=0)
    with pytest.raises(ValueError, match="Менять можно"):
        update_rule(r, "exp_menu", variants=variants("x", "y"))
    with pytest.raises(KeyError):
        update_rule(r, "нет", ramp_pct=60.0)
    with pytest.raises(LaunchConflictError):
        update_rule(rs, "exp_menu", slot_end=6000)


# --- Назначение по ruleset -----------------------------------------------------------------------

def test_evaluate_matches_engine(rs):
    r = rs.by_flag("exp_promo")
    for u in IDS[:2000]:
        a = evaluate(rs, r, u)
        b = ASG.assign_user(u, "exp_promo", global_holdout_pct=5.0, layer_id="ui", layer_slot_start=5000,
                            layer_slot_end=10000, variants=r.variants)
        assert (a.variant_id, a.is_exposed, a.reason) == (b.variant_id, b.is_exposed, b.reason)
        assert a.variant_id in {"A", "B", "C"}


def test_lanes_exclusive_layers_independent(rs):
    menu = np.array([evaluate(rs, rs.by_flag("exp_menu"), u).is_exposed for u in IDS])
    promo = np.array([evaluate(rs, rs.by_flag("exp_promo"), u).is_exposed for u in IDS])
    rank = np.array([evaluate(rs, rs.by_flag("exp_rank"), u).is_exposed for u in IDS])
    assert not (menu & promo).any()
    assert abs(menu.mean() - 0.475) < 0.01 and abs(rank.mean() - 0.95) < 0.005
    # слои ортогональны: доля exp_rank одинакова среди участников и неучастников exp_menu
    nonhold = rank | menu | promo
    assert abs(rank[menu].mean() - rank[~menu & nonhold].mean()) < 0.01


def test_asg4_holdout_out_of_all_marketing_tests(rs):
    marketing = [r for r in rs.experiments if rs.layer(r.layer_id).marketing]
    held = 0
    for u in IDS:
        res = [evaluate(rs, r, u) for r in marketing]
        if any(x.reason == REASON_HOLDOUT for x in res):
            held += 1
            assert all(x.reason == REASON_HOLDOUT and not x.is_exposed for x in res)
    assert abs(held / len(IDS) - 0.05) < 0.005
    # немаркетинговый слой holdout не учитывает
    ops = [evaluate(rs, rs.by_flag("exp_ops"), u) for u in IDS]
    assert all(x.is_exposed for x in ops)


def test_kill_switch_returns_control(rs):
    killed = update_rule(rs, "exp_promo", kill_switch=True)
    res = [evaluate(killed, killed.by_flag("exp_promo"), u) for u in IDS[:500]]
    assert {(x.variant_id, x.is_exposed, x.reason) for x in res} == {("B", False, REASON_KILL_SWITCH)}


def test_asg2_sticky_through_ruleset_versions(rs):
    r1 = rs.next_version(experiments=[rule("exp_menu", "ui", 0, 2000, ramp_pct=5.0)])
    versions = [r1]
    for kw in ({"ramp_pct": 20.0}, {"ramp_pct": 50.0, "slot_end": 4000}, {"ramp_pct": 100.0, "slot_end": 6000}):
        versions.append(update_rule(versions[-1], "exp_menu", **kw))
    prev = {}
    for v in versions:
        for u in IDS:
            a = evaluate(v, v.by_flag("exp_menu"), u)
            if u in prev:
                assert a.is_exposed and a.variant_id == prev[u]
            if a.is_exposed:
                prev[u] = a.variant_id
    assert len(prev) > 0.55 * len(IDS)


def test_custom_layer_salt_is_used(rs):
    custom = Ruleset(version=1, layers=[Layer(layer_id="ui", salt="ui_v2")],
                     experiments=[rule("exp_menu", "ui", 0, 5000)])
    a = [evaluate(custom, custom.by_flag("exp_menu"), u).is_exposed for u in IDS[:3000]]
    expected = [ASG._hash_to_10000_buckets("ui_v2", u) < 5000 and
                ASG._hash_to_10000_buckets("global_holdout_v1", u) >= 500 for u in IDS[:3000]]
    assert a == expected
    default = [evaluate(rs, rs.by_flag("exp_menu"), u).is_exposed for u in IDS[:3000]]
    assert a != default


def test_empty_salts_and_taken_flag(rs):
    with pytest.raises(ValidationError, match="salt слоя"):
        Layer(layer_id="x", salt="")
    with pytest.raises(ValidationError, match="не могут быть пустыми"):
        rule("e", salt="")
    with pytest.raises(LaunchConflictError, match="flag_key 'exp_rank' уже занят"):
        validate_launch(rule("exp_new", "ops", flag_key="exp_rank"), rs)
