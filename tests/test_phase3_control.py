"""
Фаза 3, П4: контур управления — Паспорт → правило, жизненный цикл управляет ruleset, права, аудит (LNK-1, LNK-2).
"""
from contextlib import contextmanager

import pytest

from ab_platform.control import Actor, ControlError, ExperimentService, rule_from_passport, verify_ruleset
from ab_platform.engine.ruleset_engine import evaluate
from ab_platform.schemas.passport import ExperimentPassport, PassportStatus
from ab_platform.schemas.ruleset import Ruleset
from ab_platform.storage import MemoryStorage, SqliteStorage
from tests.test_phase1_passport_v4 import base, restaurant

OWNER, OWNER2 = Actor("ivanova", "owner"), Actor("sidorov", "owner")
ANALYST, ADMIN, READER, SDK = Actor("petrov", "analyst"), Actor("root", "admin"), Actor("guest", "reader"), Actor("app", "sdk")
V2 = [{"variant_id": "control", "traffic_percentage": 50.0, "is_control": True, "params": {"title": "Меню"}},
      {"variant_id": "treatment", "traffic_percentage": 50.0, "params": {"title": "Новое меню"}}]
IOS = {"platform": "ios", "app_version": "5.4"}


@pytest.fixture(params=["memory", "sqlite"])
def svc(request, tmp_path):
    storage = MemoryStorage() if request.param == "memory" else SqliteStorage(str(tmp_path / "ab.db"))
    service = ExperimentService(storage, intersection_matrix={"exp_push": ["exp_coupon"]})
    service.add_layer(ADMIN, "ui")
    yield service
    if request.param == "sqlite":
        storage.close()


def draft(exp="exp_menu", **kw):
    data = base(experiment_id=exp, test_key=f"flag_{exp}", layer_id="ui", variants=V2, ramp_plan=[10, 50, 100],
                targeting={"platforms": ["ios"], "app_version_min": "5.2"}, targeting_audience_share=0.4)
    data.update(kw)
    return {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in data.items()} | {
        "variants": data["variants"], "guardrail_metrics": [g.model_dump() for g in data["guardrail_metrics"]]}


def approved(svc, exp="exp_menu", **kw):
    svc.save_draft(OWNER, draft(exp, **kw))
    svc.submit(OWNER, exp)
    return svc.approve(ANALYST, exp)


@contextmanager
def denied(kind, text=None):
    """Команда отклонена с нужным видом ошибки (и текстом, если задан)."""
    with pytest.raises(ControlError, match=text) as e:
        yield
    assert e.value.kind == kind, (e.value.kind, str(e.value))


def kind_of(fn):
    with pytest.raises(ControlError) as e:
        fn()
    return e.value.kind


# ---------------- Паспорт → правило (LNK-1) ----------------
def test_rule_is_built_only_from_approved_untampered_guest_passport():
    p = ExperimentPassport(**draft(owner="ivanova"))
    for not_approved in (p, p.with_status("review")):
        with denied("conflict", "не утверждён"):
            rule_from_passport(not_approved)
    a = p.with_status("review").with_status("approved", actor="petrov")
    rule = rule_from_passport(a)
    assert (rule.experiment_id, rule.flag_key, rule.salt) == ("exp_menu@v1", "flag_exp_menu", "exp_menu@v1")
    assert (rule.layer_id, rule.ramp_pct, rule.unit_type, rule.kill_switch) == ("ui", 10.0, "guest_id", False)
    assert rule.targeting == a.targeting and rule.params_of("treatment") == {"title": "Новое меню"}
    assert (rule.passport_id, rule.passport_version, rule.passport_hash) == ("exp_menu", 1, a.passport_hash)
    forged = a.model_copy(update={"targeting": {"platforms": ["android"]}})
    with denied("conflict", "изменён после утверждения"):
        rule_from_passport(forged)
    rest = ExperimentPassport(**restaurant(owner="ivanova")).locked()
    with denied("invalid", "только для гостевого A/B"):
        rule_from_passport(rest)
    b2b = ExperimentPassport(**draft(randomization_unit="b2b_account_id", targeting={}, targeting_audience_share=None,
                                     owner="ivanova")).locked()
    with denied("invalid", "только для гостевого A/B"):
        rule_from_passport(b2b)
    device = ExperimentPassport(**draft(randomization_unit="device_id", exposure_surface="pre_login", owner="i")).locked()
    assert rule_from_passport(device).unit_type == "device_id"
    # новая версия Паспорта — новый идентификатор правила и новая соль: рандомизация заново
    v2 = a.new_version(mde_relative=0.05).locked()
    assert rule_from_passport(v2).salt == "exp_menu@v2" and rule_from_passport(v2).flag_key == rule.flag_key


def test_start_publishes_rule_and_only_from_approved(svc):
    svc.save_draft(OWNER, draft())
    with denied("conflict", "только утверждённый"):
        svc.start(OWNER, "exp_menu")
    svc.submit(OWNER, "exp_menu")
    with denied("conflict", "только утверждённый"):
        svc.start(OWNER, "exp_menu")
    a = svc.approve(ANALYST, "exp_menu")
    rs = svc.start(OWNER, "exp_menu")
    assert rs.version == 2 and svc.ruleset() == rs and svc.get("exp_menu").status == PassportStatus.RUNNING
    rule = rs.by_flag("flag_exp_menu")
    assert rule == rule_from_passport(svc.get("exp_menu")) and rule.passport_hash == a.passport_hash
    assert verify_ruleset(rs, svc.storage) == []
    with denied("conflict", "только утверждённый"):
        svc.start(OWNER, "exp_menu")  # повторный запуск


def test_ruleset_without_or_against_passport_is_rejected(svc):
    """LNK-1: правило без Паспорта, с чужим хэшем или с изменёнными условиями не проходит сверку и не публикуется."""
    approved(svc)
    rs = svc.start(OWNER, "exp_menu")
    rule = rs.experiments[0].model_dump()

    def with_rule(**change):
        return Ruleset.model_validate({**rs.model_dump(exclude={"etag"}), "version": rs.version + 1,
                                       "experiments": [{**rule, **change}]})
    cases = {
        "не связано с Паспортом": with_rule(passport_id=None),
        "Паспорт не найден": with_rule(passport_id="нет_такого"),
        "расходится с Паспортом": with_rule(targeting={"platforms": ["android"]}),
        "kill-switch не соответствует": with_rule(kill_switch=True),
        "вне плана": with_rule(ramp_pct=77.0),
    }
    cases["расходится с Паспортом "] = with_rule(passport_hash="0" * 64)
    cases["расходится с Паспортом  "] = with_rule(variants=[{**V2[0], "traffic_percentage": 10.0},
                                                            {**V2[1], "traffic_percentage": 90.0}])
    cases["расходится с Паспортом   "] = with_rule(slot_end=5000)
    for text, bad in cases.items():
        problems = verify_ruleset(bad, svc.storage)
        assert len(problems) == 1 and text.strip() in problems[0], (text, problems)
        with denied("conflict", "расходится с Паспортами"):
            svc._publish(bad)
    assert svc.ruleset() == rs
    # правило от Паспорта, который ещё не запущен
    approved(svc, "exp_other", layer_slot_start=0, layer_slot_end=10000, layer_id="ui")
    other = rule_from_passport(svc.get("exp_other")).model_dump()
    orphan = Ruleset.model_validate({**rs.model_dump(exclude={"etag"}), "version": rs.version + 1, "experiments": [other]})
    assert "в статусе 'approved'" in verify_ruleset(orphan, svc.storage)[0]


# ---------------- жизненный цикл управляет правилом (LNK-2) ----------------
def test_lifecycle_drives_kill_switch_and_keeps_variants(svc):
    approved(svc, ramp_plan=[100])
    rs = svc.start(OWNER, "exp_menu")
    units = [f"g{i}" for i in range(3000)]

    def variants(ruleset):
        rule = ruleset.by_flag("flag_exp_menu")
        return {u: (r.variant_id, r.reason) for u in units for r in [evaluate(ruleset, rule, u, IOS)]}
    running = variants(rs)
    assert {v for v, _ in running.values()} == {"control", "treatment"}
    paused = svc.pause(OWNER, "exp_menu")
    assert paused.version == 3 and set(variants(paused).values()) == {("control", "kill_switch")}
    assert svc.get("exp_menu").status == PassportStatus.PAUSED
    resumed = svc.resume(OWNER, "exp_menu")
    assert variants(resumed) == running  # после паузы у тех же гостей те же варианты
    stopped = svc.stop(ADMIN, "exp_menu")
    assert set(variants(stopped).values()) == {("control", "kill_switch")}
    assert svc.get("exp_menu").status == PassportStatus.STOPPED
    for fn in (svc.pause, svc.resume, svc.stop):
        with denied("conflict", "недоступно в статусе 'stopped'"):
            fn(OWNER, "exp_menu")
    # до архива правило остаётся (флаг известен SDK, гости на контроле); в архиве — убирается
    svc.advance(ANALYST, "exp_menu", "analysis")
    svc.advance(ANALYST, "exp_menu", "decision")
    assert svc.ruleset().by_flag("flag_exp_menu").kill_switch
    assert svc.advance(ANALYST, "exp_menu", "archived").status == PassportStatus.ARCHIVED
    assert svc.ruleset().experiments == [] and verify_ruleset(svc.ruleset(), svc.storage) == []
    with denied("conflict", "не запущен"):
        svc.stop(ADMIN, "exp_menu")
    for bad in ("running", "нет"):
        with denied("invalid"):
            svc.advance(ANALYST, "exp_menu", bad)


def test_ramp_only_by_plan_and_only_up(svc):
    approved(svc)
    rs = svc.start(OWNER, "exp_menu")
    rule = rs.by_flag("flag_exp_menu")
    units = [f"g{i}" for i in range(20000)]
    in10 = {u: r.variant_id for u in units for r in [evaluate(rs, rule, u, IOS)] if r.is_exposed}
    for bad in (10, 5, 30, 150):
        with denied("conflict", "недопустима"):
            svc.ramp(OWNER, "exp_menu", bad)
    rs50 = svc.ramp(OWNER, "exp_menu", 50)
    in50 = {u: r.variant_id for u in units for r in [evaluate(rs50, rs50.by_flag("flag_exp_menu"), u, IOS)] if r.is_exposed}
    assert rs50.by_flag("flag_exp_menu").ramp_pct == 50 and len(in50) > 3 * len(in10)
    assert all(in50[u] == v for u, v in in10.items())  # уже назначенные не меняют вариант
    svc.ramp(ADMIN, "exp_menu", 100)  # шаг плана можно пропускать вперёд, но не назад
    with denied("conflict", "недопустима"):
        svc.ramp(OWNER, "exp_menu", 50)
    svc.pause(OWNER, "exp_menu")
    with denied("conflict", "только у идущего"):
        svc.ramp(OWNER, "exp_menu", 100)


def test_launch_conflicts_are_named_and_nothing_is_saved(svc):
    approved(svc, "exp_menu", layer_slot_start=0, layer_slot_end=6000)
    svc.start(OWNER, "exp_menu")
    before = (svc.ruleset(), len(svc.storage.audit()))
    approved(svc, "exp_cart", layer_slot_start=5000, layer_slot_end=10000)
    n_audit = len(svc.storage.audit())
    with denied("conflict", r"дорожка \[5000, 10000\).*пересекается с 'exp_menu@v1'"):
        svc.start(OWNER, "exp_cart")
    # отказ ничего не оставил: статус, ruleset и аудит — как до команды
    assert svc.get("exp_cart").status == PassportStatus.APPROVED and svc.ruleset() == before[0]
    assert len(svc.storage.audit()) == n_audit
    approved(svc, "exp_nolayer", layer_id="нет_слоя")
    with denied("conflict", "слой 'нет_слоя' не существует"):
        svc.start(OWNER, "exp_nolayer")
    approved(svc, "exp_dup", test_key="flag_exp_menu", layer_slot_start=6000, layer_slot_end=7000)
    with denied("conflict", "flag_key 'flag_exp_menu' уже занят"):
        svc.start(OWNER, "exp_dup")
    # матрица пересечений задана по идентификаторам Паспортов и действует в обе стороны
    svc.add_layer(ADMIN, "comm")
    approved(svc, "exp_coupon", layer_id="comm", layer_slot_start=0, layer_slot_end=5000)
    approved(svc, "exp_push", layer_id="comm", layer_slot_start=5000, layer_slot_end=10000)
    svc.start(OWNER, "exp_coupon")
    with denied("conflict", "матрица пересечений запрещает одновременный запуск с 'exp_coupon@v1'"):
        svc.start(OWNER, "exp_push")
    svc.stop(OWNER, "exp_coupon")
    [svc.advance(ANALYST, "exp_coupon", s) for s in ("analysis", "decision", "archived")]
    assert svc.start(OWNER, "exp_push").by_flag("flag_exp_push") is not None
    # обратное направление: идёт exp_push — exp_coupon (новая версия) не запускается
    svc.new_version(OWNER, "exp_coupon", {"mde_relative": 0.05})
    svc.submit(OWNER, "exp_coupon")
    svc.approve(ANALYST, "exp_coupon")
    with denied("conflict", "матрица пересечений запрещает одновременный запуск с 'exp_push@v1'"):
        svc.start(OWNER, "exp_coupon")


def test_new_version_is_new_randomization_and_old_must_be_stopped(svc):
    approved(svc, ramp_plan=[100])
    rs1 = svc.start(OWNER, "exp_menu")
    v2 = svc.new_version(OWNER, "exp_menu", {"variants": [{**V2[0], "traffic_percentage": 80.0},
                                                         {**V2[1], "traffic_percentage": 20.0}]})
    assert v2.version == 2 and v2.status == PassportStatus.DRAFT and v2.approved_by is None
    # идущая версия не тронута; пауза и остановка относятся к ней, а не к черновику v2
    assert svc.ruleset() == rs1 and svc.get("exp_menu", 1).status == PassportStatus.RUNNING
    svc.submit(OWNER, "exp_menu")
    svc.approve(ANALYST, "exp_menu")
    with denied("conflict", "Версия 1 теста ещё идёт"):
        svc.start(OWNER, "exp_menu")
    svc.pause(OWNER, "exp_menu")
    assert svc.get("exp_menu", 1).status == PassportStatus.PAUSED and svc.get("exp_menu", 2).status == PassportStatus.APPROVED
    with denied("conflict", "Версия 1 теста ещё идёт"):
        svc.start(OWNER, "exp_menu")
    svc.stop(OWNER, "exp_menu")
    rs2 = svc.start(OWNER, "exp_menu")
    assert [e.experiment_id for e in rs2.experiments] == ["exp_menu@v2"] and rs2.version == svc.storage.get_ruleset(rs2.version - 1).version + 1
    assert svc.get("exp_menu", 1).status == PassportStatus.ARCHIVED and svc.get("exp_menu", 2).status == PassportStatus.RUNNING
    units = [f"g{i}" for i in range(20000)]
    a = [evaluate(rs1, rs1.experiments[0], u, IOS).variant_id for u in units]
    b = [evaluate(rs2, rs2.experiments[0], u, IOS).variant_id for u in units]
    share_b = sum(v == "treatment" for v in b) / len(b)
    both = sum(x == "treatment" and y == "treatment" for x, y in zip(a, b)) / len(b)
    assert 0.18 < share_b < 0.22 and abs(both - 0.5 * share_b) < 0.01  # независимо от прежнего разбиения
    svc.save_draft(OWNER, draft("exp_x"))
    with denied("conflict", "ещё не утверждена"):
        svc.new_version(OWNER, "exp_x", {"mde_relative": 0.1})
    with denied("invalid"):
        svc.new_version(OWNER, "exp_menu", {})


# ---------------- права и «второй человек» ----------------
def test_roles_and_second_person(svc):
    assert kind_of(lambda: svc.save_draft(READER, draft())) == "forbidden"
    assert kind_of(lambda: svc.save_draft(SDK, draft())) == "forbidden"
    assert kind_of(lambda: svc.save_draft(ADMIN, draft())) == "forbidden"
    p = svc.save_draft(OWNER, draft(owner="кто-то другой"))
    assert p.owner == "ivanova"  # владелец — автор, подставить чужое имя нельзя
    assert kind_of(lambda: svc.save_draft(OWNER2, draft())) == "forbidden"   # чужой черновик
    assert kind_of(lambda: svc.submit(OWNER2, "exp_menu")) == "forbidden"
    svc.submit(OWNER, "exp_menu")
    for actor in (OWNER, OWNER2, ADMIN, READER, SDK):
        assert kind_of(lambda: svc.approve(actor, "exp_menu")) == "forbidden"  # утверждает только аналитик
    # аналитик — владелец собственного Паспорта: сам себя утвердить не может
    svc.save_draft(ANALYST, draft("exp_own", owner="petrov"))
    svc.submit(ANALYST, "exp_own")
    with denied("forbidden", "второй человек"):
        svc.approve(ANALYST, "exp_own")
    assert svc.approve(Actor("smirnova", "analyst"), "exp_own").approved_by == "smirnova"
    a = svc.approve(ANALYST, "exp_menu")
    assert a.approved_by == "petrov" and a.status == PassportStatus.APPROVED
    for actor in (OWNER2, ANALYST, READER, SDK):
        assert kind_of(lambda: svc.start(actor, "exp_menu")) == "forbidden"
    svc.start(OWNER, "exp_menu")
    for actor in (OWNER2, ANALYST, READER, SDK):
        for fn in (svc.pause, svc.stop):
            assert kind_of(lambda: fn(actor, "exp_menu")) == "forbidden"
        assert kind_of(lambda: svc.ramp(actor, "exp_menu", 50)) == "forbidden"
    svc.pause(ADMIN, "exp_menu")  # kill-switch доступен администратору
    for actor in (OWNER, READER, SDK, ANALYST):
        assert kind_of(lambda: svc.add_layer(actor, "x")) == "forbidden"
        assert kind_of(lambda: svc.set_holdout(actor, 10)) == "forbidden"
    assert kind_of(lambda: svc.advance(OWNER, "exp_menu", "analysis")) == "forbidden"
    with pytest.raises(ControlError):
        Actor("x", "superuser")
    with pytest.raises(ControlError):
        Actor("", "admin")


def test_draft_rules(svc):
    assert kind_of(lambda: svc.save_draft(OWNER, {**draft(), "targetting": {}})) == "invalid"   # опечатка в поле
    assert kind_of(lambda: svc.save_draft(OWNER, {**draft(), "experiment_id": ""})) == "invalid"
    assert kind_of(lambda: svc.save_draft(ANALYST, {k: v for k, v in draft().items() if k != "owner"})) == "invalid"
    assert kind_of(lambda: svc.save_draft(OWNER, draft(exposure_surface="pre_login"))) == "invalid"  # IDN-1
    assert kind_of(lambda: svc.get("нет")) == "not_found" and kind_of(lambda: svc.submit(OWNER, "нет")) == "not_found"
    # статус, версия, хэш и утверждающий из запроса игнорируются
    p = svc.save_draft(OWNER, {**draft(), "status": "running", "version": 9, "passport_hash": "ab", "approved_by": "x"})
    assert (p.status, p.version, p.passport_hash, p.approved_by) == (PassportStatus.DRAFT, 1, None, None)
    p2 = svc.save_draft(OWNER, draft(mde_relative=0.08))
    assert p2.version == 1 and svc.get("exp_menu").mde_relative == 0.08
    svc.submit(OWNER, "exp_menu")
    with denied("conflict", "только новой версией"):
        svc.save_draft(OWNER, draft(mde_relative=0.09))
    back = svc.return_to_draft(ANALYST, "exp_menu", "уточните MDE")
    assert back.status == PassportStatus.DRAFT
    svc.save_draft(OWNER, draft(mde_relative=0.09))
    assert kind_of(lambda: svc.approve(ANALYST, "exp_menu")) == "conflict"  # из черновика — нельзя


def test_layers_and_holdout(svc):
    rs = svc.ruleset()
    assert rs.version == 1 and rs.holdout.pct == 5.0 and [l.layer_id for l in rs.layers] == ["ui"]
    with denied("conflict", "уже существует"):
        svc.add_layer(ADMIN, "ui")
    rs = svc.add_layer(ADMIN, "ops", marketing=False)
    assert rs.version == 2 and not rs.layer("ops").marketing
    assert svc.set_holdout(ADMIN, 2.5).holdout.pct == 2.5
    assert kind_of(lambda: svc.set_holdout(ADMIN, 100)) == "invalid"
    approved(svc)
    svc.start(OWNER, "exp_menu")
    with denied("conflict", "есть тесты"):
        svc.set_holdout(ADMIN, 10)
    fresh = ExperimentService(MemoryStorage(), default_holdout_pct=1.0)
    assert fresh.ruleset() is None and kind_of(lambda: fresh.start(ADMIN, "x")) == "not_found"
    assert fresh.set_holdout(ADMIN, 3.0).version == 1 and fresh.add_layer(ADMIN, "a").holdout.pct == 3.0


# ---------------- аудит ----------------
def test_every_command_is_audited_and_chain_is_valid(svc):
    approved(svc)
    svc.start(OWNER, "exp_menu")
    svc.ramp(OWNER, "exp_menu", 50)
    svc.pause(ADMIN, "exp_menu")
    svc.resume(OWNER, "exp_menu")
    svc.stop(OWNER, "exp_menu")
    log = svc.storage.audit("exp_menu")
    assert [(r.action, r.actor) for r in log] == [
        ("passport.create", "ivanova"), ("passport.submit", "ivanova"), ("passport.approve", "petrov"),
        ("passport.sealed", "petrov"), ("experiment.start", "ivanova"), ("experiment.ramp", "ivanova"),
        ("experiment.pause", "root"), ("experiment.resume", "ivanova"), ("experiment.stop", "ivanova")]
    assert log[3].details["passport_hash"] == svc.get("exp_menu").passport_hash
    assert log[4].details == {"version": 1, "from": "approved", "to": "running", "ruleset_version": 2, "ramp_pct": 10.0,
                              "rule": "exp_menu@v1"}
    assert log[5].details["from"] == 10.0 and log[5].details["to"] == 50 and log[5].details["ruleset_version"] == 3
    assert svc.storage.audit()[0].action == "layer.add" and svc.storage.verify_audit() == len(svc.storage.audit())
    # версии ruleset в аудите совпадают с сохранёнными
    assert [r.details["ruleset_version"] for r in log[4:]] == [2, 3, 4, 5, 6] and svc.ruleset().version == 6


def test_current_ruleset_is_cached_per_version(svc):
    """Раздача SDK: ruleset разбирается один раз на версию, новая версия видна сразу после команды."""
    first = svc.current()
    assert first[0].version == 1 and first[1] == first[0].to_json() and svc.current() is first
    approved(svc)
    assert svc.current() is first  # Паспорт без запуска ruleset не меняет
    svc.start(OWNER, "exp_menu")
    second = svc.current()
    assert second[0].version == 2 and second[0] == svc.ruleset() and svc.current() is second
    with pytest.raises(ControlError):
        svc.ramp(OWNER, "exp_menu", 30)  # отклонённая команда версию не меняет
    assert svc.current() is second
    assert ExperimentService(MemoryStorage()).current() is None
