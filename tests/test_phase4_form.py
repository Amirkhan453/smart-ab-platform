"""
Фаза 4, П5: форма Паспорта (UI-0) — страница Streamlit через HTTP API. Паспорт гостевого теста создаётся,
утверждается и запускается через форму без правки кода (AppTest); дерево выбора дизайна; калькулятор длительности.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from ab_platform.design import TREE, choose_design
from ab_platform.ui_api import ApiError, PlatformApi
from ab_platform.ui_form import build_passport, estimate_duration, parse_numbers, values_from_passport
from tests.test_phase3_http import KEYS, make_app

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "1_passport_form.py")


@pytest.fixture
def stand():
    app, service = make_app()
    service.add_layer(service_admin(), "ui")
    client = TestClient(app)
    return service, (lambda who: PlatformApi(key=KEYS[who], client=client))


def service_admin():
    from ab_platform.control import Actor
    return Actor("root", "admin")


def page(api):
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.session_state["api"] = api
    return at.run()


def texts(elements):
    return [e.value for e in elements]


# ---------------- UI-0: весь путь через форму ----------------
def test_passport_created_approved_and_started_through_form(stand):
    service, api = stand
    at = page(api("ivanova"))
    assert not at.exception and not at.error
    at.text_input(key="experiment_id_new").set_value("exp_form")
    at.text_input(key="test_key_new").set_value("flag_exp_form")
    at.text_area(key="hypothesis_new").set_value("Если мы покажем новое меню, выручка на гостя вырастет на 3%")
    at.multiselect(key="platforms_new").set_value(["ios"])
    at.text_input(key="vmin_new").set_value("5.2")
    at.number_input(key="audience_new").set_value(40.0)
    at.button(key="save").click().run()
    assert not at.exception and not at.error, texts(at.error)
    assert any("Черновик exp_form v1 сохранён" in s for s in texts(at.success))

    p = service.get("exp_form")
    assert p.status.value == "draft" and p.owner == "ivanova" and p.estimand.value == "triggered"
    assert p.primary_metric.registry_ref == "revenue_per_guest@1"
    assert {g.metric.registry_ref for g in p.guardrail_metrics} == {"crash_share@1", "payment_error_share@1",
                                                                   "cancel_rate@1"}
    assert p.targeting.platforms == ["ios"] and p.targeting_audience_share == 0.4 and p.ramp_plan == [10, 50, 100]
    assert p.design_warnings == []

    assert at.selectbox(key="choice").value == "exp_form"  # форма перешла к сохранённому Паспорту
    assert at.text_input(key="test_key_exp_form").value == "flag_exp_form"  # поля — из сохранённого Паспорта
    assert any("Статус: **draft**" in m for m in texts(at.markdown))
    at.button(key="cmd_submit").click().run()
    assert service.get("exp_form").status.value == "review"
    assert any("статус review" in s for s in texts(at.success)) and any("**review**" in m for m in texts(at.markdown))
    at.button(key="cmd_approve").click().run()  # владелец сам себя не утверждает
    assert at.error and service.get("exp_form").status.value == "review"

    analyst = page(api("petrov"))
    analyst.selectbox(key="choice").set_value("exp_form").run()
    analyst.button(key="cmd_approve").click().run()
    assert not analyst.error, texts(analyst.error)
    assert service.get("exp_form").status.value == "approved" and service.get("exp_form").approved_by == "petrov"

    at = page(api("ivanova"))
    at.selectbox(key="choice").set_value("exp_form").run()
    at.button(key="cmd_start").click().run()
    assert not at.error, texts(at.error)
    assert service.get("exp_form").status.value == "running"
    assert service.ruleset().by_flag("flag_exp_form") is not None
    assert "Анализов ещё не было." in texts(at.caption) and "Тревог нет." in texts(at.caption)
    actions = [r.action for r in service.storage.audit("exp_form")]
    assert actions[:4] == ["passport.create", "passport.submit", "passport.approve", "experiment.start"] or \
        {"passport.create", "experiment.start"} <= set(actions)


def test_validator_errors_are_shown_as_plain_text(stand):
    service, api = stand
    at = page(api("ivanova"))
    at.text_input(key="experiment_id_new").set_value("exp_bad")
    at.text_input(key="test_key_new").set_value("flag_bad")
    at.text_area(key="hypothesis_new").set_value("коротко")  # меньше 15 знаков
    at.button(key="save").click().run()
    assert not at.exception and len(at.error) == 1
    msg = at.error[0].value
    assert "hypothesis_statement" in msg and "422" in msg
    assert "pydantic" not in msg and "input_value" not in msg  # без служебного текста библиотеки
    assert service.storage.get_passport("exp_bad") is None


def test_reader_sees_refusal_not_crash(stand):
    _, api = stand
    at = page(api("guest"))
    at.text_input(key="experiment_id_new").set_value("exp_x")
    at.button(key="save").click().run()
    assert not at.exception and any("403" in e for e in texts(at.error))


def test_page_without_connection_asks_for_key():
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception and any("личный ключ" in i for i in texts(at.info))


def test_design_hint_and_duration_warning(stand):
    _, api = stand
    at = page(api("ivanova"))
    at.checkbox(key="tree_randomize_restaurants").check().run()
    assert any("Кластерный RCT" in m for m in texts(at.markdown))
    assert any("только гостевой A/B" in w for w in texts(at.warning))
    at.number_input(key="daily_new").set_value(500).run()  # мало трафика — 28 дней не хватит
    assert any("Плановой длительности 28 дн. не хватает" in w for w in texts(at.warning))


# ---------------- функции формы ----------------
def test_choose_design_follows_tree_order():
    assert choose_design({"randomize_guests": True, "switchable": True})["experiment_type"] == "classic_ab"
    d = choose_design({"switchable": True, "whole_network": True})
    assert d["evidence_level"] == "A-" and d["auto_ship_allowed"] and "вопрос 3" in d["rule"]
    assert choose_design({"self_selected": True})["auto_ship_allowed"] is False
    none = choose_design({k["key"]: False for k in TREE})
    assert none["method"] is None and not none["auto_ship_allowed"]
    with pytest.raises(ValueError, match="Неизвестные вопросы"):
        choose_design({"magic": True})


def test_build_passport_roundtrip_and_errors(stand):
    service, api = stand
    a = api("ivanova")
    defs = {r: a.metric(r)["definition"] for r in ("revenue_per_guest@1", "crash_share@1", "conversion@1")}
    values = dict(experiment_id=" exp_rt ", test_key="flag_rt", layer_id="ui",
                  hypothesis_statement="Если мы X, то выручка вырастет на 3%", treatment_share_pct=30.0,
                  primary_ref="revenue_per_guest@1", secondary_refs=["conversion@1"], guardrail_refs=["crash_share@1"],
                  max_degradation_pct=5.0, baseline_value=450.0, mde_pct=3.0, planned_duration_days=28,
                  daily_eligible_units=10_000, platforms=[], app_version_min="", audience_share_pct=40.0,
                  ramp_plan="10; 100", attribution_window_days=7, use_cuped=False)
    body = build_passport(values, defs)
    assert body["experiment_id"] == "exp_rt" and body["targeting"] == {} and body["targeting_audience_share"] is None
    assert [v["traffic_percentage"] for v in body["variants"]] == [70.0, 30.0] and body["mde_relative"] == 0.03
    saved = a.save_draft(body)
    back = values_from_passport(saved)
    for k in ("treatment_share_pct", "primary_ref", "secondary_refs", "guardrail_refs", "baseline_value",
              "planned_duration_days", "use_cuped"):
        assert back[k] == values[k], k
    assert back["ramp_plan"] == "10, 100" and back["max_degradation_pct"] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="нет в реестре"):
        build_passport({**values, "primary_ref": "nope@1"}, defs)
    with pytest.raises(ValueError, match="Доля тестовой группы"):
        build_passport({**values, "treatment_share_pct": 100}, defs)
    assert parse_numbers("10, 50 ,100", "x") == [10, 50, 100]
    for bad in ("", "10, много"):
        with pytest.raises(ValueError, match="План"):
            parse_numbers(bad, "План")


def test_estimate_duration():
    cont = estimate_duration("continuous", 450, 0.03, 4000, cv=2.5)
    cuped = estimate_duration("continuous", 450, 0.03, 4000, cv=2.5, pre_post_correlation=0.5)
    assert cuped["n_per_variant"] < cont["n_per_variant"] and cont["days"] % 7 == 0
    more = estimate_duration("continuous", 450, 0.03, 8000, cv=2.5)
    assert more["days"] <= cont["days"] and more["n_per_variant"] == cont["n_per_variant"]
    assert estimate_duration("binary", 0.3, 0.03, 4000)["n_per_variant"] > 0
    with pytest.raises(ValueError, match="> 0"):
        estimate_duration("continuous", 450, 0.03, 4000, cv=0)


def test_api_client_errors(stand):
    _, api = stand
    with pytest.raises(ValueError, match="ключ"):
        PlatformApi(key="")
    with pytest.raises(ApiError) as e:
        api("ivanova").passport("нет")
    assert e.value.status == 404
    with pytest.raises(ApiError) as e:
        api("ivanova").command("нет", "ramp", {"ramp_pct": "x"})
    assert e.value.status == 422 and "ramp_pct" in e.value.detail


def test_invalid_text_plain():
    from pydantic import BaseModel, field_validator, model_validator

    from ab_platform.control import invalid_text

    class M(BaseModel):
        x: int

        @field_validator("x")
        @classmethod
        def positive(cls, v):
            if v <= 0:
                raise ValueError("должно быть > 0")
            return v

        @model_validator(mode="after")
        def whole(self):
            if self.x == 13:
                raise ValueError("несчастливое число")
            return self

    with pytest.raises(ValueError) as e:
        M(x=13)
    assert invalid_text(e.value) == "данные: несчастливое число"  # ошибка всей модели — без поля
    with pytest.raises(ValueError) as e:
        M(x=0)
    assert invalid_text(e.value) == "x: должно быть > 0"
    with pytest.raises(ValueError) as e:
        M.model_validate({})
    assert invalid_text(e.value) == "x: Field required"
    assert invalid_text(ValueError("просто текст")) == "просто текст"
