"""Фаза 3, П1: Паспорт v4.1 — таргетинг, параметры вариантов, тип единицы, владелец и утверждающий, план раскатки."""
import pytest
from pydantic import ValidationError

from ab_platform.schemas.passport import ExperimentPassport, PassportStatus, RandomizationUnit, VariantAllocation
from ab_platform.schemas.targeting import Targeting, ascii_lower, parse_version
from tests.test_phase1_passport_v4 import base

TGT = {"platforms": ["ios"], "app_version_min": "5.2"}


# ---------------- правила сравнения (TGT-4) ----------------
@pytest.mark.parametrize("text, expected", [
    ("5", (5,)), ("5.0", (5,)), ("5.1.0", (5, 1)), ("5.10", (5, 10)), ("05.01", (5, 1)), ("0", (0,)), ("0.0", (0,)),
    ("1.2.3.4.5.6", (1, 2, 3, 4, 5, 6)), ("999999999", (999999999,)),
    ("", None), ("5.", None), (".5", None), ("5..1", None), ("5.1-beta", None), ("v5.1", None), (" 5.1", None),
    ("5.1\n", None), ("٥.١", None), ("1.2.3.4.5.6.7", None), ("1234567890", None), (5.1, None), (None, None),
])
def test_parse_version(text, expected):
    assert parse_version(text) == expected


def test_version_order_is_numeric_not_lexicographic():
    assert parse_version("5.10") > parse_version("5.9") and parse_version("5.1") == parse_version("5.1.0")
    assert parse_version("5") < parse_version("5.0.1") < parse_version("5.1")


def test_ascii_lower_touches_only_latin():
    assert ascii_lower("IOS-Msk_1") == "ios-msk_1"
    # знак Кельвина и турецкая İ в str.lower() превращаются в латиницу — здесь не должны
    assert ascii_lower("K") == "K" and ascii_lower("İ") == "İ" and ascii_lower("МСК") == "МСК"


def test_targeting_is_normalized_and_validated():
    t = Targeting(platforms=["iOS", "android", "ios"], cities=["MSK", "spb"], regions=["RU-MOW"])
    assert t.platforms == ["android", "ios"] and t.cities == ["msk", "spb"] and t.regions == ["ru-mow"]
    # порядок и регистр значений не меняют условия, значит и хэш Паспорта
    assert t == Targeting(platforms=["android", "IOS"], cities=["spb", "msk"], regions=["ru-mow"])
    assert Targeting().is_empty and not t.is_empty
    for bad in (dict(platforms=["windows"]), dict(platforms=[]), dict(cities=["Москва"]), dict(cities=["a b"]),
                dict(cities=[""]), dict(app_version_min="5.x"), dict(app_version_min="6.0", app_version_max="5.9"),
                dict(is_new_guest=1), dict(has_delivery="true"), dict(country=["ru"])):
        with pytest.raises(ValidationError):
            Targeting(**bad)
    assert Targeting(app_version_min="5.0", app_version_max="5").app_version_max == "5"  # равные границы допустимы


def test_mismatch_reasons_and_check_order():
    t = Targeting(platforms=["ios"], app_version_min="5.2", app_version_max="5.9", cities=["msk"], regions=["ru-mow"],
                  is_new_guest=True, has_delivery=False)
    ok = {"platform": "iOS", "app_version": "5.2.0", "city": "MSK", "region": "ru-mow", "is_new_guest": True,
          "has_delivery": False, "extra": "игнорируется"}
    assert t.required_attributes == ["platform", "app_version", "city", "region", "is_new_guest", "has_delivery"]
    assert t.mismatch(ok) is None
    assert t.mismatch({**ok, "app_version": "5.9"}) is None and t.mismatch({**ok, "app_version": "5.9.0.0"}) is None
    cases = {
        "targeting:platform": {"platform": "android"}, "targeting_missing:platform": {"platform": None},
        "targeting:app_version": {"app_version": "5.1.9"}, "targeting_missing:app_version": {"app_version": "5.2-rc"},
        "targeting:city": {"city": "spb"}, "targeting_missing:city": {"city": 77},
        "targeting:region": {"region": "ru-spe"},
        "targeting:is_new_guest": {"is_new_guest": False}, "targeting_missing:is_new_guest": {"is_new_guest": 1},
        "targeting:has_delivery": {"has_delivery": True}, "targeting_missing:has_delivery": {"has_delivery": "false"},
    }
    for reason, change in cases.items():
        assert t.mismatch({**ok, **change}) == reason, reason
    assert t.mismatch({**ok, "app_version": "5.9.1"}) == "targeting:app_version"
    assert t.mismatch({**ok, "platform": " ios"}) == "targeting:platform"  # пробелы не обрезаются
    # причина — по первому невыполненному условию в фиксированном порядке
    assert t.mismatch({}) == "targeting_missing:platform" and t.mismatch(None) == "targeting_missing:platform"
    assert t.mismatch({**ok, "city": "spb", "has_delivery": True}) == "targeting:city"
    assert Targeting().mismatch(None) is None
    # has_delivery = False — тоже условие, а не «не задано»
    assert Targeting(has_delivery=False).mismatch({"has_delivery": True}) == "targeting:has_delivery"


def test_treatment_sensitive_conditions_are_flagged():
    assert Targeting(platforms=["ios"], app_version_min="5.0", cities=["msk"], has_delivery=True).treatment_sensitive == []
    flagged = Targeting(app_version_max="5.9", is_new_guest=True).treatment_sensitive
    assert len(flagged) == 2 and "app_version_max" in flagged[0] and "is_new_guest" in flagged[1]


# ---------------- Паспорт ----------------
def test_targeting_requires_audience_share_and_scales_traffic():
    with pytest.raises(ValueError, match="targeting_audience_share"):
        ExperimentPassport(**base(targeting=TGT))
    with pytest.raises(ValueError, match="только вместе с условиями"):
        ExperimentPassport(**base(targeting_audience_share=0.5))
    p = ExperimentPassport(**base(targeting=TGT, targeting_audience_share=0.35))
    assert p.target_daily_units == 3500 and ExperimentPassport(**base()).target_daily_units == 10_000
    assert p.design_warnings == []
    risky = ExperimentPassport(**base(targeting={"app_version_max": "5.9"}, targeting_audience_share=0.5))
    assert len(risky.design_warnings) == 1 and "первому показу" in risky.design_warnings[0]


def test_targeting_only_for_guest_units():
    from tests.test_phase1_passport_v4 import restaurant
    with pytest.raises(ValueError, match="только для единиц user_id и device_id"):
        ExperimentPassport(**restaurant(targeting=TGT, targeting_audience_share=0.5))


def test_targeting_is_part_of_hash_and_locked_after_approval():
    a = ExperimentPassport(**base(targeting=TGT, targeting_audience_share=0.35)).locked()
    b = ExperimentPassport(**base(targeting={**TGT, "app_version_min": "5.3"}, targeting_audience_share=0.35)).locked()
    assert a.passport_hash != b.passport_hash
    forged = a.model_copy(update={"targeting": Targeting(platforms=["android"]).model_dump()})
    with pytest.raises(ValueError, match="Хэш Паспорта не совпадает"):
        forged.with_status(PassportStatus.RUNNING)
    with pytest.raises(ValidationError):
        a.targeting.platforms = ["android"]  # frozen
    v2 = a.new_version(targeting={"platforms": ["android"]})
    assert v2.version == 2 and v2.locked().passport_hash != a.passport_hash


def test_idn1_pre_login_requires_device_unit():
    with pytest.raises(ValueError, match="IDN-1"):
        ExperimentPassport(**base(exposure_surface="pre_login"))
    p = ExperimentPassport(**base(exposure_surface="pre_login", randomization_unit=RandomizationUnit.DEVICE_ID))
    assert any("устройство" in w for w in p.design_warnings)
    assert ExperimentPassport(**base()).design_warnings == []
    with pytest.raises(ValidationError):
        ExperimentPassport(**base(exposure_surface="kiosk"))


def test_variant_params_are_json_and_part_of_hash():
    def variants(text):
        return [VariantAllocation(variant_id="control", traffic_percentage=50, is_control=True),
                VariantAllocation(variant_id="treatment", traffic_percentage=50,
                                  params={"title": text, "order": (3, 1, 2), "discount": 0.1, "nested": {"on": True}})]
    a = ExperimentPassport(**base(variants=variants("Берите соус"))).locked()
    assert a.variants[1].params == {"title": "Берите соус", "order": [3, 1, 2], "discount": 0.1, "nested": {"on": True}}
    assert a.variants[0].params == {}
    assert a.passport_hash != ExperimentPassport(**base(variants=variants("Берите два соуса"))).locked().passport_hash
    forged = a.model_copy(update={"variants": [v.model_dump() for v in variants("другой текст")]})
    with pytest.raises(ValueError, match="Хэш Паспорта не совпадает"):
        forged.with_status(PassportStatus.RUNNING)
    for bad in ({"x": float("nan")}, {"x": float("inf")}, {"x": object()}, {"x": "я" * 5000}):
        with pytest.raises(ValidationError):
            VariantAllocation(variant_id="t", traffic_percentage=50, params=bad)


def test_ramp_plan_validation():
    assert ExperimentPassport(**base()).ramp_plan == [100.0]
    assert ExperimentPassport(**base(ramp_plan=[1, 5, 20, 50, 100])).ramp_plan == [1, 5, 20, 50, 100]
    for bad in ([], [0, 50], [50, 50], [50, 20], [50, 101]):
        with pytest.raises(ValueError, match="ramp_plan"):
            ExperimentPassport(**base(ramp_plan=bad))


def test_second_person_approval():
    p = ExperimentPassport(**base(owner="ivanova")).with_status(PassportStatus.REVIEW)
    with pytest.raises(ValueError, match="второй человек"):
        p.with_status(PassportStatus.APPROVED, actor="ivanova")
    a = p.with_status(PassportStatus.APPROVED, actor="petrov")
    assert a.approved_by == "petrov" and a.passport_hash == a.compute_hash()
    # утверждающий входит в хэш: подмену после утверждения ловит проверка хэша
    with pytest.raises(ValueError, match="Хэш Паспорта не совпадает"):
        a.model_copy(update={"approved_by": "sidorov"}).with_status(PassportStatus.RUNNING)
    with pytest.raises(ValueError, match="нельзя менять"):
        a.new_version(approved_by="petrov")
    v2 = a.new_version(mde_relative=0.05)
    assert v2.approved_by is None and v2.owner == "ivanova"
