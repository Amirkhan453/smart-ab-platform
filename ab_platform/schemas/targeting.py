"""
Таргетинг гостевого теста (ТЗ v4.1: TGT-1…4).

Условия — часть Паспорта и его хэша; они одинаковы для всех вариантов и проверяются до выбора варианта.
Все заданные условия должны выполняться одновременно (И); внутри списка достаточно одного значения (ИЛИ).
Нет нужного признака или он неверного типа → гость вне теста (контроль), причина targeting_missing:<признак>.

Правила сравнения — общие для Python, JS и /decide, зафиксированы в tests/vectors/assignment_vectors.json:
- платформа, город, регион — коды из [A-Za-z0-9_.-]; сравнение без учёта регистра только для латиницы A–Z,
  без обрезки пробелов;
- версия приложения — от 1 до 6 чисел через точку, только цифры 0–9; сравнение по числам слева направо,
  недостающие части равны нулю (5.1 = 5.1.0, 5.10 > 5.9); границы включительные;
- булевы признаки — только true/false (не 1/0 и не строки).
"""
import re
from typing import Any, List, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, StrictBool, field_validator, model_validator

PLATFORMS = ("android", "ios", "web")
# Порядок проверки условий фиксирован: от него зависит причина отказа, одинаковая во всех SDK.
ATTRIBUTES = ("platform", "app_version", "city", "region", "is_new_guest", "has_delivery")

_CODE_RE = re.compile(r"[A-Za-z0-9_.\-]{1,64}")
_VERSION_RE = re.compile(r"[0-9]{1,9}(\.[0-9]{1,9}){0,5}")
_ASCII_LOWER = {c: c + 32 for c in range(ord("A"), ord("Z") + 1)}


def ascii_lower(text: str) -> str:
    """Нижний регистр только для A–Z: str.lower() зависит от Юникода и расходится между языками."""
    return text.translate(_ASCII_LOWER)


def parse_version(text: Any) -> Optional[Tuple[int, ...]]:
    """Версия → кортеж чисел без хвостовых нулей; None, если это не версия."""
    if not isinstance(text, str) or not _VERSION_RE.fullmatch(text):
        return None
    parts = [int(p) for p in text.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _codes(values: Optional[List[str]], name: str, allowed: Optional[Tuple[str, ...]] = None) -> Optional[List[str]]:
    if values is None:
        return None
    if not values:
        raise ValueError(f"{name}: пустой список; чтобы снять условие, уберите поле.")
    out = set()
    for v in values:
        if not isinstance(v, str) or not _CODE_RE.fullmatch(v):
            raise ValueError(f"{name}: '{v}' — не код (латиница, цифры, '_', '.', '-'; до 64 знаков).")
        v = ascii_lower(v)
        if allowed is not None and v not in allowed:
            raise ValueError(f"{name}: '{v}' не из {list(allowed)}.")
        out.add(v)
    return sorted(out)


class Targeting(BaseModel):
    """Условия применимости теста. Пустой объект — тест для всех гостей слоя."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    platforms: Optional[List[str]] = None
    app_version_min: Optional[str] = None
    app_version_max: Optional[str] = None
    cities: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    is_new_guest: Optional[StrictBool] = None
    has_delivery: Optional[StrictBool] = None

    @field_validator("platforms")
    @classmethod
    def _platforms(cls, v):
        return _codes(v, "platforms", PLATFORMS)

    @field_validator("cities")
    @classmethod
    def _cities(cls, v):
        return _codes(v, "cities")

    @field_validator("regions")
    @classmethod
    def _regions(cls, v):
        return _codes(v, "regions")

    @model_validator(mode="after")
    def _versions(self):
        lo, hi = self.app_version_min, self.app_version_max
        for name, value in (("app_version_min", lo), ("app_version_max", hi)):
            if value is not None and parse_version(value) is None:
                raise ValueError(f"{name}: '{value}' — не версия (числа через точку, например 5.12.0).")
        if lo is not None and hi is not None and parse_version(lo) > parse_version(hi):
            raise ValueError(f"app_version_min ({lo}) больше app_version_max ({hi}).")
        return self

    @property
    def is_empty(self) -> bool:
        return not self.required_attributes

    @property
    def required_attributes(self) -> List[str]:
        """Признаки, которые приложение обязано передать, в порядке проверки."""
        used = {
            "platform": self.platforms is not None,
            "app_version": self.app_version_min is not None or self.app_version_max is not None,
            "city": self.cities is not None,
            "region": self.regions is not None,
            "is_new_guest": self.is_new_guest is not None,
            "has_delivery": self.has_delivery is not None,
        }
        return [a for a in ATTRIBUTES if used[a]]

    @property
    def treatment_sensitive(self) -> List[str]:
        """Условия по признакам, которые сам тест может изменить (TGT-3): нужны предупреждение и анализ по первому показу."""
        out = []
        if self.app_version_max is not None:
            out.append("app_version_max: вариант может ускорить или замедлить обновление приложения")
        if self.is_new_guest is not None:
            out.append("is_new_guest: вариант может изменить срок первого заказа")
        return out

    def mismatch(self, attributes: Optional[Mapping[str, Any]]) -> Optional[str]:
        """None — гость подходит; иначе причина по первому невыполненному условию."""
        attrs = attributes or {}
        for name in self.required_attributes:
            value = attrs.get(name)
            if name in ("platform", "city", "region"):
                allowed = {"platform": self.platforms, "city": self.cities, "region": self.regions}[name]
                if not isinstance(value, str):
                    return f"targeting_missing:{name}"
                if ascii_lower(value) not in allowed:
                    return f"targeting:{name}"
            elif name == "app_version":
                version = parse_version(value)
                if version is None:
                    return f"targeting_missing:{name}"
                lo, hi = self.app_version_min, self.app_version_max
                if (lo is not None and version < parse_version(lo)) or (hi is not None and version > parse_version(hi)):
                    return f"targeting:{name}"
            else:
                if not isinstance(value, bool):
                    return f"targeting_missing:{name}"
                if value != getattr(self, name):
                    return f"targeting:{name}"
        return None
