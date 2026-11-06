"""
Конфигурация синтетического генератора «как ВИТ» (ТЗ v4, раздел 11).

Параметры по умолчанию — допущения порядка величины из публичных данных
(ТЗ v4, раздел 19.1), а не реальные данные сети. Пресеты:

- ``reference`` — ровно модель эталонной симуляции ``sim_restaurants.py``
  (оптимистичный шум, без сезонности, праздников, форматов и закрытий);
  на нём проверяется воспроизведение таблиц ТЗ 19.2 (SYN-5);
- ``reference_pessimistic`` — то же с пессимистичным шумом;
- ``optimistic`` / ``pessimistic`` — полный режим: форматы, сезонность,
  праздники, региональные шоки, закрытия и ремонты.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FORMATS = ("street", "food_court", "drive", "transit")
REGIONS = ("moscow", "spb", "million_plus", "other")
CHANNELS = ("hall", "kiosk", "drive", "delivery", "app")

# Профиль дня недели (пн..вс) из эталонной симуляции.
DEFAULT_DOW = [0.95, 0.93, 0.95, 0.98, 1.08, 1.12, 0.99]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PanelParams(_Frozen):
    """Модель log выручки ресторан-дня (ТЗ v4, раздел 11.3)."""

    base_daily_revenue: float = Field(600_000.0, gt=0, description="Медиана выручки ресторан-дня, руб.")
    level_sd: float = Field(0.45, ge=0, description="sd(log) уровня ресторана; CV ≈ 0.47")
    common_shock_sd: float = Field(0.05, ge=0, description="Общий для сети шок дня (погода, промо)")
    region_shock_sd: float = Field(0.0, ge=0, description="Шок региона в день")
    drift_sd: float = Field(0.004, ge=0, description="sd шага случайного блуждания ресторана в день")
    ar_coef: float = Field(0.5, ge=0, lt=1, description="Коэффициент AR(1) дневного шума")
    ar_sd: float = Field(0.08, ge=0, description="sd инновации AR(1)")
    local_shock_p: float = Field(0.0, ge=0, le=1, description="Вероятность локального шока в ресторан-день")
    local_shock_sd: float = Field(0.0, ge=0, description="sd локального шока")
    dow_profile: List[float] = Field(default_factory=lambda: list(DEFAULT_DOW))
    format_dow: bool = Field(False, description="Свой профиль дня недели у формата (фудкорт, транзит)")
    format_effects: Dict[str, float] = Field(
        default_factory=lambda: {f: 0.0 for f in FORMATS},
        description="Сдвиг log-уровня по формату",
    )
    annual_amplitude: float = Field(0.0, ge=0, description="Амплитуда годовой сезонности (log)")
    holidays: bool = Field(False, description="Праздники РФ влияют на выручку")
    federal_promos_per_year: int = Field(0, ge=0, description="Число общесетевых промо в год")
    federal_promo_effect: float = Field(0.04, description="Эффект общесетевого промо на log выручки")
    # Средний чек: log AC = log AC_i + w * u_it + eps_ac
    avg_check_median: float = Field(400.0, gt=0)
    avg_check_level_sd: float = Field(0.15, ge=0)
    avg_check_noise_share: float = Field(0.3, ge=0, le=1, description="Доля AR-шума выручки, уходящая в средний чек")
    avg_check_sd: float = Field(0.03, ge=0)
    discount_rate_mean: float = Field(0.08, ge=0, lt=1)
    cogs_rate_mean: float = Field(0.32, ge=0, lt=1)

    @field_validator("dow_profile")
    @classmethod
    def _dow_len(cls, v: List[float]) -> List[float]:
        if len(v) != 7 or any(x <= 0 for x in v):
            raise ValueError("dow_profile: нужно 7 положительных значений (пн..вс).")
        return v

    @field_validator("format_effects")
    @classmethod
    def _formats(cls, v: Dict[str, float]) -> Dict[str, float]:
        unknown = set(v) - set(FORMATS)
        if unknown:
            raise ValueError(f"Неизвестные форматы: {sorted(unknown)}")
        return {f: float(v.get(f, 0.0)) for f in FORMATS}


class NetworkConfig(_Frozen):
    """Конфигурация сети ресторанов и истории."""

    seed: int = Field(20261106, ge=0)
    n_restaurants: int = Field(1000, ge=2)
    start_date: date = date(2024, 11, 4)  # понедельник
    days: int = Field(730, ge=7)
    mode: Literal["reference", "full"] = "full"
    format_shares: Dict[str, float] = Field(
        default_factory=lambda: {"street": 0.55, "food_court": 0.25, "drive": 0.15, "transit": 0.05}
    )
    region_shares: Dict[str, float] = Field(
        default_factory=lambda: {"moscow": 0.30, "spb": 0.12, "million_plus": 0.30, "other": 0.28}
    )
    franchise_share: float = Field(0.10, ge=0, le=1)
    kiosk_share: float = Field(0.60, ge=0, le=1)
    delivery_share: float = Field(0.65, ge=0, le=1)
    trade_zone_km: float = Field(2.0, gt=0, description="Сторона ячейки торговой зоны, км")
    openings_share: float = Field(0.0, ge=0, le=1, description="Доля ресторанов, открывшихся внутри истории")
    closures_share: float = Field(0.0, ge=0, le=1, description="Доля ресторанов, закрывшихся внутри истории")
    renovation_share: float = Field(0.0, ge=0, le=1, description="Доля ресторанов с ремонтом 7–28 дней")
    panel: PanelParams = Field(default_factory=PanelParams)

    @field_validator("format_shares", "region_shares")
    @classmethod
    def _shares(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if any(x < 0 for x in v.values()) or abs(total - 1.0) > 1e-6:
            raise ValueError(f"Доли должны быть неотрицательны и в сумме давать 1 (получено {total}).")
        return v

    @model_validator(mode="after")
    def _keys(self) -> "NetworkConfig":
        if set(self.format_shares) - set(FORMATS):
            raise ValueError(f"format_shares: допустимы {FORMATS}")
        if set(self.region_shares) - set(REGIONS):
            raise ValueError(f"region_shares: допустимы {REGIONS}")
        return self

    # ---------- пресеты ----------
    @classmethod
    def preset(cls, name: str, **overrides) -> "NetworkConfig":
        """Пресет с переопределением полей верхнего уровня; `panel=` — словарь полей PanelParams."""
        reference_panel = dict(
            base_daily_revenue=600_000.0, level_sd=0.45, common_shock_sd=0.05, drift_sd=0.004,
            ar_coef=0.5, ar_sd=0.08,
        )
        pessimistic_noise = dict(drift_sd=0.012, ar_sd=0.15, local_shock_p=0.003, local_shock_sd=0.3)
        full_extras = dict(
            region_shock_sd=0.02, format_dow=True, annual_amplitude=0.05, holidays=True,
            federal_promos_per_year=4,
            format_effects={"street": 0.0, "food_court": -0.10, "drive": 0.15, "transit": 0.20},
        )
        full_network = dict(mode="full", openings_share=0.03, closures_share=0.02, renovation_share=0.05)
        presets = {
            "reference": (dict(mode="reference"), reference_panel),
            "reference_pessimistic": (dict(mode="reference"), {**reference_panel, **pessimistic_noise}),
            "optimistic": (full_network, {**reference_panel, **full_extras}),
            "pessimistic": (full_network, {**reference_panel, **full_extras, **pessimistic_noise}),
        }
        if name not in presets:
            raise ValueError(f"Неизвестный пресет {name!r}; доступны: {sorted(presets)}")
        net_kw, panel_kw = presets[name]
        panel_override = overrides.pop("panel", None) or {}
        return cls(**{**net_kw, **overrides, "panel": PanelParams(**{**panel_kw, **panel_override})})

    # ---------- YAML / JSON ----------
    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NetworkConfig":
        import yaml  # зависимость PyYAML (дизайн-заметка фазы 0)

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        preset: Optional[str] = data.pop("preset", None)
        return cls.preset(preset, **data) if preset else cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def config_hash(self) -> str:
        import hashlib

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class GuestParams(_Frozen):
    """
    Гости приложения (ТЗ v4, 11.2). Частота заказов: λ_g ~ Gamma(shape, scale) заказов/день,
    множитель периода exp(s·z − s²/2), заказы ~ Poisson. Параметры по умолчанию подобраны так,
    чтобы на окне 28 дней CV выручки на гостя ≈ 2.5 и корреляция пре/пост ≈ 0.5 (ТЗ 19.3);
    при этом доля гостей с ≥ 1 заказом за 28 дней ≈ 48%.
    """

    freq_shape: float = Field(0.351, gt=0)
    freq_scale: float = Field(0.259, gt=0)
    period_sd: float = Field(0.706, ge=0)
    check_median: float = Field(450.0, gt=0)
    check_sd: float = Field(0.30, ge=0)
    order_noise_sd: float = Field(0.35, ge=0)


class KioskParams(_Frozen):
    """Сессии киоска (ТЗ v4, 7.4): чек с CV внутри ресторан-дня, разброс уровня ресторан-дня."""

    orders_per_day: int = Field(150, ge=1, description="Сессий на ресторан в день")
    conversion: float = Field(0.70, gt=0, lt=1, description="Доля сессий, закончившихся заказом")
    check_cv: float = Field(0.70, gt=0, description="CV чека внутри ресторан-дня")
    cell_sd: float = Field(0.30, ge=0, description="sd(log) уровня ресторан-дня")
    check_mean: float = Field(400.0, gt=0)
    checkout_time_median_s: float = Field(75.0, gt=0)
    checkout_time_sd: float = Field(0.35, ge=0)
