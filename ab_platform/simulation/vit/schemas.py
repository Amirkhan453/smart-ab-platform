"""
Схемы выходных таблиц генератора (SYN-2). Коннекторы к реальным источникам (фаза 5)
обязаны выдавать те же колонки и типы — тогда контрактные тесты общие.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

RESTAURANTS: Dict[str, str] = {
    "restaurant_id": "string", "format": "category", "region": "category", "city": "string",
    "lat": "float", "lon": "float", "trade_zone": "string", "is_franchise": "bool",
    "has_kiosk": "bool", "has_delivery": "bool", "open_date": "datetime", "close_date": "datetime",
    "renovation_start": "datetime", "renovation_end": "datetime",
}
DAILY: Dict[str, str] = {
    "restaurant_id": "string", "date": "datetime", "gross_sales": "float", "discounts": "float",
    "revenue": "float", "cogs": "float", "margin": "float", "checks": "int",
    "service_time_p50_s": "float", "service_time_p90_s": "float", "waste_rub": "float", "avg_check": "float",
}
CHANNELS: Dict[str, str] = {
    "restaurant_id": "string", "date": "datetime", "channel": "string", "gross_sales": "float",
    "discounts": "float", "revenue": "float", "cogs": "float", "margin": "float", "checks": "int",
}
EVENTS: Dict[str, str] = {
    "event_id": "string", "event_type": "string", "name": "string", "start": "datetime",
    "end": "datetime", "effect_log": "float", "scope": "string",
}
GUESTS: Dict[str, str] = {
    "guest_id": "string", "home_restaurant_id": "string", "pre_orders": "int", "pre_revenue": "float",
    "orders": "int", "revenue": "float", "converted": "int",
}
KIOSK_SESSIONS: Dict[str, str] = {
    "session_id": "string", "restaurant_id": "string", "date": "datetime", "converted": "int",
    "check": "float", "checkout_time_s": "float",
}

_CHECKS = {
    "string": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
    "category": lambda s: isinstance(s.dtype, pd.CategoricalDtype),
    "float": pd.api.types.is_float_dtype,
    "int": pd.api.types.is_integer_dtype,
    "bool": pd.api.types.is_bool_dtype,
    "datetime": pd.api.types.is_datetime64_any_dtype,
}


def validate(df: pd.DataFrame, schema: Dict[str, str], name: str = "table") -> None:
    """Проверяет наличие колонок и типы; лишние служебные колонки (с ``_`` или latent) допустимы."""
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: нет колонок {missing}")
    bad = [f"{c} ({t})" for c, t in schema.items() if not _CHECKS[t](df[c])]
    if bad:
        raise ValueError(f"{name}: неверные типы: {bad}")
