"""Календарь внешних событий (ТЗ v4, 11.2): праздники РФ и общесетевые промо."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .config import NetworkConfig
from .rng import stream

# (месяц, день начала, день конца, название, эффект на log выручки) — допущения, не данные ВИТ.
_HOLIDAYS = [
    (1, 1, 1, "Новый год", -0.25),
    (1, 2, 8, "Новогодние каникулы", 0.05),
    (2, 23, 23, "День защитника Отечества", 0.03),
    (3, 8, 8, "Международный женский день", 0.05),
    (5, 1, 1, "Праздник Весны и Труда", 0.05),
    (5, 9, 9, "День Победы", 0.05),
    (6, 12, 12, "День России", 0.03),
    (11, 4, 4, "День народного единства", 0.03),
    (12, 31, 31, "Канун Нового года", -0.15),
]
_PROMO_DAYS = 14

EVENT_COLUMNS = ["event_id", "event_type", "name", "start", "end", "effect_log", "scope"]


def generate_events(cfg: NetworkConfig) -> pd.DataFrame:
    p = cfg.panel
    start = pd.Timestamp(cfg.start_date)
    end = start + pd.Timedelta(days=cfg.days - 1)
    rows = []
    if p.holidays:
        for year in range(start.year, end.year + 1):
            for month, d0, d1, name, eff in _HOLIDAYS:
                s, e = pd.Timestamp(date(year, month, d0)), pd.Timestamp(date(year, month, d1))
                if e >= start and s <= end:
                    rows.append(("holiday", name, max(s, start), min(e, end), eff))
    if p.federal_promos_per_year > 0:
        rng = stream(cfg.seed, "calendar", "promos")
        n_promos = int(round(p.federal_promos_per_year * cfg.days / 365.0))
        if cfg.days > _PROMO_DAYS:
            for k, off in enumerate(sorted(rng.integers(0, cfg.days - _PROMO_DAYS, n_promos))):
                s = start + pd.Timedelta(days=int(off))
                rows.append(("federal_promo", f"Федеральное промо {k + 1}", s,
                             s + pd.Timedelta(days=_PROMO_DAYS - 1), p.federal_promo_effect))
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS[1:6])
    df.insert(0, "event_id", [f"E{i + 1:04d}" for i in range(len(df))])
    df["scope"] = "all"
    for col in ("start", "end"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return df[EVENT_COLUMNS]


def calendar_effect(events: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """Суммарный эффект событий на log выручки по датам (одинаков для всех ресторанов)."""
    eff = np.zeros(len(dates))
    for _, ev in events.iterrows():
        mask = (dates >= ev["start"]) & (dates <= ev["end"])
        eff[mask] += float(ev["effect_log"])
    return eff
