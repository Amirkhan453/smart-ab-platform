"""
Панель «ресторан × день» и разбивка по каналам (ТЗ v4, 11.3).

log выручки(i, t) = уровень ресторана + формат + профиль дня недели + годовая сезонность
                    + праздники и промо + общий шок дня + шок региона + дрейф (случайное
                    блуждание) + AR(1) + редкие локальные шоки.

В режиме ``reference`` остаются только уровень, профиль дня недели, общий шок, дрейф,
AR(1) и локальные шоки — ровно модель эталонной симуляции ``sim_restaurants.py``.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .calendar import calendar_effect
from .config import CHANNELS, NetworkConfig, PanelParams
from .rng import stream


def simulate_log_components(
    n: int, n_days: int, p: PanelParams, rng: np.random.Generator, start_dow: int = 0
) -> Dict[str, np.ndarray]:
    """Компоненты эталонной модели. Порядок вызовов rng совпадает с ``sim_restaurants.make_panel``."""
    level = rng.normal(np.log(p.base_daily_revenue), p.level_sd, n)
    common = rng.normal(0.0, p.common_shock_sd, n_days)
    drift = np.cumsum(rng.normal(0.0, p.drift_sd, (n, n_days)), axis=1)
    shocks = rng.binomial(1, p.local_shock_p, (n, n_days)) * rng.normal(
        0.0, max(p.local_shock_sd, 1e-12), (n, n_days)
    )
    u = np.zeros((n, n_days))
    eps = rng.normal(0.0, p.ar_sd, (n, n_days))
    for t in range(n_days):
        u[:, t] = (p.ar_coef * u[:, t - 1] if t else 0.0) + eps[:, t]
    dow = np.log(np.asarray(p.dow_profile)[(start_dow + np.arange(n_days)) % 7])
    return {"level": level, "common": common, "drift": drift, "shocks": shocks, "ar": u, "dow": dow}


def simulate_reference_revenue(
    n: int, n_days: int, p: PanelParams, rng: np.random.Generator
) -> np.ndarray:
    """Матрица выручки (n × n_days) по эталонной модели — для быстрых симуляций A/A и MDE."""
    c = simulate_log_components(n, n_days, p, rng)
    return np.exp(c["level"][:, None] + c["dow"][None, :] + c["common"][None, :] + c["drift"] + c["ar"] + c["shocks"])


def _format_dow(p: PanelParams, fmt: str) -> np.ndarray:
    base = np.asarray(p.dow_profile, dtype=float)
    if not p.format_dow:
        return base
    if fmt == "food_court":   # ТЦ: выходные сильнее
        return base * np.array([1, 1, 1, 1, 1, 1.05, 1.05])
    if fmt == "transit":      # вокзалы и аэропорты: профиль ровнее
        return np.sqrt(base)
    return base


def operating_mask(restaurants: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """True, если ресторан работает в этот день (открыт, не закрыт, не на ремонте)."""
    d = dates.values[None, :]
    open_ = restaurants["open_date"].values[:, None] <= d
    close = restaurants["close_date"].values[:, None]
    not_closed = pd.isna(close) | (d < close)
    rs, re_ = restaurants["renovation_start"].values[:, None], restaurants["renovation_end"].values[:, None]
    in_reno = ~pd.isna(rs) & (d >= rs) & (d <= re_)
    return open_ & not_closed & ~in_reno


def generate_daily(cfg: NetworkConfig, restaurants: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    p = cfg.panel
    n, T = len(restaurants), cfg.days
    dates = pd.date_range(cfg.start_date, periods=T, freq="D")
    rng = stream(cfg.seed, "panel")
    c = simulate_log_components(n, T, p, rng, start_dow=int(dates[0].day_of_week))

    log_rev = c["level"][:, None] + c["common"][None, :] + c["drift"] + c["ar"] + c["shocks"]
    if cfg.mode == "reference":
        log_rev = log_rev + c["dow"][None, :]
    else:
        fmts = restaurants["format"].astype(str).to_numpy()
        dow_idx = (int(dates[0].day_of_week) + np.arange(T)) % 7
        for f in np.unique(fmts):
            m = fmts == f
            log_rev[m] += np.log(_format_dow(p, f))[dow_idx][None, :] + p.format_effects[f]
        if p.annual_amplitude > 0:
            doy = dates.day_of_year.to_numpy()
            log_rev += p.annual_amplitude * np.sin(2 * np.pi * (doy - 80) / 365.25)[None, :]
        log_rev += calendar_effect(events, dates)[None, :]
        if p.region_shock_sd > 0:
            r_rng = stream(cfg.seed, "panel", "region")
            regions = restaurants["region"].astype(str).to_numpy()
            for r in np.unique(regions):
                log_rev[regions == r] += r_rng.normal(0.0, p.region_shock_sd, T)[None, :]

    revenue = np.exp(log_rev)

    a_rng = stream(cfg.seed, "panel", "attributes")
    ac_level = np.log(p.avg_check_median) + a_rng.normal(0.0, p.avg_check_level_sd, n)
    log_ac = ac_level[:, None] + p.avg_check_noise_share * c["ar"] + a_rng.normal(0.0, p.avg_check_sd, (n, T))
    avg_check_raw = np.exp(log_ac)
    checks = np.maximum(1, np.rint(revenue / avg_check_raw)).astype(np.int64)

    disc_rate = np.clip(a_rng.normal(p.discount_rate_mean, 0.02, n), 0.0, 0.5)
    cogs_rate = np.clip(a_rng.normal(p.cogs_rate_mean, 0.02, n)[:, None] + a_rng.normal(0, 0.005, (n, T)), 0.05, 0.9)
    gross = revenue / (1.0 - disc_rate[:, None])
    cogs = revenue * cogs_rate

    svc_base = np.exp(a_rng.normal(np.log(180.0), 0.2, n))
    load = checks / checks.mean(axis=1, keepdims=True)
    p50 = svc_base[:, None] * load ** 0.3 * np.exp(a_rng.normal(0, 0.05, (n, T)))
    p90 = p50 * np.exp(a_rng.normal(np.log(1.9), 0.05, (n, T)))
    waste_rate = np.exp(a_rng.normal(np.log(0.02), 0.3, n))
    waste = revenue * waste_rate[:, None] * np.exp(a_rng.normal(0, 0.2, (n, T)))

    mask = operating_mask(restaurants, dates)
    ii, tt = np.nonzero(mask)
    df = pd.DataFrame({
        "restaurant_id": restaurants["restaurant_id"].to_numpy()[ii],
        "date": dates.values[tt].astype("datetime64[ns]"),
        "gross_sales": gross[ii, tt],
        "discounts": (gross - revenue)[ii, tt],
        "revenue": revenue[ii, tt],
        "cogs": cogs[ii, tt],
        "margin": (revenue - cogs)[ii, tt],
        "checks": checks[ii, tt],
        "service_time_p50_s": p50[ii, tt],
        "service_time_p90_s": p90[ii, tt],
        "waste_rub": waste[ii, tt],
    })
    df["avg_check"] = df["revenue"] / df["checks"]
    return df


_BASE_CHANNEL_WEIGHT = {"hall": 0.45, "kiosk": 0.25, "drive": 0.50, "delivery": 0.15, "app": 0.10}


def channel_split(cfg: NetworkConfig, restaurants: pd.DataFrame, daily: pd.DataFrame,
                  noise_sd: float = 0.05) -> pd.DataFrame:
    """Разбивка ресторан-дня по каналам; сумма по каналам равна значениям ресторан-дня."""
    rng = stream(cfg.seed, "panel", "channels")
    avail = np.zeros((len(restaurants), len(CHANNELS)), dtype=bool)
    avail[:, CHANNELS.index("hall")] = True
    avail[:, CHANNELS.index("app")] = True
    avail[:, CHANNELS.index("kiosk")] = restaurants["has_kiosk"].to_numpy()
    avail[:, CHANNELS.index("delivery")] = restaurants["has_delivery"].to_numpy()
    avail[:, CHANNELS.index("drive")] = (restaurants["format"].astype(str) == "drive").to_numpy()
    alpha = np.array([_BASE_CHANNEL_WEIGHT[c] for c in CHANNELS]) * 20.0
    shares = np.where(avail, rng.gamma(alpha, 1.0, avail.shape), 0.0)
    shares /= shares.sum(axis=1, keepdims=True)

    idx = pd.Index(restaurants["restaurant_id"]).get_indexer(daily["restaurant_id"])
    s = shares[idx] * np.exp(rng.normal(0.0, noise_sd, (len(daily), len(CHANNELS))))
    s = np.where(shares[idx] > 0, s, 0.0)
    s /= s.sum(axis=1, keepdims=True)

    frames = []
    for k, ch in enumerate(CHANNELS):
        m = s[:, k] > 0
        part = daily.loc[m, ["restaurant_id", "date"]].copy()
        part["channel"] = ch
        for col in ("gross_sales", "discounts", "revenue", "cogs", "margin"):
            part[col] = daily.loc[m, col].to_numpy() * s[m, k]
        part["checks_share"] = s[m, k]
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    # Целые чеки по каналам с сохранением суммы ресторан-дня (метод наибольших остатков).
    total_checks = daily.set_index(["restaurant_id", "date"])["checks"]
    raw = out["checks_share"].to_numpy() * total_checks.reindex(
        pd.MultiIndex.from_frame(out[["restaurant_id", "date"]])
    ).to_numpy()
    base = np.floor(raw).astype(np.int64)
    out["checks"] = base
    remainder = raw - base
    key = out["restaurant_id"].astype(str) + "|" + out["date"].astype(str)
    deficit = (total_checks.reindex(pd.MultiIndex.from_frame(out[["restaurant_id", "date"]])).to_numpy()
               - out.groupby(key)["checks"].transform("sum").to_numpy())
    rank = pd.Series(-remainder).groupby(key.to_numpy()).rank(method="first").to_numpy()
    out.loc[rank <= deficit, "checks"] += 1
    return out.drop(columns="checks_share").sort_values(["restaurant_id", "date", "channel"], ignore_index=True)
