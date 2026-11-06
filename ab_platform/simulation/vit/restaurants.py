"""Справочник ресторанов (ТЗ v4, 11.2): формат, регион, город, координаты, торговая зона, флаги, даты."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .config import FORMATS, REGIONS, NetworkConfig
from .rng import stream

# Город: (название, широта, долгота, радиус размещения точек, км). Координаты — центры городов.
_CITIES = {
    "moscow": [("Москва", 55.751, 37.618, 18.0)],
    "spb": [("Санкт-Петербург", 59.939, 30.316, 12.0)],
    "million_plus": [
        ("Новосибирск", 55.030, 82.920, 8.0), ("Екатеринбург", 56.838, 60.605, 8.0),
        ("Казань", 55.796, 49.106, 8.0), ("Нижний Новгород", 56.327, 44.006, 8.0),
        ("Челябинск", 55.160, 61.400, 7.0), ("Самара", 53.195, 50.100, 7.0),
        ("Краснодар", 45.035, 38.975, 7.0), ("Ростов-на-Дону", 47.222, 39.720, 7.0),
    ],
}
_N_OTHER_CITIES = 40


def _other_cities(rng: np.random.Generator):
    lat = rng.uniform(50.0, 60.0, _N_OTHER_CITIES)
    lon = rng.uniform(30.0, 50.0, _N_OTHER_CITIES)
    return [(f"Город-{i + 1:02d}", float(lat[i]), float(lon[i]), 3.0) for i in range(_N_OTHER_CITIES)]


def generate_restaurants(cfg: NetworkConfig) -> pd.DataFrame:
    rng = stream(cfg.seed, "restaurants")
    n = cfg.n_restaurants
    cities = dict(_CITIES)
    cities["other"] = _other_cities(rng)

    formats = rng.choice(list(cfg.format_shares), size=n, p=list(cfg.format_shares.values()))
    regions = rng.choice(list(cfg.region_shares), size=n, p=list(cfg.region_shares.values()))

    city_names, lats, lons, xs, ys = [], [], [], [], []
    for reg in regions:
        name, lat0, lon0, radius = cities[reg][rng.integers(len(cities[reg]))]
        # Равномерно в круге радиуса radius км.
        r = radius * np.sqrt(rng.random())
        phi = rng.uniform(0.0, 2.0 * np.pi)
        x, y = r * np.cos(phi), r * np.sin(phi)
        city_names.append(name)
        xs.append(x)
        ys.append(y)
        lats.append(lat0 + y / 111.0)
        lons.append(lon0 + x / (111.0 * np.cos(np.radians(lat0))))
    xs, ys = np.array(xs), np.array(ys)
    gx = np.floor(xs / cfg.trade_zone_km).astype(int)
    gy = np.floor(ys / cfg.trade_zone_km).astype(int)
    trade_zone = [f"{c}:{a}:{b}" for c, a, b in zip(city_names, gx, gy)]

    franchise = rng.random(n) < cfg.franchise_share
    has_kiosk = rng.random(n) < cfg.kiosk_share
    has_delivery = rng.random(n) < cfg.delivery_share

    start = pd.Timestamp(cfg.start_date)
    open_date = np.array([start - pd.Timedelta(days=int(d)) for d in rng.integers(90, 365 * 15, n)])
    close_date = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    reno_start = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    reno_end = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    if cfg.mode == "full":
        u = rng.random(n)
        opened = u < cfg.openings_share
        closed = (u >= cfg.openings_share) & (u < cfg.openings_share + cfg.closures_share)
        reno = rng.random(n) < cfg.renovation_share
        for i in np.flatnonzero(opened):
            open_date[i] = start + pd.Timedelta(days=int(rng.integers(0, cfg.days - 7)))
        for i in np.flatnonzero(closed):
            close_date[i] = (start + pd.Timedelta(days=int(rng.integers(7, cfg.days)))).to_datetime64()
        for i in np.flatnonzero(reno & ~opened & ~closed):
            s = start + pd.Timedelta(days=int(rng.integers(0, max(1, cfg.days - 28))))
            reno_start[i] = s.to_datetime64()
            reno_end[i] = (s + timedelta(days=int(rng.integers(7, 29) - 1))).to_datetime64()

    df = pd.DataFrame({
        "restaurant_id": [f"R{i + 1:04d}" for i in range(n)],
        "format": pd.Categorical(formats, categories=list(FORMATS)),
        "region": pd.Categorical(regions, categories=list(REGIONS)),
        "city": city_names,
        "lat": np.round(lats, 5),
        "lon": np.round(lons, 5),
        "trade_zone": trade_zone,
        "is_franchise": franchise,
        "has_kiosk": has_kiosk,
        "has_delivery": has_delivery,
        "open_date": pd.to_datetime(open_date).astype("datetime64[ns]"),
        "close_date": pd.to_datetime(close_date).astype("datetime64[ns]"),
        "renovation_start": pd.to_datetime(reno_start).astype("datetime64[ns]"),
        "renovation_end": pd.to_datetime(reno_end).astype("datetime64[ns]"),
    })
    return df
