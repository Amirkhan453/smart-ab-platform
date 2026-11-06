"""Сеть: справочник, календарь, дневная панель и записи о правде. Точка входа — ``generate_network``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .calendar import generate_events
from .config import GuestParams, KioskParams, NetworkConfig
from .guests import generate_guests
from .kiosk import generate_kiosk_sessions
from .panel import channel_split, generate_daily
from .restaurants import generate_restaurants
from .truth import Truth


@dataclass(frozen=True)
class Network:
    config: NetworkConfig
    restaurants: pd.DataFrame
    events: pd.DataFrame
    daily: pd.DataFrame
    truths: Tuple[Truth, ...] = field(default_factory=tuple)

    def wide(self, metric: str = "revenue") -> pd.DataFrame:
        """Матрица ресторан × дата для метрики (NaN — ресторан не работал)."""
        if metric not in self.daily.columns:
            raise ValueError(f"Неизвестная метрика {metric!r}")
        return self.daily.pivot_table(index="restaurant_id", columns="date", values=metric, aggfunc="sum")

    def channels(self) -> pd.DataFrame:
        """Разбивка по каналам (считается по запросу: ~3 млн строк для сети 1000 × 2 года)."""
        return channel_split(self.config, self.restaurants, self.daily)

    def guests(self, n: int, pre_days: int = 28, test_days: int = 28,
               params: GuestParams = GuestParams(), window: str = "default") -> pd.DataFrame:
        """Гости приложения под окно эксперимента; домашний ресторан — пропорционально выручке."""
        mean_rev = self.daily.groupby("restaurant_id")["revenue"].mean()
        w = mean_rev.reindex(self.restaurants["restaurant_id"]).fillna(0.0).to_numpy()
        return generate_guests(self.config.seed, self.restaurants, n, pre_days, test_days, params, window, w)

    def kiosk_sessions(self, start: str, days: int, restaurant_ids: Optional[list] = None,
                       params: KioskParams = KioskParams(), window: str = "default") -> pd.DataFrame:
        if restaurant_ids is None:
            restaurant_ids = self.restaurants.loc[self.restaurants["has_kiosk"], "restaurant_id"].tolist()
        return generate_kiosk_sessions(self.config.seed, restaurant_ids, start, days, params, window)


def generate_network(cfg: NetworkConfig) -> Network:
    restaurants = generate_restaurants(cfg)
    events = generate_events(cfg)
    daily = generate_daily(cfg, restaurants, events)
    return Network(config=cfg, restaurants=restaurants, events=events, daily=daily)
