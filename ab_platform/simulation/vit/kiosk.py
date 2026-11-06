"""
Сессии киоска (ТЗ v4, 7 и 11.2) и интерференция через общую очередь.

Чек сессии: уровень ресторан-дня (sd(log) = ``cell_sd``) × логнормальный чек с CV ``check_cv``
внутри ресторан-дня. На этих допущениях построена проверка MDE в ТЗ 7.4.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .config import KioskParams
from .rng import stream
from .truth import Truth


def generate_kiosk_sessions(
    seed: int, restaurant_ids: Iterable[str], start: str | pd.Timestamp, days: int,
    params: KioskParams = KioskParams(), window: str = "default",
) -> pd.DataFrame:
    ids = np.asarray(list(restaurant_ids))
    if len(ids) == 0 or days < 1:
        raise ValueError("Нужен хотя бы один ресторан и один день")
    rng = stream(seed, "kiosk", window)
    dates = pd.date_range(pd.Timestamp(start), periods=days, freq="D")
    n_cells, k = len(ids) * days, params.orders_per_day
    cell_level = rng.normal(0.0, params.cell_sd, n_cells)
    sigma = np.sqrt(np.log(1.0 + params.check_cv ** 2))
    n = n_cells * k
    converted = rng.random(n) < params.conversion
    check = params.check_mean * np.exp(np.repeat(cell_level, k) + rng.normal(-sigma ** 2 / 2, sigma, n))
    t = params.checkout_time_median_s * np.exp(rng.normal(0.0, params.checkout_time_sd, n))
    cell = np.repeat(np.arange(n_cells), k)
    return pd.DataFrame({
        "session_id": [f"S{window[:8]}_{i:09d}" for i in range(n)],
        "restaurant_id": ids[cell // days],
        "date": dates.values[cell % days].astype("datetime64[ns]"),
        "converted": converted.astype(np.int8),
        "check": np.where(converted, check, 0.0),
        "checkout_time_s": t,
        "_check_if_converted": check,
    })


def apply_kiosk_ui(
    sessions: pd.DataFrame, treated: np.ndarray, *, check_lift: float = 0.0, extra_time_s: float = 0.0,
    queue_conv_penalty: float = 0.0, seed: int = 0, name: str = "kiosk_ui",
) -> Tuple[pd.DataFrame, Truth]:
    """
    Версия B: чек ×(1 + check_lift), оформление дольше на ``extra_time_s``.
    Интерференция: в ресторан-дне конверсия **всех** сессий (обеих версий) падает на
    ``queue_conv_penalty × доля сессий B`` — общая очередь. Поэтому сравнение A и B внутри
    ресторана не видит этот вред; правда о полном эффекте — в ``Truth.params``.
    """
    treated = np.asarray(treated, dtype=bool)
    if treated.shape[0] != len(sessions):
        raise ValueError("Длина treated не совпадает с числом сессий")
    rng = stream(seed, "kiosk", name)
    s = sessions.copy()
    key = s["restaurant_id"].astype(str) + "|" + s["date"].astype(str)
    share_b = pd.Series(treated.astype(float)).groupby(key.to_numpy()).transform("mean").to_numpy()
    lose = (s["converted"].to_numpy() == 1) & (rng.random(len(s)) < queue_conv_penalty * share_b)
    conv = np.where(lose, 0, s["converted"].to_numpy()).astype(np.int8)
    full_check = s["_check_if_converted"].to_numpy() * np.where(treated, 1.0 + check_lift, 1.0)
    s["converted"] = conv
    s["check"] = np.where(conv == 1, full_check, 0.0)
    s["checkout_time_s"] = s["checkout_time_s"].to_numpy() + np.where(treated, extra_time_s, 0.0)
    s["_check_if_converted"] = full_check
    base_rev = sessions["check"].to_numpy()
    truth = Truth(
        scenario="kiosk_ui", metric="check", lift_on="avg_check", units=[], start="", end="",
        params=dict(check_lift=check_lift, extra_time_s=extra_time_s, queue_conv_penalty=queue_conv_penalty,
                    direct_effect_rel=check_lift,
                    total_effect_if_all_b_rel=(1 + check_lift) * (1 - queue_conv_penalty) - 1),
        n_treated_cells=int(treated.sum()),
        per_restaurant_mean_abs=float(s["check"].to_numpy()[treated].mean() - base_rev[treated].mean()),
        per_restaurant_mean_rel=float(check_lift),
        network_total_abs=float(s["check"].sum() - base_rev.sum()),
        network_total_rel=float(s["check"].sum() / max(base_rev.sum(), 1e-9) - 1),
        note="direct_effect_rel — эффект внутри ресторана; total_effect_if_all_b_rel — если все сессии в версии B.",
    )
    return s, truth
