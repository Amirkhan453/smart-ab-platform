"""
Внедрение сценариев в ресторанную панель (ТЗ v4, 11.4).

``inject`` — эффект на тестовых ресторанах: размер, неоднородность по формату, затухание
новизны, ступенчатый старт (``unit_starts``), последействие после окончания (``carryover_days``),
переток от соседей по торговой зоне (``spillover``). Возвращает новую сеть и ``Truth``.

Взаимодействие двух тестов задаётся отдельным вызовом ``inject`` на пересечении их
тестовых ресторанов с размером взаимодействия.

``break_data`` — поломки данных: пропуски ресторан-дней, дубли строк, закрытие точек посреди теста.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .rng import stream
from .truth import Truth

if TYPE_CHECKING:  # pragma: no cover
    from .network import Network

_MONEY = ("gross_sales", "discounts", "revenue", "cogs", "margin", "waste_rub")


def _apply_multiplier(df: pd.DataFrame, rows: np.ndarray, m: np.ndarray, lift_on: str) -> None:
    """Умножает строки ``rows`` (позиции) на ``m``; работает по позициям, а не по индексу."""
    def scale(col: str, factor: np.ndarray, integer: bool = False) -> None:
        a = df[col].to_numpy(copy=True)
        if integer:
            a[rows] = np.maximum(0, np.rint(a[rows] * factor)).astype(a.dtype)
        else:
            a[rows] = a[rows] * factor
        df[col] = a

    for col in _MONEY:
        scale(col, m)
    if lift_on == "checks":
        scale("checks", m, integer=True)
        for col in ("service_time_p50_s", "service_time_p90_s"):
            scale(col, np.maximum(m, 1e-6) ** 0.3)
    rev, checks = df["revenue"].to_numpy(), df["checks"].to_numpy()
    df["avg_check"] = np.where(checks > 0, rev / np.maximum(checks, 1), 0.0)


def inject(
    net: "Network",
    units: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    lift: float,
    *,
    lift_on: Literal["checks", "avg_check"] = "checks",
    metric: str = "revenue",
    hetero_by_format: Optional[Dict[str, float]] = None,
    novelty_extra: float = 0.0,
    novelty_decay_days: float = 7.0,
    carryover_days: int = 0,
    spillover: float = 0.0,
    unit_starts: Optional[Dict[str, str | pd.Timestamp]] = None,
    scenario: str = "effect",
) -> Tuple["Network", Truth]:
    """
    Мультипликативный эффект ``1 + lift_i(t)`` на тестовых ресторанах.

    - ``lift_on="checks"`` — растёт трафик: чеки, выручка, деньги и нагрузка (время выдачи);
      ``"avg_check"`` — растёт средний чек: деньги растут, чеки нет.
    - ``hetero_by_format`` — множитель эффекта по формату, например ``{"drive": 1.5}``.
    - ``novelty_extra`` — доп. эффект новизны в момент старта, затухает с постоянной ``novelty_decay_days``.
    - ``carryover_days`` — после ``end`` эффект линейно спадает до нуля за столько дней.
    - ``spillover`` — доля прироста выручки тестовой точки, «забираемая» у соседей по торговой
      зоне (равными долями между работающими нетестовыми соседями).
    - ``unit_starts`` — индивидуальные даты старта (ступенчатая раскатка).
    """
    if lift_on not in ("checks", "avg_check"):
        raise ValueError("lift_on: 'checks' или 'avg_check'")
    if not 0.0 <= spillover <= 1.0:
        raise ValueError("spillover должен быть в [0, 1]")
    if metric not in net.daily.columns:
        raise ValueError(f"Неизвестная метрика {metric!r}")
    units = list(dict.fromkeys(units))
    unknown = set(units) - set(net.restaurants["restaurant_id"])
    if unknown:
        raise ValueError(f"Неизвестные рестораны: {sorted(unknown)[:5]}")
    if not units:
        raise ValueError("Пустой список тестовых ресторанов")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end < start:
        raise ValueError("end раньше start")

    df = net.daily.copy()
    cf_metric = df[metric].to_numpy(copy=True)
    cf_revenue = df["revenue"].to_numpy(copy=True)
    rid = df["restaurant_id"].to_numpy()
    dates = pd.DatetimeIndex(df["date"])

    starts = {u: start for u in units}
    for u, s in (unit_starts or {}).items():
        if u not in starts:
            raise ValueError(f"unit_starts: ресторан {u} не в units")
        starts[u] = pd.Timestamp(s)
    fmt = net.restaurants.set_index("restaurant_id")["format"].astype(str)
    hetero = hetero_by_format or {}
    lift_by_unit = {u: lift * float(hetero.get(fmt[u], 1.0)) for u in units}

    is_unit = pd.Series(rid).isin(units).to_numpy()
    unit_start = pd.to_datetime(pd.Series(rid).map(starts)).to_numpy()
    lift_i = pd.Series(rid).map(lift_by_unit).fillna(0.0).to_numpy()
    in_window = is_unit & (dates.values >= unit_start) & (dates.values <= end.to_datetime64())
    carry_end = end + pd.Timedelta(days=carryover_days)
    in_carry = is_unit & (dates.values > end.to_datetime64()) & (dates.values <= carry_end.to_datetime64())

    m = np.ones(len(df))
    tau = (dates.values - unit_start).astype("timedelta64[D]").astype(float)
    m[in_window] = 1.0 + lift_i[in_window] * (1.0 + novelty_extra * np.exp(-tau[in_window] / novelty_decay_days))
    if carryover_days > 0:
        k = (dates.values - end.to_datetime64()).astype("timedelta64[D]").astype(float)
        m[in_carry] = 1.0 + lift_i[in_carry] * (1.0 - k[in_carry] / (carryover_days + 1))
    affected = np.flatnonzero(in_window | in_carry)
    _apply_multiplier(df, affected, m[affected], lift_on)

    spill_units: list = []
    spill_abs = 0.0
    if spillover > 0:
        zone = net.restaurants.set_index("restaurant_id")["trade_zone"]
        df_zone = pd.Series(rid).map(zone).to_numpy()
        gain = np.zeros(len(df))
        gain[affected] = df["revenue"].to_numpy()[affected] - cf_revenue[affected]
        key = pd.Series(df_zone).astype(str) + "|" + pd.Series(dates.astype(str))
        zone_gain = pd.Series(gain).groupby(key.to_numpy()).transform("sum").to_numpy()
        neighbor = ~is_unit & (zone_gain != 0)
        n_neighbors = pd.Series(neighbor.astype(int)).groupby(key.to_numpy()).transform("sum").to_numpy()
        rows = np.flatnonzero(neighbor & (n_neighbors > 0))
        if len(rows):
            loss = spillover * zone_gain[rows] / n_neighbors[rows]
            rev = df["revenue"].to_numpy()[rows]
            factor = np.clip(1.0 - loss / np.maximum(rev, 1e-9), 0.0, None)
            before = df[metric].to_numpy()[rows].copy()
            _apply_multiplier(df, rows, factor, "checks")
            spill_abs = float((df[metric].to_numpy()[rows] - before).sum())
            spill_units = sorted(set(rid[rows]))

    delta = df[metric].to_numpy() - cf_metric
    w = pd.DataFrame({"rid": rid[in_window], "delta": delta[in_window], "cf": cf_metric[in_window]})
    if w.empty:
        raise ValueError("В окне эффекта нет работающих ресторан-дней тестовых точек")
    per_unit = w.groupby("rid").agg(delta=("delta", "mean"), cf=("cf", "mean"))
    truth = Truth(
        scenario=scenario, metric=metric, lift_on=lift_on, units=units,
        start=str(start.date()), end=str(end.date()),
        params=dict(lift=lift, hetero_by_format=hetero, novelty_extra=novelty_extra,
                    novelty_decay_days=novelty_decay_days, carryover_days=carryover_days,
                    spillover=spillover,
                    unit_starts={u: str(s.date()) for u, s in starts.items()} if unit_starts else None),
        n_treated_cells=int(in_window.sum()),
        per_restaurant_mean_abs=float(per_unit["delta"].mean()),
        per_restaurant_mean_rel=float(per_unit["delta"].mean() / per_unit["cf"].mean()),
        network_total_abs=float(w["delta"].sum()),
        network_total_rel=float(w["delta"].sum() / w["cf"].sum()),
        spillover_units=spill_units, spillover_abs=spill_abs,
        carryover_abs=float(delta[in_carry].sum()),
    )
    return replace(net, daily=df, truths=net.truths + (truth,)), truth


def break_data(
    net: "Network",
    *,
    drop_frac: float = 0.0,
    duplicate_frac: float = 0.0,
    close_units: Iterable[str] = (),
    close_from: Optional[str | pd.Timestamp] = None,
    seed_name: str = "break",
) -> Tuple["Network", dict]:
    """Пропуски ресторан-дней, дубли строк и закрытие точек с даты ``close_from``. Возвращает журнал поломок."""
    if not (0 <= drop_frac < 1 and 0 <= duplicate_frac < 1):
        raise ValueError("drop_frac и duplicate_frac должны быть в [0, 1)")
    rng = stream(net.config.seed, "effects", seed_name)
    df = net.daily
    close_units = list(close_units)
    log = {"dropped": 0, "duplicated": 0, "closed_units": close_units, "closed_rows": 0}
    if close_units:
        if close_from is None:
            raise ValueError("close_from обязателен при close_units")
        m = df["restaurant_id"].isin(close_units) & (df["date"] >= pd.Timestamp(close_from))
        log["closed_rows"] = int(m.sum())
        df = df.loc[~m]
    if drop_frac > 0:
        keep = rng.random(len(df)) >= drop_frac
        log["dropped"] = int((~keep).sum())
        df = df.loc[keep]
    if duplicate_frac > 0:
        dup = df.sample(frac=duplicate_frac, random_state=int(rng.integers(2**31 - 1)))
        log["duplicated"] = len(dup)
        df = pd.concat([df, dup])
    return replace(net, daily=df.reset_index(drop=True)), log
