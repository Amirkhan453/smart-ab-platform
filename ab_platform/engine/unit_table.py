"""
Таблица «гость × метрики» из событий (ТЗ v4.1: CNV-2, MET-1, TGT-3).

Состав — первый показ каждого гостя в тесте (вариант и признаки — на момент первого показа).
Окно гостя — [первый показ; первый показ + W), W — окно атрибуции из Паспорта, и не дальше среза cutoff.
Заказы до показа в эффект не идут: из окна [показ − pre_days; показ) считается ковариата CUPED (колонка <метрика>__pre).
Отмены и возвраты относятся к исходному заказу: заказ в окне, отменённый позже, из выручки исключается.

Срез воспроизводим: upto_seq — номер последнего принятого события (ab_platform/events.py), cutoff — момент
анализа. Одинаковые (upto_seq, cutoff) дают одинаковую таблицу байт в байт.
"""
from datetime import datetime, timedelta
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ab_platform.engine.exposure_analysis import analysis_population
from ab_platform.metrics import MetricVersion


def _utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values, dtype="object"), utc=True, format="ISO8601")


def _period_aggregates(orders: pd.DataFrame, apps: pd.DataFrame, units: pd.Index, lo: pd.Series,
                       hi: pd.Series) -> pd.DataFrame:
    """Агрегаты гостя за [lo, hi): заказы всего, неотменённые, отменённые, выручка; события приложения."""
    out = pd.DataFrame(index=units)
    o = orders.join(pd.DataFrame({"lo": lo, "hi": hi}), on="unit_id")
    o = o[(o["order_time"] >= o["lo"]) & (o["order_time"] < o["hi"])]
    g = o.groupby("unit_id")
    out["orders_all"] = g.size()
    out["orders_valid"] = g["valid"].sum()
    out["orders_cancelled"] = g["cancelled"].sum()
    out["revenue"] = g["net"].sum()
    a = apps.join(pd.DataFrame({"lo": lo, "hi": hi}), on="unit_id")
    a = a[(a["event_time"] >= a["lo"]) & (a["event_time"] < a["hi"])]
    for event_type in ("app_crash", "payment_error"):
        out[f"has_{event_type}"] = a[a["event_type"] == event_type].groupby("unit_id").size()
    return out.fillna(0.0)


def _metric_values(agg: pd.DataFrame, m: MetricVersion):
    """(числитель, знаменатель) метрики по агрегатам гостя; знаменатель — только у ratio."""
    if m.formula == "has_order":
        return (agg["orders_valid"] > 0).astype(np.int64), None
    if m.formula == "revenue":
        return agg["revenue"].astype(float), None
    if m.formula == "order_count":
        return agg["orders_valid"].astype(float), None
    if m.formula == "avg_check":
        return agg["revenue"].astype(float), agg["orders_valid"].astype(float)
    if m.formula == "cancel_rate":
        return agg["orders_cancelled"].astype(float), agg["orders_all"].astype(float)
    return (agg[f"has_{m.event_type}"] > 0).astype(np.int64), None


def build_unit_table(
    store,
    experiment_id: str,
    metrics: Sequence[MetricVersion],
    window_days: float,
    pre_days: float,
    cutoff: datetime,
    unit_type: str = "guest_id",
    upto_seq: Optional[int] = None,
) -> pd.DataFrame:
    """
    Строки — гости с первым показом до cutoff. Колонки: unit_id, variant, first_exposure_time, attr_*,
    window_end, window_complete и по метрике: <имя> и <имя>__pre (ковариата CUPED), у ratio — <имя>__num
    и <имя>__den без ковариаты.
    """
    if window_days <= 0 or pre_days <= 0:
        raise ValueError("window_days и pre_days должны быть > 0")
    if cutoff.tzinfo is None:
        raise ValueError("cutoff должен быть с часовым поясом")
    if len({m.name for m in metrics}) != len(metrics):
        raise ValueError("Метрики в таблице не должны повторяться по имени.")
    cut = pd.Timestamp(cutoff).tz_convert("UTC")
    pop = analysis_population(store.first_exposures(experiment_id, upto_seq))
    if pop.empty:
        cols = [c for m in metrics for c in (m.columns() if m.columns()[1] else (m.name, f"{m.name}__pre"))]
        return pd.DataFrame(columns=[*pop.columns, "window_end", "window_complete", *cols])
    pop["t0"] = _utc(pop["first_exposure_time"])
    pop = pop[pop["t0"] < cut].sort_values("unit_id", kind="stable").set_index("unit_id")
    units = pop.index

    orders = store.frame("orders", upto_seq)
    orders = orders[orders["unit_id"].isin(units) & (orders["unit_type"] == unit_type)].copy()
    orders["order_time"] = _utc(orders["order_time"])
    adj = store.frame("adjustments", upto_seq)
    cancelled = set(adj.loc[adj["kind"] == "cancel", "order_id"])
    refunds = adj[adj["kind"] == "refund"].groupby("order_id")["amount"].sum()
    orders["cancelled"] = orders["order_id"].isin(cancelled)
    orders["valid"] = ~orders["cancelled"]
    refund = orders["order_id"].map(refunds).fillna(0.0)
    orders["net"] = np.where(orders["cancelled"], 0.0, np.maximum(orders["amount"] - refund, 0.0))
    apps = store.frame("app_events", upto_seq)
    apps = apps[apps["unit_id"].isin(units) & (apps["unit_type"] == unit_type)].copy()
    apps["event_time"] = _utc(apps["event_time"])

    t0 = pop["t0"]
    window_end = (t0 + timedelta(days=window_days)).where(t0 + timedelta(days=window_days) <= cut, cut)
    win = _period_aggregates(orders, apps, units, t0, window_end)
    pre = _period_aggregates(orders, apps, units, t0 - timedelta(days=pre_days), t0)

    table = pop.drop(columns=["t0"]).copy()
    table["window_end"] = window_end.map(lambda t: t.isoformat(timespec="microseconds"))
    table["window_complete"] = (t0 + timedelta(days=window_days)) <= cut
    for m in metrics:
        num_col, den_col = m.columns()
        num, den = _metric_values(win, m)
        table[num_col] = num
        if den_col is not None:
            table[den_col] = den
        else:
            table[f"{m.name}__pre"] = _metric_values(pre, m)[0]
    return table.reset_index()
