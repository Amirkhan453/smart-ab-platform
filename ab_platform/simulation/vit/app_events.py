"""
Поток событий приложения «как ВИТ» (ТЗ v4, 11.2; ТЗ v4.1: CNV-1, CNV-2, MET-2): заказы со временем, отмены
и возвраты, падения приложения, попытки и ошибки оплаты, момент, когда гость впервые доходит до экрана теста.

Модель гостя — та же, что в generate_guests (GuestParams): частота заказов λ ~ Gamma, множитель периода
exp(s·z − s²/2) отдельно для пре-периода и периода теста (отсюда корреляция пре/пост ≈ 0,5), заказы — пуассоновский
поток. На окне 28 дней CV выручки на гостя ≈ 2,5 (ТЗ 19.3) — проверяется тестом.

Двухшаговая схема, чтобы вариант назначал настоящий движок, а не генератор:
1) simulate_app_world — мир без воздействия: гости, их признаки, контрфактические заказы и события,
   время первого захода на экран теста (или NaT, если гость не дошёл);
2) AppWorld.realize(treated, ...) — события с эффектом: у гостей из treated после первого показа меняются
   частота заказов (lift) и частота падений (crash_lift). До показа поведение одинаково — так и должно быть.

Истина (Truth) считается по контрфакту в окне атрибуции [показ; показ + W).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import GuestParams
from .rng import stream
from .truth import Truth

DAY = 86_400.0


class AppParams(BaseModel):
    """Поведение в приложении. Значения — допущения порядка величины, не данные ВИТ."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    visit_base_per_day: float = Field(0.15, ge=0, description="Заходы без заказа, в день")
    visits_per_order: float = Field(2.0, ge=0, description="Дополнительных заходов на каждый заказ")
    screen_reach: float = Field(0.4, gt=0, le=1, description="Доля заходов, доходящих до экрана теста")
    p_cancel: float = Field(0.03, ge=0, le=1)
    p_refund: float = Field(0.02, ge=0, le=1)
    refund_share_min: float = Field(0.1, gt=0, le=1)
    refund_share_max: float = Field(0.5, gt=0, le=1)
    discount_rate_mean: float = Field(0.08, ge=0, lt=1)
    crash_per_visit: float = Field(0.004, ge=0, le=1)
    payment_error_per_order: float = Field(0.03, ge=0, description="Ожидаемое число ошибок оплаты на заказ")
    ios_share: float = Field(0.45, ge=0, le=1)
    web_share: float = Field(0.05, ge=0, le=1)
    has_delivery_share: float = Field(0.7, ge=0, le=1)
    new_guest_share: float = Field(0.1, ge=0, le=1)


def _utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError("start должен быть с часовым поясом")
    return t.astimezone(timezone.utc)


def _times(start: datetime, days: np.ndarray) -> pd.Series:
    """Дни от начала → время UTC с точностью до микросекунды (как в хранилище)."""
    return pd.Series(pd.Timestamp(start) + pd.to_timedelta(np.round(days * DAY * 1e6).astype("int64"), unit="us"))


def _uniform_points(rng, counts: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Пуассоновский поток при известном числе точек: равномерно на [lo, hi) у каждой единицы."""
    idx = np.repeat(np.arange(len(counts)), counts)
    return idx, lo[idx] + rng.random(len(idx)) * (hi[idx] - lo[idx])


@dataclass
class AppWorld:
    seed: int
    start: datetime
    test_start: datetime
    end: datetime
    guests: pd.DataFrame          # unit_id, признаки, lam, m_pre, m_test, check, visit_rate, exposure_day
    params: AppParams
    guest_params: GuestParams
    prefix: str

    @property
    def pre_days(self) -> float:
        return (self.test_start - self.start).total_seconds() / DAY

    @property
    def total_days(self) -> float:
        return (self.end - self.start).total_seconds() / DAY

    def exposure_times(self) -> pd.Series:
        """Время первого захода на экран теста (UTC) или NaT."""
        days = self.guests["exposure_day"].to_numpy()
        out = _times(self.start, np.nan_to_num(days, nan=0.0))
        out[np.isnan(days)] = pd.NaT
        return out.set_axis(self.guests["unit_id"])

    def attributes(self, i: int) -> Dict[str, object]:
        g = self.guests.iloc[i]
        return {"platform": g["platform"], "app_version": g["app_version"], "has_delivery": bool(g["has_delivery"]),
                "is_new_guest": bool(g["is_new_guest"])}

    def realize(self, treated: Optional[np.ndarray] = None, lift: float = 0.0, crash_lift: float = 0.0,
                window_days: float = 7.0, name: str = "realize") -> "AppEvents":
        """
        События мира. treated — маска гостей варианта; эффект действует только после их первого показа.
        lift > 0 — дополнительные заказы с частотой λ·lift, lift < 0 — прореживание заказов после показа.
        """
        g, p, gp = self.guests, self.params, self.guest_params
        n = len(g)
        treated = np.zeros(n, dtype=bool) if treated is None else np.asarray(treated, dtype=bool)
        if treated.shape[0] != n:
            raise ValueError("Длина treated не совпадает с числом гостей")
        if lift <= -1 or crash_lift <= -1:
            raise ValueError("lift и crash_lift должны быть > -1")
        rng = stream(self.seed, "app_events", name)
        lam, m_pre, m_test = g["lam"].to_numpy(), g["m_pre"].to_numpy(), g["m_test"].to_numpy()
        check, exp_day = g["check"].to_numpy(), g["exposure_day"].to_numpy()
        pre, total = self.pre_days, self.total_days
        zeros, pre_end, ends = np.zeros(n), np.full(n, pre), np.full(n, total)
        active = treated & ~np.isnan(exp_day)
        t_exp = np.where(active, exp_day, total)

        # контрфактические заказы: пре-период и период теста (до и после показа — один поток)
        rng_cf = stream(self.seed, "app_events", "counterfactual")
        i_pre, d_pre = _uniform_points(rng_cf, rng_cf.poisson(lam * m_pre * pre), zeros, pre_end)
        i_tst, d_tst = _uniform_points(rng_cf, rng_cf.poisson(lam * m_test * (total - pre)), pre_end, ends)
        idx = np.concatenate([i_pre, i_tst])
        day = np.concatenate([d_pre, d_tst])
        noise_cf = np.exp(rng_cf.normal(0.0, gp.order_noise_sd, len(idx)) - gp.order_noise_sd ** 2 / 2)
        keep = np.ones(len(idx), dtype=bool)
        if lift < 0:  # прореживание заказов после показа у тестовой группы
            after = active[idx] & (day >= t_exp[idx])
            keep &= ~(after & (rng.random(len(idx)) < -lift))
        i_add, d_add = (np.array([], dtype=int), np.array([]))
        if lift > 0:
            i_add, d_add = _uniform_points(rng, rng.poisson(lam * m_test * lift * np.maximum(total - t_exp, 0.0)),
                                           t_exp, ends)
        noise_add = np.exp(rng.normal(0.0, gp.order_noise_sd, len(i_add)) - gp.order_noise_sd ** 2 / 2)
        cf_amt = np.round(check[idx] * noise_cf, 2)
        o_idx = np.concatenate([idx[keep], i_add])
        o_day = np.concatenate([day[keep], d_add])
        o_amt = np.concatenate([cf_amt[keep], np.round(check[i_add] * noise_add, 2)])
        o_cf = np.concatenate([np.ones(keep.sum(), dtype=bool), np.zeros(len(i_add), dtype=bool)])
        # номер заказа устойчив: контрфактический заказ сохраняет id в любой реализации, добавленные — свои id
        cf_no = np.arange(len(idx))
        ids = [f"{self.prefix}o{k}" for k in cf_no[keep]] + [f"{self.prefix}x{k}" for k in range(len(i_add))]
        disc = np.round(o_amt * np.clip(rng.normal(p.discount_rate_mean, 0.03, len(o_idx)), 0.0, 0.5), 2)
        orders = pd.DataFrame({
            "order_id": ids, "unit_id": g["unit_id"].to_numpy()[o_idx],
            "unit_type": "guest_id", "order_time": _times(self.start, o_day), "amount": o_amt, "discount": disc,
            "day": o_day, "counterfactual": o_cf,
        }).sort_values(["order_time", "order_id"], kind="stable").reset_index(drop=True)

        # отмены и возвраты — по исходному заказу, приходят позже
        u = rng.random(len(orders))
        cancel, refund = u < p.p_cancel, (u >= p.p_cancel) & (u < p.p_cancel + p.p_refund)
        share = rng.uniform(p.refund_share_min, p.refund_share_max, len(orders))
        delay = np.where(cancel, rng.uniform(300, 7200, len(orders)), rng.uniform(DAY, 3 * DAY, len(orders)))
        adj = orders.loc[cancel | refund, ["order_id", "order_time", "amount"]].copy()
        adj["kind"] = np.where(cancel[cancel | refund], "cancel", "refund")
        adj["amount"] = np.where(adj["kind"] == "refund", np.maximum(np.round(adj["amount"] * share[cancel | refund], 2), 0.01),
                                 np.nan)
        adj["event_time"] = adj["order_time"] + pd.to_timedelta(np.round(delay[cancel | refund] * 1e6).astype("int64"),
                                                                unit="us")
        adj["adjustment_id"] = [f"{self.prefix}a{k}" for k in range(len(adj))]
        adjustments = adj[["adjustment_id", "order_id", "kind", "amount", "event_time"]].reset_index(drop=True)

        # события приложения: падения на заходах, ошибки оплаты на заказах
        visits = g["visit_rate"].to_numpy()
        crash_rate = visits * p.crash_per_visit
        c_idx, c_day = _uniform_points(rng, rng.poisson(crash_rate * total), zeros, ends)
        c_add_idx, c_add_day = (np.array([], dtype=int), np.array([]))
        if crash_lift > 0:
            c_add_idx, c_add_day = _uniform_points(
                rng, rng.poisson(crash_rate * crash_lift * np.maximum(total - t_exp, 0.0)), t_exp, ends)
        elif crash_lift < 0:
            keep_c = ~(active[c_idx] & (c_day >= t_exp[c_idx]) & (rng.random(len(c_idx)) < -crash_lift))
            c_idx, c_day = c_idx[keep_c], c_day[keep_c]
        crash_idx, crash_day = np.concatenate([c_idx, c_add_idx]), np.concatenate([c_day, c_add_day])
        n_err = rng.poisson(p.payment_error_per_order, len(orders))
        err_rows = np.repeat(np.arange(len(orders)), n_err)
        err_time = orders["order_time"].iloc[err_rows].reset_index(drop=True) - pd.to_timedelta(
            np.round(rng.uniform(5, 600, len(err_rows)) * 1e6).astype("int64"), unit="us")
        unit_of = g["unit_id"].to_numpy()
        app = pd.concat([
            pd.DataFrame({"unit_id": unit_of[crash_idx], "event_type": "app_crash",
                          "event_time": _times(self.start, crash_day)}),
            pd.DataFrame({"unit_id": orders["unit_id"].to_numpy()[err_rows], "event_type": "payment_error",
                          "event_time": err_time}),
            pd.DataFrame({"unit_id": orders["unit_id"], "event_type": "payment_attempt",
                          "event_time": orders["order_time"]}),
        ], ignore_index=True).sort_values(["event_time", "unit_id"], kind="stable").reset_index(drop=True)
        app["unit_type"] = "guest_id"
        app["event_id"] = [f"{self.prefix}e{k}" for k in range(len(app))]
        app = app[["event_id", "unit_id", "unit_type", "event_type", "event_time"]]

        truth = self._truth(orders, idx, day, cf_amt, active, t_exp, lift, crash_lift, window_days)
        return AppEvents(orders=orders, adjustments=adjustments, app_events=app, truth=truth)

    def _truth(self, orders, cf_idx, cf_day, cf_amt, active, t_exp, lift, crash_lift, window_days) -> Truth:
        """
        Истинный эффект на выручку (до отмен) в окне атрибуции у показанных тестовых гостей:
        фактические заказы против контрфактических (тех же заказов без воздействия).
        """
        pos = pd.Series(np.arange(len(self.guests)), index=self.guests["unit_id"])
        gi, day = pos.loc[orders["unit_id"]].to_numpy(), orders["day"].to_numpy()
        in_window = active[gi] & (day >= t_exp[gi]) & (day < t_exp[gi] + window_days)
        cf_window = active[cf_idx] & (cf_day >= t_exp[cf_idx]) & (cf_day < t_exp[cf_idx] + window_days)
        actual = float(orders.loc[in_window, "amount"].sum())
        counterfactual = float(cf_amt[cf_window].sum())
        n_active = int(active.sum())
        rel = (actual - counterfactual) / counterfactual if counterfactual > 0 else 0.0
        return Truth(
            scenario="app_order_rate", metric="revenue_in_attribution_window", lift_on="order_rate_after_exposure",
            units=[], start=self.test_start.isoformat(), end=self.end.isoformat(),
            params=dict(lift=lift, crash_lift=crash_lift, window_days=window_days, n_treated_exposed=n_active),
            n_treated_cells=n_active,
            per_restaurant_mean_abs=(actual - counterfactual) / max(n_active, 1),
            per_restaurant_mean_rel=rel,
            network_total_abs=actual - counterfactual,
            network_total_rel=rel,
            note="Гости приложения: per_restaurant_* — эффект на показанного тестового гостя в окне атрибуции; "
                 "ожидаемый относительный эффект на выручку равен lift.",
        )


@dataclass
class AppEvents:
    orders: pd.DataFrame        # + служебные day, counterfactual
    adjustments: pd.DataFrame
    app_events: pd.DataFrame
    truth: Truth

    def payload(self, kind: str) -> List[Dict[str, object]]:
        """События в формате HTTP API (POST /orders, /order-adjustments, /app-events): время — ISO 8601 UTC."""
        if kind == "orders":
            frame, time_col = self.orders[["order_id", "unit_id", "unit_type", "order_time", "amount", "discount"]], "order_time"
        elif kind == "adjustments":
            frame, time_col = self.adjustments, "event_time"
        elif kind == "app_events":
            frame, time_col = self.app_events, "event_time"
        else:
            raise ValueError(f"Неизвестный вид событий '{kind}'")
        out = frame.assign(**{time_col: frame[time_col].map(lambda t: t.isoformat())}).to_dict("records")
        if kind == "adjustments":
            for r in out:
                if r["kind"] == "cancel":
                    del r["amount"]
        return out


def simulate_app_world(
    seed: int,
    n: int,
    start: datetime,
    pre_days: int = 28,
    test_days: int = 28,
    params: AppParams = AppParams(),
    guest_params: GuestParams = GuestParams(),
    name: str = "app",
) -> AppWorld:
    """Гости приложения на окне [start, start + pre_days + test_days) и момент их первого захода на экран теста."""
    if n < 1 or pre_days < 1 or test_days < 1:
        raise ValueError("n, pre_days и test_days должны быть ≥ 1")
    start = _utc(start)
    rng = stream(seed, "app_world", name)
    gp, p = guest_params, params
    lam = rng.gamma(gp.freq_shape, gp.freq_scale, n)

    def mult():
        return np.exp(gp.period_sd * rng.standard_normal(n) - gp.period_sd ** 2 / 2)
    m_pre, m_test = mult(), mult()
    check = np.exp(rng.normal(np.log(gp.check_median), gp.check_sd, n))
    visit_rate = p.visit_base_per_day + p.visits_per_order * lam * m_test
    # первый заход на экран теста в период теста: показательное время с частотой заходов × доля дохода до экрана
    first = pre_days + rng.exponential(1.0 / (visit_rate * p.screen_reach))
    exposure_day = np.where(first < pre_days + test_days, first, np.nan)
    # признаки: у активных гостей чаще iOS, свежая версия и доставка
    activity = pd.Series(lam).rank(pct=True).to_numpy()
    u = rng.random(n)
    platform = np.where(u < p.web_share, "web", np.where(u < p.web_share + p.ios_share * (0.6 + 0.8 * activity),
                                                         "ios", "android"))
    versions = np.array(["5.0", "5.1.9", "5.2", "5.4.1", "5.10", "6.0"])
    v_idx = np.clip(np.floor(rng.random(n) * 3 + activity * 3), 0, 5).astype(int)
    prefix = f"{name}-{seed}-"
    guests = pd.DataFrame({
        "unit_id": [f"{prefix}g{i}" for i in range(n)],
        "platform": platform,
        "app_version": versions[v_idx],
        "has_delivery": rng.random(n) < p.has_delivery_share,
        "is_new_guest": rng.random(n) < p.new_guest_share,
        "lam": lam, "m_pre": m_pre, "m_test": m_test, "check": check, "visit_rate": visit_rate,
        "exposure_day": exposure_day,
    })
    return AppWorld(seed=seed, start=start, test_start=start + timedelta(days=pre_days),
                    end=start + timedelta(days=pre_days + test_days), guests=guests, params=p, guest_params=gp,
                    prefix=prefix)
