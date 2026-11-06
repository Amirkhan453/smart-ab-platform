"""
Гости приложения (ТЗ v4, 11.2): генерируются по запросу под окна эксперимента, а не на всю историю.

Модель: λ_g ~ Gamma(shape, scale) заказов/день; в каждом окне множитель exp(s·z − s²/2);
заказы ~ Poisson(λ_g · множитель · дней); выручка = заказы × чек гостя × шум.
Колонка ``latent_rate_post`` — ненаблюдаемая правда (для внедрения эффекта), в анализ её не передавать.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import GuestParams
from .rng import stream
from .truth import Truth


def generate_guests(
    seed: int,
    restaurants: pd.DataFrame,
    n: int,
    pre_days: int = 28,
    test_days: int = 28,
    params: GuestParams = GuestParams(),
    window: str = "default",
    restaurant_weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if n < 1 or pre_days < 1 or test_days < 1:
        raise ValueError("n, pre_days и test_days должны быть ≥ 1")
    rng = stream(seed, "guests", window)
    lam = rng.gamma(params.freq_shape, params.freq_scale, n)
    check = np.exp(rng.normal(np.log(params.check_median), params.check_sd, n))

    def period(days: int):
        mult = np.exp(params.period_sd * rng.standard_normal(n) - params.period_sd ** 2 / 2)
        rate = lam * mult
        k = rng.poisson(rate * days)
        noise = np.exp(rng.normal(0.0, params.order_noise_sd, n) / np.sqrt(np.maximum(k, 1)))
        return k, k * check * noise, rate

    pre_k, pre_rev, _ = period(pre_days)
    k, rev, rate_post = period(test_days)

    w = restaurant_weights
    if w is None:
        w = np.ones(len(restaurants))
    w = np.asarray(w, dtype=float)
    home = rng.choice(restaurants["restaurant_id"].to_numpy(), size=n, p=w / w.sum())
    return pd.DataFrame({
        "guest_id": [f"G{window[:8]}_{i:07d}" for i in range(n)],
        "home_restaurant_id": home,
        "pre_orders": pre_k,
        "pre_revenue": pre_rev,
        "orders": k,
        "revenue": rev,
        "converted": (k > 0).astype(np.int8),
        "avg_check_latent": check,
        "latent_rate_post": rate_post,
    })


def apply_guest_effect(
    guests: pd.DataFrame, treated: np.ndarray, lift: float, test_days: int, seed: int,
    name: str = "guest_effect", scenario: str = "guest_effect",
) -> Tuple[pd.DataFrame, Truth]:
    """
    Эффект на частоту заказов: тестовые гости получают доп. заказы ~ Poisson(λ·lift·дней)
    (при lift < 0 — биномиальное прореживание). Ожидаемый относительный эффект на выручку = lift.
    """
    treated = np.asarray(treated, dtype=bool)
    if treated.shape[0] != len(guests):
        raise ValueError("Длина treated не совпадает с числом гостей")
    if lift <= -1:
        raise ValueError("lift должен быть > -1")
    rng = stream(seed, "guests", name)
    g = guests.copy()
    k = g["orders"].to_numpy().copy()
    rev = g["revenue"].to_numpy().copy()
    unit_check = np.where(k > 0, rev / np.maximum(k, 1), g["avg_check_latent"].to_numpy())
    if lift >= 0:
        extra = rng.poisson(g["latent_rate_post"].to_numpy() * lift * test_days) * treated
        new_k = k + extra
        new_rev = rev + extra * unit_check
    else:
        new_k = np.where(treated, rng.binomial(k, 1.0 + lift), k)
        new_rev = new_k * unit_check
    g["orders"], g["revenue"], g["converted"] = new_k, new_rev, (new_k > 0).astype(np.int8)
    delta = new_rev - rev
    truth = Truth(
        scenario=scenario, metric="revenue", lift_on="frequency", units=[], start="", end="",
        params=dict(lift=lift, test_days=test_days, n_treated=int(treated.sum())),
        n_treated_cells=int(treated.sum()),
        per_restaurant_mean_abs=float(delta[treated].mean()) if treated.any() else 0.0,
        per_restaurant_mean_rel=float(delta[treated].sum() / max(rev[treated].sum(), 1e-9)) if treated.any() else 0.0,
        network_total_abs=float(delta.sum()),
        network_total_rel=float(delta.sum() / max(rev[treated].sum(), 1e-9)) if treated.any() else 0.0,
        note="Для гостей поля per_restaurant_* означают эффект на гостя (ATT в рублях и относительно).",
    )
    return g, truth


def adopt_feature(
    guests: pd.DataFrame, base_rate: float, selection_strength: float, lift: float,
    test_days: int, seed: int,
) -> Tuple[pd.DataFrame, Truth]:
    """
    Самоотбор: гость сам начинает пользоваться фичей с вероятностью
    sigmoid(logit(base_rate) + selection_strength · z(log1p(pre_revenue))). Активные гости
    чаще пользуются — наивное сравнение «пользователи vs остальные» смещено вверх.
    Колонка ``adopted`` — наблюдаемое воздействие; истинный ATT — в ``Truth``.
    """
    if not 0 < base_rate < 1:
        raise ValueError("base_rate в (0, 1)")
    rng = stream(seed, "guests", "adoption")
    x = np.log1p(guests["pre_revenue"].to_numpy())
    z = (x - x.mean()) / max(x.std(), 1e-12)
    logit = np.log(base_rate / (1 - base_rate)) + selection_strength * z
    adopted = rng.random(len(guests)) < 1.0 / (1.0 + np.exp(-logit))
    g, truth = apply_guest_effect(guests, adopted, lift, test_days, seed, name="adoption_effect",
                                  scenario="self_selection")
    g["adopted"] = adopted.astype(np.int8)
    return g, truth
