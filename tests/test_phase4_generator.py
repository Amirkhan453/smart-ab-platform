"""
Фаза 4, П2: синтетический поток событий приложения — калибровка (ТЗ 19.3), детерминизм, эффект только после
показа, истина, формат событий для HTTP-приёма.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ab_platform.events import AppEventIn, MemoryEventStore, OrderAdjustmentIn, OrderIn
from ab_platform.simulation.vit import AppParams, simulate_app_world

START = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def world():
    return simulate_app_world(1, 100_000, START)


@pytest.fixture(scope="module")
def base(world):
    return world.realize()


def by_guest(world, orders, lo, hi):
    o = orders[(orders["day"] >= lo) & (orders["day"] < hi)]
    return o.groupby("unit_id")["amount"].sum().reindex(world.guests["unit_id"], fill_value=0.0)


def test_calibration_matches_tz_19_3(world, base):
    """SYN-4: на окне 28 дней CV выручки на гостя ≈ 2,5, корреляция пре/пост ≈ 0,5 (±10%)."""
    pre, post = by_guest(world, base.orders, 0, 28), by_guest(world, base.orders, 28, 56)
    cv, rho = post.std() / post.mean(), np.corrcoef(pre, post)[0, 1]
    assert 2.25 <= cv <= 2.75 and 0.45 <= rho <= 0.55, (cv, rho)
    assert 0.43 <= (post > 0).mean() <= 0.53  # доля гостей с заказом за 28 дней ≈ 48% (как в generate_guests)
    assert 420 <= base.orders["amount"].mean() <= 520
    shown = ~np.isnan(world.guests["exposure_day"].to_numpy())
    assert 0.5 < shown.mean() < 0.98 and (world.guests.loc[shown, "exposure_day"] >= 28).all()


def test_deterministic_and_independent_of_assignment(world, base):
    again = simulate_app_world(1, 100_000, START).realize()
    for a, b in ((again.orders, base.orders), (again.adjustments, base.adjustments), (again.app_events, base.app_events)):
        assert a.equals(b)
    other = simulate_app_world(2, 1000, START)
    assert not other.guests["lam"].equals(world.guests["lam"].head(1000))
    # без эффекта события не зависят от того, кого считать тестовой группой (A/A)
    treated = np.random.default_rng(3).random(len(world.guests)) < 0.5
    aa = world.realize(treated)
    assert aa.orders.equals(base.orders) and aa.app_events.equals(base.app_events)
    assert aa.truth.network_total_rel == 0.0


def test_effect_only_after_exposure_and_truth(world, base):
    treated = np.random.default_rng(4).random(len(world.guests)) < 0.5
    exp_day = world.guests["exposure_day"].to_numpy()
    pos = pd.Series(np.arange(len(world.guests)), index=world.guests["unit_id"])
    for lift in (0.10, -0.10):
        ev = world.realize(treated, lift=lift)
        gi = pos.loc[ev.orders["unit_id"]].to_numpy()
        before = ~(treated[gi] & (ev.orders["day"].to_numpy() >= np.nan_to_num(exp_day[gi], nan=np.inf)))
        # всё, что не «тестовый гость после своего показа», совпадает с миром без воздействия
        untouched = ev.orders.loc[before, ["order_id", "amount", "day"]].set_index("order_id")
        same = base.orders.set_index("order_id").reindex(untouched.index)
        assert same["amount"].notna().all() and (same["amount"] == untouched["amount"]).all()
        assert (same["day"] == untouched["day"]).all()
        # и наоборот: ни один заказ мира без воздействия до показа не пропал
        bgi = pos.loc[base.orders["unit_id"]].to_numpy()
        base_before = ~(treated[bgi] & (base.orders["day"].to_numpy() >= np.nan_to_num(exp_day[bgi], nan=np.inf)))
        assert base.orders.loc[base_before, "order_id"].isin(untouched.index).all()
        if lift > 0:
            added = ev.orders[~ev.orders["counterfactual"]]
            ga = pos.loc[added["unit_id"]].to_numpy()
            assert len(added) > 0 and treated[ga].all() and (added["day"].to_numpy() >= exp_day[ga]).all()
        else:
            assert len(ev.orders) < len(base.orders) and ev.orders["counterfactual"].all()
        t = ev.truth
        assert abs(t.network_total_rel - lift) < 0.015, (lift, t.network_total_rel)
        assert t.n_treated_cells == int((treated & ~np.isnan(exp_day)).sum())
    with pytest.raises(ValueError):
        world.realize(treated, lift=-1.0)
    with pytest.raises(ValueError):
        world.realize(treated[:10])


def test_crash_lift_after_exposure(world, base):
    treated = np.random.default_rng(5).random(len(world.guests)) < 0.5
    ev = world.realize(treated, crash_lift=1.0)
    crashes = ev.app_events[ev.app_events["event_type"] == "app_crash"]
    base_crashes = base.app_events[base.app_events["event_type"] == "app_crash"]
    shown_t = set(world.guests.loc[treated & ~np.isnan(world.guests["exposure_day"]), "unit_id"])
    n_t, n_t0 = crashes["unit_id"].isin(shown_t).sum(), base_crashes["unit_id"].isin(shown_t).sum()
    n_c, n_c0 = (~crashes["unit_id"].isin(shown_t)).sum(), (~base_crashes["unit_id"].isin(shown_t)).sum()
    assert n_c == n_c0 and n_t > 1.2 * n_t0  # у остальных падений не прибавилось


def test_adjustments_and_app_events_are_consistent(base):
    orders = base.orders.set_index("order_id")
    adj = base.adjustments
    assert adj["order_id"].isin(orders.index).all() and adj["adjustment_id"].is_unique
    joined = adj.join(orders[["order_time", "amount"]], on="order_id", rsuffix="_order")
    assert (joined["event_time"] > joined["order_time"]).all()
    refunds = joined[joined["kind"] == "refund"]
    assert (refunds["amount"] > 0).all() and (refunds["amount"] < refunds["amount_order"]).all()
    assert joined.loc[joined["kind"] == "cancel", "amount"].isna().all()
    assert 0.025 < (adj["kind"] == "cancel").sum() / len(orders) < 0.035
    app = base.app_events
    assert app["event_id"].is_unique and set(app["event_type"]) == {"app_crash", "payment_attempt", "payment_error"}
    assert (app["event_type"] == "payment_attempt").sum() == len(orders)


def test_payload_is_accepted_by_event_store():
    world = simulate_app_world(9, 300, START, params=AppParams(p_cancel=0.2, p_refund=0.2))
    ev = world.realize()
    store = MemoryEventStore()
    r1 = store.add_orders([OrderIn(**o) for o in ev.payload("orders")])
    r2 = store.add_adjustments([OrderAdjustmentIn(**a) for a in ev.payload("adjustments")])
    r3 = store.add_app_events([AppEventIn(**e) for e in ev.payload("app_events")])
    assert (r1.accepted, r2.accepted, r3.accepted) == (len(ev.orders), len(ev.adjustments), len(ev.app_events))
    assert not (r1.conflicts or r2.conflicts or r3.conflicts) and store.orphan_adjustments() == 0
    # время в хранилище совпадает с временем генератора до микросекунды
    stored = store.frame("orders").set_index("order_id")["order_time"]
    assert all(stored[o] == t.isoformat(timespec="microseconds") for o, t in
               zip(ev.orders["order_id"], ev.orders["order_time"]))
    assert {"cancel", "refund"} == set(ev.adjustments["kind"])
    with pytest.raises(ValueError):
        ev.payload("visits")
