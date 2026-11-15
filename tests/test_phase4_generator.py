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
        # корректировки и ошибки оплаты нетронутых заказов — те же, что в мире без воздействия
        mine = ev.adjustments[ev.adjustments["order_id"].isin(untouched.index)].reset_index(drop=True)
        theirs = base.adjustments[base.adjustments["order_id"].isin(untouched.index)].reset_index(drop=True)
        assert len(mine) > 1000 and mine.equals(theirs)
        errs = lambda e: e.app_events[e.app_events["event_type"] == "payment_error"].set_index("event_id")["event_time"]
        common = errs(ev).index.intersection(errs(base).index)
        assert len(common) > 1000 and (errs(ev)[common] == errs(base)[common]).all()
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
    joined = adj.merge(orders[["order_time", "amount"]].rename(columns={"amount": "amount_order"}),
                       left_on="order_id", right_index=True)
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


def test_generator_rejects_bad_input(world):
    with pytest.raises(ValueError, match="часовым поясом"):
        simulate_app_world(1, 10, datetime(2026, 9, 1))
    for n, pre, test in ((0, 28, 28), (10, 0, 28), (10, 28, 0)):
        with pytest.raises(ValueError, match="≥ 1"):
            simulate_app_world(1, n, START, pre_days=pre, test_days=test)
    tiny = simulate_app_world(1, 1, START, pre_days=1, test_days=1)  # нижняя граница допустима
    assert len(tiny.guests) == 1 and tiny.total_days == 2
    treated = np.ones(len(world.guests), dtype=bool)
    for kw in ({"lift": -1.0}, {"crash_lift": -1.0}):
        with pytest.raises(ValueError, match="> -1"):
            world.realize(treated, **kw)
    with pytest.raises(ValueError, match="Длина treated"):
        world.realize(treated[:-1])
    with pytest.raises(ValueError):
        AppParams().p_cancel = 0.5  # параметры неизменяемы


def test_noise_and_multipliers_are_mean_one(world, base):
    """Множители периода и шум чека — логнормальные со средним 1: генератор не сдвигает уровень выручки."""
    g = world.guests
    assert abs(g["m_pre"].mean() - 1) < 0.02 and abs(g["m_test"].mean() - 1) < 0.02
    check = g.set_index("unit_id")["check"]
    noise = base.orders["amount"] / check.loc[base.orders["unit_id"]].to_numpy()
    assert abs(noise.mean() - 1) < 0.01, noise.mean()


def test_attributes_follow_params(world):
    g = world.guests
    p = world.params
    assert abs((g["platform"] == "web").mean() - p.web_share) < 0.005
    assert abs((g["platform"] == "ios").mean() - p.ios_share) < 0.01
    assert abs(g["has_delivery"].mean() - p.has_delivery_share) < 0.01
    assert abs(g["is_new_guest"].mean() - p.new_guest_share) < 0.01
    # у активных гостей чаще iOS и свежая версия
    act = g["lam"].rank(pct=True)
    top, low = act > 0.8, act < 0.2
    assert (g.loc[top, "platform"] == "ios").mean() > (g.loc[low, "platform"] == "ios").mean() + 0.2
    versions = ["5.0", "5.1.9", "5.2", "5.4.1", "5.10", "6.0"]
    v = g["app_version"].map(versions.index)
    assert set(g["app_version"]) == set(versions) and 2.4 < v.mean() < 2.6, v.mean()
    assert v[top].mean() > v[low].mean() + 1


def test_amounts_discounts_and_adjustment_delays(world, base):
    treated = np.random.default_rng(6).random(len(world.guests)) < 0.5
    ev = world.realize(treated, lift=0.2)
    assert list(ev.orders.columns) == ["order_id", "unit_id", "unit_type", "order_time", "amount", "discount", "day",
                                       "counterfactual"]
    assert isinstance(ev.orders.index, pd.RangeIndex) and (~ev.orders["counterfactual"]).sum() > 1000
    cents = lambda s: np.allclose(s * 100, np.round(s * 100), atol=1e-6)  # суммы — до копейки
    assert cents(ev.orders["amount"]) and cents(ev.orders["discount"])
    rate = ev.orders["discount"] / ev.orders["amount"]
    assert 0.075 < rate.mean() < 0.085 and 0.027 < rate.std() < 0.033, (rate.mean(), rate.std())
    adj = base.adjustments.merge(base.orders[["order_id", "order_time", "amount"]].rename(columns={"amount": "order_amount"}),
                                 on="order_id")
    delay = (adj["event_time"] - adj["order_time"]).dt.total_seconds()
    cancel, refund = adj["kind"] == "cancel", adj["kind"] == "refund"
    assert 300 <= delay[cancel].min() < 400 and 7100 < delay[cancel].max() <= 7200
    assert 86_400 <= delay[refund].min() < 1.1 * 86_400 and 2.9 * 86_400 < delay[refund].max() <= 3 * 86_400
    share = adj.loc[refund, "amount"] / adj.loc[refund, "order_amount"]
    assert cents(adj.loc[refund, "amount"]) and share.min() > 0.09 and share.max() < 0.51
    assert 0.016 < refund.sum() / len(base.orders) < 0.024


def test_payment_errors_precede_their_order(base):
    errs = base.app_events[base.app_events["event_type"] == "payment_error"]
    order_id = errs["event_id"].str.rsplit("-err", n=1).str[0]
    k = errs["event_id"].str.rsplit("-err", n=1).str[1].astype(int)
    t_order = base.orders.set_index("order_id").loc[order_id, "order_time"].to_numpy()
    gap = (t_order - errs["event_time"].to_numpy()) / np.timedelta64(1, "s")
    assert len(errs) > 1000 and (gap == (k.to_numpy() + 1) * 60).all()  # k-я ошибка — за (k + 1) минут до заказа


def test_crash_lift_negative_and_truth_arithmetic(world, base):
    treated = np.random.default_rng(7).random(len(world.guests)) < 0.5
    ev = world.realize(treated, crash_lift=-0.5)
    shown_t = set(world.guests.loc[treated & ~np.isnan(world.guests["exposure_day"]), "unit_id"])
    crash = lambda e: e.app_events[e.app_events["event_type"] == "app_crash"]["unit_id"]
    n_t, n_t0 = crash(ev).isin(shown_t).sum(), crash(base).isin(shown_t).sum()
    assert (~crash(ev).isin(shown_t)).sum() == (~crash(base).isin(shown_t)).sum() and n_t < 0.85 * n_t0
    for lift in (0.1, -0.1):
        t = world.realize(treated, lift=lift).truth
        assert np.sign(t.network_total_abs) == np.sign(lift)
        assert t.per_restaurant_mean_abs == pytest.approx(t.network_total_abs / t.n_treated_cells)
