"""Приёмка генератора «как ВИТ»: SYN-1 (повторяемость), SYN-2 (схемы), SYN-3 (правда), SYN-4 (параметры)."""
import numpy as np
import pandas as pd
import pytest

from ab_platform.simulation.vit import (
    GuestParams, KioskParams, NetworkConfig, adopt_feature, apply_guest_effect, apply_kiosk_ui,
    break_data, generate_kiosk_sessions, generate_network, inject, load_network, save_network,
)
from ab_platform.simulation.vit import schemas
from ab_platform.simulation.vit.panel import simulate_log_components
from ab_platform.simulation.vit.rng import stream


@pytest.fixture(scope="module")
def ref_net():
    return generate_network(NetworkConfig.preset("reference", n_restaurants=400, days=140, seed=7))


@pytest.fixture(scope="module")
def full_net():
    return generate_network(NetworkConfig.preset("pessimistic", n_restaurants=300, days=400, seed=11))


# ---------------- SYN-1: повторяемость ----------------
def test_same_seed_same_data_and_different_seed_differs():
    cfg = NetworkConfig.preset("pessimistic", n_restaurants=50, days=60, seed=3)
    a, b = generate_network(cfg), generate_network(cfg)
    pd.testing.assert_frame_equal(a.daily, b.daily)
    pd.testing.assert_frame_equal(a.restaurants, b.restaurants)
    c = generate_network(NetworkConfig.preset("pessimistic", n_restaurants=50, days=60, seed=4))
    assert not np.allclose(a.daily["revenue"].to_numpy()[:100], c.daily["revenue"].to_numpy()[:100])


def test_streams_are_independent_by_name():
    x1 = stream(1, "panel").random(5)
    stream(1, "guests").random(1000)          # другой компонент не сдвигает поток панели
    x2 = stream(1, "panel").random(5)
    assert np.array_equal(x1, x2)
    assert not np.array_equal(x1, stream(1, "kiosk").random(5))


def test_config_yaml_roundtrip_and_preset_override(tmp_path):
    cfg = NetworkConfig.preset("optimistic", n_restaurants=20, panel={"ar_sd": 0.1})
    assert cfg.panel.ar_sd == 0.1 and cfg.panel.holidays
    cfg.to_yaml(tmp_path / "c.yaml")
    assert NetworkConfig.from_yaml(tmp_path / "c.yaml") == cfg
    with pytest.raises(ValueError):
        NetworkConfig.preset("nope")
    with pytest.raises(ValueError):
        NetworkConfig(format_shares={"street": 0.5, "drive": 0.2})


# ---------------- SYN-2: схемы ----------------
def test_output_schemas(full_net):
    schemas.validate(full_net.restaurants, schemas.RESTAURANTS, "restaurants")
    schemas.validate(full_net.daily, schemas.DAILY, "daily")
    schemas.validate(full_net.events, schemas.EVENTS, "events")
    ch = full_net.channels()
    schemas.validate(ch, schemas.CHANNELS, "channels")
    schemas.validate(full_net.guests(1000), schemas.GUESTS, "guests")
    schemas.validate(full_net.kiosk_sessions("2025-03-03", 2), schemas.KIOSK_SESSIONS, "kiosk")
    with pytest.raises(ValueError, match="нет колонок"):
        schemas.validate(full_net.daily.drop(columns="revenue"), schemas.DAILY)


def test_accounting_identities(full_net):
    d = full_net.daily
    assert np.allclose(d["gross_sales"] - d["discounts"], d["revenue"])
    assert np.allclose(d["revenue"] - d["cogs"], d["margin"])
    assert (d["checks"] >= 1).all() and (d["revenue"] > 0).all()
    assert not d.duplicated(["restaurant_id", "date"]).any()
    ch = full_net.channels()
    agg = ch.groupby(["restaurant_id", "date"])[["revenue", "checks"]].sum()
    base = d.set_index(["restaurant_id", "date"])[["revenue", "checks"]].loc[agg.index]
    assert np.allclose(agg["revenue"], base["revenue"])
    assert (agg["checks"] == base["checks"]).all()
    # Киоск только там, где он есть; авто — только у формата drive.
    r = full_net.restaurants.set_index("restaurant_id")
    assert r.loc[ch.loc[ch.channel == "kiosk", "restaurant_id"].unique(), "has_kiosk"].all()
    assert (r.loc[ch.loc[ch.channel == "drive", "restaurant_id"].unique(), "format"] == "drive").all()


def test_closures_and_renovations_remove_rows(full_net):
    r, d = full_net.restaurants, full_net.daily
    closed = r.dropna(subset=["close_date"]).iloc[0]
    rows = d[d.restaurant_id == closed.restaurant_id]
    assert (rows["date"] < closed.close_date).all()
    reno = r.dropna(subset=["renovation_start"]).iloc[0]
    rows = d[d.restaurant_id == reno.restaurant_id]
    assert not ((rows["date"] >= reno.renovation_start) & (rows["date"] <= reno.renovation_end)).any()


# ---------------- SYN-3: правда об эффекте ----------------
def _window(net, start_day, days):
    s = pd.Timestamp(net.config.start_date) + pd.Timedelta(days=start_day)
    return s, s + pd.Timedelta(days=days - 1)


def test_inject_truth_matches_lift(ref_net):
    units = ref_net.restaurants.restaurant_id[:40].tolist()
    s, e = _window(ref_net, 70, 28)
    net2, truth = inject(ref_net, units, s, e, lift=0.03, lift_on="avg_check")
    assert truth.per_restaurant_mean_rel == pytest.approx(0.03, rel=1e-9)
    assert truth.network_total_rel == pytest.approx(0.03, rel=1e-9)
    assert truth.n_treated_cells == 40 * 28
    # Контроль и пре-период не тронуты, исходная сеть не изменена.
    untouched = ~(net2.daily.restaurant_id.isin(units) & net2.daily.date.between(s, e))
    assert np.array_equal(net2.daily.loc[untouched, "revenue"], ref_net.daily.loc[untouched, "revenue"])
    assert net2.truths == (truth,) and ref_net.truths == ()
    # По трафику: чеки растут, средний чек почти не меняется (округление чеков).
    net3, t3 = inject(ref_net, units, s, e, lift=0.05, lift_on="checks", metric="checks")
    assert t3.per_restaurant_mean_rel == pytest.approx(0.05, abs=0.002)


def test_inject_novelty_carryover_heterogeneity_staggered(ref_net):
    r = ref_net.restaurants
    units = r.restaurant_id[:20].tolist()
    s, e = _window(ref_net, 70, 28)
    _, t_nov = inject(ref_net, units, s, e, lift=0.02, lift_on="avg_check", novelty_extra=1.0, novelty_decay_days=5)
    assert t_nov.per_restaurant_mean_rel > 0.02
    _, t_car = inject(ref_net, units, s, e, lift=0.02, lift_on="avg_check", carryover_days=6)
    assert t_car.carryover_abs > 0
    fmt = r.set_index("restaurant_id").loc[units, "format"].astype(str)
    _, t_het = inject(ref_net, units, s, e, lift=0.02, lift_on="avg_check",
                      hetero_by_format={f: 2.0 for f in fmt.unique()})
    assert t_het.per_restaurant_mean_rel == pytest.approx(0.04, rel=1e-9)
    late = {u: s + pd.Timedelta(days=14) for u in units[:10]}
    _, t_st = inject(ref_net, units, s, e, lift=0.02, lift_on="avg_check", unit_starts=late)
    assert t_st.n_treated_cells == 10 * 28 + 10 * 14


def test_inject_spillover_takes_from_zone_neighbours(full_net):
    r = full_net.restaurants
    zone_sizes = r.groupby("trade_zone").restaurant_id.transform("size")
    units = r.loc[zone_sizes >= 2].groupby("trade_zone").restaurant_id.first().tolist()[:15]
    assert units, "в синтетике должны быть торговые зоны с ≥ 2 ресторанами"
    s, e = _window(full_net, 200, 28)
    _, t = inject(full_net, units, s, e, lift=0.05, lift_on="avg_check", spillover=0.4)
    assert t.spillover_units and not set(t.spillover_units) & set(units)
    assert t.spillover_abs == pytest.approx(-0.4 * t.network_total_abs, rel=0.02)


def test_inject_validates_inputs(ref_net):
    s, e = _window(ref_net, 70, 14)
    with pytest.raises(ValueError):
        inject(ref_net, [], s, e, lift=0.01)
    with pytest.raises(ValueError):
        inject(ref_net, ["R9999"], s, e, lift=0.01)
    with pytest.raises(ValueError):
        inject(ref_net, ["R0001"], e, s, lift=0.01)
    with pytest.raises(ValueError):
        inject(ref_net, ["R0001"], s, e, lift=0.01, spillover=1.5)


def test_break_data_log(ref_net):
    s, _ = _window(ref_net, 100, 1)
    net2, log = break_data(ref_net, drop_frac=0.01, duplicate_frac=0.005, close_units=["R0001"], close_from=s)
    assert log["dropped"] > 0 and log["duplicated"] > 0 and log["closed_rows"] == 40
    assert len(net2.daily) == len(ref_net.daily) - log["closed_rows"] - log["dropped"] + log["duplicated"]
    assert net2.daily.duplicated(["restaurant_id", "date"]).any()


def test_save_load_roundtrip(tmp_path, ref_net):
    s, e = _window(ref_net, 70, 14)
    net2, _ = inject(ref_net, ["R0001", "R0002"], s, e, lift=0.02)
    save_network(net2, tmp_path / "net")
    back = load_network(tmp_path / "net")
    pd.testing.assert_frame_equal(back.daily, net2.daily)
    pd.testing.assert_frame_equal(back.restaurants, net2.restaurants, check_categorical=False)
    assert back.config == net2.config and back.truths == net2.truths
    with pytest.raises(FileNotFoundError):
        load_network(tmp_path / "missing")


def test_guest_effect_and_self_selection(full_net):
    g = full_net.guests(200_000, window="w1")
    treated = stream(5, "test").random(len(g)) < 0.5
    g2, t = apply_guest_effect(g, treated, lift=0.05, test_days=28, seed=5)
    assert t.per_restaurant_mean_rel == pytest.approx(0.05, abs=0.01)
    assert (g2.loc[~treated, "revenue"].to_numpy() == g.loc[~treated, "revenue"].to_numpy()).all()
    # Самоотбор: наивная разность «пользователи − остальные» сильно больше истинного ATT.
    g3, t3 = adopt_feature(g, base_rate=0.2, selection_strength=1.5, lift=0.05, test_days=28, seed=5)
    naive = g3.loc[g3.adopted == 1, "revenue"].mean() / g3.loc[g3.adopted == 0, "revenue"].mean() - 1
    assert naive > 5 * t3.per_restaurant_mean_rel


def test_kiosk_queue_interference_truth():
    s = generate_kiosk_sessions(1, ["R0001", "R0002"], "2025-01-06", 7, KioskParams(orders_per_day=200))
    treated = stream(1, "kiosk_test").random(len(s)) < 0.5
    s2, t = apply_kiosk_ui(s, treated, check_lift=0.03, extra_time_s=20, queue_conv_penalty=0.1, seed=1)
    assert t.params["direct_effect_rel"] == 0.03
    assert t.params["total_effect_if_all_b_rel"] < 0       # вред через очередь сильнее прироста чека
    assert (s2.loc[treated, "checkout_time_s"] - s.loc[treated, "checkout_time_s"]).round(9).eq(20).all()


# ---------------- SYN-4: фактические параметры = заданным (±10%) ----------------
def test_panel_components_match_parameters():
    p = NetworkConfig.preset("reference").panel
    c = simulate_log_components(2000, 400, p, stream(9, "syn4"))
    assert np.std(c["level"]) == pytest.approx(p.level_sd, rel=0.10)
    assert np.std(c["common"]) == pytest.approx(p.common_shock_sd, rel=0.10)
    u = c["ar"][:, 50:]
    ar_hat = np.mean(u[:, 1:] * u[:, :-1]) / np.mean(u[:, :-1] ** 2)
    assert ar_hat == pytest.approx(p.ar_coef, rel=0.10)
    assert np.std(np.diff(c["drift"], axis=1)) == pytest.approx(p.drift_sd, rel=0.10)


def test_panel_dow_and_level_dispersion(ref_net):
    d = ref_net.daily
    logrev = np.log(d["revenue"])
    by_dow = logrev.groupby(d["date"].dt.day_of_week).mean()
    expected = np.log(ref_net.config.panel.dow_profile)
    # Допуск ≈ 3 SE: общий шок дня (sd 0.05) усредняется лишь по 20 дням каждого дня недели.
    assert np.allclose(by_dow - by_dow.mean(), expected - expected.mean(), atol=0.035)
    level = np.log(d.groupby("restaurant_id")["revenue"].mean())
    assert level.std() == pytest.approx(ref_net.config.panel.level_sd, rel=0.10)


def test_guest_parameters_cv_and_correlation(full_net):
    g = full_net.guests(400_000, window="syn4")
    cv = g["revenue"].std() / g["revenue"].mean()
    rho = np.corrcoef(g["pre_revenue"], g["revenue"])[0, 1]
    assert cv == pytest.approx(2.5, rel=0.10)
    assert rho == pytest.approx(0.5, rel=0.10)


def test_kiosk_parameters():
    prm = KioskParams()
    s = generate_kiosk_sessions(2, [f"R{i:04d}" for i in range(60)], "2025-01-06", 7, prm)
    assert s["converted"].mean() == pytest.approx(prm.conversion, abs=0.01)
    conv = s[s.converted == 1]
    key = conv["restaurant_id"] + conv["date"].astype(str)
    within = conv["check"] / conv.groupby(key)["check"].transform("mean")
    assert within.std() == pytest.approx(prm.check_cv, rel=0.10)


# ---------------- SYN-7: скорость ----------------
@pytest.mark.slow
def test_full_network_generation_speed():
    import time

    t0 = time.perf_counter()
    net = generate_network(NetworkConfig.preset("pessimistic", seed=1))   # 1000 ресторанов × 730 дней
    ch = net.channels()
    elapsed = time.perf_counter() - t0
    assert net.restaurants.shape[0] == 1000 and len(ch) > 2_000_000
    assert elapsed <= 120, f"{elapsed:.1f} c"
