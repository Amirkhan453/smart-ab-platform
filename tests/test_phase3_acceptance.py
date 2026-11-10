"""
Фаза 3, П6: приёмка контура управления.

- TGT-2: A/A с таргетингом боевым хэшем и боевым анализом — FPR в допуске, SRM нет.
- TGT-3: симуляция «вариант B ускоряет обновление приложения»: отбор по текущему признаку даёт SRM и смещение,
  состав по первому показу — нет.
- TGT-5: трафик калькулятора с учётом таргетинга; SRM по сегментам находит сбой в одном сегменте.
- LNK-2: смена статуса доходит до SDK ≤ 60 с на uvicorn при опросе раз в 30 с.
- NFR-B1: замеры на ruleset из 10 000 экспериментов.
"""
import math
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import stats

from ab_platform.engine.assignment import DeterministicAssignmentEngine as ASG
from ab_platform.engine.exposure_analysis import NO_VALUE, analysis_population, srm_by_segment
from ab_platform.engine.power import PowerAndSampleSizeEngine as POWER
from ab_platform.engine.ruleset_engine import evaluate, evaluate_all
from ab_platform.engine.safeguards import ExperimentSafeguardEngine as SG
from ab_platform.orchestrator import SmartABPlatformOrchestrator as ORCH
from ab_platform.schemas.passport import ExperimentPassport, MetricDefinition, MetricType
from ab_platform.schemas.ruleset import ExperimentRule, Layer, Ruleset
from ab_platform.schemas.targeting import Targeting
from ab_platform.sdk import ExperimentClient
from ab_platform.sdk_http import HttpExposureSink, HttpRulesetFetcher
from ab_platform.server import ExposureIn, ExposureStore
from ab_platform.simulation.vit import NetworkConfig, generate_network
from tests.test_phase1_passport_v4 import base
from tests.test_phase2_http import Stand
from tests.test_phase3_http import KEYS, h, make_app, running

REVENUE = MetricDefinition(name="revenue", metric_type=MetricType.CONTINUOUS, numerator_col="revenue",
                           winsorize_percentile=None)
V2 = [{"variant_id": "control", "traffic_percentage": 50.0, "is_control": True},
      {"variant_id": "treatment", "traffic_percentage": 50.0}]
TARGETING = Targeting(platforms=["ios", "android"], app_version_min="5.2", has_delivery=True)


# ---------------- TGT-2: A/A с таргетингом ----------------
@pytest.fixture(scope="module")
def targeted_guests():
    """
    Гости генератора с признаками приложения. Признаки связаны с выручкой (у активных гостей чаще iOS и свежая
    версия): таргетинг отбирает неслучайную часть базы, и тест проверяет, что это не ломает A/A внутри неё.
    """
    net = generate_network(NetworkConfig.preset("reference", n_restaurants=20, days=14, seed=3))
    g = net.guests(12000, window="tgt")
    rng = np.random.default_rng(31)
    active = (g["pre_revenue"].rank(pct=True)).to_numpy()
    g["platform"] = np.where(rng.random(len(g)) < 0.25 + 0.4 * active, "ios",
                             np.where(rng.random(len(g)) < 0.85, "android", "web"))
    g["app_version"] = np.where(rng.random(len(g)) < 0.3 + 0.5 * active, rng.choice(["5.2", "5.4.1", "5.10", "6.0"], len(g)),
                                rng.choice(["4.9", "5.0", "5.1.9"], len(g)))
    g["has_delivery"] = rng.random(len(g)) < 0.7
    attrs = g[["platform", "app_version", "has_delivery"]].to_dict("records")
    g["has_delivery"] = g["has_delivery"].astype(bool)
    g["eligible"] = [TARGETING.mismatch({**a, "has_delivery": bool(a["has_delivery"])}) is None for a in attrs]
    g["guest_id"] = g["guest_id"].astype(str)
    return g


def run_targeted_aa(guests, n_runs, seed):
    """A/A среди подходящих под таргетинг: вариант — боевым хэшем (шаг 2), анализ — боевой функцией с CUPED."""
    eligible = guests[guests["eligible"]].reset_index(drop=True)
    ids = eligible["guest_id"].tolist()
    units = eligible[["guest_id", "revenue", "pre_revenue", "platform"]]
    p_values, srm_all, srm_seg = [], [], []
    for run in range(n_runs):
        is_control = ASG.control_from_bucket(ASG.buckets_for(f"tgt_aa:{seed}:{run}", ids), 50.0)
        frame = units.assign(variant=np.where(is_control, "control", "treatment"))
        res, _ = ORCH._evaluate_pair_metric(frame, REVENUE, "control", "treatment", variant_col="variant",
                                            covariate_col="pre_revenue", alpha=0.05)
        p_values.append(res["p_value"])
        check = srm_by_segment(frame.rename(columns={"platform": "attr_platform"}),
                               [("control", 50.0), ("treatment", 50.0)], "attr_platform")
        srm_all.append(check["overall"]["srm_detected"])
        srm_seg.append(any(s["srm_detected"] for s in check["segments"].values()))
    return np.asarray(p_values), np.asarray(srm_all), np.asarray(srm_seg), len(ids)


def test_tgt2_targeting_selects_nonrandom_part_and_engine_agrees(targeted_guests):
    g = targeted_guests
    share = g["eligible"].mean()
    assert 0.15 < share < 0.6
    # таргетинг действительно смещён по выручке — иначе тест ничего бы не проверял
    assert g.loc[g["eligible"], "pre_revenue"].mean() > 1.15 * g.loc[~g["eligible"], "pre_revenue"].mean()
    # векторный путь симуляции совпадает с движком: те же гости в тесте и те же варианты
    rs = Ruleset(version=1, holdout={"pct": 0.0}, layers=[Layer(layer_id="L")], experiments=[
        ExperimentRule(experiment_id="tgt_aa:1:0", layer_id="L", variants=V2, targeting=TARGETING)])
    sample = g.head(1500)
    control = ASG.control_from_bucket(ASG.buckets_for("tgt_aa:1:0", sample["guest_id"]), 50.0)
    for (_, row), is_control in zip(sample.iterrows(), control):
        attrs = {"platform": row["platform"], "app_version": row["app_version"], "has_delivery": bool(row["has_delivery"])}
        e = evaluate(rs, rs.experiments[0], row["guest_id"], attrs)
        assert e.is_exposed == bool(row["eligible"])
        if e.is_exposed:
            assert e.variant_id == ("control" if is_control else "treatment")


def test_tgt2_targeted_aa_fast(targeted_guests):
    p, srm_all, srm_seg, n = run_targeted_aa(targeted_guests, n_runs=300, seed=1)
    # порог SRM 0,001: на 300 честных прогонов ждём 0,3 тревоги; три и больше — вероятность 0,4%
    assert n > 2000 and 0.02 <= np.mean(p < 0.05) <= 0.09 and srm_all.sum() <= 2 and srm_seg.sum() <= 2


@pytest.mark.slow
def test_tgt2_targeted_aa_10000_runs(targeted_guests):
    """TGT-2: 10 000 прогонов. FPR ∈ [4,5%; 5,5%], p-value равномерны, SRM — на уровне своего порога 0,1%."""
    p, srm_all, srm_seg, n = run_targeted_aa(targeted_guests, n_runs=10_000, seed=11)
    fpr, ks = float(np.mean(p < 0.05)), stats.kstest(p, "uniform").pvalue
    print(f"\nTGT-2: гостей в тесте {n}, FPR {fpr:.4f}, KS p {ks:.3f}, SRM общий {int(srm_all.sum())}, "
          f"по сегментам {int(srm_seg.sum())} из {len(p)}")
    assert 0.045 <= fpr <= 0.055, fpr
    assert ks > 0.01, ks
    # порог SRM 0,001: ожидается ≈ 10 ложных тревог на 10 000; P(> 25) < 10⁻⁴
    assert srm_all.sum() <= 25 and srm_seg.sum() <= 25


# ---------------- TGT-3: вариант влияет на признак таргетинга ----------------
def simulate_update_nudge(n_guests, days, seed, update_boost=3.0):
    """
    Тест только для версий приложения ≤ 5.9. Вариант treatment после первого показа втрое ускоряет обновление
    до 6.0; на выручку он не влияет (истинный эффект — ноль). Активные гости чаще заходят и чаще обновляются.
    Возвращает по гостям: вариант, был ли показ, выручку после первого показа и версию на конец теста.
    """
    rng = np.random.default_rng(seed)
    ids = [f"u{seed}_{i}" for i in range(n_guests)]
    treatment = ~ASG.control_from_bucket(ASG.buckets_for(f"nudge:{seed}", ids), 50.0)
    activity = rng.gamma(2.0, 0.5, n_guests)                      # склонность заходить и заказывать
    p_visit = 1.0 - np.exp(-0.25 * activity)
    hazard = np.clip(0.02 * activity, 0.0, 0.5)                   # обновление в день без подталкивания
    old_version = np.ones(n_guests, dtype=bool)                   # на старте у всех ≤ 5.9
    exposed = np.zeros(n_guests, dtype=bool)
    revenue = np.zeros(n_guests)
    for _ in range(days):
        visit = rng.random(n_guests) < p_visit
        # показ — только пока гость подходит под таргетинг (или уже закреплён первым показом)
        exposed |= visit & old_version
        revenue += np.where(visit & exposed, rng.gamma(2.0, 200.0, n_guests) * (0.5 + activity), 0.0)
        h = np.where(treatment & exposed, np.clip(hazard * update_boost, 0.0, 0.9), hazard)
        old_version &= ~(rng.random(n_guests) < h)
    return pd.DataFrame({"guest_id": ids, "variant": np.where(treatment, "treatment", "control"),
                         "exposed": exposed, "revenue": revenue, "still_old_version": old_version})


def analyze_nudge(df, rule):
    """rule='first_exposure' — состав по первому показу (TGT-3); 'current_attribute' — отбор по версии на конец теста."""
    frame = df[df["exposed"]] if rule == "first_exposure" else df[df["exposed"] & df["still_old_version"]]
    n_c = int((frame["variant"] == "control").sum())
    srm = SG.detect_srm_chi2(n_c, len(frame) - n_c, 0.5, alpha=0.001)["srm_detected"]
    res, _ = ORCH._evaluate_pair_metric(frame, REVENUE, "control", "treatment", variant_col="variant", alpha=0.05)
    return srm, res["p_value"], res["relative_diff"]


def run_nudge(n_sims, n_guests=6000, days=28):
    out = {"first_exposure": [], "current_attribute": []}
    for seed in range(n_sims):
        df = simulate_update_nudge(n_guests, days, seed)
        for rule in out:
            out[rule].append(analyze_nudge(df, rule))
    return {rule: (np.mean([r[0] for r in rows]), np.mean([r[1] < 0.05 for r in rows]),
                   float(np.mean([r[2] for r in rows]))) for rule, rows in out.items()}


def test_tgt3_first_exposure_rule_fast():
    res = run_nudge(40)
    srm_bad, fpr_bad, bias_bad = res["current_attribute"]
    srm_ok, fpr_ok, bias_ok = res["first_exposure"]
    assert srm_bad > 0.9 and fpr_bad > 0.5 and bias_bad < -0.03  # без правила: перекос и ложный «вред»
    assert srm_ok == 0 and fpr_ok <= 0.15 and abs(bias_ok) < 0.01


@pytest.mark.slow
def test_tgt3_first_exposure_rule_1000_sims():
    """TGT-3: истинный эффект — ноль. Без правила — SRM и смещение, с правилом — A/A в допуске."""
    res = run_nudge(1000)
    srm_bad, fpr_bad, bias_bad = res["current_attribute"]
    srm_ok, fpr_ok, bias_ok = res["first_exposure"]
    print(f"\nTGT-3 без правила: SRM {srm_bad:.1%}, ложных срабатываний {fpr_bad:.1%}, смещение {bias_bad:+.2%}"
          f"\nTGT-3 с правилом:  SRM {srm_ok:.1%}, ложных срабатываний {fpr_ok:.1%}, смещение {bias_ok:+.2%}")
    assert srm_bad > 0.95 and fpr_bad > 0.5 and bias_bad < -0.03
    assert srm_ok <= 0.005 and 0.035 <= fpr_ok <= 0.065 and abs(bias_ok) < 0.003


def test_tgt3_mechanism_through_sdk_and_exposure_store():
    """Тот же сценарий на настоящих SDK и хранилище показов: обновившийся гость остаётся в составе анализа."""
    rs = Ruleset(version=1, holdout={"pct": 0.0}, layers=[Layer(layer_id="L")], experiments=[
        ExperimentRule(experiment_id="nudge", layer_id="L", variants=V2, targeting={"app_version_max": "5.9"})])
    store = ExposureStore()
    sdk = ExperimentClient(initial_ruleset=rs, exposure_sink=lambda batch: store.add([ExposureIn(**e) for e in batch]))
    guests = [f"g{i}" for i in range(400)]
    version = {g: "5.4" for g in guests}
    variant_day1 = {}
    for g in guests[:300]:  # день 1: 300 гостей видят экран
        variant_day1[g] = sdk.get_variant("nudge", g, session_id="d1", attributes={"app_version": version[g]}).variant_id
    for g in guests:        # treatment обновляется весь, control и не видевшие экран — частично
        if variant_day1.get(g) == "treatment" or hash(g) % 5 == 0:
            version[g] = "6.0"
    day2 = {g: sdk.get_variant("nudge", g, session_id="d2", attributes={"app_version": version[g]}) for g in guests}
    sdk.flush()
    # закреплённые первым показом остаются в тесте с тем же вариантом; обновившиеся до показа — вне теста
    for g in guests[:300]:
        assert day2[g].in_experiment and day2[g].variant_id == variant_day1[g]
    for g in guests[300:]:
        assert day2[g].in_experiment == (version[g] == "5.4")
    pop = analysis_population(store.first_exposures("nudge"))
    first300 = pop[pop["unit_id"].isin(guests[:300])]
    assert len(first300) == 300 and set(first300["attr_app_version"]) == {"5.4"}  # снимок на момент первого показа
    by_first = srm_by_segment(first300, [("control", 50.0), ("treatment", 50.0)])
    assert not by_first["srm_detected"]
    # отбор по текущей версии выкинул бы весь treatment первого дня
    current = first300[first300["unit_id"].map(version) == "5.4"]
    assert set(current["variant"]) == {"control"}
    assert srm_by_segment(current, [("control", 50.0), ("treatment", 50.0)])["srm_detected"]


# ---------------- TGT-5: трафик и SRM по сегментам ----------------
def test_tgt5_power_uses_targeted_traffic():
    full = ExperimentPassport(**base(daily_eligible_units=50_000))
    part = ExperimentPassport(**base(daily_eligible_units=50_000, targeting={"platforms": ["ios"]},
                                     targeting_audience_share=0.3))
    assert (full.target_daily_units, part.target_daily_units) == (50_000, 15_000)

    def plan(p):
        return POWER.calculate_sample_size(baseline_mean=0.15, mde_relative=0.03, is_binary=True,
                                           daily_traffic=p.target_daily_units)
    # размер выборки от таргетинга не зависит, длительность — зависит: считается по целевой аудитории
    assert plan(part).total_sample_size == plan(full).total_sample_size
    for p in (full, part):
        assert plan(p).estimated_days == math.ceil(plan(p).total_sample_size / p.target_daily_units / 7) * 7
    assert plan(part).estimated_days > plan(full).estimated_days


def test_tgt5_srm_by_segment_finds_failure_in_one_segment():
    rng = np.random.default_rng(5)
    n = 60_000
    platform = rng.choice(["ios", "android", "web"], n, p=[0.45, 0.45, 0.10])
    variant = np.where(rng.random(n) < 0.5, "control", "treatment")
    pop = pd.DataFrame({"unit_id": [f"u{i}" for i in range(n)], "variant": variant, "attr_platform": platform})
    spec = [("control", 50.0), ("treatment", 50.0)]
    healthy = srm_by_segment(pop, spec, "attr_platform")
    assert not healthy["srm_detected"] and set(healthy["segments"]) == {"android", "ios", "web"}
    assert healthy["segment_alpha"] == pytest.approx(0.001 / 3)
    assert sum(s["n"] for s in healthy["segments"].values()) == healthy["overall"]["n"] == n
    # сбой: на web (10% трафика) 12% показов treatment не доходит. Общий SRM этого не видит, сегментный — видит
    lost = (pop["attr_platform"] == "web") & (pop["variant"] == "treatment") & (rng.random(n) < 0.12)
    broken = srm_by_segment(pop[~lost], spec, "attr_platform")
    assert not broken["overall"]["srm_detected"] and broken["srm_detected"]
    assert [name for name, s in broken["segments"].items() if s["srm_detected"]] == ["web"]
    assert not srm_by_segment(pop[~lost], spec, "attr_platform", alpha=1e-12)["srm_detected"]  # порог — из аргумента
    # три варианта, пустой сегмент признака, неизвестный вариант, нет колонки
    three = pd.DataFrame({"unit_id": ["a", "b", "c"], "variant": ["a", "b", "c"], "attr_platform": [NO_VALUE] * 3})
    res = srm_by_segment(three, [("a", 100 / 3), ("b", 100 / 3), ("c", 100 / 3)], "attr_platform")
    assert res["segments"][NO_VALUE]["counts"] == {"a": 1, "b": 1, "c": 1} and not res["srm_detected"]
    with pytest.raises(ValueError, match="вне Паспорта"):
        srm_by_segment(three, spec)
    with pytest.raises(ValueError, match="Нет колонки"):
        srm_by_segment(pop, spec, "attr_city")
    assert srm_by_segment(pop, spec)["segments"] == {} and len(analysis_population({})) == 0


# ---------------- LNK-2: статус доходит до SDK ----------------
def measure_status_propagation(n_clients, interval_s, timeout_s):
    """Пауза и продолжение по HTTP → время, за которое каждый SDK отдаёт kill-switch и затем прежний вариант."""
    app, _ = make_app()
    admin = TestClient(app)
    running(admin, ramp_plan=[100])
    ios = {"platform": "ios", "app_version": "5.4"}
    with Stand(app=app) as st:
        clients = []
        for _ in range(n_clients):
            c = ExperimentClient(HttpRulesetFetcher(st.url, api_key=KEYS["sdk"]),
                                 HttpExposureSink(st.url, api_key=KEYS["sdk"]), refresh_interval_s=interval_s,
                                 max_stale_s=max(300.0, interval_s))
            assert c.refresh() is True
            c.start()
            clients.append(c)
            time.sleep(interval_s / n_clients)  # разные фазы опроса
        unit = next(u for u in (f"g{i}" for i in range(100))
                    if clients[0].get_variant("flag_exp_menu", u, attributes=ios, log_exposure=False).in_experiment)
        before = clients[0].get_variant("flag_exp_menu", unit, attributes=ios, log_exposure=False).variant_id

        def wait(command, done):
            t0 = time.monotonic()
            assert admin.post(f"/passports/exp_menu/{command}", headers=h("root")).status_code == 200
            reached = {}
            while len(reached) < n_clients and time.monotonic() - t0 < timeout_s:
                for k, c in enumerate(clients):
                    if k not in reached and done(c.get_variant("flag_exp_menu", unit, attributes=ios, log_exposure=False)):
                        reached[k] = time.monotonic() - t0
                time.sleep(0.01)
            return reached
        paused = wait("pause", lambda d: d.reason == "kill_switch")
        resumed = wait("resume", lambda d: d.in_experiment and d.variant_id == before)
        for c in clients:
            c.close()
    return paused, resumed


def test_lnk2_status_reaches_sdk_fast():
    paused, resumed = measure_status_propagation(n_clients=4, interval_s=0.3, timeout_s=5)
    assert len(paused) == len(resumed) == 4 and max([*paused.values(), *resumed.values()]) < 1.3


@pytest.mark.slow
def test_lnk2_status_reaches_sdk_within_60s_default_interval():
    """LNK-2: пауза и продолжение через HTTP доходят до 10 SDK с опросом раз в 30 с не дольше чем за 60 с."""
    paused, resumed = measure_status_propagation(n_clients=10, interval_s=30.0, timeout_s=70)
    print(f"\nLNK-2: пауза дошла за {min(paused.values()):.1f}–{max(paused.values()):.1f} с, "
          f"продолжение — за {min(resumed.values()):.1f}–{max(resumed.values()):.1f} с ({len(paused)} SDK)")
    assert len(paused) == len(resumed) == 10
    assert max(paused.values()) <= 60 and max(resumed.values()) <= 60


# ---------------- NFR-B1: 10 000 экспериментов ----------------
def big_ruleset(n):
    layers = [Layer(layer_id=f"L{i}") for i in range(n // 10)]
    rules = [ExperimentRule(experiment_id=f"e{i}", layer_id=f"L{i // 10}", slot_start=(i % 10) * 1000,
                            slot_end=(i % 10 + 1) * 1000, variants=V2,
                            targeting={"platforms": ["ios"], "app_version_min": "5.2"}) for i in range(n)]
    return Ruleset(version=1, layers=layers, experiments=rules)


def p99(samples):
    return float(np.percentile(np.asarray(samples), 99))


@pytest.mark.slow
def test_nfr_b1_latency_on_10000_experiments():
    """NFR-B1: локальное решение по одному флагу — p99 ≤ 1 мс на ruleset из 10 000 экспериментов с таргетингом."""
    ios = {"platform": "ios", "app_version": "5.4"}
    lines = []
    for n in (100, 10_000):
        t = time.perf_counter()
        rs = big_ruleset(n)
        build = time.perf_counter() - t
        text = rs.to_json()
        t = time.perf_counter()
        Ruleset.from_json(text)
        parse = time.perf_counter() - t
        sdk = ExperimentClient(initial_ruleset=rs)
        one = []
        for i in range(50_000):
            t = time.perf_counter()
            sdk.get_variant(f"e{i % n}", f"g{i}", attributes=ios, log_exposure=False)
            one.append(time.perf_counter() - t)
        everything = []
        for i in range(200 if n == 100 else 30):
            t = time.perf_counter()
            evaluate_all(rs, f"g{i}", ios, "guest_id")
            everything.append(time.perf_counter() - t)
        lines.append(f"{n} экспериментов: ruleset {len(text) / 1e6:.2f} МБ, сборка {build:.2f} с, разбор {parse:.2f} с; "
                     f"один флаг p50 {np.median(one) * 1e6:.1f} мкс, p99 {p99(one) * 1e6:.1f} мкс; "
                     f"все флаги гостя — медиана {np.median(everything) * 1e3:.2f} мс")
        assert p99(one) <= 1e-3, (n, p99(one))
    print("\nNFR-B1: " + "\nNFR-B1: ".join(lines))


@pytest.mark.slow
def test_decide_throughput_on_uvicorn_100_experiments():
    """Замер POST /decide на 100 активных тестах: один поток запросов, один процесс сервера. Число — в отчёт."""
    import httpx

    from ab_platform.auth import Auth
    from ab_platform.server import RulesetStore, create_app
    app = create_app(RulesetStore(big_ruleset(100)), ExposureStore(), auth=Auth.disabled())
    with Stand(app=app) as st, httpx.Client(base_url=st.url) as http:
        body = {"unit_type": "guest_id", "attributes": {"platform": "ios", "app_version": "5.4"}}
        for i in range(200):
            http.post("/decide", json={**body, "unit_id": f"w{i}"})
        lat = []
        t0 = time.perf_counter()
        for i in range(3000):
            t = time.perf_counter()
            r = http.post("/decide", json={**body, "unit_id": f"g{i}"})
            lat.append(time.perf_counter() - t)
        total = time.perf_counter() - t0
        assert r.status_code == 200 and len(r.json()["decisions"]) == 100
    print(f"\n/decide, 100 тестов: {3000 / total:.0f} ответов/с в один поток, p50 {np.median(lat) * 1e3:.2f} мс, "
          f"p99 {p99(lat) * 1e3:.2f} мс")
    assert p99(lat) < 0.2
