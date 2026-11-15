"""
Фаза 4, П6: приёмка вехи M2.

Быстрый тест — сквозной сценарий через форму и HTTP: форма Паспорта → рецензия и утверждение → предстартовая
проверка на истории → запуск → /decide → показы, заказы, корректировки и события приложения по HTTP →
ежедневный мониторинг → остановка → итоговый отчёт → повтор байт в байт.

Медленные (pytest -m slow) — симуляции всего пути (назначение настоящим движком → показы → события → метрики →
анализ): A/A на 300 тестах (FPR ∈ [2%; 9%], SRM нет), мощность при эффекте 2 × MDE ≥ 95%,
ложные автопаузы на A/A (1000 тестов × 28 ежедневных проверок) ≤ 5%. Тесты гоняются параллельно по ядрам.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import stats

from ab_platform.analysis import run_due_analyses
from ab_platform.auth import Auth
from ab_platform.control import ExperimentService
from ab_platform.events import MemoryEventStore
from ab_platform.server import create_app
from ab_platform.simulation.vit import simulate_app_world
from ab_platform.storage import MemoryStorage
from ab_platform.ui_api import PlatformApi
from tests.test_phase3_control import OWNER
from tests.test_phase3_http import KEYS, SPEC, h
from tests.test_phase4_analysis import START, Lab
from tests.test_phase4_events import H_INGEST, INGEST
from tests.test_phase4_form import page

WORKERS = max(1, (os.cpu_count() or 2) - 1)


# ---------------- M2: сквозной сценарий через форму и HTTP ----------------
def post_all(http, path, rows, headers, size=5000):
    total = {"accepted": 0, "duplicates": 0, "conflicts": [], "rejected": []}
    for k in range(0, len(rows), size):
        r = http.post(path, json=rows[k:k + size], headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        for key in total:
            total[key] = total[key] + body.get(key, 0 if key in ("accepted", "duplicates") else [])
    return total


def test_m2_end_to_end_through_form_and_http():
    world = simulate_app_world(21, 3000, START)
    now = [world.test_start - timedelta(hours=1)]
    service = ExperimentService(MemoryStorage(), clock=lambda: now[0], require_preflight=True)
    service.ensure_default_metrics()
    events = MemoryEventStore()
    http = TestClient(create_app(exposure_store=events, service=service, auth=Auth.parse(f"{SPEC},{INGEST}")))
    api = lambda who: PlatformApi(key=KEYS[who], client=http)  # noqa: E731
    assert http.post("/layers", json={"layer_id": "ui"}, headers=h("root")).status_code == 200

    # история до старта (пре-период) — для предстартовой проверки и ковариаты CUPED
    cf = world.realize()
    before = lambda rows, col: [r for r in rows if r[col] < world.test_start.isoformat()]  # noqa: E731
    hist = post_all(http, "/orders", before(cf.payload("orders"), "order_time"), H_INGEST)
    assert hist["accepted"] > 1000 and not hist["conflicts"] and not hist["rejected"]

    # форма: владелец создаёт и отправляет, аналитик утверждает, владелец проверяет и запускает
    at = page(api("ivanova"))
    at.text_input(key="experiment_id_new").set_value("exp_m2")
    at.text_input(key="test_key_new").set_value("flag_m2")
    at.text_area(key="hypothesis_new").set_value("Если мы покажем новое меню, выручка на гостя вырастет на 3%")
    at.number_input(key="daily_new").set_value(3000)
    at.text_input(key="ramp_new").set_value("100")
    at.button(key="save").click().run()
    at.button(key="cmd_submit").click().run()
    analyst = page(api("petrov"))
    analyst.selectbox(key="choice").set_value("exp_m2").run()
    analyst.button(key="cmd_approve").click().run()
    at = page(api("ivanova"))
    at.selectbox(key="choice").set_value("exp_m2").run()
    at.button(key="cmd_start").click().run()  # без предстартовой проверки запуск отклоняется
    assert service.get("exp_m2").status.value == "approved" and any("409" in e.value for e in at.error)
    at.button(key="cmd_preflight").click().run()
    assert any("Предстартовая проверка пройдена" in s.value for s in at.success), [e.value for e in at.error]
    now[0] = world.test_start
    at.button(key="cmd_start").click().run()
    assert not at.error and service.get("exp_m2").status.value == "running"

    # приложение: решение через /decide при заходе на экран, показ — через /exposures (ключ SDK)
    ex = world.exposure_times()
    g = world.guests.set_index("unit_id")
    exposures, variant = [], {}
    for i, (u, t) in enumerate(ex.items()):
        if pd.isna(t):
            continue
        d = http.post("/decide", json={"unit_id": u, "unit_type": "guest_id", "flags": ["flag_m2"],
                                       "attributes": {"platform": g.at[u, "platform"]}}, headers=h("sdk")).json()
        (dec,) = d["decisions"]
        variant[u] = dec["variant_id"]
        if dec["in_experiment"]:
            exposures.append({"event_id": f"x{i}", "event_time": t.isoformat(), "unit_id": u, "unit_type": "guest_id",
                              "experiment_id": dec["experiment_id"], "variant_id": dec["variant_id"],
                              "ruleset_version": d["ruleset_version"], "attributes": {"platform": g.at[u, "platform"]}})
    assert post_all(http, "/exposures", exposures, h("sdk"))["accepted"] == len(exposures) > 1500
    treated = np.array([variant.get(u) == "treatment" for u in ex.index])
    ev = world.realize(treated, lift=0.30)
    orders = post_all(http, "/orders", ev.payload("orders"), H_INGEST)
    assert not orders["conflicts"] and orders["duplicates"] == hist["accepted"]  # история — те же заказы
    assert not post_all(http, "/order-adjustments", ev.payload("adjustments"), H_INGEST)["conflicts"]
    assert not post_all(http, "/app-events", ev.payload("app_events"), h("sdk"))["conflicts"]
    assert http.get("/events/health", headers=h("sdk")).json()["orphan_adjustments"] == 0

    # ежедневный мониторинг (cron), остановка, итог после закрытия окон
    now[0] = world.test_start + timedelta(days=10)
    mon = http.post("/scheduler/run", headers=h("root")).json()
    assert [m["decision"] for m in mon] == ["RUNNING_BLINDED"]
    now[0] = world.test_start + timedelta(days=28)
    assert http.post("/passports/exp_m2/stop", headers=h("ivanova")).json()["status"] == "stopped"
    now[0] = world.test_start + timedelta(days=36)
    (fin,) = http.post("/scheduler/run", headers=h("root")).json()
    assert fin["kind"] == "final"
    run = http.get(f"/analysis-runs/{fin['run_id']}", headers=h("guest")).json()["result"]
    c = run["comparisons"]["treatment"]
    lo, hi = c["ci_lower"] / c["mean_control"], c["ci_upper"] / c["mean_control"]
    assert c["is_significant"] and lo < ev.truth.network_total_rel < hi, (lo, ev.truth.network_total_rel, hi)
    assert run["can_trust_data"] and {m["ref"] for m in run["metrics"]} >= {"revenue_per_guest@1", "crash_share@1"}
    md = http.get(f"/analysis-runs/{fin['run_id']}/report.md", headers=h("guest")).text
    assert "итоговый" in md and "revenue_per_guest@1" in md and fin["run_id"] in md
    assert http.post(f"/analysis-runs/{fin['run_id']}/reproduce", headers=h("petrov")).json()["identical"]
    # форма показывает последний отчёт и тревогу «итог готов»
    at = page(api("ivanova"))
    at.selectbox(key="choice").set_value("exp_m2").run()
    assert any(fin["run_id"] in m.value for m in at.markdown)
    assert any("final_report_ready" in w.value for w in at.warning)
    audit = http.get("/audit/verify", headers=h("root")).json()
    assert audit["records"] > 10 and audit["ruleset_problems"] == []


# ---------------- симуляции (pytest -m slow) ----------------
def one_test(seed, n=3000, lift=0.0, daily_checks=False):
    """Один тест всего пути; возвращает итог. Функция модуля — чтобы её можно было отдать в другой процесс."""
    world = simulate_app_world(seed, n, START)
    lab = Lab(world, lift=lift)
    lab.start()
    paused_on, srm_alerts = None, 0
    if daily_checks:
        for day in range(1, 29):
            lab.at(day)
            run_due_analyses(lab.svc, lab.events)
            if lab.svc.get("exp_menu").status.value == "paused":
                paused_on = day
                break
        srm_alerts = sum(a.kind == "srm" for a in lab.svc.storage.list_alerts("exp_menu"))
        return {"seed": seed, "paused_on": paused_on, "srm_alerts": srm_alerts}
    lab.at(28)
    lab.svc.stop(OWNER, "exp_menu")
    lab.at(28 + 7 + 1)
    (fin,) = run_due_analyses(lab.svc, lab.events)
    r = lab.svc.storage.get_analysis_run(fin["run_id"]).result
    c = r["comparisons"]["treatment"]
    return {"seed": seed, "p": c["p_value"], "significant": bool(c["is_significant"]), "decision": r["decision"],
            "srm": bool(r["srm_detected"]), "se_rel": (c["ci_upper"] - c["ci_lower"]) / (2 * 1.96) / c["mean_control"],
            "rel": c["relative_diff"], "truth": lab.truth.network_total_rel}


def run_many(seeds, **kw):
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(_one_kw, [(s, kw) for s in seeds]))


def _one_kw(args):
    seed, kw = args
    return one_test(seed, **kw)


@pytest.fixture(scope="module")
def aa_runs():
    return run_many(range(1000, 1300))


@pytest.mark.slow
def test_aa_full_path_fpr_and_no_srm(aa_runs):
    sig = sum(r["significant"] for r in aa_runs)
    srm = sum(r["srm"] for r in aa_runs)
    fpr = sig / len(aa_runs)
    ks = stats.kstest([r["p"] for r in aa_runs], "uniform").pvalue
    print(f"\nA/A всего пути: {len(aa_runs)} тестов, FPR {fpr:.3f} ({sig}), SRM {srm}, KS p {ks:.3f}")
    assert 0.02 <= fpr <= 0.09 and srm <= 0.01 * len(aa_runs)


@pytest.mark.slow
def test_power_at_twice_mde(aa_runs):
    """MDE — по стандартной ошибке относительного эффекта на A/A (тот же путь и размер): (z_{1-α/2} + z_0.8) · SE."""
    se = float(np.median([r["se_rel"] for r in aa_runs]))
    mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se
    runs = run_many(range(2000, 2100), lift=2 * mde)
    power = np.mean([r["significant"] and r["rel"] > 0 for r in runs])
    bias = np.mean([r["rel"] - r["truth"] for r in runs])
    print(f"\nМощность: MDE {mde:.3f}, эффект {2 * mde:.3f}, {len(runs)} тестов, мощность {power:.3f}, "
          f"смещение оценки {bias:+.4f}")
    assert power >= 0.95


@pytest.mark.slow
def test_false_auto_pause_on_aa_daily_checks():
    runs = run_many(range(5000, 6000), n=2000, daily_checks=True)
    paused = sum(r["paused_on"] is not None for r in runs)
    with_srm = sum(r["srm_alerts"] > 0 for r in runs)
    print(f"\nАвтопауза на A/A: {len(runs)} тестов × 28 проверок, ложных пауз {paused} ({paused / len(runs):.3f}), "
          f"тестов с тревогой SRM {with_srm} ({with_srm / len(runs):.3f})")
    assert paused / len(runs) <= 0.05
