"""
Фаза 2, П6: приёмка гостевого домена целиком.

A/A end-to-end: генератор гостей → ruleset на сервере → Python SDK (решение, exposure) → HTTP-приём →
анализ по тому, что сохранил сервер, той же функцией, что и в оркестраторе. На каждый прогон — новый эксперимент
(новая соль) и новая версия ruleset, которую SDK забирает по HTTP.

Критерий (дизайн-заметка, раздел 5): FPR ∈ [2%; 9%] на 300 прогонах, SRM не срабатывает. SRM проверяется χ²
с α = 0,001; при 300 честных прогонах одно случайное срабатывание имеет вероятность 26%, поэтому допускается
не больше 3 (P(≥ 4) ≈ 3·10⁻⁴), фактическое число выводится.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ab_platform.engine.assignment import DeterministicAssignmentEngine as ASG
from ab_platform.engine.ruleset_engine import evaluate
from ab_platform.engine.safeguards import ExperimentSafeguardEngine as SG
from ab_platform.orchestrator import SmartABPlatformOrchestrator as ORCH
from ab_platform.schemas.passport import MetricDefinition, MetricType
from ab_platform.schemas.ruleset import ExperimentRule, Layer, Ruleset
from ab_platform.sdk import ExperimentClient
from ab_platform.sdk_http import HttpExposureSink, HttpRulesetFetcher
from ab_platform.simulation.vit import NetworkConfig, generate_network
from tests.test_phase2_http import Stand

REVENUE = MetricDefinition(name="revenue", metric_type=MetricType.CONTINUOUS, numerator_col="revenue",
                           winsorize_percentile=None)


@pytest.fixture(scope="module")
def guests():
    net = generate_network(NetworkConfig.preset("reference", n_restaurants=20, days=14, seed=2))
    return net.guests(4000, window="aa")


def ruleset_for_run(version, run):
    return Ruleset(
        version=version,
        layers=[Layer(layer_id="promo"), Layer(layer_id="ops", marketing=False)],
        experiments=[ExperimentRule(
            experiment_id=f"aa_{run}", layer_id="promo",
            variants=[{"variant_id": "control", "traffic_percentage": 50.0, "is_control": True},
                      {"variant_id": "treatment", "traffic_percentage": 50.0}],
        )],
    )


def run_e2e_aa(guests, n_runs):
    ids = guests["guest_id"].astype(str).tolist()
    units = guests[["guest_id", "revenue", "pre_revenue"]].assign(guest_id=lambda d: d["guest_id"].astype(str))
    p_values, srm, rows = [], [], []
    with Stand(ruleset_for_run(1, 0)) as st:
        sink = HttpExposureSink(st.url)
        client = ExperimentClient(HttpRulesetFetcher(st.url), sink, batch_size=500)
        for run in range(n_runs):
            if run > 0:
                st.store.publish(ruleset_for_run(run + 1, run))
            assert client.refresh() is True or run == 0
            assert client.ruleset.version == run + 1
            flag = f"aa_{run}"
            sdk_in = {}
            for u in ids:
                d = client.get_variant(flag, u, session_id=f"s{run}")
                if d.in_experiment:
                    sdk_in[u] = d.variant_id
            client.flush()
            # анализ — по тому, что сохранил сервер
            exp = [e for e in st.exposures.events() if e["experiment_id"] == flag]
            got = {e["unit_id"]: e["variant_id"] for e in exp}
            assert len(got) == len(exp) and got == sdk_in, run  # ни потерь, ни дублей
            frame = units.merge(pd.DataFrame({"guest_id": list(got), "variant": list(got.values())}), on="guest_id")
            res, _ = ORCH._evaluate_pair_metric(frame, REVENUE, "control", "treatment", variant_col="variant",
                                                covariate_col="pre_revenue", alpha=0.05)
            p_values.append(res["p_value"])
            n_c = int((frame["variant"] == "control").sum())
            srm.append(SG.detect_srm_chi2(n_c, len(frame) - n_c, 0.5, alpha=0.001)["srm_detected"])
            rows.append(len(frame))
        client.close()
        assert sink.rejected == [] and client.counters["sink_error"] == 0 and client.fallback_rate == 0.0
    return np.asarray(p_values), np.asarray(srm), np.asarray(rows), len(ids)


def test_e2e_aa_pipeline_fast(guests):
    p, srm, rows, n = run_e2e_aa(guests, n_runs=8)
    assert len(p) == 8 and np.all((p >= 0) & (p <= 1))
    # в тесте — все гости вне глобального holdout 5%; holdout не зависит от прогона
    held = int((ASG.buckets_for("global_holdout_v1", guests["guest_id"].astype(str)) < 500).sum())
    assert np.all(rows == n - held) and 0.03 < held / n < 0.07
    assert not srm.any()


@pytest.mark.slow
def test_e2e_aa_300_runs(guests):
    p, srm, _, _ = run_e2e_aa(guests, n_runs=300)
    fpr = float(np.mean(p < 0.05))
    ks = stats.kstest(p, "uniform").pvalue
    print(f"\nA/A e2e: прогонов {len(p)}, FPR {fpr:.4f}, KS p {ks:.3f}, SRM {int(srm.sum())}")
    assert 0.02 <= fpr <= 0.09, fpr
    assert int(srm.sum()) <= 3


@pytest.mark.slow
def test_asg4_holdout_guests_outside_all_marketing_tests_1e6():
    """Holdout 5% на 10⁶ гостей: доля в [4,9%; 5,1%]; ни один holdout-гость не в маркетинговом тесте любого слоя."""
    ids = np.array([f"g{i}" for i in range(1_000_000)])
    rs = Ruleset(
        version=1,
        layers=[Layer(layer_id="ui"), Layer(layer_id="algo"), Layer(layer_id="ops", marketing=False)],
        experiments=[
            ExperimentRule(experiment_id="m1", layer_id="ui", slot_end=5000, variants=_v()),
            ExperimentRule(experiment_id="m2", layer_id="ui", slot_start=5000, variants=_v()),
            ExperimentRule(experiment_id="m3", layer_id="algo", variants=_v()),
            ExperimentRule(experiment_id="o1", layer_id="ops", variants=_v()),
        ],
    )
    held = ASG.buckets_for(rs.holdout.salt, ids) < 500
    assert 0.049 <= held.mean() <= 0.051
    marketing = [r for r in rs.experiments if rs.layer(r.layer_id).marketing]
    for u in ids[held]:
        for r in marketing:
            e = evaluate(rs, r, u)
            assert not e.is_exposed and e.reason == "holdout", (u, r.experiment_id)
        assert evaluate(rs, rs.by_flag("o1"), u).is_exposed  # немаркетинговый слой holdout не учитывает
    # обратное: вне holdout гость попадает ровно в один тест слоя ui и в тест слоя algo
    for u in ids[~held][:50_000]:
        assert evaluate(rs, rs.by_flag("m1"), u).is_exposed + evaluate(rs, rs.by_flag("m2"), u).is_exposed == 1
        assert evaluate(rs, rs.by_flag("m3"), u).is_exposed


def _v():
    return [{"variant_id": "c", "traffic_percentage": 50.0, "is_control": True},
            {"variant_id": "t", "traffic_percentage": 50.0}]
