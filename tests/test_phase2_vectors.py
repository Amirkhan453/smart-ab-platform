"""
Фаза 2, П5: общие тест-векторы назначения (ASG-6) — Python и JS дают одинаковые решения на одном файле.
JS-тесты (node:test) запускаются отсюда же, если установлен node; без node они пропускаются.
"""
import importlib.util
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from ab_platform.engine.assignment import (
    allocate_variant_buckets, murmurhash3_x86_32, pct_to_buckets, ramp_range, smhasher_self_test,
)
from ab_platform.engine.ruleset_engine import evaluate, evaluate_all
from ab_platform.schemas.ruleset import Ruleset
from ab_platform.sdk import ExperimentClient
from tests.test_phase2_http import Stand

ROOT = Path(__file__).resolve().parents[1]
VEC_PATH = ROOT / "tests" / "vectors" / "assignment_vectors.json"
V = json.loads(VEC_PATH.read_text(encoding="utf-8"))
JS_DIR = ROOT / "sdk" / "js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node не установлен")


def test_hashes_and_rounding():
    assert f"{smhasher_self_test():08X}" == V["smhasher_verification"]
    for v in V["hash"]:
        assert murmurhash3_x86_32(v["key"]) == v["uint32"] and v["uint32"] % 10000 == v["bucket"], v["key"]
    for c in V["pct_to_buckets"]:
        assert pct_to_buckets(c["pct"]) == c["buckets"], c
    for c in V["ramp_end"]:
        assert ramp_range(c["slot_start"], c["slot_end"], c["ramp_pct"])[1] == c["end"], c
    for c in V["allocate"]:
        assert allocate_variant_buckets(c["percentages"]) == c["buckets"], c


def test_all_cases_match_engine_and_sdk():
    rss = [Ruleset.model_validate(r) for r in V["rulesets"]]
    # ruleset в векторах — ровно то, что отдаёт сервер: etag в файле совпадает с вычисленным
    assert [r.etag for r in rss] == [r["etag"] for r in V["rulesets"]]
    clients = [ExperimentClient(initial_ruleset=r) for r in rss]
    position = [{e.flag_key: i for i, e in enumerate(r.experiments)} for r in rss]
    for c in V["cases"]:
        rs, want = rss[c["rs"]], (c["variant"], c["in"], c["reason"])
        rule = rs.by_flag(c["flag"])
        attrs, unit_type, enrolled = c.get("attrs"), c.get("unit_type"), c.get("enrolled", False)
        e = evaluate(rs, rule, c["unit"], attrs, unit_type, enrolled)
        assert (e.variant_id, e.is_exposed, e.reason) == want, c
        assert rule.params_of(e.variant_id) == c.get("params", {}), c
        # все флаги за один проход (POST /decide) дают то же самое
        a = evaluate_all(rs, c["unit"], attrs, unit_type, {rule.experiment_id} if enrolled else ())
        assert a[position[c["rs"]][c["flag"]]] == e, c
        # SDK: «первый показ уже был» — память клиента о показах
        client, key = clients[c["rs"]], (c["unit"], rule.experiment_id)
        if enrolled:
            client._enrolled[key] = None
        d = client.get_variant(c["flag"], c["unit"], attributes=attrs, unit_type=unit_type, log_exposure=False)
        client._enrolled.pop(key, None)
        assert (d.variant_id, d.in_experiment, d.reason, d.params) == (*want, c.get("params", {})), c
    reasons = {c["reason"] for c in V["cases"]}
    assert {"assigned", "holdout", "not_in_lane", "ramp", "kill_switch", "unit_type_mismatch"} < reasons
    assert len({r for r in reasons if r.startswith("targeting")}) == 12  # 6 признаков × (не подошёл, нет признака)


@pytest.mark.skipif(shutil.which("cc") is None, reason="нет C-компилятора")
def test_vectors_file_is_reproducible(tmp_path):
    """Файл векторов не правили руками: генератор (C-референс + независимая запись правил) даёт его побайтно."""
    spec = importlib.util.spec_from_file_location("gen", ROOT / "tests" / "vectors" / "gen_assignment_vectors.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = tmp_path / "v.json"
    gen.main(out)
    assert out.read_bytes() == VEC_PATH.read_bytes()


@needs_node
def test_js_sdk_node_tests():
    r = subprocess.run([NODE, "--test", *map(str, sorted((JS_DIR / "test").glob("*.test.mjs")))],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "# fail 0" in r.stdout or "ℹ fail 0" in r.stdout


JS_E2E = r"""
import { ExperimentClient, createHttpFetcher, createHttpSink } from "%s";
const [url, n] = [process.argv[2], Number(process.argv[3])];
const sink = createHttpSink(url);
const c = new ExperimentClient({ fetcher: createHttpFetcher(url), sink, batchSize: 100 });
const first = await c.refresh();
const second = await c.refresh();
const out = [];
for (let i = 0; i < n; i++) {
  const u = i %% 7 === 0 ? `гость_${i}` : `guest_${i}`;
  for (const f of ["exp_menu", "exp_promo", "exp_rank", "exp_killed", "ops_flag"]) {
    const d = c.getVariant(f, u, { sessionId: "s1" });
    out.push([f, u, d.variantId, d.inExperiment, d.reason]);
  }
}
await c.close();
console.log(JSON.stringify({ first, second, counters: c.counters, rejected: sink.rejected, out }));
"""


@needs_node
def test_js_sdk_against_real_server(tmp_path):
    """JS SDK ↔ сервер uvicorn: ETag/304, решения как у Python, exposure принимаются без отказов."""
    rs = Ruleset.model_validate(V["rulesets"][0])
    script = tmp_path / "e2e.mjs"
    script.write_text(JS_E2E % (JS_DIR / "ab-sdk.mjs").as_uri(), encoding="utf-8")
    n = 1500
    with Stand(rs) as st:
        r = subprocess.run([NODE, str(script), st.url, str(n)], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-3000:]
        res = json.loads(r.stdout)
        stored = st.exposures.events()
    assert res["first"] is True and res["second"] is False and res["counters"]["not_modified"] == 1
    assert res["rejected"] == [] and res["counters"]["sink_error"] == 0
    exposed = []
    for f, u, variant, inexp, reason in res["out"]:
        e = evaluate(rs, rs.by_flag(f), u)
        assert (variant, inexp, reason) == (e.variant_id, e.is_exposed, e.reason), (f, u)
        if inexp:
            exposed.append((u, rs.by_flag(f).experiment_id, variant))
    assert len(res["out"]) == 5 * n and len(exposed) > n
    # сервер получил ровно по одному exposure на каждое участие, с верными полями
    assert Counter((e["unit_id"], e["experiment_id"], e["variant_id"]) for e in stored) == Counter(exposed)
    assert all(e["ruleset_version"] == rs.version and e["session_id"] == "s1" for e in stored)
