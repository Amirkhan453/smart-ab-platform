"""Фаза 2, П1: MurmurHash3 и двухшаговое назначение гостей (ТЗ v4 4.1; ASG-1, ASG-2, ASG-4; K8)."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from ab_platform.engine.assignment import (
    N_BUCKETS, REASON_ASSIGNED, REASON_HOLDOUT, REASON_NOT_IN_LANE, REASON_RAMP,
    DeterministicAssignmentEngine as ASG, allocate_variant_buckets, bucket, murmurhash3_x86_32, smhasher_self_test,
)
from ab_platform.schemas.passport import VariantAllocation

VECTORS = json.loads((Path(__file__).parent / "vectors" / "murmur3_x86_32.json").read_text(encoding="utf-8"))


# --- Хэш ---------------------------------------------------------------------------------------

def test_smhasher_verification():
    assert smhasher_self_test() == int(VECTORS["smhasher_verification"], 16) == 0xB0F57EE3


@pytest.mark.parametrize("v", VECTORS["vectors"], ids=lambda v: v["key"][:20])
def test_shared_vectors_from_c_reference(v):
    assert murmurhash3_x86_32(v["key"]) == v["uint32"]
    assert murmurhash3_x86_32(v["key"]) % N_BUCKETS == v["bucket"]


def test_public_known_values():
    # Общеизвестные значения MurmurHash3_x86_32 (seed 0 и seed 1)
    assert murmurhash3_x86_32("") == 0
    assert murmurhash3_x86_32("", seed=1) == 0x514E28B7
    assert murmurhash3_x86_32("a") == 0x3C2569B2
    assert murmurhash3_x86_32("test") == 0xBA6BD213
    assert murmurhash3_x86_32("Hello, world!") == 0xC0363E43


def test_str_and_bytes_agree_and_unsigned():
    for k in ["Гость", "🍔", "abc", "x" * 101]:
        h = murmurhash3_x86_32(k)
        assert h == murmurhash3_x86_32(k.encode("utf-8"))
        assert 0 <= h <= 0xFFFFFFFF


def test_bucket_key_format_and_separator():
    assert bucket("salt", "unit123") == murmurhash3_x86_32("salt:unit123") % N_BUCKETS
    # разделитель не даёт склеить соль и ID: "1"+"23" и "12"+"3" — разные ключи
    assert murmurhash3_x86_32("1:23") != murmurhash3_x86_32("12:3")
    for bad in ["", None, 5]:
        with pytest.raises(ValueError):
            bucket("s", bad)
        with pytest.raises(ValueError):
            bucket(bad, "u")


# --- Назначение ----------------------------------------------------------------------------------

IDS = [f"guest_{i}" for i in range(20000)]


def test_combat_split_equals_buckets_for():
    z = ASG.buckets_for("exp_x", IDS[:2000])
    ctrl = ASG.control_from_bucket(z, 30.0)
    combat = [ASG.assign_user(u, "exp_x", control_split_pct=30.0).variant_id == "control" for u in IDS[:2000]]
    assert ctrl.tolist() == combat
    assert abs(ctrl.mean() - 0.30) < 0.035


def test_allocate_variant_buckets_exact():
    assert allocate_variant_buckets([100 / 3] * 3) == [3334, 3333, 3333]
    assert sum(allocate_variant_buckets([16.665, 16.665, 16.67, 50.0])) == N_BUCKETS
    assert allocate_variant_buckets([50.0, 50.0]) == [5000, 5000]


def test_multi_variant_shares_and_ids():
    variants = [VariantAllocation(variant_id=v, traffic_percentage=p, is_control=(v == "A"))
                for v, p in [("A", 50.0), ("B", 25.0), ("C", 25.0)]]
    got = [ASG.assign_user(u, "exp_abc", variants=variants).variant_id for u in IDS]
    counts = np.array([got.count(v) for v in "ABC"])
    assert stats.chisquare(counts, f_exp=np.array([0.5, 0.25, 0.25]) * len(IDS)).pvalue > 0.001


def test_outside_returns_control_variant_and_reason():
    variants = [VariantAllocation(variant_id="B", traffic_percentage=50.0),
                VariantAllocation(variant_id="A", traffic_percentage=50.0, is_control=True)]
    reasons = set()
    for u in IDS[:5000]:
        r = ASG.assign_user(u, "e", variants=variants, exposure_rate_pct=30.0, global_holdout_pct=5.0,
                            layer_slot_start=0, layer_slot_end=5000)
        reasons.add(r.reason)
        if not r.is_exposed:
            assert r.variant_id == "A"
        assert r.variant_id in {"A", "B"}
    assert reasons == {REASON_ASSIGNED, REASON_HOLDOUT, REASON_NOT_IN_LANE, REASON_RAMP}
    # без вариантов контроль называется "control"
    assert {ASG.assign_user(u, "e", exposure_rate_pct=0.0).variant_id for u in IDS[:100]} == {"control"}


def test_lane_share_and_ramp_share():
    res = [ASG.assign_user(u, "e", layer_id="L", layer_slot_start=2000, layer_slot_end=6000,
                           exposure_rate_pct=25.0) for u in IDS]
    exposed = np.mean([r.is_exposed for r in res])  # 40% дорожки × 25% ramp = 10%
    assert abs(exposed - 0.10) < 0.01
    assert all(2000 <= ASG._hash_to_10000_buckets("layer:L", r.user_id) < 3000 for r in res if r.is_exposed)


def test_lanes_in_one_layer_are_exclusive():
    a = [ASG.assign_user(u, "e1", layer_id="L", layer_slot_start=0, layer_slot_end=5000).is_exposed for u in IDS]
    b = [ASG.assign_user(u, "e2", layer_id="L", layer_slot_start=5000, layer_slot_end=10000).is_exposed for u in IDS]
    assert not any(x and y for x, y in zip(a, b))
    assert all(x or y for x, y in zip(a, b))


def _sticky_violations(ids, ramps, **kw):
    prev, changed = {}, 0
    for ramp in ramps:
        for u in ids:
            r = ASG.assign_user(u, "exp_ramp", exposure_rate_pct=ramp, **kw)
            if u in prev:
                changed += (not r.is_exposed) or r.variant_id != prev[u]
            if r.is_exposed:
                prev[u] = r.variant_id
    return changed, len(prev)


def test_sticky_ramp_up_fast():
    changed, n_final = _sticky_violations(IDS[:10000], [1, 5, 20, 50, 100], layer_id="L",
                                          layer_slot_start=1000, layer_slot_end=9000)
    assert changed == 0 and n_final > 7000


def test_holdout_share_and_never_exposed():
    res = [ASG.assign_user(u, f"e{i % 3}", global_holdout_pct=5.0) for i, u in enumerate(IDS)]
    held = np.array([r.is_in_global_holdout for r in res])
    assert abs(held.mean() - 0.05) < 0.005
    assert not any(r.is_exposed for r in res if r.is_in_global_holdout)
    # holdout один на все эксперименты: зависит только от соли holdout, не от эксперимента
    held_other = np.array([ASG.assign_user(u, "другой", global_holdout_pct=5.0).is_in_global_holdout for u in IDS])
    assert (held == held_other).all()


def test_control_defaults_to_first_variant_without_flag():
    variants = [VariantAllocation(variant_id=v, traffic_percentage=50.0) for v in ("X", "Y")]
    assert ASG.assign_user("u1", "e", variants=variants, exposure_rate_pct=0.0).variant_id == "X"


def test_validation_errors():
    for kw in ({"exposure_rate_pct": 100.5}, {"control_split_pct": 100.0}, {"global_holdout_pct": 100.0}):
        with pytest.raises(ValueError):
            ASG.assign_user("u", "e", **kw)
    with pytest.raises(ValueError):
        ASG.assign_user("u", "e", layer_slot_start=5000, layer_slot_end=5000)
    with pytest.raises(ValueError):
        ASG.assign_user("u", "e", variants=[VariantAllocation(variant_id="A", traffic_percentage=100.0)])
    with pytest.raises(ValueError):
        ASG.assign_user("u", "e", variants=[VariantAllocation(variant_id="A", traffic_percentage=60.0),
                                            VariantAllocation(variant_id="B", traffic_percentage=30.0)])
    with pytest.raises(ValueError):
        ASG.assign_user("u", "e", variants=[VariantAllocation(variant_id=v, traffic_percentage=50.0, is_control=True)
                                            for v in "AB"])
    with pytest.raises(ValueError):
        ASG.validate_orthogonality_and_uniformity("a", "b", n_sim_users=10)


def test_orthogonality_report():
    rep = ASG.validate_orthogonality_and_uniformity("exp_ui", "exp_algo", n_sim_users=20000)
    assert rep["is_orthogonal_and_valid"] is True


# --- Приёмка на больших объёмах (slow) ----------------------------------------------------------

@pytest.mark.slow
def test_asg1_bucket_uniformity_1e7():
    counts = np.zeros(N_BUCKETS, dtype=np.int64)
    for start in range(0, 10_000_000, 1_000_000):
        z = ASG.buckets_for("asg1_uniformity", (f"g{i}" for i in range(start, start + 1_000_000)))
        counts += np.bincount(z, minlength=N_BUCKETS)
    assert stats.chisquare(counts).pvalue > 0.001


@pytest.mark.slow
def test_asg1_layer_and_variant_independence_1e6():
    ids = [f"g{i}" for i in range(1_000_000)]
    l1 = ASG.buckets_for("layer:ui", ids) // 1000
    l2 = ASG.buckets_for("layer:algo", ids) // 1000
    v1 = ASG.buckets_for("exp_ui_1", ids) // 1000
    for a, b in [(l1, l2), (l1, v1)]:
        table = np.zeros((10, 10), dtype=np.int64)
        np.add.at(table, (a, b), 1)
        assert stats.chi2_contingency(table)[1] > 0.001


@pytest.mark.slow
def test_asg2_sticky_ramp_up_1e6():
    changed, n_final = _sticky_violations([f"g{i}" for i in range(1_000_000)], [1, 5, 20, 50, 100])
    assert changed == 0 and n_final > 990_000


@pytest.mark.slow
def test_asg4_holdout_share_1e6():
    held = ASG.buckets_for("global_holdout_v1", (f"g{i}" for i in range(1_000_000))) < 500
    assert 0.049 <= held.mean() <= 0.051
