"""Фаза 1, П6: K5 — явные имена групп в multivariate_debiasing_ate."""
import numpy as np
import pytest


def test_k5_debiasing_uses_explicit_variant_ids():
    import pandas as pd
    from ab_platform.engine.causal_quasi import CausalAndMLDebiasingEngine as C

    rng = np.random.default_rng(5)
    n = 600
    x = rng.normal(0, 1, n)
    v = np.array(["A", "B", "C"])[np.arange(n) % 3]
    y = 10 + 2 * x + (v == "B") * 0.5 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x": x, "variant": v})
    with pytest.raises(ValueError, match="нет вариантов"):
        C.multivariate_debiasing_ate(df, "y", ["x"])
    with pytest.raises(ValueError, match="другие варианты"):
        C.multivariate_debiasing_ate(df, "y", ["x"], control_id="A", treatment_id="B")
    pair = df[df.variant.isin(["A", "B"])]
    r = C.multivariate_debiasing_ate(pair, "y", ["x"], control_id="A", treatment_id="B")
    assert r["ate_debiased"] == pytest.approx(0.5, abs=0.25) and r["variance_reduction_pct"] > 50
