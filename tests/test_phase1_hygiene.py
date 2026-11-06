"""Фаза 1, П6: тесты гигиены (код «Фазы 1А» пришёл без тестов) — DAT-3."""
import numpy as np
import pandas as pd
import pytest

from ab_platform.engine.hygiene import DataHygieneProcessor as H


def frame():
    return pd.DataFrame({
        "user_id": ["a", "a", "b", "c", "c", "d", "e"],
        "variant": ["control", "treatment", "control", "treatment", "treatment", "control", "treatment"],
        "event_count": [1, 1, 3, 100, 50, 2, 4],
        "revenue": [1.0, 2.0, 3.0, 1000.0, 5.0, 4.0, 6.0],
    })


def test_double_exposure_diagnostics():
    clean, frac, diag = H.remove_double_exposure(frame(), return_diagnostics=True)
    assert "a" not in set(clean.user_id) and frac == pytest.approx(1 / 5)
    assert isinstance(diag, dict)
    with pytest.raises(ValueError):
        H.remove_double_exposure(frame().drop(columns="variant"))


def test_winsorization_diagnostics_by_variant():
    out, diag = H.apply_winsorization(frame(), "revenue", upper_quantile=0.8, return_diagnostics=True)
    assert out["revenue"].max() < 1000.0
    assert isinstance(diag, dict)
    with pytest.raises(ValueError):
        H.apply_winsorization(frame(), "nope")


def test_bot_filter_is_variant_blind():
    clean, diag = H.filter_bots_variant_blind(frame(), max_threshold=60)
    assert diag["n_bot_units"] == 1 and "c" not in set(clean.user_id)
    assert diag["removed_by_variant"] == {"treatment": 2} and diag["threshold_source"] == "absolute"
    # Порог по квантилю считается по всему пулу: перестановка рук не меняет порог.
    swapped = frame().assign(variant=lambda d: np.where(d.variant == "control", "treatment", "control"))
    _, d1 = H.filter_bots_variant_blind(frame(), quantile_threshold=0.8)
    _, d2 = H.filter_bots_variant_blind(swapped, quantile_threshold=0.8)
    assert d1["threshold_value"] == d2["threshold_value"] and d1["n_bot_units"] == d2["n_bot_units"]
    for bad in (dict(), dict(max_threshold=1, quantile_threshold=0.5), dict(quantile_threshold=1.5),
                dict(max_threshold=0)):
        with pytest.raises(ValueError):
            H.filter_bots_variant_blind(frame(), **bad)
    with pytest.raises(ValueError):
        H.filter_bots_variant_blind(frame(), activity_col="missing", max_threshold=1)


def test_log_transformation_warns_about_estimand():
    out, diag = H.apply_log_transformation(frame(), "revenue")
    assert np.allclose(out["revenue_log"], np.log1p(frame()["revenue"]))
    assert "оцениваемую величину" in diag["warning"]
    with pytest.raises(ValueError):
        H.apply_log_transformation(frame().assign(revenue=-1.0), "revenue")
    with pytest.raises(ValueError):
        H.apply_log_transformation(frame(), "nope")
