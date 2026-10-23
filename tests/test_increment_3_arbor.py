import pytest
from ab_platform.copilot.arbor_engine import ArborDecisionCopilot, CopilotSessionState
from ab_platform.schemas.passport import ExperimentType, RandomizationUnit


def test_arbor_graph_integrity():
    copilot = ArborDecisionCopilot()
    assert copilot.validate_graph_integrity() is True


def test_arbor_marketplace_switchback_routing():
    """Если онлайн-маркетплейс имеет динамические сетевые эффекты (такси/доставка), ИИ выбирает Switchback."""
    copilot = ArborDecisionCopilot()
    state = CopilotSessionState()

    state = copilot.advance_tree(
        state,
        {
            "can_randomize": True,
            "is_offline": False,
            "has_network_effects": True,
            "interference_type": "temporal_marketplace",
            "has_cannibalization_risk": True,
        },
    )

    assert state.recommended_design == ExperimentType.SWITCHBACK
    assert state.recommended_randomization_unit == RandomizationUnit.CLUSTER_TIME_CELL
    assert state.p_suff == 1.0
    assert any("каннибализации" in w for w in state.warnings)


def test_arbor_offline_retail_geo_routing():
    """Если тест идет в оффлайн-магазинах, ИИ выбирает Geo Experiment / Matched Markets."""
    copilot = ArborDecisionCopilot()
    state = CopilotSessionState()

    state = copilot.advance_tree(
        state,
        {
            "can_randomize": True,
            "is_offline": True,
            "daily_traffic": 1500,
            "mde_relative": 0.02,
        },
    )

    assert state.recommended_design == ExperimentType.GEO_EXPERIMENT
    assert state.recommended_randomization_unit == RandomizationUnit.STORE_ID
    assert state.p_suff == 1.0
    assert any("Kohavi Rule" in w for w in state.warnings)


def test_linkedin_site_wide_impact_calculation():
    """Проверка расчета Site Wide Impact (SWI = Delta * alpha)."""
    swi = ArborDecisionCopilot.compute_site_wide_impact(
        local_lift_rel=0.05, triggered_share_alpha=0.20
    )
    assert swi["local_lift_pct"] == 5.0
    assert swi["site_wide_impact_pct"] == 1.0
