import pytest
from pydantic import ValidationError
from ab_platform.schemas.passport import ExperimentPassport, Metric, MetricType, TrafficSplit


def _valid_metric(**overrides) -> Metric:
    defaults = dict(name="ctr", metric_type=MetricType.BINARY, mde=0.01)
    defaults.update(overrides)
    return Metric(**defaults)


def _valid_passport(**overrides) -> ExperimentPassport:
    defaults = dict(
        experiment_id="exp-001", name="Test", hypothesis="H0",
        primary_metric=_valid_metric(),
        guardrail_metrics=[_valid_metric(name="errors", is_guardrail=True)],
        traffic_split=TrafficSplit(control=0.5, treatment=0.5),
        duration_days=14,
    )
    defaults.update(overrides)
    return ExperimentPassport(**defaults)


class TestTrafficSplit:
    def test_valid_split(self):
        split = TrafficSplit(control=0.5, treatment=0.5)
        assert split.control == 0.5

    def test_invalid_split_sum(self):
        with pytest.raises(ValidationError, match="Сумма сплитов"):
            TrafficSplit(control=0.3, treatment=0.3)


class TestExperimentPassport:
    def test_valid_passport(self):
        p = _valid_passport()
        assert p.experiment_id == "exp-001"

    def test_no_guardrail_raises(self):
        with pytest.raises(ValidationError, match="Guardrail"):
            _valid_passport(guardrail_metrics=[])

    def test_no_mde_raises(self):
        with pytest.raises(ValidationError, match="MDE"):
            _valid_passport(primary_metric=_valid_metric(mde=None))

    def test_network_effects_risk(self):
        p = _valid_passport(unit_of_randomization="cluster_id")
        assert p.has_network_effects_risk() is True
        p2 = _valid_passport(unit_of_randomization="user_id")
        assert p2.has_network_effects_risk() is False