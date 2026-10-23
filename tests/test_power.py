from ab_platform.engine.power import PowerAndSampleSizeEngine, PowerConfig


class TestPowerEngine:
    def setup_method(self):
        self.engine = PowerAndSampleSizeEngine(PowerConfig(alpha=0.05, power=0.80))

    def test_binary_sample_size(self):
        result = self.engine.binary(baseline_rate=0.10, mde=0.01, daily_users=1000)
        assert result.sample_size_per_group > 0
        assert result.total_sample_size == result.sample_size_per_group * 2
        assert result.duration_days > 0

    def test_continuous_sample_size(self):
        result = self.engine.continuous(mean=100.0, std=30.0, mde=3.0, daily_users=500)
        assert result.sample_size_per_group > 0

    def test_cuped_reduces_sample(self):
        base = self.engine.binary(0.10, 0.01, 1000)
        adjusted = self.engine.cuped_adjustment(base.sample_size_per_group, correlation_with_covariate=0.5)
        assert adjusted < base.sample_size_per_group