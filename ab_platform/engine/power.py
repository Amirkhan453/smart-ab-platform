"""Power & Sample Size Engine — расчёт выборки, MDE и длительности."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from scipy import stats


class MetricKind(str, Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    RATIO = "ratio"


@dataclass(frozen=True)
class PowerConfig:
    alpha: float = 0.05
    power: float = 0.80
    two_sided: bool = True


@dataclass(frozen=True)
class PowerResult:
    sample_size_per_group: int
    total_sample_size: int
    mde: float
    duration_days: float
    notes: str = ""


class PowerAndSampleSizeEngine:
    def __init__(self, config: PowerConfig | None = None):
        self.config = config or PowerConfig()

    def _z_values(self) -> tuple[float, float]:
        alpha = self.config.alpha
        if self.config.two_sided:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(self.config.power)
        return z_alpha, z_beta

    def binary(self, baseline_rate: float, mde: float, daily_users: int) -> PowerResult:
        z_alpha, z_beta = self._z_values()
        p1 = baseline_rate
        p2 = baseline_rate + mde
        pooled = (p1 + p2) / 2.0
        numerator = (z_alpha * math.sqrt(2 * pooled * (1 - pooled)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2
        n_per_group = math.ceil(numerator / denominator)
        total = n_per_group * 2
        duration = math.ceil(total / daily_users)
        return PowerResult(sample_size_per_group=n_per_group, total_sample_size=total, mde=mde, duration_days=float(duration), notes=f"Baseline={p1:.4f}, Treatment={p2:.4f}")

    def continuous(self, mean: float, std: float, mde: float, daily_users: int) -> PowerResult:
        z_alpha, z_beta = self._z_values()
        effect_size = mde / std
        n_per_group = math.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2)
        total = n_per_group * 2
        duration = math.ceil(total / daily_users)
        return PowerResult(sample_size_per_group=n_per_group, total_sample_size=total, mde=mde, duration_days=float(duration), notes=f"Mean={mean:.2f}, Std={std:.2f}, Cohen's d={effect_size:.4f}")

    def ratio_delta_method(self, numerator_mean: float, numerator_var: float, denominator_mean: float, denominator_var: float, covariance: float, mde_relative: float, daily_users: int) -> PowerResult:
        z_alpha, z_beta = self._z_values()
        d = denominator_mean
        ratio_var = numerator_var / d**2 - 2 * numerator_mean * covariance / d**3 + (numerator_mean / d)**2 * denominator_var / d**2
        ratio_std = math.sqrt(max(ratio_var, 1e-12))
        ratio_mean = numerator_mean / d
        effect_size = (mde_relative * ratio_mean) / ratio_std
        n_per_group = math.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2)
        total = n_per_group * 2
        duration = math.ceil(total / daily_users)
        return PowerResult(sample_size_per_group=n_per_group, total_sample_size=total, mde=mde_relative, duration_days=float(duration), notes=f"Delta-method ratio: mean={ratio_mean:.4f}, std={ratio_std:.4f}")

    def cuped_adjustment(self, base_n: int, correlation_with_covariate: float) -> int:
        rho_sq = correlation_with_covariate ** 2
        adjusted = math.ceil(base_n * (1 - rho_sq))
        return max(adjusted, 1)