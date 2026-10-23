"""Synthetic Data Generator — реалистичные датасеты для тестирования."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    n_users: int = 10_000
    true_effect: float = 0.0
    baseline_rate: float = 0.10
    covariate_effect: float = 0.3
    whale_fraction: float = 0.01
    whale_multiplier: float = 50.0
    double_exposure_rate: float = 0.0
    seed: Optional[int] = 42


class SyntheticDataGenerator:
    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()

    def generate_binary(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.config.seed)
        n = self.config.n_users
        covariate = rng.beta(2, 5, size=n)
        group = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])
        base_prob = self.config.baseline_rate * (1 + self.config.covariate_effect * covariate)
        base_prob = np.clip(base_prob, 0.01, 0.99)
        treatment_mask = group == "treatment"
        base_prob[treatment_mask] += self.config.true_effect
        base_prob = np.clip(base_prob, 0.01, 0.99)
        conversion = rng.binomial(1, base_prob).astype(float)
        revenue = rng.exponential(10.0, size=n)
        whale_mask = rng.random(n) < self.config.whale_fraction
        revenue[whale_mask] *= self.config.whale_multiplier
        if self.config.double_exposure_rate > 0:
            de_mask = rng.random(n) < self.config.double_exposure_rate
            group[de_mask] = "both"
        return pd.DataFrame({"user_id": range(n), "group": group, "conversion": conversion, "revenue": revenue, "covariate": covariate})

    def generate_continuous(self, mean: float = 100.0, std: float = 30.0) -> pd.DataFrame:
        rng = np.random.default_rng(self.config.seed)
        n = self.config.n_users
        covariate = rng.normal(0, 1, size=n)
        group = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])
        value = rng.normal(mean, std, size=n)
        value += self.config.covariate_effect * std * covariate
        treatment_mask = group == "treatment"
        value[treatment_mask] += self.config.true_effect * std
        whale_mask = rng.random(n) < self.config.whale_fraction
        value[whale_mask] *= self.config.whale_multiplier
        return pd.DataFrame({"user_id": range(n), "group": group, "value": value, "covariate": covariate})