"""Experiment Passport — строгий контракт запуска A/B-теста."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class MetricType(str, Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    RATIO = "ratio"


class TrafficSplit(BaseModel):
    control: float = Field(ge=0.0, le=1.0)
    treatment: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_total(self) -> "TrafficSplit":
        total = self.control + self.treatment
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Сумма сплитов должна быть 1.0, получено {total}")
        return self


class Metric(BaseModel):
    name: str
    metric_type: MetricType
    is_guardrail: bool = False
    baseline_value: Optional[float] = None
    mde: Optional[float] = Field(default=None, gt=0.0, description="Minimum Detectable Effect")


class ExperimentPassport(BaseModel):
    experiment_id: str
    name: str
    hypothesis: str
    primary_metric: Metric
    guardrail_metrics: list[Metric] = Field(default_factory=list)
    traffic_split: TrafficSplit
    duration_days: int = Field(ge=1, le=90)
    unit_of_randomization: str = "user_id"

    @field_validator("guardrail_metrics")
    @classmethod
    def at_least_one_guardrail(cls, v: list[Metric]) -> list[Metric]:
        if not v:
            raise ValueError("Обязательно укажите хотя бы одну Guardrail-метрику")
        return v

    @field_validator("primary_metric")
    @classmethod
    def primary_must_have_mde(cls, v: Metric) -> Metric:
        if v.mde is None:
            raise ValueError("У primary_metric должен быть задан MDE")
        return v

    def has_network_effects_risk(self) -> bool:
        """Проверка риска нарушения SUTVA."""
        return self.unit_of_randomization not in ("user_id", "session_id")
