"""Запись об истинном эффекте (ground truth), SYN-3."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Truth:
    """
    Истинный эффект внедрённого воздействия, посчитанный по контрфактическим данным.

    - ``per_restaurant_mean_abs`` — средний по тестовым ресторанам прирост метрики в день
      (estimand ``per_restaurant_mean``);
    - ``network_total_rel`` — суммарный прирост / суммарный контрфактический уровень
      тестовых ресторанов за период (estimand ``network_total``);
    - ``spillover_abs`` — суммарное изменение метрики у соседей (обычно отрицательное).
    """

    scenario: str
    metric: str
    lift_on: str
    units: List[str]
    start: str
    end: str
    params: Dict[str, object]
    n_treated_cells: int
    per_restaurant_mean_abs: float
    per_restaurant_mean_rel: float
    network_total_abs: float
    network_total_rel: float
    spillover_units: List[str] = field(default_factory=list)
    spillover_abs: float = 0.0
    carryover_abs: float = 0.0
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Optional[str | Path] = None) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, d: dict) -> "Truth":
        return cls(**d)
