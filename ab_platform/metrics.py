"""
Реестр метрик приложения (ТЗ v4.1: MET-1, MET-2; ТЗ v4: раздел 8, DAT-5).

Метрика — версия определения: имя, формула из закрытого списка, тип, направление, винзоризация, владелец.
Версия неизменяема: правка — новая версия, прошлые отчёты ссылаются на свою. Паспорт ссылается на метрику
строкой «имя@версия» (MetricDefinition.registry_ref). Свои SQL-определения — фаза 10.

Формулы считаются по событиям гостя в окне атрибуции (ab_platform/engine/unit_table.py):
- has_order — был хотя бы один неотменённый заказ (бинарная);
- revenue — сумма неотменённых заказов минус возвраты, по дате исходного заказа (непрерывная);
- order_count — число неотменённых заказов (непрерывная);
- avg_check — revenue / order_count, дельта-метод на агрегатах гостей (ratio);
- cancel_rate — отменённые заказы / все заказы (ratio);
- has_app_event — было событие приложения event_type: падение или ошибка оплаты (бинарная).
"""
import hashlib
import json
import re
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ab_platform.schemas.passport import APP_GUARDRAIL_NAMES, Direction, MetricDefinition, MetricType

FORMULAS: Dict[str, MetricType] = {
    "has_order": MetricType.BINARY,
    "revenue": MetricType.CONTINUOUS,
    "order_count": MetricType.CONTINUOUS,
    "avg_check": MetricType.RATIO,
    "cancel_rate": MetricType.RATIO,
    "has_app_event": MetricType.BINARY,
}
REF_RE = re.compile(r"^([a-z][a-z0-9_]{1,62})@([1-9][0-9]{0,5})$")


class MetricVersion(BaseModel):
    """Версия метрики. Хэш — по определению (без владельца и описания): одинаковое определение — один хэш."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]{1,62}$")
    version: int = Field(..., ge=1)
    title: str = Field(..., min_length=3, max_length=200)
    formula: Literal["has_order", "revenue", "order_count", "avg_check", "cancel_rate", "has_app_event"]
    event_type: Optional[Literal["app_crash", "payment_error"]] = None
    direction: Direction = Direction.INCREASE
    winsorize_percentile: Optional[float] = Field(None, ge=0.9, le=1.0)
    owner: str = Field(..., min_length=1)
    description: str = ""

    @model_validator(mode="after")
    def _event_type(self):
        if (self.formula == "has_app_event") != (self.event_type is not None):
            raise ValueError("event_type задаётся только для формулы has_app_event, и для неё обязателен.")
        if self.winsorize_percentile is not None and self.metric_type != MetricType.CONTINUOUS:
            raise ValueError("Винзоризация применима только к непрерывной метрике.")
        return self

    @property
    def metric_type(self) -> MetricType:
        return FORMULAS[self.formula]

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def definition_hash(self) -> str:
        body = self.model_dump(mode="json", include={"formula", "event_type", "direction", "winsorize_percentile"})
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def columns(self) -> Tuple[str, Optional[str]]:
        """Колонки таблицы «гость × метрики»: (числитель, знаменатель) — знаменатель только у ratio."""
        if self.metric_type == MetricType.RATIO:
            return f"{self.name}__num", f"{self.name}__den"
        return self.name, None

    def definition(self) -> MetricDefinition:
        """MetricDefinition для Паспорта и стат-ядра."""
        num, den = self.columns()
        return MetricDefinition(name=self.name, metric_type=self.metric_type, numerator_col=num, denominator_col=den,
                                expected_direction=self.direction, winsorize_percentile=self.winsorize_percentile,
                                registry_ref=self.ref)


def parse_ref(ref: str) -> Tuple[str, int]:
    m = REF_RE.match(ref or "")
    if not m:
        raise ValueError(f"Ссылка на метрику '{ref}' должна иметь вид имя@версия, например revenue_per_guest@1.")
    return m.group(1), int(m.group(2))


SYSTEM_OWNER = "platform"
DEFAULT_METRICS: List[MetricVersion] = [
    MetricVersion(name="conversion", version=1, title="Конверсия в заказ", formula="has_order", owner=SYSTEM_OWNER,
                  description="Доля показанных гостей с неотменённым заказом в окне атрибуции"),
    MetricVersion(name="revenue_per_guest", version=1, title="Выручка на гостя", formula="revenue",
                  winsorize_percentile=0.995, owner=SYSTEM_OWNER,
                  description="Выручка без НДС за вычетом отмен и возвратов, по дате исходного заказа"),
    MetricVersion(name="avg_check", version=1, title="Средний чек", formula="avg_check", owner=SYSTEM_OWNER,
                  description="Выручка / число заказов, дельта-метод на агрегатах гостей"),
    MetricVersion(name="order_frequency", version=1, title="Частота заказов", formula="order_count",
                  owner=SYSTEM_OWNER, description="Число неотменённых заказов на гостя в окне атрибуции"),
    MetricVersion(name="crash_share", version=1, title="Доля гостей с падением приложения", formula="has_app_event",
                  event_type="app_crash", direction=Direction.DECREASE, owner=SYSTEM_OWNER),
    MetricVersion(name="payment_error_share", version=1, title="Доля гостей с ошибкой оплаты",
                  formula="has_app_event", event_type="payment_error", direction=Direction.DECREASE,
                  owner=SYSTEM_OWNER),
    MetricVersion(name="cancel_rate", version=1, title="Доля отменённых заказов", formula="cancel_rate",
                  direction=Direction.DECREASE, owner=SYSTEM_OWNER),
]
# защитные метрики приложения по умолчанию (MET-2): их отсутствие в гостевом Паспорте — предупреждение
APP_GUARDRAILS = APP_GUARDRAIL_NAMES
assert set(APP_GUARDRAILS) <= {m.name for m in DEFAULT_METRICS}
