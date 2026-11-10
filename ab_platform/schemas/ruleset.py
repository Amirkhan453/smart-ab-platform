"""
Ruleset гостевого домена (ТЗ v4: 4.1, 10.3, ASG-3…5) — версионируемый JSON, который читают все SDK.

Ruleset неизменяем: любое изменение (запуск, ramp-up, kill-switch) — новая версия с новым ETag.
ETag считается по содержимому (sha256 канонического JSON), поэтому одинаковое содержимое даёт одинаковый ETag.
"""
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, field_validator, model_validator

from ab_platform.engine.assignment import N_BUCKETS
from ab_platform.schemas.passport import VariantAllocation
from ab_platform.schemas.targeting import Targeting


class HoldoutConfig(BaseModel):
    """Глобальный holdout гостей: вне всех тестов маркетинговых слоёв (ASG-4)."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    salt: str = Field("global_holdout_v1", min_length=1)
    pct: float = Field(5.0, ge=0.0, lt=100.0)


class Layer(BaseModel):
    """
    Слой: гость участвует не больше чем в одном тесте слоя; разные слои ортогональны.
    marketing — на слой действует глобальный holdout; kind = external_flags — существующие флаги приложения.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    layer_id: str = Field(..., min_length=1)
    salt: Optional[str] = None
    marketing: bool = True
    kind: Literal["platform", "external_flags"] = "platform"

    @model_validator(mode="after")
    def _default_salt(self):
        if self.salt is None:
            object.__setattr__(self, "salt", f"layer:{self.layer_id}")
        elif not self.salt:
            raise ValueError("salt слоя не может быть пустым.")
        return self


class ExperimentRule(BaseModel):
    """
    Правило эксперимента: дорожка [slot_start, slot_end) в слое, таргетинг, ramp-up внутри дорожки, варианты
    с параметрами, kill-switch. passport_* — из какого утверждённого Паспорта построено правило (ТЗ v4.1, LNK-1).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    experiment_id: str = Field(..., min_length=1)
    flag_key: Optional[str] = None
    layer_id: str = Field(..., min_length=1)
    slot_start: int = Field(0, ge=0, lt=N_BUCKETS)
    slot_end: int = Field(N_BUCKETS, gt=0, le=N_BUCKETS)
    ramp_pct: float = Field(100.0, ge=0.0, le=100.0)
    salt: Optional[str] = None
    variants: List[VariantAllocation]
    kill_switch: bool = False
    unit_type: Literal["guest_id", "device_id"] = "guest_id"
    targeting: Targeting = Field(default_factory=Targeting)
    passport_id: Optional[str] = None
    passport_version: Optional[int] = Field(None, ge=1)
    passport_hash: Optional[str] = None

    @field_validator("variants")
    @classmethod
    def _check_variants(cls, v: List[VariantAllocation]) -> List[VariantAllocation]:
        if len(v) < 2:
            raise ValueError("Нужно не меньше двух вариантов.")
        ids = [x.variant_id for x in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Повторяющиеся variant_id: {ids}.")
        total = sum(x.traffic_percentage for x in v)
        if abs(total - 100.0) > 1e-4:
            raise ValueError(f"Сумма долей вариантов должна быть 100%, получено: {total}.")
        if sum(x.is_control for x in v) != 1:
            raise ValueError("Ровно один вариант должен быть контрольным (is_control = true).")
        return v

    @model_validator(mode="after")
    def _defaults_and_lane(self):
        if self.slot_start >= self.slot_end:
            raise ValueError(f"Пустая дорожка: [{self.slot_start}, {self.slot_end}).")
        if self.flag_key is None:
            object.__setattr__(self, "flag_key", self.experiment_id)
        if self.salt is None:
            object.__setattr__(self, "salt", self.experiment_id)
        if not self.flag_key or not self.salt:
            raise ValueError("flag_key и salt не могут быть пустыми.")
        return self

    @property
    def control_variant_id(self) -> str:
        return next(v.variant_id for v in self.variants if v.is_control)

    def params_of(self, variant_id: str) -> Dict[str, Any]:
        """Параметры варианта (PRM-1); для неизвестного варианта — пусто."""
        return next((v.params for v in self.variants if v.variant_id == variant_id), {})

    def overlaps(self, other: "ExperimentRule") -> bool:
        return self.layer_id == other.layer_id and self.slot_start < other.slot_end and other.slot_start < self.slot_end


def lane_conflicts(rule: ExperimentRule, others: List[ExperimentRule]) -> List[str]:
    """Тексты конфликтов rule с другими правилами: дорожки, соли, идентификаторы."""
    out = []
    for o in others:
        if o.experiment_id == rule.experiment_id:
            out.append(f"experiment_id '{rule.experiment_id}' уже есть в ruleset")
            continue
        if o.flag_key == rule.flag_key:
            out.append(f"flag_key '{rule.flag_key}' уже занят экспериментом '{o.experiment_id}'")
        if o.salt == rule.salt:
            out.append(f"соль '{rule.salt}' уже использует эксперимент '{o.experiment_id}' — варианты будут зависимы")
        if rule.overlaps(o):
            out.append(
                f"дорожка [{rule.slot_start}, {rule.slot_end}) эксперимента '{rule.experiment_id}' пересекается "
                f"с '{o.experiment_id}' [{o.slot_start}, {o.slot_end}) в слое '{rule.layer_id}'"
            )
    return out


class Ruleset(BaseModel):
    """Версия ruleset. Поиск правила по flag_key — O(1) через by_flag."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: int = Field(..., ge=1)
    holdout: HoldoutConfig = HoldoutConfig()
    layers: List[Layer] = Field(default_factory=list)
    experiments: List[ExperimentRule] = Field(default_factory=list)
    _layers: Dict[str, Layer] = PrivateAttr(default_factory=dict)
    _flags: Dict[str, ExperimentRule] = PrivateAttr(default_factory=dict)
    _by_layer: Dict[str, List[ExperimentRule]] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _drop_etag(cls, data: Any) -> Any:
        # etag вычисляется из содержимого; во входном JSON допускается, но не хранится
        if isinstance(data, dict) and "etag" in data:
            data = {k: v for k, v in data.items() if k != "etag"}
        return data

    @model_validator(mode="after")
    def _check_consistency(self):
        layer_ids = [l.layer_id for l in self.layers]
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(f"Повторяющиеся layer_id: {layer_ids}.")
        layer_salts = [l.salt for l in self.layers]
        if len(set(layer_salts)) != len(layer_salts):
            raise ValueError("Соли слоёв должны различаться, иначе слои не ортогональны.")
        if set(layer_salts) & {e.salt for e in self.experiments} or self.holdout.salt in set(layer_salts) | {
            e.salt for e in self.experiments
        }:
            raise ValueError("Соли holdout, слоёв и экспериментов не должны совпадать.")
        problems = [f"эксперимент '{e.experiment_id}' ссылается на неизвестный слой '{e.layer_id}'"
                    for e in self.experiments if e.layer_id not in layer_ids]
        by_layer: Dict[str, List[ExperimentRule]] = {}
        for e in self.experiments:
            by_layer.setdefault(e.layer_id, []).append(e)
        if self._has_conflicts(by_layer):
            # подробные тексты — попарным сравнением; до него доходит только некорректный ruleset
            for i, e in enumerate(self.experiments):
                problems += lane_conflicts(e, self.experiments[:i])
        if problems:
            raise ValueError("Некорректный ruleset: " + "; ".join(problems))
        self._layers = {l.layer_id: l for l in self.layers}
        self._flags = {e.flag_key: e for e in self.experiments}
        self._by_layer = by_layer
        return self

    def _has_conflicts(self, by_layer: Dict[str, List[ExperimentRule]]) -> bool:
        """Есть ли повторы идентификаторов, флагов, солей или пересечения дорожек — за O(n log n)."""
        n = len(self.experiments)
        if len({e.experiment_id for e in self.experiments}) != n or len({e.flag_key for e in self.experiments}) != n \
                or len({e.salt for e in self.experiments}) != n:
            return True
        for rules in by_layer.values():
            lanes = sorted((e.slot_start, e.slot_end) for e in rules)
            if any(prev_end > start for (_, prev_end), (start, _) in zip(lanes, lanes[1:])):
                return True
        return False

    def rules_by_layer(self) -> Dict[str, List[ExperimentRule]]:
        return self._by_layer

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json", exclude={"etag"}), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    @computed_field
    @property
    def etag(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:32]

    def by_flag(self, flag_key: str) -> Optional[ExperimentRule]:
        return self._flags.get(flag_key)

    def layer(self, layer_id: str) -> Layer:
        return self._layers[layer_id]

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, text: str) -> "Ruleset":
        return cls.model_validate_json(text)

    def next_version(self, **changes: Dict[str, Any]) -> "Ruleset":
        """Новая версия с изменениями; проходит все валидаторы заново."""
        data = self.model_dump(exclude={"etag"})
        data.update(changes)
        data["version"] = self.version + 1
        return Ruleset.model_validate(data)
