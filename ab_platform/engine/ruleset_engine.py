"""
Назначение по ruleset и проверка запуска (ТЗ v4: 4.1, ASG-3, ASG-4).

evaluate — одна точка вычисления варианта для SDK и сервера: holdout действует только на маркетинговые слои,
kill-switch отдаёт контроль, таргетинг проверяется после дорожки и до ramp-up (ТЗ v4.1, TGT-2).
evaluate_all — то же по всем правилам за один проход, для API решений.
validate_launch отклоняет запуск, конфликтующий по дорожкам, солям или матрице.
"""
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence

from ab_platform.engine.assignment import (
    REASON_HOLDOUT, REASON_NOT_IN_LANE, AssignmentResult, DeterministicAssignmentEngine, bucket, pct_to_buckets,
)
from ab_platform.schemas.ruleset import ExperimentRule, Ruleset, lane_conflicts

REASON_KILL_SWITCH = "kill_switch"
REASON_UNIT_TYPE = "unit_type_mismatch"


class LaunchConflictError(ValueError):
    """Запуск эксперимента отклонён; conflicts — список конфликтов обычным языком."""

    def __init__(self, experiment_id: str, conflicts: List[str]):
        self.experiment_id = experiment_id
        self.conflicts = conflicts
        super().__init__(f"Запуск '{experiment_id}' отклонён: " + "; ".join(conflicts))


def _outside(rule: ExperimentRule, unit_id: str, reason: str, z: int = -1, in_holdout: bool = False) -> AssignmentResult:
    return AssignmentResult(unit_id, rule.experiment_id, rule.layer_id, in_holdout, False, rule.control_variant_id, z,
                            reason)


def evaluate(
    ruleset: Ruleset,
    rule: ExperimentRule,
    unit_id: str,
    attributes: Optional[Mapping[str, Any]] = None,
    unit_type: Optional[str] = None,
    enrolled: bool = False,
    holdout_bucket: Optional[int] = None,
    layer_bucket: Optional[int] = None,
) -> AssignmentResult:
    """
    Вариант гостя по правилу ruleset. Вне теста — контрольный вариант правила и причина.
    Порядок (ТЗ v4.1): kill-switch → тип единицы → holdout → слой и дорожка → таргетинг → ramp-up → вариант.
    unit_type — тип переданного идентификатора; None — вызывающий отвечает за тип сам (локальный SDK).
    enrolled — у гостя уже был первый показ в этом тесте: таргетинг не перепроверяется (TGT-3).
    """
    if rule.kill_switch:
        return _outside(rule, unit_id, REASON_KILL_SWITCH)
    if unit_type is not None and unit_type != rule.unit_type:
        return _outside(rule, unit_id, REASON_UNIT_TYPE)
    layer = ruleset.layer(rule.layer_id)
    return DeterministicAssignmentEngine.assign_user(
        user_id=unit_id,
        experiment_id=rule.experiment_id,
        exposure_rate_pct=rule.ramp_pct,
        global_holdout_pct=ruleset.holdout.pct if layer.marketing else 0.0,
        holdout_salt=ruleset.holdout.salt,
        shared_hash_id=rule.salt,
        layer_id=rule.layer_id,
        layer_salt=layer.salt,
        layer_slot_start=rule.slot_start,
        layer_slot_end=rule.slot_end,
        variants=rule.variants,
        ineligible_reason=None if enrolled else rule.targeting.mismatch(attributes),
        holdout_bucket=holdout_bucket,
        layer_bucket=layer_bucket,
    )


def evaluate_all(
    ruleset: Ruleset,
    unit_id: str,
    attributes: Optional[Mapping[str, Any]] = None,
    unit_type: Optional[str] = None,
    enrolled: Collection[str] = (),
) -> List[AssignmentResult]:
    """
    Решения гостя по всем правилам ruleset в порядке ruleset.experiments — для POST /decide (DEC-1).
    Результат совпадает с evaluate по каждому правилу; holdout и бакет слоя считаются один раз,
    полный расчёт идёт только для правила, в дорожку которого гость попал.
    enrolled — experiment_id тестов, где у гостя уже был первый показ.
    """
    holdout_z = bucket(ruleset.holdout.salt, unit_id)  # заодно проверка unit_id до любых решений
    holdout_buckets = pct_to_buckets(ruleset.holdout.pct)
    out: Dict[str, AssignmentResult] = {}
    for layer_id, rules in ruleset.rules_by_layer().items():
        layer = ruleset.layer(layer_id)
        in_holdout = layer.marketing and holdout_z < holdout_buckets
        layer_z = None if in_holdout else bucket(layer.salt, unit_id)
        for rule in rules:
            if rule.kill_switch:
                res = _outside(rule, unit_id, REASON_KILL_SWITCH)
            elif unit_type is not None and unit_type != rule.unit_type:
                res = _outside(rule, unit_id, REASON_UNIT_TYPE)
            elif in_holdout:
                res = _outside(rule, unit_id, REASON_HOLDOUT, holdout_z, in_holdout=True)
            elif not (rule.slot_start <= layer_z < rule.slot_end):
                res = _outside(rule, unit_id, REASON_NOT_IN_LANE, layer_z)
            else:
                res = evaluate(ruleset, rule, unit_id, attributes, unit_type, rule.experiment_id in enrolled,
                               holdout_bucket=holdout_z, layer_bucket=layer_z)
            out[rule.experiment_id] = res
    return [out[e.experiment_id] for e in ruleset.experiments]


def validate_launch(
    candidate: ExperimentRule,
    ruleset: Ruleset,
    intersection_matrix: Optional[Dict[str, Sequence[str]]] = None,
) -> None:
    """
    ASG-3: проверка запуска candidate поверх действующего ruleset.
    intersection_matrix — запреты одновременного запуска {experiment_id: [несовместимые experiment_id]},
    действует в обе стороны и независимо от слоёв. Конфликт → LaunchConflictError с перечнем причин.
    """
    conflicts = []
    if candidate.layer_id not in {l.layer_id for l in ruleset.layers}:
        conflicts.append(f"слой '{candidate.layer_id}' не существует")
    if candidate.salt in {l.salt for l in ruleset.layers} | {ruleset.holdout.salt}:
        conflicts.append(f"соль '{candidate.salt}' совпадает с солью слоя или holdout")
    conflicts += lane_conflicts(candidate, list(ruleset.experiments))
    matrix = intersection_matrix or {}
    for active in ruleset.experiments:
        if active.experiment_id == candidate.experiment_id:
            continue
        if active.experiment_id in set(matrix.get(candidate.experiment_id, ())) or \
                candidate.experiment_id in set(matrix.get(active.experiment_id, ())):
            conflicts.append(
                f"матрица пересечений запрещает одновременный запуск с '{active.experiment_id}'"
            )
    if conflicts:
        raise LaunchConflictError(candidate.experiment_id, conflicts)


def launch(
    candidate: ExperimentRule,
    ruleset: Ruleset,
    intersection_matrix: Optional[Dict[str, Sequence[str]]] = None,
) -> Ruleset:
    """Проверка запуска и новая версия ruleset с candidate."""
    validate_launch(candidate, ruleset, intersection_matrix)
    return ruleset.next_version(experiments=[*ruleset.experiments, candidate])


def update_rule(ruleset: Ruleset, experiment_id: str, **changes) -> Ruleset:
    """
    Новая версия ruleset с изменённым правилом (ramp_pct, kill_switch).
    Ramp-up только растёт, дорожка только расширяется — иначе назначенные гости сменят вариант (ASG-2).
    """
    allowed = {"ramp_pct", "kill_switch", "slot_start", "slot_end"}
    if set(changes) - allowed:
        raise ValueError(f"Менять можно только {sorted(allowed)}; остальное — новый эксперимент.")
    rules = {e.experiment_id: e for e in ruleset.experiments}
    if experiment_id not in rules:
        raise KeyError(f"Эксперимент '{experiment_id}' не найден в ruleset.")
    old = rules[experiment_id]
    new = old.model_copy(update=changes)
    new = ExperimentRule.model_validate(new.model_dump())
    if new.ramp_pct < old.ramp_pct or new.slot_start > old.slot_start or new.slot_end < old.slot_end:
        raise ValueError("Ramp-up и дорожка могут только расти (sticky); для сужения — kill-switch и новый эксперимент.")
    if new.slot_start != old.slot_start and new.ramp_pct < 100.0:
        raise ValueError("Сдвиг начала дорожки при ramp < 100% меняет активный поддиапазон; расширяйте дорожку вправо.")
    others = [e for e in ruleset.experiments if e.experiment_id != experiment_id]
    conflicts = lane_conflicts(new, others)
    if conflicts:
        raise LaunchConflictError(experiment_id, conflicts)
    return ruleset.next_version(experiments=[new if e.experiment_id == experiment_id else e for e in ruleset.experiments])
