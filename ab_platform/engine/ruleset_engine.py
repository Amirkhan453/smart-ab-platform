"""
Назначение по ruleset и проверка запуска (ТЗ v4: 4.1, ASG-3, ASG-4).

evaluate — одна точка вычисления варианта для SDK и сервера: holdout действует только на маркетинговые слои,
kill-switch отдаёт контроль. validate_launch отклоняет запуск, конфликтующий по дорожкам, солям или матрице.
"""
from typing import Dict, List, Optional, Sequence

from ab_platform.engine.assignment import AssignmentResult, DeterministicAssignmentEngine
from ab_platform.schemas.ruleset import ExperimentRule, Ruleset, lane_conflicts

REASON_KILL_SWITCH = "kill_switch"


class LaunchConflictError(ValueError):
    """Запуск эксперимента отклонён; conflicts — список конфликтов обычным языком."""

    def __init__(self, experiment_id: str, conflicts: List[str]):
        self.experiment_id = experiment_id
        self.conflicts = conflicts
        super().__init__(f"Запуск '{experiment_id}' отклонён: " + "; ".join(conflicts))


def evaluate(ruleset: Ruleset, rule: ExperimentRule, unit_id: str) -> AssignmentResult:
    """Вариант гостя по правилу ruleset. Вне теста и при kill-switch — контрольный вариант правила."""
    layer = ruleset.layer(rule.layer_id)
    if rule.kill_switch:
        return AssignmentResult(unit_id, rule.experiment_id, rule.layer_id, False, False,
                                rule.control_variant_id, -1, REASON_KILL_SWITCH)
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
    )


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
