"""
Контур управления гостевыми тестами (ТЗ v4.1: LNK-1, LNK-2; ТЗ v4: 2.1, 2.2).

Единственный путь к ruleset — команды над Паспортом: правило строится только из утверждённого Паспорта
(rule_from_passport), статус теста управляет правилом, каждая команда пишется в аудит. Команда выполняется
в одной транзакции хранилища: статус Паспорта, новая версия ruleset и запись аудита сохраняются вместе.

Соответствие статуса и правила:
  Running → правило активно; Paused, Stopped, Analysis, Decision → kill-switch (гости получают контроль);
  Archived → правило убрано, дорожка свободна.
Новая версия Паспорта — новая рандомизация: experiment_id правила = "<id>@v<версия>", он же соль варианта.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ab_platform.engine.ruleset_engine import launch, update_rule
from ab_platform.schemas.passport import (
    Domain, ExperimentPassport, ExperimentType, PassportStatus, RandomizationUnit,
)
from ab_platform.schemas.ruleset import ExperimentRule, Ruleset
from ab_platform.storage import Storage

ROLES = ("owner", "analyst", "coordinator", "finance", "admin", "reader", "sdk")
_UNIT_TYPE = {RandomizationUnit.USER_ID: "guest_id", RandomizationUnit.DEVICE_ID: "device_id"}
# статусы, при которых правило Паспорта находится в ruleset
_IN_RULESET = (PassportStatus.RUNNING, PassportStatus.PAUSED, PassportStatus.STOPPED, PassportStatus.ANALYSIS,
               PassportStatus.DECISION)


class ControlError(ValueError):
    """Команда отклонена. kind: forbidden — нет прав; not_found; conflict — состояние не позволяет; invalid — данные."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(message)


@dataclass(frozen=True)
class Actor:
    subject: str
    role: str

    def __post_init__(self):
        if not self.subject or self.role not in ROLES:
            raise ControlError("invalid", f"Некорректный субъект: '{self.subject}', роль '{self.role}'.")


def rule_id(passport: ExperimentPassport) -> str:
    return f"{passport.experiment_id}@v{passport.version}"


def rule_from_passport(passport: ExperimentPassport, ramp_pct: Optional[float] = None,
                       kill_switch: bool = False) -> ExperimentRule:
    """
    Правило ruleset из утверждённого Паспорта (LNK-1). Отклоняет Паспорт без утверждения, с хэшем,
    не совпадающим с содержимым, и не гостевой домен (ресторанные тесты назначаются таблицей, фаза 5).
    """
    if passport.passport_hash is None or passport.status in (PassportStatus.DRAFT, PassportStatus.REVIEW):
        raise ControlError("conflict", f"Паспорт '{passport.experiment_id}' v{passport.version} не утверждён.")
    if passport.passport_hash != passport.compute_hash():
        raise ControlError("conflict", f"Паспорт '{passport.experiment_id}' v{passport.version} изменён после "
                                       "утверждения: хэш не совпадает с содержимым.")
    if passport.domain != Domain.GUEST or passport.experiment_type != ExperimentType.CLASSIC_AB \
            or passport.randomization_unit not in _UNIT_TYPE:
        raise ControlError("invalid", "Правило для SDK строится только для гостевого A/B (единица user_id или "
                                      "device_id); ресторанные и квазитесты назначаются иначе.")
    return ExperimentRule(
        experiment_id=rule_id(passport),
        flag_key=passport.test_key,
        layer_id=passport.layer_id,
        slot_start=passport.layer_slot_start,
        slot_end=passport.layer_slot_end,
        ramp_pct=passport.ramp_plan[0] if ramp_pct is None else ramp_pct,
        variants=[v.model_dump() for v in passport.variants],
        kill_switch=kill_switch,
        unit_type=_UNIT_TYPE[passport.randomization_unit],
        targeting=passport.targeting.model_dump(),
        passport_id=passport.experiment_id,
        passport_version=passport.version,
        passport_hash=passport.passport_hash,
    )


def verify_ruleset(ruleset: Ruleset, storage: Storage) -> List[str]:
    """
    Сверка ruleset с Паспортами (LNK-1): у каждого правила есть утверждённый Паспорт с тем же хэшем, статус
    Паспорта соответствует правилу, и правило совпадает с построенным из Паспорта. Возвращает список расхождений.
    """
    problems = []
    for rule in ruleset.experiments:
        if rule.passport_id is None or rule.passport_version is None:
            problems.append(f"правило '{rule.experiment_id}' не связано с Паспортом")
            continue
        passport = storage.get_passport(rule.passport_id, rule.passport_version)
        if passport is None:
            problems.append(f"правило '{rule.experiment_id}': Паспорт не найден")
            continue
        if passport.status not in _IN_RULESET:
            problems.append(f"правило '{rule.experiment_id}': Паспорт в статусе '{passport.status.value}'")
            continue
        if rule.kill_switch != (passport.status != PassportStatus.RUNNING):
            problems.append(f"правило '{rule.experiment_id}': kill-switch не соответствует статусу "
                            f"'{passport.status.value}'")
        try:
            expected = rule_from_passport(passport, ramp_pct=rule.ramp_pct, kill_switch=rule.kill_switch)
        except ControlError as exc:
            problems.append(f"правило '{rule.experiment_id}': {exc}")
            continue
        if expected != rule:
            problems.append(f"правило '{rule.experiment_id}' расходится с Паспортом v{passport.version}")
        elif rule.ramp_pct not in passport.ramp_plan:
            problems.append(f"правило '{rule.experiment_id}': ramp-up {rule.ramp_pct}% вне плана {passport.ramp_plan}")
    return problems


class ExperimentService:
    """Команды жизненного цикла. intersection_matrix — запреты одновременного запуска по experiment_id Паспортов."""

    def __init__(self, storage: Storage, intersection_matrix: Optional[Dict[str, Sequence[str]]] = None,
                 default_holdout_pct: float = 5.0):
        self.storage = storage
        self._matrix = {k: list(v) for k, v in (intersection_matrix or {}).items()}
        self._default_holdout_pct = default_holdout_pct

    # ---------------- чтение ----------------
    def ruleset(self) -> Optional[Ruleset]:
        return self.storage.latest_ruleset()

    def get(self, experiment_id: str, version: Optional[int] = None) -> ExperimentPassport:
        passport = self.storage.get_passport(experiment_id, version)
        if passport is None:
            suffix = f" v{version}" if version is not None else ""
            raise ControlError("not_found", f"Паспорт '{experiment_id}'{suffix} не найден.")
        return passport

    # ---------------- права ----------------
    @staticmethod
    def _require(actor: Actor, *roles: str) -> None:
        if actor.role not in roles:
            raise ControlError("forbidden", f"Роль '{actor.role}' не может выполнить это действие; нужна: "
                                            f"{', '.join(roles)}.")

    def _require_owner_or(self, actor: Actor, passport: ExperimentPassport, *roles: str) -> None:
        if actor.role in roles or (actor.role == "owner" and passport.owner == actor.subject):
            return
        raise ControlError("forbidden", f"'{actor.subject}' ({actor.role}) не владелец теста "
                                        f"'{passport.experiment_id}' и не {', '.join(roles)}.")

    # ---------------- слои и holdout (администратор) ----------------
    def _next(self, current: Optional[Ruleset], **changes: Any) -> Ruleset:
        """Следующая версия ruleset; первая версия — с holdout по умолчанию."""
        try:
            if current is None:
                return Ruleset.model_validate({"version": 1, "holdout": {"pct": self._default_holdout_pct}, **changes})
            return current.next_version(**changes)
        except ValueError as exc:
            raise ControlError("invalid", str(exc)) from exc

    def _publish(self, new: Ruleset) -> Ruleset:
        """Сохранить версию ruleset; перед записью — сверка каждого правила с его Паспортом (LNK-1)."""
        problems = verify_ruleset(new, self.storage)
        if problems:
            raise ControlError("conflict", "Ruleset расходится с Паспортами: " + "; ".join(problems))
        self.storage.save_ruleset(new)
        return new

    def add_layer(self, actor: Actor, layer_id: str, marketing: bool = True, kind: str = "platform") -> Ruleset:
        self._require(actor, "admin")
        with self.storage.transaction():
            current = self.storage.latest_ruleset()
            layers = list(current.layers) if current else []
            if any(l.layer_id == layer_id for l in layers):
                raise ControlError("conflict", f"Слой '{layer_id}' уже существует.")
            new = self._publish(self._next(current, layers=[*layers, {"layer_id": layer_id, "marketing": marketing,
                                                                      "kind": kind}]))
            self.storage.append_audit(actor.subject, "layer.add", None,
                                      {"layer_id": layer_id, "marketing": marketing, "kind": kind,
                                       "ruleset_version": new.version})
            return new

    def set_holdout(self, actor: Actor, pct: float) -> Ruleset:
        """Долю holdout можно менять, только пока в ruleset нет тестов: иначе гости сменят группу."""
        self._require(actor, "admin")
        with self.storage.transaction():
            current = self.storage.latest_ruleset()
            if current is not None and current.experiments:
                raise ControlError("conflict", "В ruleset есть тесты: смена holdout изменила бы их аудиторию.")
            salt = current.holdout.salt if current is not None else "global_holdout_v1"
            new = self._publish(self._next(current, holdout={"salt": salt, "pct": pct}))
            self.storage.append_audit(actor.subject, "holdout.set", None, {"pct": pct, "ruleset_version": new.version})
            return new

    # ---------------- Паспорт до утверждения ----------------
    def save_draft(self, actor: Actor, data: Dict[str, Any]) -> ExperimentPassport:
        """
        Создать Паспорт или изменить черновик. Версия, статус, хэш и утверждающий задаются платформой;
        владелец — автор, если он в роли owner, иначе берётся из data.
        """
        self._require(actor, "owner", "analyst")
        data = {k: v for k, v in data.items() if k not in ("version", "status", "passport_hash", "approved_by", "created_at")}
        if actor.role == "owner":
            data["owner"] = actor.subject
        if not data.get("owner"):
            raise ControlError("invalid", "У Паспорта должен быть владелец (owner).")
        if not data.get("experiment_id"):
            raise ControlError("invalid", "Не задан experiment_id.")
        unknown = sorted(set(data) - set(ExperimentPassport.model_fields))
        if unknown:
            raise ControlError("invalid", f"Неизвестные поля Паспорта: {unknown}.")
        with self.storage.transaction():
            latest = self.storage.get_passport(data["experiment_id"])
            if latest is not None and latest.status != PassportStatus.DRAFT:
                raise ControlError("conflict", f"Паспорт '{latest.experiment_id}' v{latest.version} в статусе "
                                               f"'{latest.status.value}': изменения — только новой версией.")
            if latest is not None:
                self._require_owner_or(actor, latest, "analyst")
            try:
                passport = ExperimentPassport(**data, version=latest.version if latest else 1)
            except ValueError as exc:
                raise ControlError("invalid", str(exc)) from exc
            self.storage.save_passport(passport)
            self.storage.append_audit(actor.subject, "passport.update" if latest else "passport.create",
                                      passport.experiment_id, {"version": passport.version})
            return passport

    def new_version(self, actor: Actor, experiment_id: str, changes: Dict[str, Any]) -> ExperimentPassport:
        """Новая версия утверждённого Паспорта (черновик). Идущая версия при этом не меняется."""
        with self.storage.transaction():
            latest = self.get(experiment_id)
            self._require_owner_or(actor, latest, "analyst")
            if latest.status in (PassportStatus.DRAFT, PassportStatus.REVIEW):
                raise ControlError("conflict", f"Версия {latest.version} ещё не утверждена: правьте черновик.")
            try:
                passport = latest.new_version(**changes)
            except ValueError as exc:
                raise ControlError("invalid", str(exc)) from exc
            self.storage.save_passport(passport)
            self.storage.append_audit(actor.subject, "passport.new_version", experiment_id,
                                      {"version": passport.version, "changed": sorted(changes)})
            return passport

    def _transition(self, actor: Actor, passport: ExperimentPassport, to: PassportStatus, action: str,
                    details: Optional[Dict[str, Any]] = None, approve: bool = False) -> ExperimentPassport:
        try:
            new = passport.with_status(to, actor=actor.subject if approve else None)
        except ValueError as exc:
            raise ControlError("conflict", str(exc)) from exc
        self.storage.save_passport(new)
        self.storage.append_audit(actor.subject, action, passport.experiment_id,
                                  {"version": passport.version, "from": passport.status.value, "to": to.value,
                                   **(details or {})})
        return new

    def submit(self, actor: Actor, experiment_id: str) -> ExperimentPassport:
        with self.storage.transaction():
            passport = self.get(experiment_id)
            self._require_owner_or(actor, passport, "analyst")
            return self._transition(actor, passport, PassportStatus.REVIEW, "passport.submit")

    def return_to_draft(self, actor: Actor, experiment_id: str, comment: str = "") -> ExperimentPassport:
        self._require(actor, "analyst")
        with self.storage.transaction():
            return self._transition(actor, self.get(experiment_id), PassportStatus.DRAFT, "passport.return",
                                    {"comment": comment})

    def approve(self, actor: Actor, experiment_id: str) -> ExperimentPassport:
        """Утверждение: только аналитик и только не владелец этого Паспорта (ТЗ v4, 2.1, 2.2)."""
        self._require(actor, "analyst")
        with self.storage.transaction():
            passport = self.get(experiment_id)
            if passport.owner is None:
                raise ControlError("invalid", "У Паспорта нет владельца: утвердить нельзя.")
            if passport.owner == actor.subject:
                raise ControlError("forbidden", "Паспорт не может утвердить его владелец: нужен второй человек.")
            approved = self._transition(actor, passport, PassportStatus.APPROVED, "passport.approve", approve=True)
            self.storage.append_audit(actor.subject, "passport.sealed", experiment_id,
                                      {"version": approved.version, "passport_hash": approved.passport_hash})
            return approved

    # ---------------- запуск и управление правилом ----------------
    def _rule_of(self, ruleset: Optional[Ruleset], experiment_id: str) -> Optional[ExperimentRule]:
        return next((r for r in (ruleset.experiments if ruleset else ()) if r.passport_id == experiment_id), None)

    def _running_passport(self, experiment_id: str) -> tuple:
        """Правило теста в ruleset и его Паспорт (той версии, что в ruleset, а не последней)."""
        ruleset = self.storage.latest_ruleset()
        rule = self._rule_of(ruleset, experiment_id)
        if rule is None:
            raise ControlError("conflict", f"Тест '{experiment_id}' не запущен: правила в ruleset нет.")
        return ruleset, rule, self.get(experiment_id, rule.passport_version)

    def start(self, actor: Actor, experiment_id: str) -> Ruleset:
        """Approved → Running: правило из Паспорта, проверка конфликтов, новая версия ruleset."""
        with self.storage.transaction():
            passport = self.get(experiment_id)
            self._require_owner_or(actor, passport, "admin")
            if passport.status != PassportStatus.APPROVED:
                raise ControlError("conflict", f"Запустить можно только утверждённый Паспорт; сейчас "
                                               f"'{passport.status.value}'.")
            ruleset = self.storage.latest_ruleset()
            if ruleset is None:
                raise ControlError("conflict", "Нет ни одного слоя: администратор должен создать слой.")
            previous = self._rule_of(ruleset, experiment_id)
            if previous is not None:
                old = self.get(experiment_id, previous.passport_version)
                if old.status in (PassportStatus.RUNNING, PassportStatus.PAUSED):
                    raise ControlError("conflict", f"Версия {old.version} теста ещё идёт: сначала остановите её.")
                # прежняя версия остановлена: её правило уступает место новой рандомизации (в той же версии ruleset)
                ruleset = Ruleset.model_validate({
                    **ruleset.model_dump(exclude={"etag"}),
                    "experiments": [e.model_dump() for e in ruleset.experiments
                                    if e.experiment_id != previous.experiment_id]})
                self._close_previous(actor, old)
            rule = rule_from_passport(passport)
            matrix = self._rule_matrix(ruleset, experiment_id)
            try:
                new = launch(rule, ruleset, matrix)
            except ValueError as exc:
                raise ControlError("conflict", str(exc)) from exc
            self._transition(actor, passport, PassportStatus.RUNNING, "experiment.start",
                             {"ruleset_version": new.version, "ramp_pct": rule.ramp_pct, "rule": rule.experiment_id})
            return self._publish(new)

    def _close_previous(self, actor: Actor, old: ExperimentPassport) -> None:
        """Довести остановленную прежнюю версию до Archived: её правило убирается из ruleset."""
        while old.status != PassportStatus.ARCHIVED:
            nxt = {PassportStatus.STOPPED: PassportStatus.ANALYSIS, PassportStatus.ANALYSIS: PassportStatus.DECISION,
                   PassportStatus.DECISION: PassportStatus.ARCHIVED}[old.status]
            old = self._transition(actor, old, nxt, "experiment.superseded", {"reason": "запущена новая версия"})

    def _rule_matrix(self, ruleset: Ruleset, experiment_id: str) -> Dict[str, List[str]]:
        """Матрица запретов в идентификаторах правил: запреты заданы по experiment_id Паспортов."""
        by_passport = {r.passport_id: r.experiment_id for r in ruleset.experiments}
        banned = set(self._matrix.get(experiment_id, ())) | {k for k, v in self._matrix.items() if experiment_id in v}
        candidate = rule_id(self.get(experiment_id))
        return {candidate: [by_passport[b] for b in banned if b in by_passport]}

    def _switch(self, actor: Actor, experiment_id: str, to: PassportStatus, action: str, kill: bool,
                allowed_from: tuple, *roles: str) -> Ruleset:
        with self.storage.transaction():
            ruleset, rule, passport = self._running_passport(experiment_id)
            self._require_owner_or(actor, passport, *roles)
            if passport.status not in allowed_from:
                raise ControlError("conflict", f"Действие недоступно в статусе '{passport.status.value}'.")
            new = update_rule(ruleset, rule.experiment_id, kill_switch=kill)
            self._transition(actor, passport, to, action, {"ruleset_version": new.version})
            return self._publish(new)

    def pause(self, actor: Actor, experiment_id: str) -> Ruleset:
        return self._switch(actor, experiment_id, PassportStatus.PAUSED, "experiment.pause", True,
                            (PassportStatus.RUNNING,), "admin")

    def resume(self, actor: Actor, experiment_id: str) -> Ruleset:
        return self._switch(actor, experiment_id, PassportStatus.RUNNING, "experiment.resume", False,
                            (PassportStatus.PAUSED,), "admin")

    def stop(self, actor: Actor, experiment_id: str) -> Ruleset:
        return self._switch(actor, experiment_id, PassportStatus.STOPPED, "experiment.stop", True,
                            (PassportStatus.RUNNING, PassportStatus.PAUSED), "admin")

    def ramp(self, actor: Actor, experiment_id: str, ramp_pct: float) -> Ruleset:
        """Следующий шаг раскатки: только значение из плана Паспорта и только вверх."""
        with self.storage.transaction():
            ruleset, rule, passport = self._running_passport(experiment_id)
            self._require_owner_or(actor, passport, "admin")
            if passport.status != PassportStatus.RUNNING:
                raise ControlError("conflict", f"Раскатка меняется только у идущего теста; сейчас "
                                               f"'{passport.status.value}'.")
            if ramp_pct not in passport.ramp_plan or ramp_pct <= rule.ramp_pct:
                raise ControlError("conflict", f"Раскатка {ramp_pct}% недопустима: сейчас {rule.ramp_pct}%, план "
                                               f"Паспорта — {passport.ramp_plan}.")
            new = update_rule(ruleset, rule.experiment_id, ramp_pct=ramp_pct)
            self.storage.append_audit(actor.subject, "experiment.ramp", experiment_id,
                                      {"version": passport.version, "from": rule.ramp_pct, "to": ramp_pct,
                                       "ruleset_version": new.version})
            return self._publish(new)

    def advance(self, actor: Actor, experiment_id: str, to: str) -> ExperimentPassport:
        """Stopped → Analysis → Decision → Archived (аналитик). В Archived правило убирается из ruleset."""
        self._require(actor, "analyst")
        try:
            target = PassportStatus(to)
        except ValueError as exc:
            raise ControlError("invalid", f"Неизвестный статус '{to}'.") from exc
        if target not in (PassportStatus.ANALYSIS, PassportStatus.DECISION, PassportStatus.ARCHIVED):
            raise ControlError("invalid", "advance переводит только в analysis, decision или archived.")
        with self.storage.transaction():
            ruleset, rule, passport = self._running_passport(experiment_id)
            details = {}
            if target == PassportStatus.ARCHIVED:
                new = ruleset.next_version(
                    experiments=[e for e in ruleset.experiments if e.experiment_id != rule.experiment_id])
                details["ruleset_version"] = new.version
            moved = self._transition(actor, passport, target, "experiment.advance", details)
            if target == PassportStatus.ARCHIVED:
                self._publish(new)
            return moved
