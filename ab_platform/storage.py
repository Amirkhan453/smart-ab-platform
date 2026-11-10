"""
Хранилище контура управления (ТЗ v4.1: LNK-3; ТЗ v4: PAS-1, AUD-1): Паспорта и их версии, версии ruleset, аудит.

Storage — интерфейс. Реализации: MemoryStorage (тесты и стенды) и SqliteStorage (файл; перезапуск ничего не теряет).
Реализация на PostgreSQL (фаза 6) обязана пройти те же контрактные тесты tests/test_phase3_storage.py.

Правила, общие для всех реализаций:
- утверждённая версия Паспорта неизменяема: после появления хэша меняется только статус,
  и только по допустимым переходам;
- версии ruleset только добавляются, строго по порядку (v, v+1, …);
- аудит только дополняется; записи связаны хэш-цепочкой, verify_audit находит правку, вставку и удаление;
- transaction() — всё или ничего: смена статуса, новая версия ruleset и запись аудита сохраняются вместе.
"""
import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Protocol

from ab_platform.schemas.passport import ALLOWED_TRANSITIONS, ExperimentPassport
from ab_platform.schemas.ruleset import Ruleset

SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64


class StorageError(ValueError):
    """Запись отклонена хранилищем: нарушено правило неизменяемости или порядка версий."""


class AuditIntegrityError(RuntimeError):
    """Хэш-цепочка аудита нарушена."""


@dataclass(frozen=True)
class AuditRecord:
    seq: int
    ts: str
    actor: str
    action: str
    experiment_id: Optional[str]
    details: Dict[str, Any]
    prev_hash: str
    hash: str


def audit_hash(seq: int, ts: str, actor: str, action: str, experiment_id: Optional[str], details: Dict[str, Any],
               prev_hash: str) -> str:
    payload = json.dumps([seq, ts, actor, action, experiment_id, details, prev_hash], ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_passport_write(old: Optional[ExperimentPassport], new: ExperimentPassport, max_version: int) -> None:
    """Правила записи Паспорта (PAS-1), одинаковые для всех реализаций."""
    if new.passport_hash is not None and new.passport_hash != new.compute_hash():
        raise StorageError(f"Паспорт '{new.experiment_id}' v{new.version}: хэш не совпадает с содержимым.")
    if old is None:
        if new.version != max_version + 1:
            raise StorageError(
                f"Паспорт '{new.experiment_id}': ожидается версия {max_version + 1}, получена {new.version}."
            )
        return
    if old.passport_hash is not None and new.passport_hash != old.passport_hash:
        raise StorageError(
            f"Паспорт '{new.experiment_id}' v{new.version} утверждён и неизменяем; изменения — только новой версией."
        )
    if new.status != old.status and new.status not in ALLOWED_TRANSITIONS[old.status]:
        raise StorageError(
            f"Паспорт '{new.experiment_id}' v{new.version}: переход {old.status.value} → {new.status.value} недопустим."
        )


def _verify_chain(records: List[AuditRecord]) -> None:
    prev, expected_seq = GENESIS_HASH, 1
    for r in records:
        if r.seq != expected_seq:
            raise AuditIntegrityError(f"Аудит: ожидалась запись №{expected_seq}, найдена №{r.seq} — запись удалена или вставлена.")
        if r.prev_hash != prev or r.hash != audit_hash(r.seq, r.ts, r.actor, r.action, r.experiment_id, r.details, prev):
            raise AuditIntegrityError(f"Аудит: запись №{r.seq} изменена или цепочка разорвана.")
        prev, expected_seq = r.hash, expected_seq + 1


class Storage(Protocol):
    def transaction(self): ...
    def save_passport(self, passport: ExperimentPassport) -> None: ...
    def get_passport(self, experiment_id: str, version: Optional[int] = None) -> Optional[ExperimentPassport]: ...
    def list_passports(self) -> List[ExperimentPassport]: ...
    def save_ruleset(self, ruleset: Ruleset) -> None: ...
    def latest_ruleset(self) -> Optional[Ruleset]: ...
    def latest_ruleset_version(self) -> Optional[int]: ...
    def get_ruleset(self, version: int) -> Optional[Ruleset]: ...
    def append_audit(self, actor: str, action: str, experiment_id: Optional[str] = None,
                     details: Optional[Dict[str, Any]] = None) -> AuditRecord: ...
    def audit(self, experiment_id: Optional[str] = None) -> List[AuditRecord]: ...
    def verify_audit(self) -> int: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStorage:
    """Хранилище в памяти: те же правила, что у SqliteStorage; после перезапуска пусто."""

    def __init__(self):
        self._lock = threading.RLock()
        self._passports: Dict[str, Dict[int, ExperimentPassport]] = {}
        self._rulesets: List[Ruleset] = []
        self._audit: List[AuditRecord] = []
        self._depth = 0

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            if self._depth:
                raise StorageError("Вложенные транзакции не поддерживаются.")
            snapshot = ({k: dict(v) for k, v in self._passports.items()}, list(self._rulesets), list(self._audit))
            self._depth = 1
            try:
                yield
            except BaseException:
                self._passports, self._rulesets, self._audit = snapshot
                raise
            finally:
                self._depth = 0

    def save_passport(self, passport: ExperimentPassport) -> None:
        with self._lock:
            versions = self._passports.setdefault(passport.experiment_id, {})
            _check_passport_write(versions.get(passport.version), passport, max(versions, default=0))
            versions[passport.version] = passport

    def get_passport(self, experiment_id: str, version: Optional[int] = None) -> Optional[ExperimentPassport]:
        with self._lock:
            versions = self._passports.get(experiment_id) or {}
            return versions.get(max(versions) if version is None and versions else version)

    def list_passports(self) -> List[ExperimentPassport]:
        with self._lock:
            return [v[max(v)] for _, v in sorted(self._passports.items()) if v]

    def save_ruleset(self, ruleset: Ruleset) -> None:
        with self._lock:
            expected = self._rulesets[-1].version + 1 if self._rulesets else None
            if expected is not None and ruleset.version != expected:
                raise StorageError(f"Ruleset: ожидается версия {expected}, получена {ruleset.version}.")
            self._rulesets.append(ruleset)

    def latest_ruleset(self) -> Optional[Ruleset]:
        with self._lock:
            return self._rulesets[-1] if self._rulesets else None

    def latest_ruleset_version(self) -> Optional[int]:
        with self._lock:
            return self._rulesets[-1].version if self._rulesets else None

    def get_ruleset(self, version: int) -> Optional[Ruleset]:
        with self._lock:
            return next((r for r in self._rulesets if r.version == version), None)

    def append_audit(self, actor: str, action: str, experiment_id: Optional[str] = None,
                     details: Optional[Dict[str, Any]] = None) -> AuditRecord:
        with self._lock:
            seq, prev, ts = len(self._audit) + 1, self._audit[-1].hash if self._audit else GENESIS_HASH, _now()
            details = json.loads(json.dumps(details or {}, ensure_ascii=False))
            rec = AuditRecord(seq, ts, actor, action, experiment_id, details, prev,
                              audit_hash(seq, ts, actor, action, experiment_id, details, prev))
            self._audit.append(rec)
            return rec

    def audit(self, experiment_id: Optional[str] = None) -> List[AuditRecord]:
        with self._lock:
            return [r for r in self._audit if experiment_id is None or r.experiment_id == experiment_id]

    def verify_audit(self) -> int:
        with self._lock:
            _verify_chain(self._audit)
            return len(self._audit)


_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS passports (
    experiment_id TEXT NOT NULL,
    version       INTEGER NOT NULL,
    status        TEXT NOT NULL,
    passport_hash TEXT,
    body          TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (experiment_id, version)
);
CREATE TABLE IF NOT EXISTS rulesets (
    version    INTEGER PRIMARY KEY,
    etag       TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    seq           INTEGER PRIMARY KEY,
    ts            TEXT NOT NULL,
    actor         TEXT NOT NULL,
    action        TEXT NOT NULL,
    experiment_id TEXT,
    details       TEXT NOT NULL,
    prev_hash     TEXT NOT NULL,
    hash          TEXT NOT NULL
);
"""


class SqliteStorage:
    """
    Хранилище в файле SQLite. Одно соединение на процесс под блокировкой; запись — в транзакции BEGIN IMMEDIATE,
    поэтому второй процесс с тем же файлом ждёт, а не пишет поверх. SQL — без особенностей SQLite, кроме DDL.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30.0)
        self._in_tx = False
        with self._lock:
            self._conn.executescript(_DDL)
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif int(row[0]) != SCHEMA_VERSION:
                raise StorageError(f"Схема базы версии {row[0]}, код ожидает {SCHEMA_VERSION}: нужна миграция.")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            if self._in_tx:
                raise StorageError("Вложенные транзакции не поддерживаются.")
            self._conn.execute("BEGIN IMMEDIATE")
            self._in_tx = True
            try:
                yield
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            finally:
                self._in_tx = False

    @contextmanager
    def _write(self) -> Iterator[None]:
        """Одиночная запись: своя транзакция, если внешней нет."""
        with self._lock:
            if self._in_tx:
                yield
            else:
                with self.transaction():
                    yield

    def save_passport(self, passport: ExperimentPassport) -> None:
        with self._write():
            old = self.get_passport(passport.experiment_id, passport.version)
            top = self._conn.execute("SELECT COALESCE(MAX(version), 0) FROM passports WHERE experiment_id = ?",
                                     (passport.experiment_id,)).fetchone()[0]
            _check_passport_write(old, passport, top)
            self._conn.execute(
                "INSERT INTO passports (experiment_id, version, status, passport_hash, body, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (experiment_id, version) DO UPDATE SET "
                "status = excluded.status, passport_hash = excluded.passport_hash, body = excluded.body, "
                "updated_at = excluded.updated_at",
                (passport.experiment_id, passport.version, passport.status.value, passport.passport_hash,
                 passport.model_dump_json(), _now()),
            )

    def get_passport(self, experiment_id: str, version: Optional[int] = None) -> Optional[ExperimentPassport]:
        with self._lock:
            if version is None:
                row = self._conn.execute(
                    "SELECT body FROM passports WHERE experiment_id = ? ORDER BY version DESC LIMIT 1", (experiment_id,)
                ).fetchone()
            else:
                row = self._conn.execute("SELECT body FROM passports WHERE experiment_id = ? AND version = ?",
                                         (experiment_id, version)).fetchone()
        return ExperimentPassport.model_validate_json(row[0]) if row else None

    def list_passports(self) -> List[ExperimentPassport]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.body FROM passports p JOIN (SELECT experiment_id, MAX(version) AS v FROM passports "
                "GROUP BY experiment_id) m ON p.experiment_id = m.experiment_id AND p.version = m.v "
                "ORDER BY p.experiment_id"
            ).fetchall()
        return [ExperimentPassport.model_validate_json(r[0]) for r in rows]

    def save_ruleset(self, ruleset: Ruleset) -> None:
        with self._write():
            top = self._conn.execute("SELECT MAX(version) FROM rulesets").fetchone()[0]
            if top is not None and ruleset.version != top + 1:
                raise StorageError(f"Ruleset: ожидается версия {top + 1}, получена {ruleset.version}.")
            self._conn.execute("INSERT INTO rulesets (version, etag, body, created_at) VALUES (?, ?, ?, ?)",
                               (ruleset.version, ruleset.etag, ruleset.to_json(), _now()))

    def latest_ruleset(self) -> Optional[Ruleset]:
        with self._lock:
            row = self._conn.execute("SELECT body FROM rulesets ORDER BY version DESC LIMIT 1").fetchone()
        return Ruleset.from_json(row[0]) if row else None

    def latest_ruleset_version(self) -> Optional[int]:
        """Дешёвая проверка «не вышла ли новая версия» — без разбора ruleset."""
        with self._lock:
            return self._conn.execute("SELECT MAX(version) FROM rulesets").fetchone()[0]

    def get_ruleset(self, version: int) -> Optional[Ruleset]:
        with self._lock:
            row = self._conn.execute("SELECT body FROM rulesets WHERE version = ?", (version,)).fetchone()
        return Ruleset.from_json(row[0]) if row else None

    def append_audit(self, actor: str, action: str, experiment_id: Optional[str] = None,
                     details: Optional[Dict[str, Any]] = None) -> AuditRecord:
        with self._write():
            last = self._conn.execute("SELECT seq, hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            seq, prev = (last[0] + 1, last[1]) if last else (1, GENESIS_HASH)
            ts = _now()
            details = json.loads(json.dumps(details or {}, ensure_ascii=False))
            rec = AuditRecord(seq, ts, actor, action, experiment_id, details, prev,
                              audit_hash(seq, ts, actor, action, experiment_id, details, prev))
            self._conn.execute(
                "INSERT INTO audit (seq, ts, actor, action, experiment_id, details, prev_hash, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (seq, ts, actor, action, experiment_id, json.dumps(details, ensure_ascii=False, sort_keys=True), prev,
                 rec.hash),
            )
            return rec

    def _audit_rows(self, where: str = "", args: tuple = ()) -> List[AuditRecord]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT seq, ts, actor, action, experiment_id, details, prev_hash, hash FROM audit {where} ORDER BY seq",
                args,
            ).fetchall()
        return [AuditRecord(r[0], r[1], r[2], r[3], r[4], json.loads(r[5]), r[6], r[7]) for r in rows]

    def audit(self, experiment_id: Optional[str] = None) -> List[AuditRecord]:
        if experiment_id is None:
            return self._audit_rows()
        return self._audit_rows("WHERE experiment_id = ?", (experiment_id,))

    def verify_audit(self) -> int:
        records = self._audit_rows()
        _verify_chain(records)
        return len(records)
