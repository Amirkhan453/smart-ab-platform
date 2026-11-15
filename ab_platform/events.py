"""
Хранилище событий гостевого домена (ТЗ v4.1: CNV-1, DEC-2, TGT-3; ТЗ v4: 10.2).

Четыре вида событий:
- показы (exposure) — от приложения, идемпотентно по event_id;
- заказы — от сервера заказов, идемпотентно по order_id. Повтор с тем же содержимым — дубль, с другим —
  конфликт: заказ не переписывается, отличие возвращается с причиной;
- корректировки заказов (cancel — отмена целиком, refund — частичный возврат) — по adjustment_id;
  в метриках учитываются по дате исходного заказа (ТЗ v4, раздел 8);
- события приложения (app_crash, payment_attempt, payment_error) — по event_id, для защитных метрик.

Время событий хранится в UTC в формате фиксированной длины: строковый порядок совпадает с временным.
Каждая принятая запись получает сквозной номер seq. Срез «всё, что принято до номера N» воспроизводим:
поздние события в него не попадают. На этом держится воспроизводимость отчёта (RPT-1).

EventStore — интерфейс. Реализации: MemoryEventStore (стенды) и SqliteEventStore (файл); ClickHouse (фаза 6)
обязан пройти те же контрактные тесты tests/test_phase4_events.py. ExposureStore — прежний стенд с jsonl
для показов (фаза 2), теперь на основе MemoryEventStore.
"""
import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, Iterator, List, Literal, Optional, Sequence, Set, Tuple, Union

import pandas as pd
from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator,
)

from ab_platform.schemas.targeting import ATTRIBUTES

UnitType = Literal["guest_id", "device_id"]
APP_EVENT_TYPES = ("app_crash", "payment_attempt", "payment_error")
MAX_ITEMS = 200
SCHEMA_VERSION = 1


def utc_text(t: datetime) -> str:
    """Время в UTC, фиксированная длина: '2026-11-10T10:00:00.000000+00:00'."""
    return t.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _canonical(row: Dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExposureIn(_Event):
    """Показ: event_id, event_time (с часовым поясом) и ruleset_version обязательны."""
    event_id: str = Field(..., min_length=1, max_length=128)
    event_time: AwareDatetime
    unit_id: str = Field(..., min_length=1, max_length=1024)
    experiment_id: str = Field(..., min_length=1)
    variant_id: str = Field(..., min_length=1)
    ruleset_version: int = Field(..., ge=1)
    session_id: Optional[str] = None
    unit_type: Optional[UnitType] = None
    attributes: Optional[Dict[str, Union[StrictBool, StrictStr]]] = Field(
        None, description="Снимок признаков таргетинга на момент показа — для SRM по сегментам (TGT-5)"
    )

    @field_validator("attributes")
    @classmethod
    def _only_targeting_attributes(cls, v):
        if v is not None and (set(v) - set(ATTRIBUTES) or any(isinstance(x, str) and len(x) > 64 for x in v.values())):
            raise ValueError(f"допустимы только признаки {list(ATTRIBUTES)} длиной до 64 знаков")
        return v

    def row(self) -> Dict[str, Any]:
        return {**self.model_dump(mode="json"), "event_time": utc_text(self.event_time)}


class OrderItem(_Event):
    sku: str = Field(..., min_length=1, max_length=64)
    qty: int = Field(..., ge=1, le=1000)
    price: float = Field(..., ge=0.0)


class OrderIn(_Event):
    """Заказ приложения. amount — выручка без НДС после скидки, discount — сумма скидки, руб."""
    order_id: str = Field(..., min_length=1, max_length=128)
    unit_id: str = Field(..., min_length=1, max_length=1024)
    unit_type: UnitType
    order_time: AwareDatetime
    amount: float = Field(..., ge=0.0, le=10_000_000.0)
    discount: float = Field(0.0, ge=0.0, le=10_000_000.0)
    items: Optional[List[OrderItem]] = Field(None, max_length=MAX_ITEMS)

    def row(self) -> Dict[str, Any]:
        return {**self.model_dump(mode="json"), "order_time": utc_text(self.order_time)}


class OrderAdjustmentIn(_Event):
    """
    Корректировка заказа: cancel — заказ отменён целиком (amount не задаётся); refund — частичный возврат
    на amount руб. Учитывается по дате исходного заказа.
    """
    adjustment_id: str = Field(..., min_length=1, max_length=128)
    order_id: str = Field(..., min_length=1, max_length=128)
    kind: Literal["cancel", "refund"]
    amount: Optional[float] = Field(None, gt=0.0, le=10_000_000.0)
    event_time: AwareDatetime

    @model_validator(mode="after")
    def _amount_by_kind(self):
        if self.kind == "refund" and self.amount is None:
            raise ValueError("для refund нужна сумма возврата amount")
        if self.kind == "cancel" and self.amount is not None:
            raise ValueError("cancel отменяет заказ целиком: amount не задаётся")
        return self

    def row(self) -> Dict[str, Any]:
        return {**self.model_dump(mode="json"), "event_time": utc_text(self.event_time)}


class AppEventIn(_Event):
    """Событие приложения для защитных метрик (MET-2). Список типов закрытый."""
    event_id: str = Field(..., min_length=1, max_length=128)
    unit_id: str = Field(..., min_length=1, max_length=1024)
    unit_type: UnitType
    event_type: Literal["app_crash", "payment_attempt", "payment_error"]
    event_time: AwareDatetime

    def row(self) -> Dict[str, Any]:
        return {**self.model_dump(mode="json"), "event_time": utc_text(self.event_time)}


class IngestResult(BaseModel):
    accepted: int = 0
    duplicates: int = 0
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)


# имя вида → (поле ключа, колонки таблицы); порядок колонок — порядок в DataFrame
KINDS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "exposures": ("event_id", ("event_id", "event_time", "unit_id", "unit_type", "experiment_id", "variant_id",
                               "ruleset_version", "session_id", "attributes")),
    "orders": ("order_id", ("order_id", "unit_id", "unit_type", "order_time", "amount", "discount", "items")),
    "adjustments": ("adjustment_id", ("adjustment_id", "order_id", "kind", "amount", "event_time")),
    "app_events": ("event_id", ("event_id", "unit_id", "unit_type", "event_type", "event_time")),
}
_JSON_COLS = {"attributes", "items"}


def _frame(kind: str, rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    cols = ("seq",) + KINDS[kind][1]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows], columns=list(cols))


class MemoryEventStore:
    """Хранилище событий в памяти: те же правила, что у SqliteEventStore; после перезапуска пусто."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rows: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k in KINDS}
        self._bodies: Dict[str, Dict[str, str]] = {k: {} for k in KINDS}
        self._seq = 0
        self._by_unit: Dict[str, Set[str]] = {}

    # ---------------- запись ----------------
    def _ingest(self, kind: str, events: Sequence[BaseModel], conflicts_matter: bool) -> IngestResult:
        key_field = KINDS[kind][0]
        res, new = IngestResult(), []
        with self._lock:
            for i, e in enumerate(events):
                row = e.row()
                key, body = row[key_field], _canonical(row)
                old = self._bodies[kind].get(key)
                if old is not None:
                    if old == body or not conflicts_matter:
                        res.duplicates += 1
                    else:
                        res.conflicts.append({"index": i, key_field: key,
                                              "error": f"{key_field} '{key}' уже принят с другим содержимым"})
                    continue
                self._seq += 1
                stored = {"seq": self._seq, **row}
                self._rows[kind][key], self._bodies[kind][key] = stored, body
                if kind == "exposures":
                    self._by_unit.setdefault(row["unit_id"], set()).add(row["experiment_id"])
                new.append(stored)
                res.accepted += 1
            self._after_write(kind, new)
        return res

    def _after_write(self, kind: str, rows: List[Dict[str, Any]]) -> None:
        """Точка расширения для ExposureStore (jsonl)."""

    def add(self, events: Sequence[ExposureIn]) -> Tuple[int, int]:
        """Показы: (принято, дублей). Показ с тем же event_id — дубль, даже если содержимое другое (фаза 2)."""
        r = self._ingest("exposures", events, conflicts_matter=False)
        return r.accepted, r.duplicates

    def add_orders(self, orders: Sequence[OrderIn]) -> IngestResult:
        return self._ingest("orders", orders, conflicts_matter=True)

    def add_adjustments(self, adjustments: Sequence[OrderAdjustmentIn]) -> IngestResult:
        return self._ingest("adjustments", adjustments, conflicts_matter=True)

    def add_app_events(self, events: Sequence[AppEventIn]) -> IngestResult:
        return self._ingest("app_events", events, conflicts_matter=True)

    # ---------------- чтение ----------------
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    def _snapshot(self, kind: str, upto_seq: Optional[int]) -> List[Dict[str, Any]]:
        with self._lock:
            rows = list(self._rows[kind].values())
        return [r for r in rows if upto_seq is None or r["seq"] <= upto_seq]

    def frame(self, kind: str, upto_seq: Optional[int] = None) -> pd.DataFrame:
        """События вида kind, принятые до номера upto_seq включительно, в порядке приёма."""
        if kind not in KINDS:
            raise ValueError(f"Неизвестный вид событий '{kind}'; есть {sorted(KINDS)}.")
        return _frame(kind, sorted(self._snapshot(kind, upto_seq), key=lambda r: r["seq"]))

    def counts_by_kind(self) -> Dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._rows.items()}

    def orphan_adjustments(self) -> int:
        """Корректировки, для которых заказ ещё не пришёл: признак задержки или сбоя потока заказов."""
        with self._lock:
            return sum(r["order_id"] not in self._rows["orders"] for r in self._rows["adjustments"].values())

    # совместимость с ExposureStore фазы 2–3
    def __len__(self) -> int:
        with self._lock:
            return len(self._rows["exposures"])

    def events(self) -> List[dict]:
        return [{k: v for k, v in r.items() if k != "seq"}
                for r in sorted(self._snapshot("exposures", None), key=lambda r: r["seq"])]

    def enrolled(self, unit_id: str) -> FrozenSet[str]:
        """Тесты, в которых у гостя уже был показ: для них таргетинг не перепроверяется (TGT-3)."""
        with self._lock:
            return frozenset(self._by_unit.get(unit_id, ()))

    def first_exposures(self, experiment_id: str, upto_seq: Optional[int] = None) -> Dict[str, dict]:
        """
        Первый показ каждого гостя в тесте — состав анализа (TGT-3). Самый ранний по времени события;
        при равном времени — принятый раньше.
        """
        first: Dict[str, Dict[str, Any]] = {}
        for r in self._snapshot("exposures", upto_seq):
            if r["experiment_id"] != experiment_id:
                continue
            cur = first.get(r["unit_id"])
            if cur is None or (r["event_time"], r["seq"]) < (cur["event_time"], cur["seq"]):
                first[r["unit_id"]] = r
        return {u: {k: v for k, v in r.items() if k != "seq"} for u, r in first.items()}

    def counts(self) -> Dict[str, int]:
        """Число показов по (эксперимент, вариант)."""
        out: Dict[str, int] = {}
        for e in self._snapshot("exposures", None):
            k = f"{e['experiment_id']}:{e['variant_id']}"
            out[k] = out.get(k, 0) + 1
        return out


class ExposureStore(MemoryEventStore):
    """Стенд фаз 2–3: события в памяти, показы — ещё и в jsonl; при перезапуске показы читаются из файла."""

    def __init__(self, path: Optional[str] = None):
        super().__init__()
        self.path = path
        self.corrupt_lines = 0
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = ExposureIn.model_validate_json(line)
                    except ValueError:
                        self.corrupt_lines += 1  # недописанная строка при аварийной остановке
                        continue
                    self.add([ev])

    def _after_write(self, kind: str, rows: List[Dict[str, Any]]) -> None:
        if self.path and kind == "exposures" and rows:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("".join(_canonical({k: v for k, v in r.items() if k != "seq"}) + "\n" for r in rows))


_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS exposures (
    seq INTEGER NOT NULL UNIQUE, event_id TEXT PRIMARY KEY, event_time TEXT NOT NULL, unit_id TEXT NOT NULL,
    unit_type TEXT, experiment_id TEXT NOT NULL, variant_id TEXT NOT NULL, ruleset_version INTEGER NOT NULL,
    session_id TEXT, attributes TEXT, body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS exposures_unit ON exposures (unit_id);
CREATE INDEX IF NOT EXISTS exposures_exp ON exposures (experiment_id, unit_id, event_time);
CREATE TABLE IF NOT EXISTS orders (
    seq INTEGER NOT NULL UNIQUE, order_id TEXT PRIMARY KEY, unit_id TEXT NOT NULL, unit_type TEXT NOT NULL,
    order_time TEXT NOT NULL, amount REAL NOT NULL, discount REAL NOT NULL, items TEXT, body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS orders_unit ON orders (unit_id, order_time);
CREATE TABLE IF NOT EXISTS adjustments (
    seq INTEGER NOT NULL UNIQUE, adjustment_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, kind TEXT NOT NULL,
    amount REAL, event_time TEXT NOT NULL, body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS adjustments_order ON adjustments (order_id);
CREATE TABLE IF NOT EXISTS app_events (
    seq INTEGER NOT NULL UNIQUE, event_id TEXT PRIMARY KEY, unit_id TEXT NOT NULL, unit_type TEXT NOT NULL,
    event_type TEXT NOT NULL, event_time TEXT NOT NULL, body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS app_events_unit ON app_events (unit_id, event_time);
"""


class SqliteEventStore(MemoryEventStore):
    """
    События в файле SQLite. Запись пачки — одна транзакция BEGIN IMMEDIATE: номера seq идут подряд без дыр,
    второй процесс с тем же файлом ждёт. Чтение — запросами, без загрузки всего в память.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30.0)
        with self._lock:
            self._conn.executescript(_DDL)
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
                self._conn.execute("INSERT INTO meta (key, value) VALUES ('seq', '0')")
            elif int(row[0]) != SCHEMA_VERSION:
                raise ValueError(f"Схема базы событий версии {row[0]}, код ожидает {SCHEMA_VERSION}: нужна миграция.")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    def _ingest(self, kind: str, events: Sequence[BaseModel], conflicts_matter: bool) -> IngestResult:
        key_field, cols = KINDS[kind]
        res = IngestResult()
        with self._lock, self._tx():
            seq = int(self._conn.execute("SELECT value FROM meta WHERE key = 'seq'").fetchone()[0])
            for i, e in enumerate(events):
                row = e.row()
                key, body = row[key_field], _canonical(row)
                # строки этой же пачки видны: запрос идёт внутри той же транзакции
                found = self._conn.execute(f"SELECT body FROM {kind} WHERE {key_field} = ?", (key,)).fetchone()
                old = found[0] if found else None
                if old is not None:
                    if old == body or not conflicts_matter:
                        res.duplicates += 1
                    else:
                        res.conflicts.append({"index": i, key_field: key,
                                              "error": f"{key_field} '{key}' уже принят с другим содержимым"})
                    continue
                seq += 1
                values = [json.dumps(row[c], ensure_ascii=False, sort_keys=True) if c in _JSON_COLS and row[c] is not None
                          else row[c] for c in cols]
                self._conn.execute(
                    f"INSERT INTO {kind} (seq, {', '.join(cols)}, body) VALUES (?, {', '.join('?' * len(cols))}, ?)",
                    (seq, *values, body))
                res.accepted += 1
            self._conn.execute("UPDATE meta SET value = ? WHERE key = 'seq'", (str(seq),))
        return res

    def last_seq(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT value FROM meta WHERE key = 'seq'").fetchone()[0])

    def _rows(self, kind: str, where: str = "", args: tuple = ()) -> List[Dict[str, Any]]:
        cols = ("seq",) + KINDS[kind][1]
        with self._lock:
            found = self._conn.execute(f"SELECT {', '.join(cols)} FROM {kind} {where} ORDER BY seq", args).fetchall()
        return [{c: (json.loads(v) if c in _JSON_COLS and v is not None else v) for c, v in zip(cols, r)} for r in found]

    def _snapshot(self, kind: str, upto_seq: Optional[int]) -> List[Dict[str, Any]]:
        return self._rows(kind, "" if upto_seq is None else "WHERE seq <= ?", () if upto_seq is None else (upto_seq,))

    def counts_by_kind(self) -> Dict[str, int]:
        with self._lock:
            return {k: self._conn.execute(f"SELECT COUNT(*) FROM {k}").fetchone()[0] for k in KINDS}

    def orphan_adjustments(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM adjustments a WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = a.order_id)"
            ).fetchone()[0]

    def __len__(self) -> int:
        return self.counts_by_kind()["exposures"]

    def enrolled(self, unit_id: str) -> FrozenSet[str]:
        with self._lock:
            found = self._conn.execute("SELECT DISTINCT experiment_id FROM exposures WHERE unit_id = ?", (unit_id,))
            return frozenset(r[0] for r in found.fetchall())

    def first_exposures(self, experiment_id: str, upto_seq: Optional[int] = None) -> Dict[str, dict]:
        where, args = "WHERE experiment_id = ?", (experiment_id,)
        if upto_seq is not None:
            where, args = where + " AND seq <= ?", args + (upto_seq,)
        first: Dict[str, Dict[str, Any]] = {}
        for r in self._rows("exposures", where, args):
            cur = first.get(r["unit_id"])
            if cur is None or (r["event_time"], r["seq"]) < (cur["event_time"], cur["seq"]):
                first[r["unit_id"]] = r
        return {u: {k: v for k, v in r.items() if k != "seq"} for u, r in first.items()}
