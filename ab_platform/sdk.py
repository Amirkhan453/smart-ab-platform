"""
Эталонный Python SDK гостевого домена (ТЗ v4: ASG-5, ASG-7, 10.2).

- Решение считается локально по ruleset (без сети), p99 ≤ 1 мс.
- Ruleset обновляется по ETag (If-None-Match → 304) с периодом ≤ 30 с — kill-switch доходит ≤ 60 с.
- Fail-open: при любой ошибке, неизвестном флаге, отсутствии или устаревании ruleset отдаётся контроль
  с причиной; доля таких ответов видна в метриках (fallback_rate).
- Exposure создаётся только для гостя в тесте, один раз на (юнит, эксперимент, сессию) в процессе,
  копится в буфере и уходит пачками в sink.
"""
import threading
import time
import uuid
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from ab_platform.engine.ruleset_engine import evaluate
from ab_platform.schemas.ruleset import Ruleset

FALLBACK_NO_RULESET = "fallback:no_ruleset"
FALLBACK_STALE = "fallback:stale_ruleset"
FALLBACK_UNKNOWN_FLAG = "fallback:unknown_flag"
FALLBACK_INVALID_UNIT = "fallback:invalid_unit_id"


class RulesetFetcher(Protocol):
    """Источник ruleset. fetch(etag) → (304, None, etag) без изменений или (200, json, новый etag)."""

    def fetch(self, if_none_match: Optional[str]) -> Tuple[int, Optional[str], Optional[str]]: ...


@dataclass(frozen=True)
class Decision:
    flag_key: str
    unit_id: str
    variant_id: str
    in_experiment: bool
    reason: str
    experiment_id: Optional[str]
    ruleset_version: Optional[int]


@dataclass(frozen=True)
class ExposureEvent:
    event_id: str
    event_time: str
    unit_id: str
    experiment_id: str
    variant_id: str
    ruleset_version: int
    session_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentClient:
    """
    Клиент SDK. Типовой цикл: client = ExperimentClient(fetcher, sink); client.refresh(); client.start();
    variant = client.get_variant("flag", guest_id, session_id=...).variant_id; ... client.close().
    """

    def __init__(
        self,
        fetcher: Optional[RulesetFetcher] = None,
        exposure_sink: Optional[Callable[[List[dict]], None]] = None,
        initial_ruleset: Optional[Ruleset] = None,
        refresh_interval_s: float = 30.0,
        max_stale_s: float = 300.0,
        batch_size: int = 100,
        max_buffer: int = 100_000,
        dedup_capacity: int = 1_000_000,
        clock: Callable[[], float] = time.monotonic,
    ):
        if refresh_interval_s <= 0 or refresh_interval_s > 30:
            raise ValueError("refresh_interval_s должен быть в (0, 30] с — иначе kill-switch не уложится в 60 с.")
        if max_stale_s < refresh_interval_s:
            raise ValueError("max_stale_s не может быть меньше refresh_interval_s.")
        self._fetcher = fetcher
        self._sink = exposure_sink
        self._clock = clock
        self.refresh_interval_s = refresh_interval_s
        self.max_stale_s = max_stale_s
        self._batch_size = batch_size
        self._max_buffer = max_buffer
        self._dedup_capacity = dedup_capacity

        self._ruleset: Optional[Ruleset] = initial_ruleset
        self._fetched_at: Optional[float] = clock() if initial_ruleset is not None else None
        self._lock = threading.Lock()
        self._count_lock = threading.Lock()
        self._buffer: List[ExposureEvent] = []
        self._seen: "OrderedDict[tuple, None]" = OrderedDict()
        self.decisions: Counter = Counter()
        self.counters: Counter = Counter()  # refresh_ok / not_modified / refresh_error / sent / sink_error / dropped
        self.last_refresh_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- ruleset ------------------------------------------------------------------------------------

    @property
    def ruleset(self) -> Optional[Ruleset]:
        return self._ruleset

    def refresh(self) -> bool:
        """Один запрос к источнику. True — ruleset заменён. Ошибки не выбрасываются: старый ruleset остаётся."""
        if self._fetcher is None:
            return False
        current = self._ruleset
        try:
            status, body, _etag = self._fetcher.fetch(current.etag if current else None)
            if status == 304 and current is not None:
                self._fetched_at = self._clock()
                self.counters["not_modified"] += 1
                return False
            if status != 200 or body is None:
                raise RuntimeError(f"статус {status}")
            new = Ruleset.from_json(body)
            if current is not None and new.version < current.version:
                raise RuntimeError(f"версия {new.version} старше текущей {current.version}")
            self._ruleset = new
            self._fetched_at = self._clock()
            self.counters["refresh_ok"] += 1
            return True
        except Exception as exc:  # fail-open: остаёмся на прежнем ruleset
            self.counters["refresh_error"] += 1
            self.last_refresh_error = f"{type(exc).__name__}: {exc}"
            return False

    def ruleset_age_s(self) -> Optional[float]:
        return None if self._fetched_at is None else self._clock() - self._fetched_at

    def start(self) -> None:
        """Фоновое обновление ruleset и отправка exposure каждые refresh_interval_s."""
        if self._thread is not None:
            return
        self._stop.clear()

        def loop():
            while not self._stop.wait(self.refresh_interval_s):
                self.refresh()
                self.flush()

        self._thread = threading.Thread(target=loop, name="ab-sdk-refresh", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.flush()

    # --- решение -----------------------------------------------------------------------------------

    def _fallback(self, flag_key, unit_id, reason, default, rule=None, version=None) -> Decision:
        variant = rule.control_variant_id if rule is not None else default
        self._count(reason)
        return Decision(flag_key, str(unit_id), variant, False, reason,
                        rule.experiment_id if rule is not None else None, version)

    def get_variant(self, flag_key: str, unit_id: str, session_id: Optional[str] = None,
                    default: str = "control") -> Decision:
        """
        Вариант гостя. Никогда не бросает исключений. default — что отдать, если ruleset или флаг неизвестны
        (контрольный вариант известного флага берётся из ruleset).
        """
        rs = self._ruleset
        if not unit_id or not isinstance(unit_id, str):
            return self._fallback(flag_key, unit_id, FALLBACK_INVALID_UNIT, default)
        if rs is None:
            return self._fallback(flag_key, unit_id, FALLBACK_NO_RULESET, default)
        rule = rs.by_flag(flag_key)
        if rule is None:
            return self._fallback(flag_key, unit_id, FALLBACK_UNKNOWN_FLAG, default, version=rs.version)
        age = self.ruleset_age_s()
        if self._fetcher is not None and age is not None and age > self.max_stale_s:
            return self._fallback(flag_key, unit_id, FALLBACK_STALE, default, rule, rs.version)
        try:
            res = evaluate(rs, rule, unit_id)
        except Exception as exc:
            return self._fallback(flag_key, unit_id, f"fallback:exception:{type(exc).__name__}", default, rule,
                                  rs.version)
        self._count(res.reason)
        if res.is_exposed:
            self._log_exposure(unit_id, rule.experiment_id, res.variant_id, rs.version, session_id)
        return Decision(flag_key, unit_id, res.variant_id, res.is_exposed, res.reason, rule.experiment_id,
                        rs.version)

    def _count(self, reason: str) -> None:
        with self._count_lock:
            self.decisions[reason] += 1

    @property
    def fallback_rate(self) -> float:
        total = sum(self.decisions.values())
        fb = sum(n for r, n in self.decisions.items() if r.startswith("fallback:"))
        return fb / total if total else 0.0

    def metrics(self) -> Dict[str, object]:
        rs = self._ruleset
        return {
            "decisions": dict(self.decisions),
            "fallback_rate": self.fallback_rate,
            "ruleset_version": rs.version if rs else None,
            "ruleset_etag": rs.etag if rs else None,
            "ruleset_age_s": self.ruleset_age_s(),
            "buffered_exposures": len(self._buffer),
            **dict(self.counters),
        }

    # --- exposure ----------------------------------------------------------------------------------

    def _log_exposure(self, unit_id, experiment_id, variant_id, version, session_id) -> None:
        key = (unit_id, experiment_id, session_id)
        with self._lock:
            if key in self._seen:
                self._seen.move_to_end(key)
                return
            self._seen[key] = None
            if len(self._seen) > self._dedup_capacity:
                self._seen.popitem(last=False)
            if len(self._buffer) >= self._max_buffer:
                self._buffer.pop(0)
                self.counters["dropped"] += 1
            self._buffer.append(ExposureEvent(
                event_id=str(uuid.uuid4()),
                event_time=datetime.now(timezone.utc).isoformat(),
                unit_id=unit_id, experiment_id=experiment_id, variant_id=variant_id,
                ruleset_version=version, session_id=session_id,
            ))
            # отправка на границе пачки: после сбоя приёма не долбим его на каждом событии
            full = len(self._buffer) % self._batch_size == 0
        if full:
            self.flush()

    def flush(self) -> int:
        """Отправить буфер в sink. При ошибке sink события возвращаются в буфер (повтор идемпотентен по event_id)."""
        if self._sink is None:
            return 0
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return 0
        try:
            self._sink([e.to_dict() for e in batch])
            self.counters["sent"] += len(batch)
            return len(batch)
        except Exception:
            self.counters["sink_error"] += 1
            with self._lock:
                merged = batch + self._buffer
                self.counters["dropped"] += max(0, len(merged) - self._max_buffer)
                self._buffer = merged[-self._max_buffer:]
            return 0

    def pending_exposures(self) -> List[ExposureEvent]:
        with self._lock:
            return list(self._buffer)
