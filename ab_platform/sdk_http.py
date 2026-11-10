"""
HTTP-транспорт Python SDK: источник ruleset (ETag/304) и отправка exposure на сервер (ab_platform.server),
а также тонкий клиент API решений POST /decide. Вынесен отдельно, чтобы ab_platform.sdk не зависел от httpx.
Ключ доступа передаётся параметром api_key (заголовок Authorization: Bearer); в коде ключи не хранятся.

    client = ExperimentClient(HttpRulesetFetcher(url), HttpExposureSink(url)); client.refresh(); client.start()
"""
import threading
import time
import uuid
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

import httpx

from ab_platform.sdk import Decision, known_attributes

FALLBACK_DECIDE_UNAVAILABLE = "fallback:decide_unavailable"
FALLBACK_STALE_DECISION = "fallback:stale_decision"
FALLBACK_UNKNOWN_FLAG = "fallback:unknown_flag"


class _Http:
    def __init__(self, base_url: str, timeout_s: float, client: Optional[httpx.Client], api_key: Optional[str] = None):
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)
        self._owns = client is None
        self._auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def close(self) -> None:
        if self._owns:
            self._client.close()


class HttpRulesetFetcher(_Http):
    """fetch(etag) → (304, None, etag) или (200, json, etag). Сетевые ошибки пробрасываются — SDK их ловит."""

    def __init__(self, base_url: str = "", timeout_s: float = 5.0, client: Optional[httpx.Client] = None,
                 api_key: Optional[str] = None):
        super().__init__(base_url, timeout_s, client, api_key)

    def fetch(self, if_none_match: Optional[str]) -> Tuple[int, Optional[str], Optional[str]]:
        headers = {**self._auth, **({"If-None-Match": f'"{if_none_match}"'} if if_none_match else {})}
        r = self._client.get("/ruleset", headers=headers)
        etag = r.headers.get("etag")
        etag = etag.removeprefix("W/").strip('"') if etag else None
        return r.status_code, (r.text if r.status_code == 200 else None), etag


class HttpExposureSink(_Http):
    """Отправка пачки exposure. Не-2xx и сетевые ошибки → исключение: SDK вернёт пачку в буфер и повторит."""

    def __init__(self, base_url: str = "", timeout_s: float = 5.0, client: Optional[httpx.Client] = None,
                 api_key: Optional[str] = None):
        super().__init__(base_url, timeout_s, client, api_key)
        self.rejected: List[dict] = []

    def __call__(self, batch: List[dict]) -> None:
        r = self._client.post("/exposures", json=batch, headers=self._auth)
        r.raise_for_status()
        # отклонённые сервером события повтором не исправить — сохраняем для диагностики
        self.rejected.extend(r.json().get("rejected", []))


class RemoteDecisionClient(_Http):
    """
    Тонкий клиент API решений (ТЗ v4.1: DEC-1…3) для приложений без локального SDK.

        client = RemoteDecisionClient(url, api_key=...)
        d = client.decide(guest_id, "guest_id", {"platform": "ios", "app_version": "5.4"})   # один запрос на гостя
        variant = d["flag_menu"]            # Decision: variant_id, params, reason
        client.expose(d["flag_menu"], session_id)   # когда гость реально увидел изменение — и в контроле тоже
        client.flush()

    Fail-open (DEC-3): тайм-аут или сбой → последние решения этого гостя из кэша, если они не старше max_stale_s;
    иначе контроль с причиной fallback:*. Никогда не бросает исключений. cache — словарь или хранилище устройства.
    Показ отправляется только для гостя в тесте, один раз на (гость, эксперимент, сессию).
    """

    def __init__(self, base_url: str = "", api_key: Optional[str] = None, timeout_s: float = 1.0,
                 client: Optional[httpx.Client] = None, cache: Optional[MutableMapping[str, dict]] = None,
                 max_stale_s: float = 3600.0, batch_size: int = 100, max_buffer: int = 10_000,
                 clock: Callable[[], float] = time.time):
        super().__init__(base_url, timeout_s, client, api_key)
        self._cache: MutableMapping[str, dict] = {} if cache is None else cache
        self.max_stale_s = max_stale_s
        self._batch_size, self._max_buffer, self._clock = batch_size, max_buffer, clock
        self._lock = threading.Lock()
        self._buffer: List[dict] = []
        self._seen: "OrderedDict[tuple, None]" = OrderedDict()
        self.decisions: Counter = Counter()
        self.counters: Counter = Counter()  # decide_ok / decide_error / cache_hit / sent / sink_error / dropped
        self.last_error: Optional[str] = None
        self.rejected: List[dict] = []

    def decide(self, unit_id: str, unit_type: str, attributes: Optional[Mapping[str, Any]] = None) -> "RemoteDecisions":
        key = f"{unit_type}:{unit_id}"
        try:
            r = self._client.post("/decide", headers=self._auth, json={
                "unit_id": unit_id, "unit_type": unit_type, "attributes": dict(attributes) if attributes else None})
            r.raise_for_status()
            body = r.json()
            snapshot = {"at": self._clock(), "ruleset_version": body["ruleset_version"], "decisions": body["decisions"]}
            self._cache[key] = snapshot
            self.counters["decide_ok"] += 1
            return self._wrap(unit_id, unit_type, attributes, snapshot, None)
        except Exception as exc:  # noqa: BLE001 — fail-open: приложение не должно упасть из-за платформы
            self.counters["decide_error"] += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
        try:
            snapshot = self._cache.get(key)
        except Exception:  # noqa: BLE001 — сбой хранилища устройства тоже не должен ронять приложение
            snapshot = None
        if not isinstance(snapshot, dict) or "decisions" not in snapshot:
            return self._wrap(unit_id, unit_type, attributes, None, FALLBACK_DECIDE_UNAVAILABLE)
        if self._clock() - snapshot.get("at", 0.0) > self.max_stale_s:
            return self._wrap(unit_id, unit_type, attributes, None, FALLBACK_STALE_DECISION)
        self.counters["cache_hit"] += 1
        return self._wrap(unit_id, unit_type, attributes, snapshot, None, from_cache=True)

    def _wrap(self, unit_id, unit_type, attributes, snapshot, fallback, from_cache=False) -> "RemoteDecisions":
        items: Dict[str, Decision] = {}
        for d in (snapshot or {}).get("decisions", []):
            items[d["flag_key"]] = Decision(d["flag_key"], unit_id, d["variant_id"], d["in_experiment"], d["reason"],
                                            d["experiment_id"], snapshot["ruleset_version"], d.get("params") or {},
                                            unit_type, known_attributes(attributes))
        return RemoteDecisions(self, unit_id, unit_type, items, fallback, from_cache)

    def _count(self, reason: str) -> None:
        with self._lock:
            self.decisions[reason] += 1

    @property
    def fallback_rate(self) -> float:
        total = sum(self.decisions.values())
        return sum(n for r, n in self.decisions.items() if r.startswith("fallback:")) / total if total else 0.0

    def expose(self, decision: Decision, session_id: Optional[str] = None) -> bool:
        """Показ по решению. False — гость вне теста или показ в этой сессии уже учтён."""
        if not decision.in_experiment or decision.experiment_id is None or decision.ruleset_version is None:
            return False
        key = (decision.unit_id, decision.experiment_id, session_id)
        with self._lock:
            if key in self._seen:
                return False
            self._seen[key] = None
            if len(self._seen) > 100_000:
                self._seen.popitem(last=False)
            if len(self._buffer) >= self._max_buffer:
                self._buffer.pop(0)
                self.counters["dropped"] += 1
            self._buffer.append({
                "event_id": str(uuid.uuid4()), "event_time": datetime.now(timezone.utc).isoformat(),
                "unit_id": decision.unit_id, "experiment_id": decision.experiment_id,
                "variant_id": decision.variant_id, "ruleset_version": decision.ruleset_version,
                "session_id": session_id, "unit_type": decision.unit_type, "attributes": decision.attributes,
            })
            full = len(self._buffer) % self._batch_size == 0
        if full:
            self.flush()
        return True

    def flush(self) -> int:
        """Отправить показы. При сбое они остаются в буфере; повтор идемпотентен по event_id."""
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return 0
        try:
            r = self._client.post("/exposures", json=batch, headers=self._auth)
            r.raise_for_status()
            self.rejected.extend(r.json().get("rejected", []))
            self.counters["sent"] += len(batch)
            return len(batch)
        except Exception:  # noqa: BLE001
            self.counters["sink_error"] += 1
            with self._lock:
                merged = batch + self._buffer
                self.counters["dropped"] += max(0, len(merged) - self._max_buffer)
                self._buffer = merged[-self._max_buffer:]
            return 0

    def pending_exposures(self) -> List[dict]:
        with self._lock:
            return list(self._buffer)


class RemoteDecisions:
    """Решения одного гостя. d[flag] или d.get(flag, default) → Decision; неизвестный флаг и сбой → контроль."""

    def __init__(self, client: RemoteDecisionClient, unit_id: str, unit_type: str, items: Dict[str, Decision],
                 fallback: Optional[str], from_cache: bool):
        self._client, self.unit_id, self.unit_type = client, unit_id, unit_type
        self._items, self.fallback, self.from_cache = items, fallback, from_cache

    def get(self, flag_key: str, default: str = "control") -> Decision:
        d = self._items.get(flag_key)
        if d is None:
            reason = self.fallback or FALLBACK_UNKNOWN_FLAG
            d = Decision(flag_key, self.unit_id, default, False, reason, None, None, {}, self.unit_type)
        self._client._count(d.reason)
        return d

    def __getitem__(self, flag_key: str) -> Decision:
        return self.get(flag_key)

    def flags(self) -> List[str]:
        return list(self._items)
