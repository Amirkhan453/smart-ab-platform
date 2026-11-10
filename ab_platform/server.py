"""
Минимальный HTTP-сервер гостевого домена (ТЗ v4: ASG-7, DAT-1 мин.).

- GET /ruleset — текущая версия ruleset с ETag; If-None-Match с тем же ETag → 304 без тела.
- POST /exposures — пачка exposure. Приём идемпотентный по event_id: повтор не меняет счётчики.
  Каждое событие проверяется отдельно: битое событие отклоняется с причиной и не блокирует остальные.
- Хранение — в памяти и (необязательно) в jsonl; при перезапуске индекс event_id восстанавливается из файла.
  ClickHouse — фаза 5.

Запуск: uvicorn "ab_platform.server:create_app" --factory (ruleset публикуется через RulesetStore.publish).
"""
import json
import os
import threading
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError, field_validator

from ab_platform.schemas.ruleset import Ruleset
from ab_platform.schemas.targeting import ATTRIBUTES

MAX_BATCH = 10_000


class RulesetStore:
    """Текущая опубликованная версия ruleset. Публикуются только версии новее текущей."""

    def __init__(self, ruleset: Optional[Ruleset] = None):
        self._lock = threading.Lock()
        self._current: Optional[Tuple[Ruleset, str]] = None
        if ruleset is not None:
            self.publish(ruleset)

    def publish(self, ruleset: Ruleset) -> Ruleset:
        with self._lock:
            if self._current is not None and ruleset.version <= self._current[0].version:
                raise ValueError(
                    f"Версия {ruleset.version} не новее опубликованной {self._current[0].version}."
                )
            self._current = (ruleset, ruleset.to_json())
            return ruleset

    def current(self) -> Optional[Tuple[Ruleset, str]]:
        return self._current


class ExposureIn(BaseModel):
    """Exposure на входе: event_id, event_time (с часовым поясом) и ruleset_version обязательны."""
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(..., min_length=1, max_length=128)
    event_time: AwareDatetime
    unit_id: str = Field(..., min_length=1)
    experiment_id: str = Field(..., min_length=1)
    variant_id: str = Field(..., min_length=1)
    ruleset_version: int = Field(..., ge=1)
    session_id: Optional[str] = None
    unit_type: Optional[Literal["guest_id", "device_id"]] = None
    attributes: Optional[Dict[str, Union[StrictBool, StrictStr]]] = Field(
        None, description="Снимок признаков таргетинга на момент показа — для SRM по сегментам (TGT-5)"
    )

    @field_validator("attributes")
    @classmethod
    def _only_targeting_attributes(cls, v):
        if v is not None and (set(v) - set(ATTRIBUTES) or any(isinstance(x, str) and len(x) > 64 for x in v.values())):
            raise ValueError(f"допустимы только признаки {list(ATTRIBUTES)} длиной до 64 знаков")
        return v


class ExposureStore:
    """Хранилище exposure, идемпотентное по event_id. path — jsonl для сохранения между перезапусками."""

    def __init__(self, path: Optional[str] = None):
        self._lock = threading.Lock()
        self._events: Dict[str, dict] = {}
        self.path = path
        self.corrupt_lines = 0
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                        self._events.setdefault(ev["event_id"], ev)
                    except (ValueError, KeyError, TypeError):
                        self.corrupt_lines += 1  # недописанная строка при аварийной остановке

    def add(self, events: List[ExposureIn]) -> Tuple[int, int]:
        """Возвращает (принято новых, дубликатов). Дубликат внутри пачки тоже считается дубликатом."""
        accepted, duplicates, new = 0, 0, []
        with self._lock:
            for e in events:
                if e.event_id in self._events:
                    duplicates += 1
                    continue
                row = e.model_dump(mode="json")
                self._events[e.event_id] = row
                new.append(row)
                accepted += 1
            if self.path and new:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new))
        return accepted, duplicates

    def __len__(self) -> int:
        return len(self._events)

    def events(self) -> List[dict]:
        with self._lock:
            return list(self._events.values())

    def counts(self) -> Dict[str, int]:
        """Число exposure по (эксперимент, вариант) — то, что идёт в анализ."""
        out: Dict[str, int] = {}
        for e in self.events():
            k = f"{e['experiment_id']}:{e['variant_id']}"
            out[k] = out.get(k, 0) + 1
        return out


def _etag_matches(header: Optional[str], etag: str) -> bool:
    if not header:
        return False
    for tag in header.split(","):
        tag = tag.strip()
        if tag == "*":
            return True
        if tag.startswith("W/"):
            tag = tag[2:]
        if tag.strip('"') == etag:
            return True
    return False


def create_app(ruleset_store: Optional[RulesetStore] = None,
               exposure_store: Optional[ExposureStore] = None) -> FastAPI:
    # is None, а не or: пустое ExposureStore ложно (__len__ == 0) и подменилось бы новым
    rulesets = RulesetStore() if ruleset_store is None else ruleset_store
    exposures = ExposureStore() if exposure_store is None else exposure_store
    app = FastAPI(title="smart-ab-platform: гостевой домен", version="2")
    app.state.rulesets, app.state.exposures = rulesets, exposures

    @app.get("/ruleset")
    def get_ruleset(request: Request) -> Response:
        cur = rulesets.current()
        if cur is None:
            return JSONResponse({"detail": "ruleset ещё не опубликован"}, status_code=503)
        rs, body = cur
        headers = {"ETag": f'"{rs.etag}"', "Cache-Control": "no-cache"}
        if _etag_matches(request.headers.get("if-none-match"), rs.etag):
            return Response(status_code=304, headers=headers)
        return Response(body, media_type="application/json", headers=headers)

    @app.post("/exposures")
    async def post_exposures(request: Request) -> Response:
        try:
            payload: Any = json.loads(await request.body())
        except ValueError:
            return JSONResponse({"detail": "тело не JSON"}, status_code=400)
        if isinstance(payload, dict):
            payload = payload.get("events")
        if not isinstance(payload, list):
            return JSONResponse({"detail": "ожидается список событий или {\"events\": [...]}"}, status_code=400)
        if len(payload) > MAX_BATCH:
            return JSONResponse({"detail": f"пачка больше {MAX_BATCH} событий"}, status_code=413)
        valid, rejected = [], []
        for i, raw in enumerate(payload):
            try:
                valid.append(ExposureIn.model_validate(raw))
            except ValidationError as exc:
                rejected.append({"index": i, "error": "; ".join(
                    f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in exc.errors())})
        accepted, duplicates = exposures.add(valid)
        return JSONResponse({"accepted": accepted, "duplicates": duplicates, "rejected": rejected})

    @app.get("/health")
    def health() -> Dict[str, Any]:
        cur = rulesets.current()
        return {"ruleset_version": cur[0].version if cur else None, "exposures": len(exposures)}

    return app
