"""
HTTP-сервер гостевого домена (ТЗ v4: ASG-7, DAT-1; ТЗ v4.1: AUTH-1, DEC-1…2, LNK-1…3).

Маршруты SDK (ключ любой роли):
- GET /ruleset — текущая версия ruleset с ETag; If-None-Match с тем же ETag → 304 без тела.
- POST /decide — решения гостя по всем активным тестам: вариант, параметры, причина. Показ не пишет (DEC-2):
  его отправляет приложение в момент, когда гость реально увидел изменение — и в контроле тоже.
- POST /exposures — пачка показов. Приём идемпотентный по event_id; битое событие отклоняется с причиной
  и не блокирует остальные.
Маршруты управления (ключи людей, права — по ролям ТЗ 2.1): Паспорта, команды жизненного цикла, слои, аудит.
Маршрута «опубликовать ruleset» нет: правило появляется только из утверждённого Паспорта (LNK-1).

Показы хранятся в памяти и (необязательно) в jsonl; общее хранилище событий — фаза 4, ClickHouse — фаза 6.

Запуск: uvicorn "ab_platform.server:create_app_from_env" --factory
  ABP_DB_PATH — файл базы контура управления; ABP_API_KEYS — ключи (см. ab_platform/auth.py);
  ABP_EXPOSURES_PATH — jsonl показов (необязательно).
"""
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union

from fastapi import Body, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError, field_validator

from ab_platform.auth import Auth
from ab_platform.control import Actor, ControlError, ExperimentService, verify_ruleset
from ab_platform.engine.ruleset_engine import evaluate_all
from ab_platform.schemas.ruleset import Ruleset
from ab_platform.storage import SqliteStorage
from ab_platform.schemas.targeting import ATTRIBUTES

MAX_BATCH = 10_000
MAX_UNIT_ID_LENGTH = 1024


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
        self._by_unit: Dict[str, Set[str]] = {}  # гость → тесты, где у него уже был показ (TGT-3)
        self.path = path
        self.corrupt_lines = 0
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                        if ev["event_id"] not in self._events:
                            self._events[ev["event_id"]] = ev
                            self._by_unit.setdefault(ev["unit_id"], set()).add(ev["experiment_id"])
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
                self._by_unit.setdefault(e.unit_id, set()).add(e.experiment_id)
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

    def enrolled(self, unit_id: str) -> FrozenSet[str]:
        """Тесты, в которых у гостя уже был показ: для них таргетинг не перепроверяется (TGT-3)."""
        with self._lock:
            return frozenset(self._by_unit.get(unit_id, ()))

    def first_exposures(self, experiment_id: str) -> Dict[str, dict]:
        """
        Первый показ каждого гостя в тесте (по времени события) — состав анализа (TGT-3): гость входит в тест
        в момент первого показа и остаётся в нём с тем вариантом и теми признаками, что были тогда.
        """
        first: Dict[str, Tuple[datetime, dict]] = {}
        for e in self.events():
            if e["experiment_id"] != experiment_id:
                continue
            t = datetime.fromisoformat(e["event_time"].replace("Z", "+00:00"))
            if e["unit_id"] not in first or t < first[e["unit_id"]][0]:
                first[e["unit_id"]] = (t, e)
        return {u: e for u, (_, e) in first.items()}

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


class DecideIn(BaseModel):
    """Запрос решения: идентификатор, его тип и признаки для таргетинга."""
    model_config = ConfigDict(extra="forbid")
    unit_id: str = Field(..., min_length=1, max_length=MAX_UNIT_ID_LENGTH)
    unit_type: Literal["guest_id", "device_id"]
    attributes: Optional[Dict[str, Any]] = None
    flags: Optional[List[str]] = Field(None, description="Только эти флаги; по умолчанию — все активные тесты")


class _HttpError(Exception):
    def __init__(self, status: int, detail: str, headers: Optional[Dict[str, str]] = None):
        self.status, self.detail, self.headers = status, detail, headers


_CONTROL_STATUS = {"forbidden": 403, "not_found": 404, "conflict": 409, "invalid": 422}


def create_app(ruleset_store: Optional[RulesetStore] = None,
               exposure_store: Optional[ExposureStore] = None,
               service: Optional[ExperimentService] = None,
               auth: Optional[Auth] = None) -> FastAPI:
    """
    service — контур управления: ruleset берётся из его хранилища и включаются маршруты управления.
    Без service ruleset отдаёт ruleset_store (стенды SDK). auth=None — ключи из окружения; без ключей
    и без явного режима стенда приложение не создаётся.
    """
    auth = Auth.from_env() if auth is None else auth
    # is None, а не or: пустое ExposureStore ложно (__len__ == 0) и подменилось бы новым
    rulesets = RulesetStore() if ruleset_store is None else ruleset_store
    exposures = ExposureStore() if exposure_store is None else exposure_store
    source = service if service is not None else rulesets
    app = FastAPI(title="smart-ab-platform: гостевой домен", version="3")
    app.state.rulesets, app.state.exposures, app.state.service, app.state.auth = rulesets, exposures, service, auth

    @app.exception_handler(_HttpError)
    async def _http_error(_: Request, exc: _HttpError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        # без поля input: в нём может быть строка, которую нельзя закодировать в UTF-8 (одиночный суррогат)
        return JSONResponse({"detail": [{"loc": [str(x) for x in e["loc"]], "msg": e["msg"]} for e in exc.errors()]},
                            status_code=422)

    def caller(request: Request) -> Actor:
        """Любой известный ключ — маршруты SDK."""
        actor = auth.authenticate(request.headers.get("authorization"))
        if actor is None:
            raise _HttpError(401, "нужен ключ: заголовок Authorization: Bearer <ключ>", {"WWW-Authenticate": "Bearer"})
        return actor

    def manager(request: Request) -> Actor:
        """Ключ человека — маршруты управления. Ключ SDK и режим стенда сюда не допускаются."""
        if service is None:
            raise _HttpError(404, "контур управления не подключён")
        if auth.is_disabled:
            raise _HttpError(403, "управление недоступно в режиме стенда без ключей")
        actor = caller(request)
        if actor.role == "sdk":
            service.storage.append_audit(actor.subject, "auth.denied", None,
                                         {"path": request.url.path, "reason": "ключ SDK на маршруте управления"})
            raise _HttpError(403, "ключ SDK не даёт доступа к управлению")
        return actor

    def command(request: Request, experiment_id: Optional[str], fn) -> Any:
        """Выполнить команду контура от имени владельца ключа; отказ по правам пишется в аудит."""
        actor = manager(request)
        try:
            return fn(actor)
        except ControlError as exc:
            if exc.kind == "forbidden":
                service.storage.append_audit(actor.subject, "auth.denied", experiment_id,
                                             {"path": request.url.path, "role": actor.role, "reason": str(exc)})
            raise _HttpError(_CONTROL_STATUS[exc.kind], str(exc)) from exc

    def dump(passport) -> Dict[str, Any]:
        return {**passport.model_dump(mode="json"), "design_warnings": passport.design_warnings,
                "target_daily_units": passport.target_daily_units}

    # ---------------- маршруты SDK ----------------
    @app.get("/ruleset")
    def get_ruleset(request: Request) -> Response:
        caller(request)
        cur = source.current()
        if cur is None:
            return JSONResponse({"detail": "ruleset ещё не опубликован"}, status_code=503)
        rs, body = cur
        headers = {"ETag": f'"{rs.etag}"', "Cache-Control": "no-cache"}
        if _etag_matches(request.headers.get("if-none-match"), rs.etag):
            return Response(status_code=304, headers=headers)
        return Response(body, media_type="application/json", headers=headers)

    @app.post("/decide")
    def decide(request: Request, body: DecideIn) -> Response:
        caller(request)
        cur = source.current()
        if cur is None:
            return JSONResponse({"detail": "ruleset ещё не опубликован"}, status_code=503)
        rs = cur[0]
        try:
            results = evaluate_all(rs, body.unit_id, body.attributes, body.unit_type, exposures.enrolled(body.unit_id))
        except (ValueError, UnicodeError) as exc:
            raise _HttpError(422, f"некорректный unit_id: {exc}") from exc
        wanted = None if body.flags is None else set(body.flags)
        decisions = [
            {"flag_key": rule.flag_key, "experiment_id": rule.experiment_id, "variant_id": res.variant_id,
             "in_experiment": res.is_exposed, "reason": res.reason, "params": rule.params_of(res.variant_id)}
            for rule, res in zip(rs.experiments, results) if wanted is None or rule.flag_key in wanted
        ]
        return JSONResponse({"ruleset_version": rs.version, "unit_id": body.unit_id, "unit_type": body.unit_type,
                             "decisions": decisions})

    @app.post("/exposures")
    async def post_exposures(request: Request) -> Response:
        caller(request)
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
        cur = source.current()
        return {"ruleset_version": cur[0].version if cur else None, "exposures": len(exposures),
                "auth": "disabled" if auth.is_disabled else "keys", "control_plane": service is not None}

    # ---------------- маршруты управления ----------------
    @app.get("/passports")
    def list_passports(request: Request) -> List[Dict[str, Any]]:
        manager(request)
        return [{"experiment_id": p.experiment_id, "version": p.version, "status": p.status.value, "owner": p.owner,
                 "test_key": p.test_key, "layer_id": p.layer_id} for p in service.storage.list_passports()]

    @app.get("/passports/{experiment_id}")
    def get_passport(request: Request, experiment_id: str, version: Optional[int] = None) -> Dict[str, Any]:
        return dump(command(request, experiment_id, lambda actor: service.get(experiment_id, version)))

    @app.post("/passports")
    def save_draft(request: Request, data: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        return dump(command(request, data.get("experiment_id"), lambda actor: service.save_draft(actor, data)))

    @app.post("/passports/{experiment_id}/versions")
    def new_version(request: Request, experiment_id: str, changes: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        return dump(command(request, experiment_id, lambda actor: service.new_version(actor, experiment_id, changes)))

    @app.post("/passports/{experiment_id}/{action}")
    def passport_command(request: Request, experiment_id: str, action: str,
                         body: Optional[Dict[str, Any]] = Body(None)) -> Dict[str, Any]:
        body = body or {}
        commands = {
            "submit": lambda a: service.submit(a, experiment_id),
            "return": lambda a: service.return_to_draft(a, experiment_id, str(body.get("comment", ""))),
            "approve": lambda a: service.approve(a, experiment_id),
            "start": lambda a: service.start(a, experiment_id),
            "pause": lambda a: service.pause(a, experiment_id),
            "resume": lambda a: service.resume(a, experiment_id),
            "stop": lambda a: service.stop(a, experiment_id),
            "ramp": lambda a: service.ramp(a, experiment_id, body.get("ramp_pct")),
            "advance": lambda a: service.advance(a, experiment_id, str(body.get("to", ""))),
        }
        if action not in commands:
            manager(request)
            raise _HttpError(404, f"неизвестная команда '{action}'; есть: {sorted(commands)}")
        if action == "ramp" and not isinstance(body.get("ramp_pct"), (int, float)):
            manager(request)
            raise _HttpError(422, "нужно число ramp_pct")
        result = command(request, experiment_id, commands[action])
        if isinstance(result, Ruleset):
            passport = service.get(experiment_id, next(
                (r.passport_version for r in result.experiments if r.passport_id == experiment_id), None))
            return {"ruleset_version": result.version, "status": passport.status.value, "version": passport.version}
        return dump(result)

    @app.post("/layers")
    def add_layer(request: Request, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(body.get("layer_id"), str):
            manager(request)
            raise _HttpError(422, "нужна строка layer_id")
        rs = command(request, None, lambda a: service.add_layer(
            a, body["layer_id"], bool(body.get("marketing", True)), str(body.get("kind", "platform"))))
        return {"ruleset_version": rs.version, "layers": [l.layer_id for l in rs.layers]}

    @app.put("/holdout")
    def set_holdout(request: Request, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if isinstance(body.get("pct"), bool) or not isinstance(body.get("pct"), (int, float)):
            manager(request)
            raise _HttpError(422, "нужно число pct")
        rs = command(request, None, lambda a: service.set_holdout(a, float(body["pct"])))
        return {"ruleset_version": rs.version, "holdout_pct": rs.holdout.pct}

    @app.get("/audit")
    def get_audit(request: Request, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        manager(request)
        return [r.__dict__ for r in service.storage.audit(experiment_id)]

    @app.get("/audit/verify")
    def verify_audit(request: Request) -> Dict[str, Any]:
        manager(request)
        return {"records": service.storage.verify_audit(), "ruleset_problems":
                verify_ruleset(service.ruleset(), service.storage) if service.ruleset() else []}

    return app


def create_app_from_env() -> FastAPI:
    """
    Боевой запуск: база из ABP_DB_PATH, ключи из ABP_API_KEYS. До приёма запросов проверяются цепочка аудита
    и соответствие ruleset Паспортам — с повреждённой базой сервер не стартует.
    """
    path = os.environ.get("ABP_DB_PATH", "").strip()
    if not path:
        raise RuntimeError("Не задан ABP_DB_PATH — файл базы контура управления.")
    auth = Auth.from_env()
    storage = SqliteStorage(path)
    storage.verify_audit()
    service = ExperimentService(storage)
    current = service.ruleset()
    problems = verify_ruleset(current, storage) if current is not None else []
    if problems:
        raise RuntimeError("Ruleset расходится с Паспортами: " + "; ".join(problems))
    return create_app(exposure_store=ExposureStore(os.environ.get("ABP_EXPOSURES_PATH") or None), service=service,
                      auth=auth)
