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

- POST /orders, /order-adjustments (ключ ingest) и /app-events (ключи sdk и ingest) — события для метрик.
События хранятся в ab_platform/events.py (SQLite; ClickHouse — фаза 6).

Запуск: uvicorn "ab_platform.server:create_app_from_env" --factory
  ABP_DB_PATH — файл базы контура управления; ABP_API_KEYS — ключи (см. ab_platform/auth.py);
  ABP_EVENTS_DB_PATH — файл базы событий (показы, заказы, корректировки, события приложения);
  ABP_PREFLIGHT_DISABLED=1 — стенд без истории: запуск без предстартовой проверки (видно в /health).
"""
import json
import os
import threading
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from fastapi import Body, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ab_platform.analysis import preflight, report_markdown, reproduce, run_analysis, run_due_analyses
from ab_platform.auth import Auth
from ab_platform.control import Actor, ControlError, ExperimentService, verify_ruleset
from ab_platform.engine.ruleset_engine import evaluate_all
from ab_platform.schemas.analysis import AnalysisRun
from ab_platform.schemas.ruleset import Ruleset
from ab_platform.events import (  # noqa: F401 — ExposureIn и ExposureStore импортируют отсюда тесты фаз 2–3
    AppEventIn, ExposureIn, ExposureStore, MemoryEventStore, OrderAdjustmentIn, OrderIn, SqliteEventStore,
)
from ab_platform.storage import SqliteStorage

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
# Показы и события приложения: приложение (sdk) и серверная интеграция (ingest); люди событий не пишут.
EVENT_WRITERS = ("sdk", "ingest")


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
        if actor.role in EVENT_WRITERS:
            service.storage.append_audit(actor.subject, "auth.denied", None,
                                         {"path": request.url.path, "reason": f"ключ {actor.role} на маршруте управления"})
            raise _HttpError(403, f"ключ {actor.role} не даёт доступа к управлению")
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

    async def batch(request: Request, model) -> Tuple[List[Any], List[Dict[str, Any]], List[int]]:
        """Пачка событий: (корректные, отклонённые с причиной, исходные номера корректных)."""
        try:
            payload: Any = json.loads(await request.body())
        except ValueError:
            raise _HttpError(400, "тело не JSON")
        if isinstance(payload, dict):
            payload = payload.get("events")
        if not isinstance(payload, list):
            raise _HttpError(400, "ожидается список событий или {\"events\": [...]}")
        if len(payload) > MAX_BATCH:
            raise _HttpError(413, f"пачка больше {MAX_BATCH} событий")
        valid, rejected, index = [], [], []
        for i, raw in enumerate(payload):
            try:
                valid.append(model.model_validate(raw))
                index.append(i)
            except ValidationError as exc:
                rejected.append({"index": i, "error": "; ".join(
                    f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in exc.errors())})
        return valid, rejected, index

    def writer(request: Request, roles: Tuple[str, ...], what: str) -> Actor:
        """Запись событий: только перечисленные роли. Ключ приложения не пишет заказы (его можно извлечь)."""
        actor = caller(request)
        if actor.role not in roles:
            if service is not None:
                service.storage.append_audit(actor.subject, "auth.denied", None,
                                             {"path": request.url.path, "role": actor.role, "reason": what})
            raise _HttpError(403, f"роль '{actor.role}' не может {what}; нужна: {', '.join(roles)}")
        return actor

    def ingested(result, rejected: List[Dict[str, Any]], index: List[int]) -> JSONResponse:
        # номера конфликтов — в исходной пачке, а не среди корректных
        conflicts = [{**c, "index": index[c["index"]]} for c in result.conflicts]
        return JSONResponse({"accepted": result.accepted, "duplicates": result.duplicates, "conflicts": conflicts,
                             "rejected": rejected})

    @app.post("/exposures")
    async def post_exposures(request: Request) -> Response:
        writer(request, EVENT_WRITERS, "записывать показы")
        valid, rejected, _ = await batch(request, ExposureIn)
        accepted, duplicates = exposures.add(valid)
        return JSONResponse({"accepted": accepted, "duplicates": duplicates, "rejected": rejected})

    @app.post("/orders")
    async def post_orders(request: Request) -> Response:
        writer(request, ("ingest",), "записывать заказы")
        valid, rejected, index = await batch(request, OrderIn)
        return ingested(exposures.add_orders(valid), rejected, index)

    @app.post("/order-adjustments")
    async def post_adjustments(request: Request) -> Response:
        writer(request, ("ingest",), "записывать отмены и возвраты")
        valid, rejected, index = await batch(request, OrderAdjustmentIn)
        return ingested(exposures.add_adjustments(valid), rejected, index)

    @app.post("/app-events")
    async def post_app_events(request: Request) -> Response:
        writer(request, EVENT_WRITERS, "записывать события приложения")
        valid, rejected, index = await batch(request, AppEventIn)
        return ingested(exposures.add_app_events(valid), rejected, index)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        cur = source.current()
        return {"ruleset_version": cur[0].version if cur else None, "exposures": len(exposures),
                "auth": "disabled" if auth.is_disabled else "keys", "control_plane": service is not None,
                "preflight": None if service is None else ("required" if service.require_preflight else "disabled")}

    @app.get("/events/health")
    def events_health(request: Request) -> Dict[str, Any]:
        """Сколько событий каждого вида и сколько корректировок пришло раньше своего заказа."""
        caller(request)
        return {"counts": exposures.counts_by_kind(), "orphan_adjustments": exposures.orphan_adjustments(),
                "last_seq": exposures.last_seq()}

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
            "preflight": lambda a: preflight(service, exposures, experiment_id, a),
            "advance": lambda a: service.advance(a, experiment_id, str(body.get("to", ""))),
        }
        if action not in commands:
            manager(request)
            raise _HttpError(404, f"неизвестная команда '{action}'; есть: {sorted(commands)}")
        if action == "ramp" and not isinstance(body.get("ramp_pct"), (int, float)):
            manager(request)
            raise _HttpError(422, "нужно число ramp_pct")
        result = command(request, experiment_id, commands[action])
        if isinstance(result, AnalysisRun):
            return run_summary(result)
        if isinstance(result, Ruleset):
            passport = service.get(experiment_id, next(
                (r.passport_version for r in result.experiments if r.passport_id == experiment_id), None))
            return {"ruleset_version": result.version, "status": passport.status.value, "version": passport.version}
        return dump(result)

    def run_summary(run: AnalysisRun) -> Dict[str, Any]:
        r = run.result
        return {"run_id": run.run_id, "kind": run.inputs.kind, "experiment_id": run.inputs.experiment_id,
                "passport_version": run.inputs.passport_version, "cutoff": run.inputs.cutoff,
                "created_at": run.created_at, "decision": r.get("decision"), "passed": r.get("passed"),
                "problems": r.get("problems")}

    def need_run(request: Request, run_id: str) -> AnalysisRun:
        manager(request)
        run = service.storage.get_analysis_run(run_id)
        if run is None:
            raise _HttpError(404, f"запуска {run_id} нет")
        return run

    @app.get("/analysis-runs")
    def list_runs(request: Request, experiment_id: Optional[str] = None, kind: Optional[str] = None):
        manager(request)
        return [run_summary(r) for r in service.storage.list_analysis_runs(experiment_id, kind)]

    @app.get("/analysis-runs/{run_id}")
    def get_run(request: Request, run_id: str) -> Dict[str, Any]:
        return need_run(request, run_id).model_dump(mode="json")

    @app.get("/analysis-runs/{run_id}/report.md")
    def get_report(request: Request, run_id: str) -> Response:
        run = need_run(request, run_id)
        if run.inputs.kind == "preflight":
            raise _HttpError(404, "у предстартовой проверки нет отчёта: смотрите сам запуск")
        return Response(report_markdown(run), media_type="text/markdown; charset=utf-8")

    @app.post("/analysis-runs")
    def new_run(request: Request, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Анализ по запросу человека (аналитик или администратор): monitoring или final."""
        eid, kind = str(body.get("experiment_id", "")), str(body.get("kind", "monitoring"))

        def go(actor):
            ExperimentService._require(actor, "analyst", "admin")
            return run_analysis(service, exposures, eid, kind, actor=actor.subject)
        return run_summary(command(request, eid, go))

    @app.post("/analysis-runs/{run_id}/reproduce")
    def reproduce_run(request: Request, run_id: str) -> Dict[str, Any]:
        def go(actor):
            ExperimentService._require(actor, "analyst", "admin")
            return reproduce(service, exposures, run_id)
        return command(request, None, go)

    @app.post("/scheduler/run")
    def scheduler_run(request: Request) -> List[Dict[str, Any]]:
        """Ежедневный проход по тестам (cron вызывает с ключом администратора)."""
        def go(actor):
            ExperimentService._require(actor, "admin")
            return run_due_analyses(service, exposures)
        return command(request, None, go)

    @app.get("/alerts")
    def list_alerts(request: Request, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        manager(request)
        return [a.model_dump(mode="json") for a in service.storage.list_alerts(experiment_id)]

    @app.get("/metrics")
    def list_metrics(request: Request) -> List[Dict[str, Any]]:
        manager(request)
        return [{**m.model_dump(mode="json"), "ref": m.ref, "metric_type": m.metric_type.value}
                for m in service.storage.list_metrics()]

    @app.get("/metrics/{ref}")
    def get_metric(request: Request, ref: str) -> Dict[str, Any]:
        m = command(request, None, lambda actor: service.metric(ref))
        return {**m.model_dump(mode="json"), "ref": m.ref, "metric_type": m.metric_type.value,
                "definition": m.definition().model_dump(mode="json")}

    @app.post("/metrics")
    def define_metric(request: Request, data: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        m = command(request, None, lambda actor: service.define_metric(actor, data))
        return {**m.model_dump(mode="json"), "ref": m.ref, "metric_type": m.metric_type.value}

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
    Боевой запуск: база контура из ABP_DB_PATH, события — из ABP_EVENTS_DB_PATH, ключи из ABP_API_KEYS.
    До приёма запросов проверяются цепочка аудита
    и соответствие ruleset Паспортам — с повреждённой базой сервер не стартует.
    """
    path = os.environ.get("ABP_DB_PATH", "").strip()
    if not path:
        raise RuntimeError("Не задан ABP_DB_PATH — файл базы контура управления.")
    auth = Auth.from_env()
    storage = SqliteStorage(path)
    storage.verify_audit()
    # предстартовая проверка обязательна; стенд без истории заказов отключает её только явно — как и ключи
    service = ExperimentService(storage, require_preflight=os.environ.get("ABP_PREFLIGHT_DISABLED") != "1")
    service.ensure_default_metrics()
    current = service.ruleset()
    problems = verify_ruleset(current, storage) if current is not None else []
    if problems:
        raise RuntimeError("Ruleset расходится с Паспортами: " + "; ".join(problems))
    events_path = os.environ.get("ABP_EVENTS_DB_PATH", "").strip()
    if not events_path:
        raise RuntimeError("Не задан ABP_EVENTS_DB_PATH — файл базы событий (показы, заказы, события приложения).")
    return create_app(exposure_store=SqliteEventStore(events_path), service=service, auth=auth)
