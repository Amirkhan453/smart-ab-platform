"""
Запуск анализа и тревога (ТЗ v4.1: RPT-1; ТЗ v4: принцип 4, 12.1 — каждое число ссылается на analysis_run_id).

AnalysisRun неизменяем. run_id — хэш входов: вид, Паспорт (версия и хэш), правило, версии метрик, срез событий
(upto_seq, cutoff) и версия кода. Повтор с теми же входами даёт тот же run_id и тот же result байт в байт.
"""
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RunInputs(BaseModel):
    """Всё, от чего зависит результат. Из этого считается run_id."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["monitoring", "final", "preflight"]
    experiment_id: str
    passport_version: int
    passport_hash: str
    rule_id: str
    metrics: List[Dict[str, str]]          # [{"ref": ..., "definition_hash": ...}]
    cutoff: str                            # ISO 8601 UTC
    upto_seq: int
    code_version: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()[:24]


class AnalysisRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: str
    inputs: RunInputs
    input_hash: str                        # sha256 таблицы «гость × метрики»
    result: Dict[str, Any]
    created_at: str
    actor: str

    def result_json(self) -> str:
        return canonical_json(self.result)


class Alert(BaseModel):
    """Тревога владельцу теста: SRM, автопауза по guardrail, готовность итогового отчёта, сбой анализа."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    alert_id: str
    experiment_id: str
    kind: Literal["srm", "guardrail_auto_pause", "final_report_ready", "analysis_failed"]
    message: str
    run_id: Optional[str] = None
    created_at: str
