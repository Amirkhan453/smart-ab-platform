"""
Клиент HTTP API для формы Паспорта (UI-0): всё через сервер с личным ключом, поэтому роли и аудит те же,
что у любого другого клиента. Форма не открывает базу напрямую.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class ApiError(Exception):
    """Отказ сервера: код и текст причины, пригодный для показа человеку."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def _detail(resp: httpx.Response) -> str:
    try:
        d = resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text or resp.reason_phrase
    if isinstance(d, list):  # ошибки разбора запроса FastAPI: [{loc, msg}]
        return "\n".join(f"{'.'.join(e.get('loc', [])[1:]) or 'запрос'}: {e.get('msg', '')}" for e in d)
    return str(d)


class PlatformApi:
    """
    client — готовый httpx.Client (в тестах — TestClient приложения); иначе создаётся по base_url.
    Ключ передаётся только в заголовке Authorization.
    """

    def __init__(self, base_url: str = "", key: str = "", client: Optional[httpx.Client] = None, timeout: float = 30.0):
        if not key:
            raise ValueError("Нужен личный ключ API")
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self._headers = {"Authorization": f"Bearer {key}"}

    def _call(self, method: str, path: str, **kw) -> httpx.Response:
        resp = self._client.request(method, path, headers=self._headers, **kw)
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, _detail(resp))
        return resp

    # ---------------- чтение ----------------
    def metrics(self) -> List[Dict[str, Any]]:
        return self._call("GET", "/metrics").json()

    def metric(self, ref: str) -> Dict[str, Any]:
        return self._call("GET", f"/metrics/{ref}").json()

    def passports(self) -> List[Dict[str, Any]]:
        return self._call("GET", "/passports").json()

    def passport(self, experiment_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/passports/{experiment_id}").json()

    def runs(self, experiment_id: str) -> List[Dict[str, Any]]:
        return self._call("GET", "/analysis-runs", params={"experiment_id": experiment_id}).json()

    def report(self, run_id: str) -> str:
        return self._call("GET", f"/analysis-runs/{run_id}/report.md").text

    def alerts(self, experiment_id: str) -> List[Dict[str, Any]]:
        return self._call("GET", "/alerts", params={"experiment_id": experiment_id}).json()

    # ---------------- команды ----------------
    def save_draft(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._call("POST", "/passports", json=data).json()

    def new_version(self, experiment_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
        return self._call("POST", f"/passports/{experiment_id}/versions", json=changes).json()

    def command(self, experiment_id: str, action: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._call("POST", f"/passports/{experiment_id}/{action}", json=body).json()
