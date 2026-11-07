"""
HTTP-транспорт Python SDK: источник ruleset (ETag/304) и отправка exposure на сервер (ab_platform.server).
Вынесен отдельно, чтобы ab_platform.sdk не зависел от httpx.

    client = ExperimentClient(HttpRulesetFetcher(url), HttpExposureSink(url)); client.refresh(); client.start()
"""
from typing import List, Optional, Tuple

import httpx


class _Http:
    def __init__(self, base_url: str, timeout_s: float, client: Optional[httpx.Client]):
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)
        self._owns = client is None

    def close(self) -> None:
        if self._owns:
            self._client.close()


class HttpRulesetFetcher(_Http):
    """fetch(etag) → (304, None, etag) или (200, json, etag). Сетевые ошибки пробрасываются — SDK их ловит."""

    def __init__(self, base_url: str = "", timeout_s: float = 5.0, client: Optional[httpx.Client] = None):
        super().__init__(base_url, timeout_s, client)

    def fetch(self, if_none_match: Optional[str]) -> Tuple[int, Optional[str], Optional[str]]:
        headers = {"If-None-Match": f'"{if_none_match}"'} if if_none_match else {}
        r = self._client.get("/ruleset", headers=headers)
        etag = r.headers.get("etag")
        etag = etag.removeprefix("W/").strip('"') if etag else None
        return r.status_code, (r.text if r.status_code == 200 else None), etag


class HttpExposureSink(_Http):
    """Отправка пачки exposure. Не-2xx и сетевые ошибки → исключение: SDK вернёт пачку в буфер и повторит."""

    def __init__(self, base_url: str = "", timeout_s: float = 5.0, client: Optional[httpx.Client] = None):
        super().__init__(base_url, timeout_s, client)
        self.rejected: List[dict] = []

    def __call__(self, batch: List[dict]) -> None:
        r = self._client.post("/exposures", json=batch)
        r.raise_for_status()
        # отклонённые сервером события повтором не исправить — сохраняем для диагностики
        self.rejected.extend(r.json().get("rejected", []))
