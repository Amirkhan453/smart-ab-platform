"""
Ключи доступа к HTTP API (ТЗ v4.1: AUTH-1; ТЗ v4: раздел 15).

Ключи задаются только переменной окружения ABP_API_KEYS: записи "роль:имя:ключ" через запятую, например
    ABP_API_KEYS="sdk:mobile-app:<ключ>,analyst:petrov:<ключ>,admin:root:<ключ>"
Роли — из ТЗ 2.1 плюс sdk. Ключ передаётся заголовком "Authorization: Bearer <ключ>".
В памяти хранятся только sha256 ключей; сравнение — за постоянное время.

Без ключей сервер не запускается. Режим стенда без ключей включается только явно: ABP_AUTH_DISABLED=1
или Auth.disabled() в коде. В нём открыты маршруты SDK, а маршруты управления отвечают 403.
"""
import hashlib
import hmac
import os
from typing import Dict, Mapping, Optional

from ab_platform.control import ROLES, Actor

MIN_KEY_LENGTH = 24
ENV_KEYS = "ABP_API_KEYS"
ENV_DISABLED = "ABP_AUTH_DISABLED"
ANONYMOUS = Actor("anonymous", "sdk")


class AuthConfigError(RuntimeError):
    """Ключи не заданы или заданы неверно: сервер не должен стартовать."""


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8")).digest()


class Auth:
    def __init__(self, keys: Mapping[bytes, Actor], disabled: bool = False):
        self._keys: Dict[bytes, Actor] = dict(keys)
        self.is_disabled = disabled

    @classmethod
    def disabled(cls) -> "Auth":
        """Режим стенда: маршруты SDK открыты, управление закрыто."""
        return cls({}, disabled=True)

    @classmethod
    def parse(cls, spec: str) -> "Auth":
        keys: Dict[bytes, Actor] = {}
        subjects = set()
        for i, item in enumerate(filter(None, (x.strip() for x in spec.split(","))), start=1):
            parts = item.split(":", 2)
            if len(parts) != 3 or not all(parts):
                raise AuthConfigError(f"{ENV_KEYS}: запись №{i} должна иметь вид роль:имя:ключ.")
            role, subject, key = parts
            if role not in ROLES:
                raise AuthConfigError(f"{ENV_KEYS}: запись №{i} — неизвестная роль '{role}'; допустимы {list(ROLES)}.")
            if len(key) < MIN_KEY_LENGTH:
                raise AuthConfigError(f"{ENV_KEYS}: ключ записи №{i} короче {MIN_KEY_LENGTH} знаков.")
            if _digest(key) in keys:
                raise AuthConfigError(f"{ENV_KEYS}: ключ записи №{i} повторяется.")
            if subject in subjects:
                raise AuthConfigError(f"{ENV_KEYS}: имя '{subject}' повторяется — у каждого свой ключ и одна роль.")
            subjects.add(subject)
            keys[_digest(key)] = Actor(subject, role)
        if not keys:
            raise AuthConfigError(f"{ENV_KEYS} не содержит ни одного ключа.")
        return cls(keys)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Auth":
        env = os.environ if env is None else env
        spec = env.get(ENV_KEYS, "").strip()
        if spec:
            return cls.parse(spec)
        if env.get(ENV_DISABLED) == "1":
            return cls.disabled()
        raise AuthConfigError(
            f"Не заданы ключи доступа: укажите {ENV_KEYS} (роль:имя:ключ через запятую). "
            f"Для стенда без ключей — {ENV_DISABLED}=1."
        )

    def authenticate(self, authorization: Optional[str]) -> Optional[Actor]:
        """Субъект по заголовку Authorization или None. В режиме стенда — анонимный SDK."""
        if self.is_disabled:
            return ANONYMOUS
        if not authorization or not authorization.startswith("Bearer "):
            return None
        digest = _digest(authorization[len("Bearer "):].strip())
        found = None
        for known, actor in self._keys.items():  # без раннего выхода: время не зависит от того, какой ключ совпал
            if hmac.compare_digest(known, digest):
                found = actor
        return found
