from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, Iterator, Tuple

from app.config import settings
from app.defaults import (
    DEFAULT_PRINCIPALS,
    DEFAULT_PRINCIPAL_CREDENTIALS,
)

from .service import (
    AuthService,
    InMemoryAuthBackend,
    PasswordHasher,
    Principal,
    SessionTokenManager,
)

_QUOTE_CHARS = {"'", '"'}


def _strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTE_CHARS:
        value = value[1:-1].strip()
    return value


def _iter_user_chunks(raw: str) -> Iterator[str]:
    prepared = _strip_wrapping_quotes(raw)
    for chunk in re.split(r"[,\n;]+", prepared):
        entry = _strip_wrapping_quotes(chunk)
        if entry:
            yield entry


def _parse_env_users(raw: str) -> Iterable[Tuple[str, str, str | None]]:
    for chunk in _iter_user_chunks(raw):
        parts = chunk.split(":", 2)
        if len(parts) < 2:
            continue
        username = _strip_wrapping_quotes(parts[0])
        password = _strip_wrapping_quotes(parts[1])
        role = _strip_wrapping_quotes(parts[2]) if len(parts) > 2 else ""
        if not username or not password:
            continue
        yield username, password, role.lower() or None


def _resolve_user_specs() -> Dict[str, Tuple[str, str]]:
    principal_roles = {p["username"]: p["role"].lower() for p in DEFAULT_PRINCIPALS}
    configured: Dict[str, Tuple[str, str]] = {}

    cleaned_auth_users = settings.AUTH_USERS or ""
    if cleaned_auth_users.strip():
        for username, password, role_override in _parse_env_users(cleaned_auth_users):
            role = role_override or principal_roles.get(username, "admin")
            configured[username] = (role, password)
    else:
        for username, role in principal_roles.items():
            password = DEFAULT_PRINCIPAL_CREDENTIALS.get(username, "changeme")
            configured[username] = (role, password)

    if not configured:
        raise RuntimeError("No authentication users configured")

    return configured


@lru_cache(maxsize=1)
def get_auth_service() -> AuthService:
    user_specs = _resolve_user_specs()
    hasher = PasswordHasher(iterations=settings.AUTH_HASH_ITERATIONS)
    backend = InMemoryAuthBackend(hasher)
    for username, (role, password) in user_specs.items():
        backend.add_user(username=username, role=role, password=password)
    token_manager = SessionTokenManager(settings.AUTH_TOKEN_SECRET, settings.AUTH_TOKEN_TTL_SECONDS)
    return AuthService(backend, token_manager)


__all__ = [
    "get_auth_service",
    "Principal",
]
