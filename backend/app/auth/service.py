from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Principal:
    username: str
    role: str


@dataclass
class UserRecord:
    username: str
    role: str
    password_hash: str
    salt: str


class PasswordHasher:
    """PBKDF2-based password hasher."""

    def __init__(self, iterations: int = 120_000, salt_bytes: int = 16) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if salt_bytes <= 0:
            raise ValueError("salt_bytes must be positive")
        self.iterations = iterations
        self.salt_bytes = salt_bytes

    def _normalise_salt(self, salt: str) -> str:
        return salt.rstrip("=")

    def _salt_to_bytes(self, salt: str) -> bytes:
        padded = salt + "=" * ((4 - len(salt) % 4) % 4)
        return base64.urlsafe_b64decode(padded.encode("utf-8"))

    def hash(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        if salt is None:
            salt_bytes = secrets.token_bytes(self.salt_bytes)
            salt = base64.urlsafe_b64encode(salt_bytes).decode("utf-8")
        else:
            salt_bytes = self._salt_to_bytes(salt)
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt_bytes,
            self.iterations,
        )
        password_hash = base64.urlsafe_b64encode(dk).decode("utf-8")
        return self._normalise_salt(password_hash), self._normalise_salt(salt)

    def verify(self, password: str, password_hash: str, salt: str) -> bool:
        derived, _ = self.hash(password, salt)
        return hmac.compare_digest(derived, self._normalise_salt(password_hash))


class InMemoryAuthBackend:
    """Simple user-store backed by process memory."""

    def __init__(self, hasher: PasswordHasher) -> None:
        self._hasher = hasher
        self._users: Dict[str, UserRecord] = {}

    def add_user(self, username: str, role: str, password: Optional[str] = None, *, password_hash: Optional[str] = None, salt: Optional[str] = None) -> None:
        if username in self._users:
            raise ValueError(f"duplicate user '{username}'")
        if password is not None:
            password_hash, salt = self._hasher.hash(password)
        if not password_hash or not salt:
            raise ValueError("password or password_hash+salt must be provided")
        self._users[username] = UserRecord(
            username=username,
            role=role,
            password_hash=password_hash,
            salt=salt,
        )

    def authenticate(self, username: str, password: str) -> Optional[Principal]:
        record = self._users.get(username)
        if not record:
            return None
        if not self._hasher.verify(password, record.password_hash, record.salt):
            return None
        return Principal(username=record.username, role=record.role)

    def get_principal(self, username: str) -> Optional[Principal]:
        record = self._users.get(username)
        if not record:
            return None
        return Principal(username=record.username, role=record.role)


class SessionTokenManager:
    """Issues and verifies signed bearer tokens."""

    def __init__(self, secret: str, ttl_seconds: int) -> None:
        if not secret:
            raise ValueError("secret must be set")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._secret = secret.encode("utf-8")
        self._ttl = ttl_seconds

    def _sign(self, payload: bytes) -> str:
        digest = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")

    def _pack(self, data: Dict[str, object]) -> str:
        payload = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        body = base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")
        signature = self._sign(payload)
        return f"{body}.{signature}"

    def _unpack(self, token: str) -> Optional[Dict[str, object]]:
        try:
            body_b64, signature = token.split(".", 1)
        except ValueError:
            return None
        payload_bytes = base64.urlsafe_b64decode(body_b64 + "=" * ((4 - len(body_b64) % 4) % 4))
        expected_sig = self._sign(payload_bytes)
        if not hmac.compare_digest(signature, expected_sig):
            return None
        data = json.loads(payload_bytes.decode("utf-8"))
        return data

    def issue(self, principal: Principal) -> str:
        now = int(time.time())
        payload = {
            "sub": principal.username,
            "role": principal.role,
            "iat": now,
            "exp": now + self._ttl,
        }
        return self._pack(payload)

    def verify(self, token: str) -> Optional[Principal]:
        data = self._unpack(token)
        if not data:
            return None
        exp = data.get("exp")
        if not isinstance(exp, int):
            return None
        if exp < int(time.time()):
            return None
        sub = data.get("sub")
        role = data.get("role")
        if not isinstance(sub, str) or not isinstance(role, str):
            return None
        return Principal(username=sub, role=role)


class AuthService:
    """Facade combining credential verification and token issuance."""

    def __init__(self, backend: InMemoryAuthBackend, token_manager: SessionTokenManager) -> None:
        self._backend = backend
        self._tokens = token_manager

    def login(self, username: str, password: str) -> Optional[tuple[str, Principal]]:
        principal = self._backend.authenticate(username, password)
        if not principal:
            return None
        token = self._tokens.issue(principal)
        return token, principal

    def get_principal(self, username: str) -> Optional[Principal]:
        return self._backend.get_principal(username)

    def verify_bearer(self, authorization_header: Optional[str]) -> Optional[Principal]:
        if not authorization_header:
            return None
        if not authorization_header.lower().startswith("bearer "):
            return None
        token = authorization_header[7:].strip()
        if not token:
            return None
        principal = self._tokens.verify(token)
        if not principal:
            return None
        stored = self._backend.get_principal(principal.username)
        if not stored:
            return None
        return stored

    def issue_token_for_principal(self, principal: Principal) -> str:
        return self._tokens.issue(principal)


__all__ = [
    "AuthService",
    "InMemoryAuthBackend",
    "PasswordHasher",
    "Principal",
    "SessionTokenManager",
    "UserRecord",
]

