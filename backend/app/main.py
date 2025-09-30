# app/main.py
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

from app.defaults import DEFAULT_AGENT_DEFINITIONS, DEFAULT_USERS, DEFAULT_AGENT_ACLS
from app.models import Base, Agent, User, AgentACL
from app.db import init_db, session_scope
from app.routers import documents, chat, agents as agents_router, auth_stub

DEBUGPY = os.getenv("DEBUGPY", "0") in {"1", "true", "True", "YES", "yes"}

def _ensure_default_agents() -> None:
    with session_scope() as db:
        existing = {slug for (slug,) in db.query(Agent.slug).all()}
        for agent in DEFAULT_AGENT_DEFINITIONS:
            if agent["slug"] not in existing:
                db.add(Agent(**agent))


def _ensure_default_users() -> None:
    with session_scope() as db:
        existing = {username for (username,) in db.query(User.username).all()}
        for user in DEFAULT_USERS:
            if user["username"] not in existing:
                db.add(User(**user))


def _ensure_default_agent_acl() -> None:
    with session_scope() as db:
        existing_pairs = {
            (username, agent_slug)
            for username, agent_slug in db.query(AgentACL.username, AgentACL.agent_slug).all()
        }
        existing_users = {username for (username,) in db.query(User.username).all()}
        existing_agents = {slug for (slug,) in db.query(Agent.slug).all()}
        for entry in DEFAULT_AGENT_ACLS:
            key = (entry["username"], entry["agent_slug"])
            if entry["username"] not in existing_users:
                continue
            if entry["agent_slug"] not in existing_agents:
                continue
            if key not in existing_pairs:
                db.add(AgentACL(**entry))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEBUGPY:
        try:
            import debugpy
            # IMPORTANT: bind 0.0.0.0 for Docker, not 127.0.0.1
            debugpy.listen(("0.0.0.0", 5678))
            # Optional: block startup until the IDE attaches
            # debugpy.wait_for_client()
            print("🔌 debugpy listening on 0.0.0.0:5678")
        except Exception as e:
            print(f"debugpy failed to start: {e}")

    # --- Startup logic ---
    init_db()
    _ensure_default_agents()
    _ensure_default_users()
    _ensure_default_agent_acl()

    yield  # 👈 app runs here

    # --- Shutdown logic (optional) ---
    # e.g. close connections, cleanup, etc.

app = FastAPI(title="Multi-Agent RAG Backend", lifespan=lifespan)


def _parse_allowed_origins(raw: str) -> list[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]

allowed_origins = _parse_allowed_origins(settings.ALLOWED_ORIGINS)
allow_all_origins = "*" in allowed_origins

cors_kwargs = {
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

if allow_all_origins:
    cors_kwargs.update(allow_origins=["*"], allow_credentials=False)
else:
    if not allowed_origins:
        allowed_origins = ["http://localhost:3000"]
    cors_kwargs.update(allow_origins=allowed_origins, allow_credentials=True)

app.add_middleware(CORSMiddleware, **cors_kwargs)

@app.get("/healthz")
def healthz():
    return {"ok": True}


app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(agents_router.router)
app.include_router(auth_stub.router)
