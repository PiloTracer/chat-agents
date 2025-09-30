from __future__ import annotations
from fastapi import APIRouter

from ..defaults import DEMO_PRINCIPAL_OPTIONS

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/whoami")
def whoami():
    # In real life use proper auth (NextAuth/OAuth/JWT). This is a stub.
    return {"user": "demo", "role": "owner"}


@router.get("/demo-users")
def list_demo_users():
    return DEMO_PRINCIPAL_OPTIONS
