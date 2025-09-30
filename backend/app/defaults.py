# app/defaults.py
from __future__ import annotations
from typing import Iterable, List, Literal

DEFAULT_AGENT_DEFINITIONS = [
    {
        "slug": "local-regs",
        "title": "Local Regulations Expert",
        "description": "Expert in Costa Rican and other local regulatory frameworks, permits, and compliance steps.",
    },
    {
        "slug": "intl-regs",
        "title": "International Regulations Expert",
        "description": "Expert in global regulations, treaties, and compliance standards across regions.",
    },
    {
        "slug": "logistics",
        "title": "Logistics Procedures Expert",
        "description": "Expert in import/export logistics processes, documentation, and operational workflows.",
    },
    {
        "slug": "internal",
        "title": "Internal SOPs Expert",
        "description": "Expert in internal company policies, SOPs, knowledge bases, and best practices.",
    },
]

_PrincipalAgentSpec = list[str] | Literal["all"]

DEFAULT_PRINCIPALS = [
    {
        "username": "owner",
        "role": "owner",
        "display_name": "Owner (all access)",
        "agent_slugs": "all",
    },
    {
        "username": "demo",
        "role": "admin",
        "display_name": "Demo Admin (all agents)",
        "agent_slugs": "all",
    },
    {
        "username": "admin-local",
        "role": "admin",
        "display_name": "Local Regulations Admin",
        "agent_slugs": ["local-regs", "internal"],
    },
    {
        "username": "admin-intl",
        "role": "admin",
        "display_name": "International Regulations Admin",
        "agent_slugs": ["intl-regs"],
    },
    {
        "username": "admin-logistics",
        "role": "admin",
        "display_name": "Logistics Admin",
        "agent_slugs": ["logistics"],
    },
]


def _all_agent_slugs() -> List[str]:
    return [agent["slug"] for agent in DEFAULT_AGENT_DEFINITIONS]


def _resolve_agent_list(spec: _PrincipalAgentSpec) -> List[str]:
    if spec == "all":
        return _all_agent_slugs()
    return list(spec)


DEFAULT_USERS = [
    {"username": principal["username"], "role": principal["role"]}
    for principal in DEFAULT_PRINCIPALS
]

DEFAULT_AGENT_ACLS = [
    {"username": principal["username"], "agent_slug": agent_slug}
    for principal in DEFAULT_PRINCIPALS
    if principal["role"] != "owner"
    for agent_slug in _resolve_agent_list(principal["agent_slugs"])
]

DEMO_PRINCIPAL_OPTIONS = [
    {
        "username": principal["username"],
        "role": principal["role"],
        "display_name": principal.get("display_name", principal["username"]),
        "agent_slugs": _resolve_agent_list(principal["agent_slugs"]),
    }
    for principal in DEFAULT_PRINCIPALS
]
