from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import httpx


def getenv_first(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def fetch_models(api_base: str, api_key: str, page_size: int = 200) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/models?key={api_key}&pageSize={int(page_size)}"
    with httpx.Client(timeout=30) as client:
        resp = client.get(url)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail: Optional[str] = None
            try:
                detail = exc.response.json().get("error", {}).get("message")
            except Exception:
                try:
                    detail = exc.response.text
                except Exception:
                    detail = str(exc)
            raise SystemExit(f"ListModels request failed ({resp.status_code}): {detail}") from exc
        return resp.json()


def format_model_line(m: Dict[str, Any]) -> str:
    name = m.get("name", "?")
    disp = m.get("displayName") or ""
    methods = m.get("supportedGenerationMethods") or []
    in_lim = m.get("inputTokenLimit")
    out_lim = m.get("outputTokenLimit")
    bits = [name]
    if disp:
        bits.append(f"({disp})")
    if methods:
        bits.append("methods=" + ",".join(methods))
    if in_lim:
        bits.append(f"in={in_lim}")
    if out_lim:
        bits.append(f"out={out_lim}")
    return "  - " + " | ".join(bits)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="List Gemini models available to the configured API key")
    p.add_argument("--json", action="store_true", help="Print raw JSON response")
    p.add_argument("--filter", metavar="SUBSTR", help="Filter by substring in model name", default=None)
    p.add_argument("--supports", metavar="METHOD", help="Require a supportedGenerationMethods entry", default=None)
    p.add_argument("--api-base", metavar="URL", default=None, help="Override API base URL")
    args = p.parse_args(argv)

    api_key = getenv_first("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.", file=sys.stderr)
        return 2

    api_base = (
        args.api_base
        or getenv_first("GEMINI_API_BASE")
        or "https://generativelanguage.googleapis.com/v1beta"
    )

    data = fetch_models(api_base, api_key)
    models: List[Dict[str, Any]] = list(data.get("models") or [])

    # Sort by name for stable output
    models.sort(key=lambda m: str(m.get("name", "")))

    if args.filter:
        needle = args.filter.lower()
        models = [m for m in models if needle in str(m.get("name", "")).lower()]

    if args.supports:
        req = args.supports
        models = [m for m in models if req in (m.get("supportedGenerationMethods") or [])]

    if args.json:
        print(json.dumps({"models": models}, ensure_ascii=False, indent=2))
        return 0

    if not models:
        print("No models returned. Verify API key and permissions.")
        return 1

    print("Available Gemini models (ListModels):")
    for m in models:
        print(format_model_line(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

