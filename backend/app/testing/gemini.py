from __future__ import annotations
import os
import sys
from google import genai

try:
    # Prefer app settings when executed via package/module
    from app.config import settings
except Exception:
    class _S:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-pro")
    settings = _S()  # type: ignore


def main() -> int:
    api_key = (getattr(settings, "GEMINI_API_KEY", None) or "").strip()
    if not api_key:
        api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        print("GEMINI_API_KEY/GOOGLE_API_KEY not set", file=sys.stderr)
        return 2

    model = (getattr(settings, "GEMINI_CHAT_MODEL", None) or os.getenv("GEMINI_CHAT_MODEL") or "gemini-1.5-pro").strip()
    prompt = " ".join(sys.argv[1:]).strip() or "Say hello and include today's date."

    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(model=model, contents=[prompt])
    except Exception as exc:
        print(f"Gemini request failed: {exc}", file=sys.stderr)
        return 1

    text = getattr(resp, "text", None)
    if isinstance(text, str):
        print(text, end="")
    else:
        print(str(resp), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
