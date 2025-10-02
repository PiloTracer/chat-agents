# app/llm_provider.py
from __future__ import annotations
from typing import Dict, List, Optional
import logging
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

class LLMProvider:
    """Unified chat completion client for OpenAI and DeepSeek."""
    def __init__(self) -> None:
        self.providers = {
            "openai": {
                "base_url": getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
                "api_key": getattr(settings, "OPENAI_API_KEY", settings.API_KEY),
                "default_model": getattr(settings, "CHAT_MODEL", "gpt-4.1"),
            },
            "deepseek": {
                "base_url": getattr(settings, "DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").rstrip("/"),
                "api_key": getattr(settings, "DEEPSEEK_API_KEY", ""),
                "default_model": getattr(settings, "DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
            },
        }

    def _normalize_provider(self, p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        pl = p.strip().lower()
        if pl in {"gpt", "openai"}:
            return "openai"
        if pl in {"deepseek", "ds"}:
            return "deepseek"
        return None

    def _default_provider(self) -> str:
        # Prefer env-configured default; attempt heuristic based on base URL; fall back to OpenAI
        env_default = self._normalize_provider(getattr(settings, "DEFAULT_CHAT_PROVIDER", None))
        if env_default:
            return env_default
        try:
            base = str(getattr(settings, "CHAT_PROVIDER_BASE_URL", "") or "").lower()
            if "deepseek" in base:
                return "deepseek"
            if "openai" in base:
                return "openai"
        except Exception:
            pass
        return "openai"

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 1,
    ) -> Dict[str, str]:
        """
        Returns:
            {"content": <str>, "provider": "openai"|"deepseek"}
        """
        # Prefer request parameter; if missing/invalid, fall back to env default
        primary = self._normalize_provider(provider) or self._default_provider()
        order = [primary]
        if retries:
            order += [p for p in ("openai", "deepseek") if p != primary]

        last_err: Optional[Exception] = None
        for prov in order:
            try:
                content = await self._call_provider(
                    prov, messages, model=model, temperature=temperature, max_tokens=max_tokens
                )
                return {"content": content, "provider": prov}
            except Exception as e:
                last_err = e
                logger.warning("LLM call failed on %s: %s", prov, e)

        assert last_err is not None
        raise last_err

    async def _call_provider(
        self,
        prov: str,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        cfg = self.providers.get(prov)
        if not cfg:
            raise RuntimeError(f"Unknown provider '{prov}'")
        base_url = cfg["base_url"]
        api_key = cfg["api_key"]
        use_model = model or cfg["default_model"]
        if not api_key:
            raise RuntimeError(f"Missing API key for provider '{prov}'")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {"model": use_model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

llm_provider = LLMProvider()
