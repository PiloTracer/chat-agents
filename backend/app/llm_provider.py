# app/llm_provider.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import logging
import httpx
import hashlib
import time
import asyncio

from app.config import settings

logger = logging.getLogger(__name__)

class LLMProvider:
    """Unified chat completion client for OpenAI, DeepSeek, and Gemini."""
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
            "gemini": {
                "base_url": getattr(settings, "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").rstrip("/"),
                "api_key": getattr(settings, "GEMINI_API_KEY", ""),
                "default_model": getattr(settings, "GEMINI_CHAT_MODEL", "gemini-1.5-pro"),
            },
        }
        # In-memory cache for Gemini context caching (resource name + expiry)
        self._gemini_cache: Dict[str, Tuple[str, float]] = {}
        # Cache for Gemini model limits (e.g., outputTokenLimit) keyed by model path
        self._gemini_limits: Dict[str, int] = {}

    def _normalize_provider(self, p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        pl = p.strip().lower()
        if pl in {"gpt", "openai"}:
            return "openai"
        if pl in {"deepseek", "ds"}:
            return "deepseek"
        if pl in {"gemini", "google", "vertex"}:
            return "gemini"
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
            if "generativelanguage" in base or "googleapis.com" in base:
                return "gemini"
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
            {"content": <str>, "provider": "openai"|"deepseek", "model": <str>}
        """
        # Prefer request parameter; if missing/invalid, fall back to env default
        requested = self._normalize_provider(provider)
        primary = requested or self._default_provider()
        order = [primary]
        # Only consider cross-provider fallback when caller did not request a specific provider
        if retries and getattr(settings, "ENABLE_PROVIDER_FALLBACK", False) and not requested:
            order += [p for p in ("openai", "deepseek", "gemini") if p != primary]

        last_err: Optional[Exception] = None
        for prov in order:
            try:
                content, used_model = await self._call_provider(
                    prov, messages, model=model, temperature=temperature, max_tokens=max_tokens
                )
                return {"content": content, "provider": prov, "model": used_model}
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
    ) -> tuple[str, str]:
        cfg = self.providers.get(prov)
        if not cfg:
            raise RuntimeError(f"Unknown provider '{prov}'")
        base_url = cfg["base_url"]
        api_key = cfg["api_key"]
        use_model = model or cfg["default_model"]
        if not api_key:
            raise RuntimeError(f"Missing API key for provider '{prov}'")

        if prov == "gemini":
            return await self._call_gemini(base_url, api_key, messages, use_model, temperature, max_tokens)

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
            try:
                resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                detail = None
                try:
                    errj = exc.response.json()
                    detail = errj.get("error", {}).get("message") or errj.get("message") or str(errj)
                except Exception:
                    try:
                        detail = exc.response.text
                    except Exception:
                        detail = str(exc)
                raise RuntimeError(
                    f"{prov} request failed ({status}) for model '{use_model}': {detail}"
                ) from exc

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content, use_model

    def _gemini_model_path(self, model: str) -> str:
        m = (model or "").strip()
        if not m:
            m = "gemini-1.5-pro"
        return m if m.startswith("models/") else f"models/{m}"

    def _gemini_convert_messages(self, messages: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str], List[dict]]:
        """Split out guidance (systemInstruction) and an optional large context block.

        Returns (system_instruction_text, context_text, contents_without_system)
        """
        system_text: Optional[str] = None
        contents: List[dict] = []
        for msg in messages:
            role = (msg.get("role") or "").strip().lower()
            text = str(msg.get("content") or "")
            if role == "system":
                if system_text is None:
                    system_text = text
                else:
                    system_text += "\n" + text
                continue
            g_role = "user" if role == "user" else "model"
            contents.append({"role": g_role, "parts": [{"text": text}]})

        context_text: Optional[str] = None
        if isinstance(system_text, str) and system_text:
            marker = "\nContext:\n"
            idx = system_text.find(marker)
            if idx != -1:
                pre = system_text[:idx]
                rest = system_text[idx + len(marker):]
                # Preserve trailing guidance if present after the context block
                tail_key = "Every factual statement"
                tail_idx = rest.find(tail_key)
                if tail_idx != -1:
                    context_text = rest[:tail_idx].strip()
                    tail = rest[tail_idx:]
                    system_text = (pre + "\n" + tail).strip()
                else:
                    context_text = rest.strip()
                    system_text = pre.strip()

        return system_text, context_text, contents

    async def _gemini_get_or_create_cached(self, client: httpx.AsyncClient, base_url: str, api_key: str, text: str) -> Optional[str]:
        # Generative Language API does not support cachedContents; skip when using that API
        if "generativelanguage.googleapis.com" in (base_url or ""):
            return None
        ttl = getattr(settings, "GEMINI_CACHE_TTL_SECONDS", 1800)
        min_chars = getattr(settings, "GEMINI_CACHE_MIN_CHARS", 4000)
        if len(text or "") < max(0, int(min_chars)):
            return None
        # Keyed by hash of content
        key = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        now = time.time()
        cached = self._gemini_cache.get(key)
        if cached and cached[1] > now:
            return cached[0]
        # Create cached content resource via API
        url = f"{base_url}/cachedContents?key={api_key}"
        payload = {
            "displayName": f"cached-context-{key[:8]}",
            "ttl": f"{int(ttl)}s",
            "contents": [
                {"role": "user", "parts": [{"text": text}]}
            ],
        }
        try:
            resp = await client.post(url, headers={"Content-Type": "application/json"}, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # If caching fails (e.g., not enabled for project), fall back to sending full text
            logger.warning("Gemini cachedContents creation failed: %s", exc)
            return None
        data = resp.json()
        name = data.get("name")
        if isinstance(name, str) and name:
            self._gemini_cache[key] = (name, now + int(ttl))
            return name
        return None

    async def _gemini_get_model_output_limit(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        model_path: str,
    ) -> Optional[int]:
        # Try cached
        if model_path in self._gemini_limits:
            return self._gemini_limits.get(model_path)
        try:
            url = f"{base_url}/{model_path}?key={api_key}"
            resp = await client.get(url, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            limit = data.get("outputTokenLimit")
            if isinstance(limit, int) and limit > 0:
                self._gemini_limits[model_path] = int(limit)
                return int(limit)
        except Exception:
            return None
        return None

    async def _call_gemini(
        self,
        base_url: str,
        api_key: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> tuple[str, str]:
        model_path = self._gemini_model_path(model)
        # Split incoming messages into guidance (system), optional large context, and the turn contents
        initial_system_text, initial_context_text, initial_contents = self._gemini_convert_messages(messages)
        gen_cfg: dict = {"responseMimeType": "text/plain"}
        if temperature is not None:
            gen_cfg["temperature"] = float(temperature)
        # Always honor GEMINI_MAX_TOKENS cap; if caller provided a lower max, keep the lower one.
        try:
            cap = int(getattr(settings, "GEMINI_MAX_TOKENS", 8192))
        except Exception:
            cap = 8192
        if max_tokens is not None:
            try:
                provided = int(max_tokens)
            except Exception:
                provided = cap
            gen_cfg["maxOutputTokens"] = max(1, min(cap, provided))
        else:
            gen_cfg["maxOutputTokens"] = max(1, cap)
        top_p = getattr(settings, "CHAT_TOP_P", None)
        if top_p is not None:
            gen_cfg["topP"] = float(top_p)

        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            # At most two attempts: initial + retry on MAX_TOKENS with reduced output and truncated payload
            attempt = 0
            sys_text_for_attempt = initial_system_text
            context_text_for_attempt = initial_context_text
            # Align maxOutputTokens with model's output limit if discoverable
            model_out_limit = await self._gemini_get_model_output_limit(client, base_url, api_key, model_path)
            cfg_for_attempt = dict(gen_cfg)
            try:
                if isinstance(model_out_limit, int) and model_out_limit > 0:
                    current = int(cfg_for_attempt.get("maxOutputTokens", 0) or 0)
                    if current <= 0:
                        cfg_for_attempt["maxOutputTokens"] = int(model_out_limit)
                    else:
                        cfg_for_attempt["maxOutputTokens"] = min(int(model_out_limit), current)
            except Exception:
                pass
            while attempt < 2:
                # Prefer systemInstruction for guidance; attach large context as user content
                system_instruction: Optional[dict] = None
                local_contents = list(initial_contents)

                if context_text_for_attempt:
                    can_cache = (
                        getattr(settings, "GEMINI_ENABLE_CONTEXT_CACHE", True)
                        and attempt == 0
                        and "generativelanguage.googleapis.com" not in (base_url or "")
                    )
                    if can_cache:
                        cached_name = await self._gemini_get_or_create_cached(client, base_url, api_key, context_text_for_attempt)
                        if cached_name:
                            local_contents = [{"role": "user", "parts": [{"cachedContent": cached_name}]}] + local_contents
                        else:
                            local_contents = [{"role": "user", "parts": [{"text": context_text_for_attempt}]}] + local_contents
                    else:
                        local_contents = [{"role": "user", "parts": [{"text": context_text_for_attempt}]}] + local_contents

                if sys_text_for_attempt:
                    system_instruction = {"parts": [{"text": sys_text_for_attempt}]}

                url = f"{base_url}/{model_path}:generateContent?key={api_key}"
                payload: dict = {"contents": local_contents}
                if system_instruction is not None:
                    payload["systemInstruction"] = system_instruction
                if cfg_for_attempt:
                    payload["generationConfig"] = cfg_for_attempt

                try:
                    resp = await client.post(url, headers={"Content-Type": "application/json"}, json=payload)
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code if exc.response is not None else "?"
                    detail = None
                    try:
                        errj = exc.response.json()
                        detail = (
                            errj.get("error", {}).get("message")
                            or errj.get("message")
                            or str(errj)
                        )
                    except Exception:
                        try:
                            detail = exc.response.text
                        except Exception:
                            detail = str(exc)
                    raise RuntimeError(
                        f"gemini request failed ({status}) for model '{model_path}': {detail}"
                    ) from exc

                data = resp.json()
                # Extract first candidate text; fall back to top-level text if present
                content_text = ""
                finish = None
                try:
                    cand0 = (data.get("candidates") or [{}])[0] or {}
                    finish = cand0.get("finishReason")
                    parts = (cand0.get("content") or {}).get("parts") or []
                    texts = []
                    for p in parts:
                        if not isinstance(p, dict):
                            continue
                        t = p.get("text")
                        if isinstance(t, str) and t.strip():
                            texts.append(t)
                    content_text = "".join(texts).strip()
                    if not content_text and isinstance(data.get("text"), str):
                        content_text = data.get("text").strip()
                except Exception:
                    content_text = ""

                if content_text:
                    return content_text, model_path

                # Retry strategy on MAX_TOKENS: shrink output, truncate payload
                if finish == "MAX_TOKENS" and attempt == 0:
                    try:
                        current_max = int(cfg_for_attempt.get("maxOutputTokens", 1024))
                        cfg_for_attempt["maxOutputTokens"] = max(256, current_max // 2)
                    except Exception:
                        cfg_for_attempt["maxOutputTokens"] = 512
                    try:
                        limit = int(getattr(settings, "GEMINI_TRUNCATE_SYSTEM_CHARS", 60000))
                        if isinstance(context_text_for_attempt, str) and len(context_text_for_attempt) > limit:
                            context_text_for_attempt = context_text_for_attempt[:limit] + "\n\n[... truncated for length ...]"
                        elif isinstance(sys_text_for_attempt, str) and len(sys_text_for_attempt) > limit:
                            sys_text_for_attempt = sys_text_for_attempt[:limit] + "\n\n[... truncated for length ...]"
                    except Exception:
                        pass
                    attempt += 1
                    continue

                # No content and not retryable
                msg = f"gemini returned empty text (finish={finish})"
                raise RuntimeError(msg)

    async def chat_completion_batch(
        self,
        batches: List[List[Dict[str, str]]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Process multiple chat requests concurrently using asyncio.gather.

        Returns a result list aligned with input order. Exceptions are converted
        to error dicts so the batch always resolves.
        """
        tasks = [
            self.chat_completion(
                messages=msgs,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=getattr(settings, "CHAT_MAX_RETRIES", 2),
            )
            for msgs in batches
        ]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)
        final: List[Dict[str, str]] = []
        for res in results_raw:
            if isinstance(res, Exception):
                logger.error("Batch chat task failed: %s", res)
                final.append({
                    "content": f"Error: {res}",
                    "provider": provider or "unknown",
                    "model": model or "unknown",
                })
            else:
                final.append(res)
        return final

llm_provider = LLMProvider()
