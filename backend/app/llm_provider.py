# app/llm_provider.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import logging
import httpx
import hashlib
import time
import asyncio
from functools import lru_cache

try:
    import google.auth  # type: ignore
    from google.auth.transport.requests import Request  # type: ignore
    from google.auth.exceptions import DefaultCredentialsError, RefreshError  # type: ignore
except Exception:  # pragma: no cover
    google = None  # type: ignore

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
            "vertex": {
                # Base is derived from location below when used
                "base_url": f"https://{getattr(settings, 'VERTEX_LOCATION', 'us-east1')}-aiplatform.googleapis.com/v1",
                "api_key": "",  # OAuth via google.auth
                "default_model": getattr(settings, "VERTEX_CHAT_MODEL", "gemini-1.5-pro-002"),
            },
            "grok": {
                "base_url": getattr(settings, "GROK_API_BASE", "https://api.x.ai/v1").rstrip("/"),
                "api_key": getattr(settings, "GROK_API_KEY", ""),
                "default_model": getattr(settings, "GROK_CHAT_MODEL", "grok-4-fast-reasoning"),
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
        if pl in {"gemini", "google"}:
            return "gemini"
        if pl in {"vertex", "gcp", "google-vertex"}:
            return "vertex"
        if pl in {"grok", "xai"}:
            return "grok"
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
            if "x.ai" in base:
                return "grok"
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
            order += [p for p in ("openai", "deepseek", "gemini", "vertex", "grok") if p != primary]

        last_err: Optional[Exception] = None
        for prov in order:
            try:
                content, used_model = await self._call_provider(
                    prov, messages, model=model, temperature=temperature, max_tokens=max_tokens
                )
                return {"content": content, "provider": prov, "model": used_model}
            except Exception as e:
                last_err = e
                # Log provider failures with robust context for diagnostics
                try:
                    emsg = str(e) if str(e) else repr(e)
                except Exception:
                    emsg = repr(e)
                logger.warning("LLM call failed on %s: %s (%s)", prov, emsg, type(e).__name__)

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

        # Vertex uses OAuth (ADC), not API keys
        if prov == "vertex":
            return await self._call_vertex(messages, use_model, temperature, max_tokens)

        # Gemini GL requires API key
        if prov == "gemini":
            if not api_key:
                raise RuntimeError(f"Missing API key for provider '{prov}'")
            return await self._call_gemini(base_url, api_key, messages, use_model, temperature, max_tokens)

        if not api_key:
            raise RuntimeError(f"Missing API key for provider '{prov}'")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {"model": use_model, "messages": list(messages)}
        if temperature is not None:
            payload["temperature"] = temperature
        # Prefer provider-specific caps when available
        if prov == "grok":
            try:
                grok_cap = int(getattr(settings, "GROK_MAX_TOKENS", 16384))
            except Exception:
                grok_cap = 16384
            if max_tokens is None:
                payload["max_tokens"] = grok_cap
            else:
                try:
                    # Favor Grok's larger window for fuller responses
                    payload["max_tokens"] = max(1, max(int(max_tokens), grok_cap))
                except Exception:
                    payload["max_tokens"] = grok_cap
        elif max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            # Helper to perform a single chat.completions call
            async def _single_call(msgs: List[Dict[str, str]]) -> tuple[str, Optional[str]]:
                pl = {"model": use_model, "messages": msgs}
                if temperature is not None:
                    pl["temperature"] = temperature
                if max_tokens is not None:
                    pl["max_tokens"] = max_tokens
                try:
                    r = await client.post(f"{base_url}/chat/completions", headers=headers, json=pl)
                    r.raise_for_status()
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
                dj = r.json()
                try:
                    choice0 = (dj.get("choices") or [{}])[0] or {}
                    txt = ((choice0.get("message") or {}).get("content") or "").strip()
                    finish = choice0.get("finish_reason")
                except Exception:
                    txt = ""
                    finish = None
                return txt, finish if isinstance(finish, str) else None

            # First call
            acc, finish_reason = await _single_call(list(messages))
            if not acc:
                return "", use_model

            # Auto-continue for OpenAI-like providers when cut off by length
            if getattr(settings, "CHAT_AUTO_CONTINUE", True):
                calls = 1
                max_calls = max(1, int(getattr(settings, "CHAT_CONTINUE_MAX_CALLS", 12)))
                continue_prompt = str(getattr(settings, "CHAT_CONTINUE_PROMPT", "Continue"))
                base_history: List[Dict[str, str]] = list(messages)
                last_chunk = acc
                while isinstance(finish_reason, str) and finish_reason.lower() in {"length", "max_tokens"} and calls < max_calls:
                    # Extend conversation with only the last model chunk and a continue cue
                    extended: List[Dict[str, str]] = base_history + [
                        {"role": "assistant", "content": last_chunk},
                        {"role": "user", "content": continue_prompt},
                    ]
                    nxt, finish_reason = await _single_call(extended)
                    if not nxt:
                        break
                    acc += nxt
                    last_chunk = nxt
                    calls += 1
            return acc, use_model

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

    # ---------------- Vertex AI helpers ----------------
    def _vertex_endpoints(self, model: Optional[str]) -> Dict[str, str]:
        location = getattr(settings, "VERTEX_LOCATION", "us-east1").strip() or "us-east1"
        project = getattr(settings, "VERTEX_PROJECT_ID", "").strip()
        publisher = getattr(settings, "VERTEX_PUBLISHER", "google")
        model_name = (model or getattr(settings, "VERTEX_CHAT_MODEL", "gemini-1.5-pro-002")).strip()
        if not project:
            raise RuntimeError(
                "VERTEX_PROJECT_ID is not configured. Set VERTEX_PROJECT_ID to your GCP project ID in the backend environment."
            )
        base = f"https://{location}-aiplatform.googleapis.com/v1"
        model_path = f"projects/{project}/locations/{location}/publishers/{publisher}/models/{model_name}"
        return {
            "base": base,
            "model": model_path,
            "gen": f"{base}/{model_path}:generateContent",
            "count": f"{base}/{model_path}:countTokens",
            "cache": f"{base}/projects/{project}/locations/{location}/cachedContents",
        }

    def _vertex_access_token(self) -> str:
        if not getattr(settings, "VERTEX_USE_OAUTH", True):
            raise RuntimeError("VERTEX_USE_OAUTH=false is not supported; provide OAuth via ADC")
        if google is None:
            raise RuntimeError("google-auth is required for Vertex AI; install google-auth and configure ADC")
        # Optional diagnostics for service account file
        try:
            import os, json  # local import to avoid module-level dependency
            adc = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if adc and not os.path.exists(adc):
                raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS points to missing file: {adc}")
            if adc:
                try:
                    with open(adc, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if str(data.get("type")) != "service_account":
                        raise RuntimeError(
                            f"GOOGLE_APPLICATION_CREDENTIALS must be a service account key; found type='{data.get('type')}'."
                        )
                except Exception:
                    # Non-fatal; continue to default() which will surface errors
                    pass
        except Exception:
            pass

        try:
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])  # type: ignore
        except DefaultCredentialsError as e:
            raise RuntimeError(
                f"Failed to obtain Google ADC: {e}. Provide a valid service account JSON via GOOGLE_APPLICATION_CREDENTIALS, "
                "or run 'gcloud auth application-default login' in dev environments."
            ) from e
        try:
            if not credentials.valid:
                request = Request()
                credentials.refresh(request)
        except RefreshError as e:
            raise RuntimeError(
                "Vertex OAuth refresh failed (invalid_grant). Likely causes: invalid/old service account key, clock skew, or revoked key."
            ) from e
        return str(credentials.token)

    async def _vertex_get_model_output_limit(self, client: httpx.AsyncClient, endpoints: Dict[str, str], token: str) -> Optional[int]:
        try:
            url = f"{endpoints['base']}/{endpoints['model']}"
            project = endpoints["model"].split("/")[1] if "/" in endpoints["model"] else None
            headers = {"Authorization": f"Bearer {token}"}
            if project:
                headers["x-goog-user-project"] = project
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            lim = data.get("outputTokenLimit")
            return int(lim) if isinstance(lim, int) and lim > 0 else None
        except Exception:
            return None

    async def _vertex_create_cached(self, client: httpx.AsyncClient, endpoints: Dict[str, str], token: str, text: str) -> Optional[str]:
        ttl = getattr(settings, "VERTEX_CACHE_TTL_SECONDS", 1800)
        min_chars = getattr(settings, "VERTEX_CACHE_MIN_CHARS", 4000)
        if len(text or "") < max(0, int(min_chars)):
            return None
        payload = {
            "displayName": f"cached-context-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}",
            "ttl": f"{int(ttl)}s",
            "contents": [{"role": "user", "parts": [{"text": text}]}],
        }
        try:
            project = endpoints["model"].split("/")[1] if "/" in endpoints["model"] else None
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            if project:
                headers["x-goog-user-project"] = project
            resp = await client.post(endpoints["cache"], headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            name = data.get("name")
            return str(name) if isinstance(name, str) and name else None
        except Exception:
            return None

    async def _vertex_count_tokens(self, client: httpx.AsyncClient, endpoints: Dict[str, str], token: str, payload: Dict[str, object]) -> Optional[int]:
        try:
            project = endpoints["model"].split("/")[1] if "/" in endpoints["model"] else None
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            if project:
                headers["x-goog-user-project"] = project
            resp = await client.post(endpoints["count"], headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # totalTokens or usage metadata fields may exist
            total = data.get("totalTokens") or data.get("inputTokenCount")
            return int(total) if isinstance(total, int) and total > 0 else None
        except Exception:
            return None

    async def _call_vertex(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> tuple[str, str]:
        endpoints = self._vertex_endpoints(model)
        token = self._vertex_access_token()

        initial_system_text, initial_context_text, initial_contents = self._gemini_convert_messages(messages)

        gen_cfg: dict = {"responseMimeType": "text/plain"}
        if temperature is not None:
            gen_cfg["temperature"] = float(temperature)
        top_p = getattr(settings, "CHAT_TOP_P", None)
        if top_p is not None:
            gen_cfg["topP"] = float(top_p)
        # Cap with VERTEX_CHAT_MAX_TOKENS (Vertex-specific), not global CHAT_MAX_TOKENS
        try:
            cap = int(getattr(settings, "VERTEX_CHAT_MAX_TOKENS", getattr(settings, "VERTEX_MAX_TOKENS", 8192)))
        except Exception:
            cap = 8192

        # Larger timeout for Vertex to avoid ReadTimeout on long generations
        try:
            vtimeout = float(getattr(settings, "VERTEX_HTTP_TIMEOUT_SECONDS", max(60.0, float(settings.HTTP_TIMEOUT_SECONDS))))
        except Exception:
            vtimeout = max(60.0, float(settings.HTTP_TIMEOUT_SECONDS))
        http_timeout = httpx.Timeout(timeout=vtimeout, connect=min(30.0, vtimeout), read=vtimeout, write=min(60.0, vtimeout), pool=min(30.0, vtimeout))
        async with httpx.AsyncClient(timeout=http_timeout) as client:
            # Base payload for counting
            base_payload: Dict[str, object] = {"contents": list(initial_contents)}
            if initial_system_text:
                base_payload["systemInstruction"] = {"parts": [{"text": initial_system_text}]}
            # Attach context for counting
            if initial_context_text:
                base_payload["contents"] = [{"role": "user", "parts": [{"text": initial_context_text}]}] + list(initial_contents)

            model_limit = await self._vertex_get_model_output_limit(client, endpoints, token)
            # Optionally call countTokens (disabled by default to reduce latency)
            if getattr(settings, "VERTEX_ENABLE_COUNT_TOKENS", False):
                _ = await self._vertex_count_tokens(client, endpoints, token, base_payload)
            effective_cap = min([v for v in [cap, model_limit] if isinstance(v, int) and v > 0] or [cap])
            available = max(256, effective_cap)
            gen_cfg["maxOutputTokens"] = available

            # Attempt, with one retry on MAX_TOKENS
            attempt = 0
            sys_text_for_attempt = initial_system_text
            context_text_for_attempt = initial_context_text
            cfg_for_attempt = dict(gen_cfg)
            cached_context_name: Optional[str] = None
            while attempt < 2:
                local_contents = list(initial_contents)
                system_instruction = None
                if context_text_for_attempt and getattr(settings, "VERTEX_ENABLE_CONTEXT_CACHE", True):
                    if cached_context_name is None and attempt == 0:
                        cached_context_name = await self._vertex_create_cached(client, endpoints, token, context_text_for_attempt)
                    if cached_context_name:
                        local_contents = [{"role": "user", "parts": [{"cachedContent": cached_context_name}]}] + local_contents
                    else:
                        local_contents = [{"role": "user", "parts": [{"text": context_text_for_attempt}]}] + local_contents
                elif context_text_for_attempt:
                    local_contents = [{"role": "user", "parts": [{"text": context_text_for_attempt}]}] + local_contents

                if sys_text_for_attempt:
                    system_instruction = {"parts": [{"text": sys_text_for_attempt}]}

                payload: Dict[str, object] = {"contents": local_contents, "generationConfig": cfg_for_attempt}
                if system_instruction is not None:
                    payload["systemInstruction"] = system_instruction
                try:
                    # Include project header to aid quota attribution when required
                    project = endpoints["model"].split("/")[1] if "/" in endpoints["model"] else None
                    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                    if project:
                        headers["x-goog-user-project"] = project
                    resp = await client.post(endpoints["gen"], headers=headers, json=payload)
                    resp.raise_for_status()
                except httpx.TimeoutException as exc:
                    raise RuntimeError(
                        f"vertex request timed out after {int(vtimeout)}s calling '{endpoints['gen']}'. "
                        f"Consider increasing VERTEX_HTTP_TIMEOUT_SECONDS or lowering max tokens."
                    ) from exc
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code if exc.response is not None else "?"
                    detail = None
                    try:
                        errj = exc.response.json()
                        # Prefer rich Google error format when available
                        if isinstance(errj, dict) and "error" in errj:
                            gerr = errj.get("error") or {}
                            msg = gerr.get("message") or ""
                            status_txt = gerr.get("status") or ""
                            code_num = gerr.get("code")
                            detail = f"{msg} (status={status_txt}, code={code_num})"
                        else:
                            detail = errj.get("message") or str(errj)
                    except Exception:
                        try:
                            detail = exc.response.text
                        except Exception:
                            detail = str(exc)
                    raise RuntimeError(
                        f"vertex request failed ({status}) for model '{endpoints['model']}': {detail}"
                    ) from exc

                data = resp.json()
                # Extract text
                content_text = ""
                finish = None
                try:
                    cand0 = (data.get("candidates") or [{}])[0] or {}
                    finish = cand0.get("finishReason")
                    parts = (cand0.get("content") or {}).get("parts") or []
                    texts = [p.get("text") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
                    content_text = "".join(texts).strip()
                except Exception:
                    content_text = ""

                if content_text:
                    # Auto-continue when hitting MAX_TOKENS by making follow-up calls
                    if getattr(settings, "CHAT_AUTO_CONTINUE", True) and finish == "MAX_TOKENS":
                        acc = content_text
                        calls = 1
                        max_calls = max(1, int(getattr(settings, "CHAT_CONTINUE_MAX_CALLS", 12)))
                        continue_prompt = str(getattr(settings, "CHAT_CONTINUE_PROMPT", "Continue"))
                        last_chunk = content_text
                        while finish == "MAX_TOKENS" and calls < max_calls:
                            cont_contents = []
                            if context_text_for_attempt:
                                if cached_context_name:
                                    cont_contents.append({"role": "user", "parts": [{"cachedContent": cached_context_name}]})
                                else:
                                    cont_contents.append({"role": "user", "parts": [{"text": context_text_for_attempt}]})
                            cont_contents += list(initial_contents)
                            cont_contents.append({"role": "model", "parts": [{"text": last_chunk}]})
                            cont_contents.append({"role": "user", "parts": [{"text": continue_prompt}]})

                            payload2: Dict[str, object] = {"contents": cont_contents, "generationConfig": cfg_for_attempt}
                            if system_instruction is not None:
                                payload2["systemInstruction"] = system_instruction
                            try:
                                r2 = await client.post(endpoints["gen"], headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=payload2)
                                r2.raise_for_status()
                            except httpx.TimeoutException as exc:
                                raise RuntimeError(
                                    f"vertex request timed out after {int(vtimeout)}s during continuation calling '{endpoints['gen']}'."
                                ) from exc
                            d2 = r2.json()
                            try:
                                cand2 = (d2.get("candidates") or [{}])[0] or {}
                                finish = cand2.get("finishReason")
                                parts2 = (cand2.get("content") or {}).get("parts") or []
                                more = "".join([p.get("text") for p in parts2 if isinstance(p, dict) and isinstance(p.get("text"), str)]).strip()
                            except Exception:
                                more = ""
                                finish = None
                            if not more:
                                break
                            acc += more
                            last_chunk = more
                            calls += 1
                        return acc, endpoints["model"]
                    return content_text, endpoints["model"]

                if finish == "MAX_TOKENS" and attempt == 0:
                    # Retry with smaller output and truncated payload
                    try:
                        current_max = int(cfg_for_attempt.get("maxOutputTokens", 1024))
                        cfg_for_attempt["maxOutputTokens"] = max(256, current_max // 2)
                    except Exception:
                        cfg_for_attempt["maxOutputTokens"] = 512
                    try:
                        limit = int(getattr(settings, "VERTEX_TRUNCATE_SYSTEM_CHARS", 60000))
                        if isinstance(context_text_for_attempt, str) and len(context_text_for_attempt) > limit:
                            context_text_for_attempt = context_text_for_attempt[:limit] + "\n\n[... truncated for length ...]"
                        elif isinstance(sys_text_for_attempt, str) and len(sys_text_for_attempt) > limit:
                            sys_text_for_attempt = sys_text_for_attempt[:limit] + "\n\n[... truncated for length ...]"
                    except Exception:
                        pass
                    attempt += 1
                    continue

                raise RuntimeError(f"vertex returned empty text (finish={finish})")

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
            # Persist cached context name across attempts and continuations
            cached_context_name: Optional[str] = None
            while attempt < 2:
                # Prefer systemInstruction for guidance; attach large context as user content
                system_instruction: Optional[dict] = None
                local_contents = list(initial_contents)

                if context_text_for_attempt:
                    can_cache = (
                        getattr(settings, "GEMINI_ENABLE_CONTEXT_CACHE", True)
                        and "generativelanguage.googleapis.com" not in (base_url or "")
                    )
                    if can_cache and cached_context_name is None:
                        cached_context_name = await self._gemini_get_or_create_cached(client, base_url, api_key, context_text_for_attempt)
                    if cached_context_name:
                        local_contents = [{"role": "user", "parts": [{"cachedContent": cached_context_name}]}] + local_contents
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
                chunk_text = ""
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
                    chunk_text = "".join(texts).strip()
                    if not chunk_text and isinstance(data.get("text"), str):
                        chunk_text = data.get("text").strip()
                except Exception:
                    chunk_text = ""

                if not chunk_text:
                    # Retry strategy when no content was produced and MAX_TOKENS signaled
                    if finish == "MAX_TOKENS" and attempt == 0:
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
                    msg = f"gemini returned empty text (finish={finish})"
                    raise RuntimeError(msg)

                # If content exists and was cut due to MAX_TOKENS, auto-continue across calls
                acc = chunk_text
                if getattr(settings, "CHAT_AUTO_CONTINUE", True):
                    calls = 1
                    max_calls = max(1, int(getattr(settings, "CHAT_CONTINUE_MAX_CALLS", 12)))
                    continue_prompt = str(getattr(settings, "CHAT_CONTINUE_PROMPT", "Continue"))
                    last_chunk = chunk_text
                    while finish == "MAX_TOKENS" and calls < max_calls:
                        # Build minimal history to continue: context + initial turn + last model chunk + 'continue'
                        cont_contents = []
                        if context_text_for_attempt:
                            if cached_context_name:
                                cont_contents.append({"role": "user", "parts": [{"cachedContent": cached_context_name}]})
                            else:
                                cont_contents.append({"role": "user", "parts": [{"text": context_text_for_attempt}]})
                        cont_contents += list(initial_contents)
                        cont_contents.append({"role": "model", "parts": [{"text": last_chunk}]})
                        cont_contents.append({"role": "user", "parts": [{"text": continue_prompt}]})

                        payload2: dict = {"contents": cont_contents, "generationConfig": cfg_for_attempt}
                        if sys_text_for_attempt:
                            payload2["systemInstruction"] = {"parts": [{"text": sys_text_for_attempt}]}
                        r2 = await client.post(f"{base_url}/{model_path}:generateContent?key={api_key}", headers={"Content-Type": "application/json"}, json=payload2)
                        r2.raise_for_status()
                        dj2 = r2.json()
                        try:
                            cand = (dj2.get("candidates") or [{}])[0] or {}
                            finish = cand.get("finishReason")
                            parts2 = (cand.get("content") or {}).get("parts") or []
                            more = "".join([p.get("text") for p in parts2 if isinstance(p, dict) and isinstance(p.get("text"), str)])
                            more = (more or "").strip()
                        except Exception:
                            more = ""
                            finish = None
                        if not more:
                            break
                        acc += more
                        last_chunk = more
                        calls += 1

                return acc, model_path

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
