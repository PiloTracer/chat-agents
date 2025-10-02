# deepseek_integration_reference.py
"""
DEEPSEEK INTEGRATION REFERENCE FOR MULTI-AGENT RAG CHATBOT

This reference document shows how to integrate DeepSeek API into the existing
OpenAI-based multi-agent RAG system. The integration maintains the same architecture
while adding DeepSeek as a configurable provider option.

Key Integration Points:
1. Configuration updates for multiple providers
2. Provider-agnostic embedding service
3. Provider-agnostic chat completion service
4. Fallback mechanisms between providers
5. Consistent error handling and retries
"""

# =============================================================================
# 1. ENVIRONMENT CONFIGURATION UPDATES
# =============================================================================

"""
Add to your .env.example.high.txt:

# Provider selection (openai/deepseek)
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai

# DeepSeek specific settings
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
DEEPSEEK_EMBEDDING_MODEL=deepseek-embedder
DEEPSEEK_CHAT_MODEL=deepseek-chat

# Fallback configuration
ENABLE_PROVIDER_FALLBACK=true
PRIMARY_CHAT_PROVIDER=openai
PRIMARY_EMBEDDING_PROVIDER=openai
"""

# =============================================================================
# 2. UPDATED CONFIGURATION (config.py)
# =============================================================================

"""
Update your config.py to support multiple providers:

class Settings:
    # Provider selection
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
    ENABLE_PROVIDER_FALLBACK = os.getenv("ENABLE_PROVIDER_FALLBACK", "true").lower() == "true"
    
    # Provider base URLs
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    
    # Model configuration with provider support
    @property
    def CHAT_PROVIDER_BASE_URL(self):
        if self.LLM_PROVIDER == "deepseek":
            return self.DEEPSEEK_BASE_URL
        return self.OPENAI_BASE_URL
    
    @property
    def EMBEDDING_PROVIDER_BASE_URL(self):
        if self.EMBEDDING_PROVIDER == "deepseek":
            return self.DEEPSEEK_BASE_URL
        return self.OPENAI_BASE_URL
    
    @property
    def ACTIVE_API_KEY(self):
        if self.LLM_PROVIDER == "deepseek":
            return self.DEEPSEEK_API_KEY
        return self.OPENAI_API_KEY
    
    @property
    def ACTIVE_EMBEDDING_KEY(self):
        if self.EMBEDDING_PROVIDER == "deepseek":
            return self.DEEPSEEK_API_KEY
        return self.OPENAI_API_KEY

    # Model names with provider mapping
    @property
    def EMBEDDING_MODEL(self):
        provider_models = {
            "openai": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            "deepseek": os.getenv("DEEPSEEK_EMBEDDING_MODEL", "deepseek-embedder")
        }
        return provider_models.get(self.EMBEDDING_PROVIDER, "text-embedding-3-large")
    
    @property
    def CHAT_MODEL(self):
        provider_models = {
            "openai": os.getenv("CHAT_MODEL", "gpt-4"),
            "deepseek": os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
        }
        return provider_models.get(self.LLM_PROVIDER, "gpt-4")
"""

# =============================================================================
# 3. ENHANCED EMBEDDINGS SERVICE (embeddings.py)
# =============================================================================

"""
Update your embeddings.py to support multiple providers with fallback:

class EmbeddingProvider:
    def __init__(self):
        self.providers = [
            {
                'name': 'openai',
                'base_url': settings.OPENAI_BASE_URL,
                'api_key': settings.OPENAI_API_KEY,
                'models': ['text-embedding-3-large', 'text-embedding-3-small']
            },
            {
                'name': 'deepseek', 
                'base_url': settings.DEEPSEEK_BASE_URL,
                'api_key': settings.DEEPSEEK_API_KEY,
                'models': ['deepseek-embedder']
            }
        ]
        self.current_provider = settings.EMBEDDING_PROVIDER
        self.current_model = settings.EMBEDDING_MODEL
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(3):  # Retry with different providers
            try:
                provider_config = next(p for p in self.providers if p['name'] == self.current_provider)
                return await self._call_provider(provider_config, texts)
            except Exception as e:
                if settings.ENABLE_PROVIDER_FALLBACK and attempt < 2:
                    self._switch_to_fallback()
                    continue
                raise
        return []
    
    def _switch_to_fallback(self):
        # Switch to the other provider
        self.current_provider = 'openai' if self.current_provider == 'deepseek' else 'deepseek'
        # Update model accordingly
        self.current_model = settings.EMBEDDING_MODEL  # This will use the appropriate model for the provider
        logger.warning(f"Switched embedding provider to {self.current_provider}")
    
    async def _call_provider(self, provider_config: dict, texts: List[str]) -> List[List[float]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider_config['api_key']}"
        }
        
        payload = {
            "model": self.current_model,
            "input": texts
        }
        
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{provider_config['base_url']}/embeddings",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data:
                raise RuntimeError(f"Embedding response missing data field: {data}")
            
            embeddings = []
            for item in data["data"]:
                embedding = item.get("embedding")
                if embedding is None:
                    raise RuntimeError(f"Embedding item missing vector for model '{self.current_model}'")
                embeddings.append(embedding)
            
            return embeddings

# Global embedding provider instance
embedding_provider = EmbeddingProvider()

# Updated embed_texts function
async def embed_texts(texts: List[str]) -> List[List[float]]:
    return await embedding_provider.get_embeddings(texts)
"""

# =============================================================================
# 4. ENHANCED CHAT COMPLETION SERVICE
# =============================================================================

"""
Create a new file: app/llm_provider.py

This provides a unified interface for both OpenAI and DeepSeek chat completions.
"""

class LLMProvider:
    def __init__(self):
        self.providers = {
            'openai': {
                'base_url': settings.OPENAI_BASE_URL,
                'api_key': settings.OPENAI_API_KEY,
                'models': ['gpt-4', 'gpt-3.5-turbo']
            },
            'deepseek': {
                'base_url': settings.DEEPSEEK_BASE_URL, 
                'api_key': settings.DEEPSEEK_API_KEY,
                'models': ['deepseek-chat', 'deepseek-coder']
            }
        }
        self.current_provider = settings.LLM_PROVIDER
        self.current_model = settings.CHAT_MODEL
    
    async def chat_completion(
        self,
        messages: List[Dict],
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False
    ) -> str:
        for attempt in range(3):  # Retry with different providers
            try:
                provider_config = self.providers[self.current_provider]
                return await self._call_provider(provider_config, messages, temperature, max_tokens, stream)
            except Exception as e:
                if settings.ENABLE_PROVIDER_FALLBACK and attempt < 2:
                    self._switch_to_fallback()
                    continue
                raise
        
        raise RuntimeError("All LLM providers failed")
    
    def _switch_to_fallback(self):
        self.current_provider = 'openai' if self.current_provider == 'deepseek' else 'deepseek'
        self.current_model = settings.CHAT_MODEL  # Will use appropriate model for provider
        logger.warning(f"Switched LLM provider to {self.current_provider}")
    
    async def _call_provider(
        self,
        provider_config: dict,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        stream: bool
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider_config['api_key']}"
        }
        
        payload = {
            "model": self.current_model,
            "messages": messages,
            "temperature": temperature or settings.CHAT_TEMPERATURE,
            "max_tokens": max_tokens or settings.CHAT_MAX_TOKENS,
            "top_p": settings.CHAT_TOP_P,
            "stream": stream
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{provider_config['base_url']}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            return data["choices"][0]["message"]["content"]

# Global LLM provider instance
llm_provider = LLMProvider()

# =============================================================================
# 5. UPDATED AGENTS ROUTING (agents.py)
# =============================================================================

"""
Update your agents.py to use the provider-agnostic LLM service:

async def route_question(question: str, agents: Dict[str, str]) -> str:
    if not agents:
        raise ValueError("No agents available for routing.")

    system_prompt = _router_system_prompt(agents)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    try:
        # Use the provider-agnostic LLM service
        slug = await llm_provider.chat_completion(
            messages=messages,
            temperature=0,  # Deterministic routing
            max_tokens=50   # Short response for slug only
        )
        slug = slug.strip()
        
        return slug if slug in agents else next(iter(agents))
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        # Fallback to first available agent
        return next(iter(agents))
"""

# =============================================================================
# 6. UPDATED CHAT ENDPOINT (chat.py)
# =============================================================================

"""
Update your chat.py to use the provider-agnostic services:

@router.post("/ask")
async def ask(
    payload: AskPayload,
    principal: Principal = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
):
    # ... existing context preparation code ...
    
    # Build the system prompt (same as before)
    system_prompt = build_system_prompt(agent_slug, context_blocks, agent_map)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    # Use provider-agnostic LLM service with retries
    try:
        answer = await llm_provider.chat_completion(
            messages=messages,
            temperature=settings.CHAT_TEMPERATURE,
            max_tokens=settings.CHAT_MAX_TOKENS
        )
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM service error: {e}")
    
    return {
        "ok": True,
        "agent": agent_slug,
        "answer": answer,
        "sources": response_sources,
        "sources_meta": sources_meta,
        "provider": llm_provider.current_provider  # Include provider info in response
    }
"""

# =============================================================================
# 7. DEEPSEEK-SPECIFIC OPTIMIZATIONS
# =============================================================================

"""
DeepSeek-specific optimizations for better performance:

1. CONTEXT WINDOW OPTIMIZATION:
   - DeepSeek supports 128K context window vs OpenAI's smaller windows
   - Adjust chunking strategy to leverage larger context

2. COST OPTIMIZATION:
   - DeepSeek is typically more cost-effective for large contexts
   - Consider increasing TOP_K when using DeepSeek to leverage larger context

3. PROMPT ENGINEERING:
   - DeepSeek may respond better to slightly different prompt structures
   - Consider provider-specific prompt tuning
"""

class DeepSeekOptimizer:
    @staticmethod
    def optimize_system_prompt(prompt: str, provider: str) -> str:
        """Optimize system prompts for specific providers."""
        if provider == "deepseek":
            # Add DeepSeek-specific instructions if needed
            prompt += "\n\nPlease provide comprehensive, well-structured answers with proper citations."
        return prompt
    
    @staticmethod
    def get_optimal_parameters(provider: str) -> Dict:
        """Get optimal parameters for each provider."""
        base_params = {
            "temperature": settings.CHAT_TEMPERATURE,
            "max_tokens": settings.CHAT_MAX_TOKENS,
            "top_p": settings.CHAT_TOP_P
        }
        
        if provider == "deepseek":
            # DeepSeek-specific optimizations
            base_params.update({
                "temperature": min(0.7, base_params["temperature"]),  # Slightly higher creativity
                "top_p": 0.9,  # Slightly more focused
            })
        
        return base_params

# =============================================================================
# 8. HEALTH CHECK AND PROVIDER MONITORING
# =============================================================================

"""
Add provider health monitoring:

async def check_provider_health() -> Dict[str, bool]:
    health_status = {}
    
    # Check OpenAI
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
            response = await client.get(f"{settings.OPENAI_BASE_URL}/models", headers=headers)
            health_status['openai'] = response.status_code == 200
    except:
        health_status['openai'] = False
    
    # Check DeepSeek
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"}
            response = await client.get(f"{settings.DEEPSEEK_BASE_URL}/models", headers=headers)
            health_status['deepseek'] = response.status_code == 200
    except:
        health_status['deepseek'] = False
    
    return health_status
"""

# =============================================================================
# 9. IMPLEMENTATION CHECKLIST
# =============================================================================

"""
IMPLEMENTATION CHECKLIST FOR YOUR CODING ASSISTANT:

✅ CONFIGURATION:
   - Add new environment variables for DeepSeek
   - Update config.py to support multiple providers
   - Add provider selection logic

✅ EMBEDDINGS SERVICE:
   - Create provider-agnostic embedding service
   - Implement fallback mechanism
   - Update embed_texts function

✅ CHAT COMPLETION SERVICE:
   - Create LLMProvider class
   - Implement provider-agnostic chat completion
   - Add retry and fallback logic

✅ AGENTS & ROUTING:
   - Update route_question to use new LLM service
   - Maintain existing agent logic

✅ CHAT ENDPOINT:
   - Update /ask endpoint to use provider-agnostic service
   - Add provider info to response

✅ OPTIMIZATIONS:
   - Implement provider-specific optimizations
   - Add health monitoring

✅ TESTING:
   - Test both providers independently
   - Test fallback scenarios
   - Verify embedding compatibility
   - Test large context handling

✅ DEPLOYMENT:
   - Update Docker configurations
   - Add DeepSeek API key to deployment
   - Configure provider selection
"""

# =============================================================================
# 10. DOCKER COMPOSE UPDATES (if needed)
# =============================================================================

"""
If you need to add any new services for DeepSeek integration, update docker-compose.yml:

# No additional services needed - DeepSeek is API-based
# Just ensure environment variables are properly set

environment:
  - LLM_PROVIDER=${LLM_PROVIDER:-openai}
  - EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER:-openai}
  - DEEPSEEK_API_BASE=${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}
  - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
  - DEEPSEEK_EMBEDDING_MODEL=${DEEPSEEK_EMBEDDING_MODEL:-deepseek-embedder}
  - DEEPSEEK_CHAT_MODEL=${DEEPSEEK_CHAT_MODEL:-deepseek-chat}
  - ENABLE_PROVIDER_FALLBACK=${ENABLE_PROVIDER_FALLBACK:-true}
"""

print("DeepSeek integration reference completed. Provide this to your coding assistant for implementation.")