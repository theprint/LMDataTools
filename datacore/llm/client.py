# datacore/llm/client.py

import os
import time
import random
from openai import OpenAI
import httpx
from datacore.config.settings import config
from typing import Optional, Dict, Any, Union


# Errors that are worth retrying (transient / recoverable)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors that should trigger a retry."""
    msg = str(exc).lower()
    # httpx timeouts
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    # OpenAI API errors carry a status_code attribute
    status = getattr(exc, "status_code", None)
    if status in _RETRYABLE_STATUS_CODES:
        return True
    # Rate-limit signals embedded in message text
    if "rate limit" in msg or "too many requests" in msg:
        return True
    if "timeout" in msg or "timed out" in msg:
        return True
    if "connection" in msg and ("reset" in msg or "refused" in msg or "error" in msg):
        return True
    return False


class LLMClient:
    """Unified LLM client using OpenAI SDK.

    v2 additions
    ------------
    * Automatic retry with exponential backoff on transient errors (timeouts,
      rate limits, 5xx).  ``max_retries`` defaults to 3; set to 0 to disable.
    * Cumulative token-usage counters accessible via ``get_usage_stats()``.
      These are populated from the API's usage object on non-streaming calls.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 6000,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        self.base_url = base_url if base_url is not None else config.LLM_BASE_URL
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY

        # The OpenAI client treats an empty string as an invalid key.
        if self.api_key == "":
            self.api_key = None

        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        # Cumulative token counters (populated from non-streaming responses)
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_calls: int = 0

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

        if self.api_key and len(self.api_key) >= 12:
            key_display = f"{self.api_key[:4]}...{self.api_key[-8:]}"
        elif self.api_key:
            key_display = f"({len(self.api_key)} chars)"
        else:
            key_display = repr(self.api_key)
        print(f"[LLM] init base_url={self.client.base_url} key={key_display} model={self.default_model}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        model: Optional[str] = None,
        return_dict: bool = False,
        stream: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Make an LLM call with automatic retry on transient errors.

        Parameters
        ----------
        prompt:
            User prompt text.
        system_prompt:
            Optional system prompt.
        temperature:
            Overrides ``default_temperature`` for this call.
        max_tokens:
            Overrides ``default_max_tokens`` for this call.
        top_p:
            Optional top-p sampling parameter.
        model:
            Overrides ``default_model`` for this call.
        return_dict:
            If ``True``, return a dict with keys ``response``, ``model``,
            ``prompt_tokens``, ``completion_tokens`` instead of a plain string.
        stream:
            If ``True``, stream the response (no token counts available).
        extra_body:
            Extra fields forwarded to the API (e.g. ``enable_thinking``).

        Returns
        -------
        str or dict
            Response text, or dict when ``return_dict=True``.
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        model = model or self.default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model is not None:
            kwargs["model"] = model
        if top_p is not None:
            kwargs["top_p"] = top_p
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt <= self.max_retries:
            try:
                if stream:
                    result = self._call_stream(kwargs, model)
                    if return_dict:
                        return result
                    return result["response"]
                else:
                    result = self._call_blocking(kwargs, model)
                    if return_dict:
                        return result
                    return result["response"]

            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries and _is_retryable(exc):
                    # Jittered exponential back-off: base * 2^attempt ± 10 %
                    delay = self.retry_base_delay * (2 ** attempt)
                    delay *= random.uniform(0.9, 1.1)
                    print(
                        f"  [LLM] Attempt {attempt + 1}/{self.max_retries + 1} failed "
                        f"({type(exc).__name__}: {str(exc)[:100]}). "
                        f"Retrying in {delay:.1f}s…",
                        flush=True,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                # Non-retryable or exhausted retries
                print(f"  [LLM] API call failed after {attempt + 1} attempt(s): {type(exc).__name__}: {exc}")
                raise

        # Should be unreachable, but satisfies type-checkers
        raise last_exc  # type: ignore[misc]

    def get_usage_stats(self) -> Dict[str, int]:
        """Return cumulative token usage since this client was created.

        Only populated for non-streaming calls (streaming responses don't
        expose usage in the OpenAI SDK's chunk stream by default).

        Returns
        -------
        dict with keys:
            ``prompt_tokens``, ``completion_tokens``, ``total_tokens``, ``calls``
        """
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "calls": self._total_calls,
        }

    def list_models(self):
        """List available models synchronously."""
        models_response = self.client.models.list()
        return [model.id for model in models_response.data]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_blocking(self, kwargs: dict, model: Optional[str]) -> dict:
        """Execute a single non-streaming API call and update usage counters."""
        completion_obj = self.client.chat.completions.create(**kwargs)
        if completion_obj is None or not completion_obj.choices:
            raise ValueError(f"Invalid response from LLM API: {completion_obj}")

        msg = completion_obj.choices[0].message
        content = (msg.content or "").strip("\n")
        # Thinking models (Qwen3, DeepSeek-R1) may return empty content with the
        # actual response inside reasoning_content when content is exhausted.
        if not content:
            content = (getattr(msg, "reasoning_content", None) or "").strip("\n")

        model_name = completion_obj.model

        # Update cumulative counters when usage info is available
        self._total_calls += 1
        usage = getattr(completion_obj, "usage", None)
        if usage is not None:
            self._total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        return {
            "response": content,
            "model": model_name,
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
        }

    def _call_stream(self, kwargs: dict, model: Optional[str]) -> dict:
        """Execute a single streaming API call."""
        completion_stream = self.client.chat.completions.create(**kwargs, stream=True)
        full_response = ""
        for chunk in completion_stream:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            print(".", end="", flush=True)

        model_name = model or self.default_model
        self._total_calls += 1
        return {
            "response": full_response,
            "model": model_name,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


# ---------------------------------------------------------------------------
# Global singleton + convenience wrapper
# ---------------------------------------------------------------------------

llm_model_from_env = os.getenv("LLM_MODEL_NAME") or os.getenv("LLM_MODEL")

llm_client = LLMClient(default_model=llm_model_from_env)


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 6000,
    top_p: Optional[float] = None,
    client: Optional[LLMClient] = None,
    **kwargs,
) -> str:
    """Quick LLM call using the global default client (or a provided one)."""
    if client is None:
        client = llm_client
    return client.call(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        **kwargs,
    )
