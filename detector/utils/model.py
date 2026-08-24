import json
import os
import re
from typing import Any, Dict, Optional

import httpx

from .cache import Cache

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _extract_usage_from_completion(chat_completion: Any) -> Optional[Dict[str, int]]:
    """Return the stable usage shape expected by the pipeline stages."""
    usage = getattr(chat_completion, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
    if reasoning_tokens == 0:
        reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
    return {
        "input_tokens": input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
    }


def _normalize_base_url(base_url: str) -> str:
    """Convert endpoint URLs into the base URL expected by the OpenAI SDK."""
    if base_url is None:
        raise ValueError("base_url must be supplied by the caller.")
    url = re.sub(r"[\x00-\x1f\x7f]", "", str(base_url)).strip().rstrip("/")
    if not url:
        raise ValueError("base_url must be supplied by the caller.")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")].rstrip("/")
    return url


class APIModel:
    """OpenAI-compatible model client with the pipeline's legacy API surface."""

    def __init__(
        self,
        cache_url,
        base_url,
        model_name,
        api_key="EMPTY",
        extra_params=None,
    ):
        if OpenAI is None:
            raise ImportError(
                "Missing dependency `openai` for the OpenAI-compatible backend."
            )
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be supplied by the caller.")
        if api_key is None or not str(api_key).strip():
            raise ValueError(
                "api_key must be supplied by the caller; use 'EMPTY' for a "
                "self-hosted endpoint that does not authenticate."
            )

        self.base_url = _normalize_base_url(base_url)
        self.model_name = model_name.strip()
        self.extra_params = dict(extra_params or {})

        trust_env_raw = os.getenv("DETECTOR_HTTP_TRUST_ENV", "").strip().lower()
        trust_env = trust_env_raw in {"1", "true", "yes", "on"}
        self.http_client = httpx.Client(
            follow_redirects=True,
            trust_env=trust_env,
        )
        self.client = OpenAI(
            api_key=str(api_key),
            base_url=self.base_url,
            http_client=self.http_client,
        )
        self.cache = Cache(cache_url)

    def _create_completion(
        self,
        messages,
        max_tokens,
        temperature,
        response_format,
    ):
        create_kwargs = dict(self.extra_params)
        create_kwargs.update(
            {
                "messages": messages,
                "model": self.model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        return self.client.chat.completions.create(**create_kwargs)

    def _cache_key(
        self,
        *,
        query=None,
        messages=None,
        max_tokens,
        temperature,
        response_format,
    ) -> str:
        data = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
            # Keep a stable cache namespace across OpenAI-compatible endpoints.
            "backend": "openai",
        }
        if messages is None:
            data["query"] = query
        else:
            data["messages"] = messages
        return json.dumps(data, sort_keys=True, ensure_ascii=False)

    def generate(
        self,
        query,
        max_tokens=8192,
        temperature=0.0,
        response_format: Optional[dict] = None,
        return_usage=False,
    ):
        cache_key = self._cache_key(
            query=query,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        cached_response = self.cache.check_prompt(cache_key)
        usage_info = None

        if cached_response is not None:
            response = cached_response
        else:
            response = None
            for retry_count in range(1, 4):
                try:
                    completion = self._create_completion(
                        [{"role": "user", "content": query}],
                        max_tokens,
                        temperature,
                        response_format,
                    )
                    if completion.choices[0].finish_reason != "stop":
                        max_tokens += 2048
                        raise RuntimeError(
                            "Model stopped with reason: "
                            f"{completion.choices[0].finish_reason}: {max_tokens}"
                        )
                    response = completion.choices[0].message.content
                    usage_info = _extract_usage_from_completion(completion)
                    if temperature == 0.0:
                        self.cache.save_prompt(cache_key, response)
                    self.cache.maybe_flush(threshold=1)
                    break
                except Exception as exc:
                    print(f"Attempt {retry_count} failed: {exc}")
                    if retry_count == 3:
                        print(
                            f"All 3 retries failed for model={self.model_name}"
                        )

        if return_usage:
            return response, usage_info
        return response

    def generate_chat(
        self,
        messages,
        max_tokens=8192,
        temperature=0.0,
        response_format: Optional[dict] = None,
        return_usage=False,
    ):
        cache_key = self._cache_key(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        cached_response = self.cache.check_prompt(cache_key)
        usage_info = None

        if cached_response is not None:
            response = cached_response
        else:
            response = None
            for retry_count in range(1, 4):
                try:
                    completion = self._create_completion(
                        messages,
                        max_tokens,
                        temperature,
                        response_format,
                    )
                    response = completion.choices[0].message.content
                    usage_info = _extract_usage_from_completion(completion)
                    # Preserve generate_chat's historical behavior: cache all
                    # successful calls, regardless of temperature.
                    self.cache.save_prompt(cache_key, response)
                    self.cache.maybe_flush()
                    break
                except Exception as exc:
                    print(f"Attempt {retry_count} failed: {exc}")
                    if retry_count == 3:
                        print(
                            f"All 3 retries failed for model={self.model_name}"
                        )

        if return_usage:
            return response, usage_info
        return response

    def save_cache(self):
        self.cache.flush()

    def add_n(self):
        return self.cache.add_n

    def close(self):
        self.cache.close()
        try:
            self.client.close()
        finally:
            self.http_client.close()
