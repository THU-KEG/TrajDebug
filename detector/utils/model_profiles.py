"""Generic profiles for OpenAI and OpenAI-compatible model servers."""

import copy
import os
import re
from typing import Any, Dict, Optional


MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "openai": {
        "backend": "openai",
        "base_url": "${OPENAI_BASE_URL:-https://api.openai.com/v1}",
        "model": "${OPENAI_MODEL}",
        "api_key": "${OPENAI_API_KEY}",
        "params": {},
        "description": "OpenAI API using caller/environment-provided credentials",
    },
    "self_hosted": {
        "backend": "self_hosted",
        "base_url": "${MODEL_BASE_URL}",
        "model": "${MODEL_NAME}",
        "api_key": "${MODEL_API_KEY:-EMPTY}",
        "params": {},
        "description": "Any self-hosted OpenAI-compatible endpoint",
    },
}

_ENV_PLACEHOLDER = re.compile(
    r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>.*))?\}$"
)


def _resolve_env_value(value: Any) -> Any:
    """Resolve a whole-value ${VAR} or ${VAR:-default} placeholder.

    Missing placeholders without defaults become ``None``. This prevents
    strings such as ``"${OPENAI_API_KEY}"`` from being passed as credentials.
    """
    if isinstance(value, dict):
        return {key: _resolve_env_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_value(item) for item in value]
    if not isinstance(value, str):
        return value
    match = _ENV_PLACEHOLDER.fullmatch(value)
    if match is None:
        return value
    name = match.group("name")
    default: Optional[str] = match.group("default")
    resolved = os.getenv(name)
    if resolved is not None:
        return resolved
    return default


def get_model_profile(profile_name: str) -> Dict[str, Any]:
    """Return an isolated profile with environment placeholders expanded."""
    if profile_name not in MODEL_PROFILES:
        available = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(
            f"Unknown model profile: {profile_name}. Available profiles: {available}"
        )
    return _resolve_env_value(copy.deepcopy(MODEL_PROFILES[profile_name]))


def get_available_profiles() -> Dict[str, str]:
    """Return profile names and descriptions."""
    return {
        name: str(config["description"])
        for name, config in MODEL_PROFILES.items()
    }
