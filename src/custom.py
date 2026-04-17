"""
DataRobot custom inference hook: proxy OpenAI Chat Completions to an on-prem
OpenAI-compatible API.

Environment variables (each checked in uppercase, then lowercase):
  API_BASE / api_base   — upstream base URL including /v1 (e.g. for a local
                          server: http://127.0.0.1:12345/v1)
  API_TOKEN / api_token — API key sent as Bearer to the upstream
  MODEL / model         — optional; if set, overrides the request's model field

Requires: openai>=1.0 (``pip install openai`` in the custom environment).

Local OpenAI + LangChain streaming/non-streaming examples: see LOCAL_TESTING.md.
"""

from __future__ import annotations

import os
from typing import Any, Dict
import httpx

from openai import OpenAI

from datarobot_drum import RuntimeParameters

API_TOKEN=RuntimeParameters.get("API_TOKEN")["apiToken"]
LLM = RuntimeParameters.get("MODEL")
BASE_URL = RuntimeParameters.get("API_BASE_URL")
SSL_VERIFY = RuntimeParameters.get("SSL_VERIFY")

# DRUM may add this key to the request body for moderations / MLOps association.
_DR_INTERNAL_KEYS = frozenset({"datarobot_association_id"})

_client: OpenAI | None = None



def _env_first(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None

def _get_openai_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    if not SSL_VERIFY:
        print("SSL_VERIFY is false")
        custom_http_client = httpx.Client(verify=False)
        _client = OpenAI(base_url=BASE_URL.rstrip("/"), api_key=API_TOKEN, http_client=custom_http_client)
    else:
        _client = OpenAI(base_url=BASE_URL.rstrip("/"), api_key=API_TOKEN)
    return _client


def _params_as_dict(completion_create_params: Any) -> Dict[str, Any]:
    if isinstance(completion_create_params, dict):
        return dict(completion_create_params)
    model_dump = getattr(completion_create_params, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(exclude_none=True))
    return dict(completion_create_params)


def _upstream_body(params: Dict[str, Any]) -> Dict[str, Any]:
    body = {k: v for k, v in params.items() if k not in _DR_INTERNAL_KEYS}
    model_override = _env_first("MODEL", "model")
    if model_override:
        body["model"] = model_override
    return body


def load_model(code_dir: str) -> Any:
    """Return a handle for DRUM; the chat hook uses the OpenAI client from env."""
    return {"code_dir": code_dir}


def chat(completion_create_params: Any, model: Any, **kwargs: Any) -> Any:
    """
    Forward chat completion requests to the configured OpenAI-compatible server.

    Returns a ``ChatCompletion`` when ``stream`` is false, or a stream iterator
    of ``ChatCompletionChunk`` when ``stream`` is true (as required by DRUM).
    """
    print("="*100)
    print(completion_create_params) 
    completion_create_params["model"] = LLM
    print("fixed model")
    print(completion_create_params)
    print("="*100)
    client = _get_openai_client()
    body = _upstream_body(_params_as_dict(completion_create_params))
    return client.chat.completions.create(**body)
