"""Load LLM configuration and handle API calls with fallback.

Usage:
    config = load_llm_config()
    if config.enabled:
        response = call_llm(prompt)
    else:
        response = None  # caller falls back to deterministic
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

try:
    import yaml
except ImportError:
    yaml = None


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "llm_config.yaml"


@dataclass
class LLMConfig:
    enabled: bool = False
    endpoint: str = "http://localhost:11434"
    api_key: str = ""
    request_format: str = "ollama-generate"
    model: str = "llama3.2"
    system_prompt: str = ""
    temperature: float = 0.2
    max_tokens: int = 220
    request_body: dict[str, Any] = field(default_factory=dict)


def load_llm_config(path: str | Path = CONFIG_PATH) -> LLMConfig:
    """Load LLM configuration from YAML file.

    Falls back to environment variables OLLAMA_URL / LLM_ENDPOINT / LLM_API_KEY
    for backward compatibility.
    """
    config = LLMConfig()

    if os.path.exists(path) and yaml is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        llm_raw = raw.get("llm", {})
        config.enabled = bool(llm_raw.get("enabled", False))
        config.endpoint = str(llm_raw.get("endpoint", config.endpoint))
        config.api_key = str(llm_raw.get("api_key", ""))
        config.request_format = str(llm_raw.get("request_format", config.request_format))
        config.model = str(llm_raw.get("model", config.model))
        config.system_prompt = str(llm_raw.get("system_prompt", config.system_prompt))
        config.temperature = float(llm_raw.get("temperature", config.temperature))
        config.max_tokens = int(llm_raw.get("max_tokens", config.max_tokens))
        config.request_body = dict(llm_raw.get("request_body", {}))

    # Environment variable overrides
    env_endpoint = os.environ.get("LLM_ENDPOINT") or os.environ.get("OLLAMA_URL")
    if env_endpoint:
        config.endpoint = env_endpoint
        config.enabled = True
    env_key = os.environ.get("LLM_API_KEY")
    if env_key:
        config.api_key = env_key
        config.enabled = True

    return config


def test_endpoint(config: LLMConfig) -> dict[str, Any]:
    """Test the LLM endpoint and return status info."""
    result = {
        "success": False,
        "model": None,
        "latency_sec": None,
        "error": None,
        "sample": None,
    }

    if not config.enabled:
        result["error"] = "LLM not enabled"
        return result

    import time
    t0 = time.time()

    try:
        if config.request_format == "ollama-generate":
            # Check tags first
            tags_r = requests.get(f"{config.endpoint}/api/tags", timeout=8)
            tags_r.raise_for_status()
            models = [m.get("name") for m in tags_r.json().get("models", []) if m.get("name")]
            if not models:
                result["error"] = "No models found on Ollama endpoint"
                return result
            model_name = config.model if config.model in models else models[0]

            gen_r = requests.post(
                f"{config.endpoint}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Reply with: ok",
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 20},
                },
                timeout=20,
            )
            gen_r.raise_for_status()
            out = gen_r.json()
            result["success"] = isinstance(out, dict) and ("response" in out)
            result["model"] = model_name
            result["sample"] = str(out.get("response", ""))[:200]

        elif config.request_format == "openai-chat":
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            gen_r = requests.post(
                f"{config.endpoint.rstrip('/')}/chat/completions",
                json={
                    "model": config.model,
                    "messages": [{"role": "user", "content": "Reply with: ok"}],
                    "temperature": 0.0,
                    "max_tokens": 20,
                },
                headers=headers,
                timeout=20,
            )
            gen_r.raise_for_status()
            out = gen_r.json()
            choices = out.get("choices", [])
            result["success"] = len(choices) > 0
            result["model"] = config.model
            if choices:
                result["sample"] = str(choices[0].get("message", {}).get("content", ""))[:200]

        elif config.request_format == "raw-json":
            body = _build_raw_body(config, "Reply with: ok")
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            gen_r = requests.post(config.endpoint, json=body, headers=headers, timeout=20)
            gen_r.raise_for_status()
            result["success"] = True
            result["model"] = config.model
            result["sample"] = str(gen_r.text)[:200]

        else:
            result["error"] = f"Unknown request_format: {config.request_format}"

    except Exception as e:
        result["error"] = str(e)

    result["latency_sec"] = round(time.time() - t0, 3)
    return result


def assemble_prompt(config: LLMConfig, evidence: str) -> str:
    """Assemble the full prompt from system prompt + evidence."""
    system = config.system_prompt.strip()
    if system:
        return f"{system}\n\nEvidence:\n{evidence}"
    return evidence


def _build_raw_body(config: LLMConfig, prompt: str) -> dict[str, Any]:
    """Build a raw JSON body, substituting {prompt} placeholder."""
    body = {}
    for k, v in config.request_body.items():
        if isinstance(v, str):
            body[k] = v.replace("{prompt}", prompt)
        elif isinstance(v, dict):
            body[k] = {
                sk: sv.replace("{prompt}", prompt) if isinstance(sv, str) else sv
                for sk, sv in v.items()
            }
        else:
            body[k] = v
    return body


def call_llm(prompt: str, config: LLMConfig | None = None) -> str | None:
    """Call the LLM endpoint with a prompt.

    Returns the response text, or None if LLM is disabled or the call fails.
    The caller should fall back to deterministic explanation when None.
    """
    if config is None:
        config = load_llm_config()
    if not config.enabled:
        return None

    try:
        if config.request_format == "ollama-generate":
            resp = requests.post(
                f"{config.endpoint}/api/generate",
                json={
                    "model": config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": config.temperature,
                        "num_predict": config.max_tokens,
                    },
                },
                timeout=60,
            )
            resp.raise_for_status()
            return str(resp.json().get("response", "")).strip()

        elif config.request_format == "openai-chat":
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            resp = requests.post(
                f"{config.endpoint.rstrip('/')}/chat/completions",
                json={
                    "model": config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                },
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            out = resp.json()
            choices = out.get("choices", [])
            if choices:
                return str(choices[0].get("message", {}).get("content", "")).strip()
            return None

        elif config.request_format == "raw-json":
            body = _build_raw_body(config, prompt)
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            resp = requests.post(config.endpoint, json=body, headers=headers, timeout=60)
            resp.raise_for_status()
            # Try to parse as JSON (Ollama/OpenAI style), fallback to raw text
            try:
                data = resp.json()
                if "message" in data and isinstance(data["message"], dict):
                    return str(data["message"].get("content", "")).strip()
                if "response" in data:
                    return str(data["response"]).strip()
                if "choices" in data and len(data["choices"]) > 0:
                    return str(data["choices"][0].get("message", {}).get("content", "")).strip()
            except Exception:
                pass
            return resp.text.strip()

        else:
            return None

    except Exception:
        return None
