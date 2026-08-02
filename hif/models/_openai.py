"""OpenAI-compatible client for HIF.

Handles OpenAI models (GPT-4o, GPT-4o-mini), DeepSeek models (via their
OpenAI-compatible endpoint), and local models (Ollama, LM Studio, vLLM).

The `base_url` parameter is the single mechanism for all non-OpenAI targets —
no separate client is needed for local servers.
"""

from __future__ import annotations


def complete_openai(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_text: str,
    base_url: str | None = None,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint.

    Args:
        api_key:       API key for the target service. For local servers without
                       auth, pass any non-empty string (e.g. "local").
        model_name:    Model identifier as the target API expects it.
        system_prompt: System role content.
        user_text:     User role content.
        base_url:      Override the default OpenAI base URL. Pass the full URL
                       including /v1, e.g. "http://localhost:11434/v1" for Ollama.

    Returns:
        Text content of the first response choice, or empty string.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    return response.choices[0].message.content or ""
