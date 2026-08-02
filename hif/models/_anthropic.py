"""Anthropic client for HIF."""

from __future__ import annotations


def complete_anthropic(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int = 4096,
) -> str:
    """Call the Anthropic Messages API.

    Args:
        api_key:       Anthropic API key.
        model_name:    Claude model identifier (e.g. "claude-sonnet-4-6").
        system_prompt: System prompt passed as the top-level `system` parameter.
        user_text:     User message content.
        max_tokens:    Maximum tokens to generate. Default 4096 allows full responses.

    Returns:
        Text of the first content block, or empty string.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_text}],
    )
    return message.content[0].text if message.content else ""
