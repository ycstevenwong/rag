from __future__ import annotations

from .base import LLMProvider


def get_provider(name: str, model: str) -> LLMProvider:
    name = name.lower()
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model)
    if name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(model=model)
    if name == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider(model=model)
    raise ValueError(f"Unknown LLM provider: {name!r}")
