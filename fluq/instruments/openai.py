"""Auto-instrumentation for OpenAI clients (sync and async)."""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Callable

from ..pricing import estimate_cost


class _CompletionsProxy:
    """Intercepts chat.completions.create() on a sync OpenAI client."""

    def __init__(self, completions: Any, emit: Callable[..., None]) -> None:
        self._completions = completions
        self._emit = emit

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages")
        start = time.monotonic()
        try:
            response = self._completions.create(**kwargs)
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            self._emit(
                model=model,
                messages=messages,
                response=None,
                duration_ms=duration_ms,
                error=exc,
            )
            raise
        duration_ms = (time.monotonic() - start) * 1000
        self._emit(
            model=model,
            messages=messages,
            response=response,
            duration_ms=duration_ms,
            error=None,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _ChatProxy:
    def __init__(self, chat: Any, emit: Callable[..., None]) -> None:
        self._chat = chat
        self._emit = emit
        self.completions = _CompletionsProxy(chat.completions, emit)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class OpenAIProxy:
    """Proxy around a sync OpenAI client that captures LLM calls."""

    def __init__(self, client: Any, emit: Callable[..., None]) -> None:
        self._client = client
        self._emit = emit
        self.chat = _ChatProxy(client.chat, self._handle_call)

    def _handle_call(
        self,
        *,
        model: str,
        messages: Any,
        response: Any,
        duration_ms: float,
        error: Exception | None,
    ) -> None:
        tokens_in = tokens_out = 0
        output = None
        error_message = None

        if error is not None:
            error_message = f"{type(error).__name__}: {error}"
        elif response is not None:
            usage = getattr(response, "usage", None)
            if usage:
                tokens_in = getattr(usage, "prompt_tokens", 0) or 0
                tokens_out = getattr(usage, "completion_tokens", 0) or 0
            choices = getattr(response, "choices", None)
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg:
                    output = {"content": getattr(msg, "content", None), "role": getattr(msg, "role", None)}

        cost = estimate_cost(model, tokens_in, tokens_out)
        self._emit(
            model=model,
            provider="openai",
            input_data=_serialize_messages(messages),
            output_data=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_cost_usd=cost,
            duration_ms=duration_ms,
            error_message=error_message,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# --------------- Async ---------------


class _AsyncCompletionsProxy:
    """Intercepts chat.completions.create() on an async OpenAI client."""

    def __init__(self, completions: Any, emit: Callable[..., None]) -> None:
        self._completions = completions
        self._emit = emit

    async def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages")
        start = time.monotonic()
        try:
            response = await self._completions.create(**kwargs)
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            self._emit(
                model=model,
                messages=messages,
                response=None,
                duration_ms=duration_ms,
                error=exc,
            )
            raise
        duration_ms = (time.monotonic() - start) * 1000
        self._emit(
            model=model,
            messages=messages,
            response=response,
            duration_ms=duration_ms,
            error=None,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _AsyncChatProxy:
    def __init__(self, chat: Any, emit: Callable[..., None]) -> None:
        self._chat = chat
        self._emit = emit
        self.completions = _AsyncCompletionsProxy(chat.completions, emit)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class AsyncOpenAIProxy:
    """Proxy around an async OpenAI client that captures LLM calls."""

    def __init__(self, client: Any, emit: Callable[..., None]) -> None:
        self._client = client
        self._emit = emit
        self.chat = _AsyncChatProxy(client.chat, self._handle_call)

    def _handle_call(
        self,
        *,
        model: str,
        messages: Any,
        response: Any,
        duration_ms: float,
        error: Exception | None,
    ) -> None:
        tokens_in = tokens_out = 0
        output = None
        error_message = None

        if error is not None:
            error_message = f"{type(error).__name__}: {error}"
        elif response is not None:
            usage = getattr(response, "usage", None)
            if usage:
                tokens_in = getattr(usage, "prompt_tokens", 0) or 0
                tokens_out = getattr(usage, "completion_tokens", 0) or 0
            choices = getattr(response, "choices", None)
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg:
                    output = {"content": getattr(msg, "content", None), "role": getattr(msg, "role", None)}

        cost = estimate_cost(model, tokens_in, tokens_out)
        self._emit(
            model=model,
            provider="openai",
            input_data=_serialize_messages(messages),
            output_data=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_cost_usd=cost,
            duration_ms=duration_ms,
            error_message=error_message,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _serialize_messages(messages: Any) -> dict[str, Any] | None:
    if messages is None:
        return None
    try:
        return {"messages": [dict(m) if not isinstance(m, dict) else m for m in messages]}
    except Exception:
        return {"messages": str(messages)}
