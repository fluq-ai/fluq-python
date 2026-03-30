"""Auto-instrumentation for Anthropic clients (sync and async)."""

from __future__ import annotations

import time
from typing import Any, Callable

from ..pricing import estimate_cost


class _MessagesProxy:
    """Intercepts messages.create() on a sync Anthropic client."""

    def __init__(self, messages: Any, emit: Callable[..., None]) -> None:
        self._messages = messages
        self._emit = emit

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        input_messages = kwargs.get("messages")
        system = kwargs.get("system")
        start = time.monotonic()
        try:
            response = self._messages.create(**kwargs)
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            self._emit(
                model=model,
                input_messages=input_messages,
                system=system,
                response=None,
                duration_ms=duration_ms,
                error=exc,
            )
            raise
        duration_ms = (time.monotonic() - start) * 1000
        self._emit(
            model=model,
            input_messages=input_messages,
            system=system,
            response=response,
            duration_ms=duration_ms,
            error=None,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class AnthropicProxy:
    """Proxy around a sync Anthropic client that captures LLM calls."""

    def __init__(self, client: Any, emit: Callable[..., None]) -> None:
        self._client = client
        self._emit = emit
        self.messages = _MessagesProxy(client.messages, self._handle_call)

    def _handle_call(
        self,
        *,
        model: str,
        input_messages: Any,
        system: Any,
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
                tokens_in = getattr(usage, "input_tokens", 0) or 0
                tokens_out = getattr(usage, "output_tokens", 0) or 0
            content = getattr(response, "content", None)
            if content:
                output = {"content": [_serialize_block(b) for b in content]}

        cost = estimate_cost(model, tokens_in, tokens_out)
        self._emit(
            model=model,
            provider="anthropic",
            input_data=_serialize_input(input_messages, system),
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


class _AsyncMessagesProxy:
    """Intercepts messages.create() on an async Anthropic client."""

    def __init__(self, messages: Any, emit: Callable[..., None]) -> None:
        self._messages = messages
        self._emit = emit

    async def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        input_messages = kwargs.get("messages")
        system = kwargs.get("system")
        start = time.monotonic()
        try:
            response = await self._messages.create(**kwargs)
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            self._emit(
                model=model,
                input_messages=input_messages,
                system=system,
                response=None,
                duration_ms=duration_ms,
                error=exc,
            )
            raise
        duration_ms = (time.monotonic() - start) * 1000
        self._emit(
            model=model,
            input_messages=input_messages,
            system=system,
            response=response,
            duration_ms=duration_ms,
            error=None,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class AsyncAnthropicProxy:
    """Proxy around an async Anthropic client that captures LLM calls."""

    def __init__(self, client: Any, emit: Callable[..., None]) -> None:
        self._client = client
        self._emit = emit
        self.messages = _AsyncMessagesProxy(client.messages, self._handle_call)

    def _handle_call(
        self,
        *,
        model: str,
        input_messages: Any,
        system: Any,
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
                tokens_in = getattr(usage, "input_tokens", 0) or 0
                tokens_out = getattr(usage, "output_tokens", 0) or 0
            content = getattr(response, "content", None)
            if content:
                output = {"content": [_serialize_block(b) for b in content]}

        cost = estimate_cost(model, tokens_in, tokens_out)
        self._emit(
            model=model,
            provider="anthropic",
            input_data=_serialize_input(input_messages, system),
            output_data=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_cost_usd=cost,
            duration_ms=duration_ms,
            error_message=error_message,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _serialize_block(block: Any) -> dict[str, Any]:
    if hasattr(block, "text"):
        return {"type": "text", "text": block.text}
    if hasattr(block, "type"):
        return {"type": getattr(block, "type", "unknown")}
    return {"type": "unknown"}


def _serialize_input(messages: Any, system: Any) -> dict[str, Any] | None:
    if messages is None:
        return None
    data: dict[str, Any] = {}
    try:
        data["messages"] = [dict(m) if not isinstance(m, dict) else m for m in messages]
    except Exception:
        data["messages"] = str(messages)
    if system is not None:
        data["system"] = system
    return data
