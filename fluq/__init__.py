"""Fluq Python SDK — observability client for agent fleets.

Context manager for traces, auto-batched events, and task management.
Auto-instrumentation: ``fluq.init()`` + ``fluq.watch(client)`` captures LLM calls.
"""

from __future__ import annotations

import atexit
import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urljoin

import httpx


class EventType(str, Enum):
    LLM_CALL = "llm_call"
    STATUS_CHANGE = "status_change"
    ACTION = "action"
    ERROR = "error"
    COST = "cost"
    HEARTBEAT = "heartbeat"
    TOOL_USE = "tool_use"
    DECISION = "decision"
    SPAWN = "spawn"
    API_CALL = "api_call"
    FILE_WRITE = "file_write"
    FILE_READ = "file_read"
    CONFLICT = "conflict"


class TraceStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    DEAD = "dead"


class FluqError(Exception):
    """Error returned by the Fluq API."""

    def __init__(self, message: str, status: int, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class FluqConfig:
    api_key: str
    agent_id: str
    capabilities: list[str]
    base_url: str = "https://api.fluq.dev"
    flush_interval_ms: int = 1000
    flush_batch_size: int = 50
    max_retries: int = 3
    retry_base_delay_ms: int = 500


@dataclass
class TraceInput:
    name: str
    input: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    parent_trace_id: str | None = None
    environment: str | None = None


@dataclass
class TraceResult:
    id: str
    name: str
    agent_id: str
    status: TraceStatus
    created_at: str


@dataclass
class EventInput:
    trace_id: str
    event_type: EventType
    payload: dict[str, Any] | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    resource: str | None = None
    duration_ms: float | None = None
    error_message: str | None = None
    estimated_cost_usd: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass
class Task:
    id: str
    fleet_id: str
    name: str
    priority: TaskPriority
    status: TaskStatus
    required_capabilities: list[str]
    created_at: str
    description: str | None = None
    input: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    deadline: str | None = None
    retries: int = 0


@dataclass
class CompleteTaskInput:
    output: dict[str, Any] | None = None


@dataclass
class _TraceContext:
    """Context manager returned by Fluq.trace() for use with `async with`."""

    _client: Fluq
    _result: TraceResult

    @property
    def id(self) -> str:
        return self._result.id

    @property
    def name(self) -> str:
        return self._result.name

    @property
    def agent_id(self) -> str:
        return self._result.agent_id

    @property
    def status(self) -> TraceStatus:
        return self._result.status

    @property
    def created_at(self) -> str:
        return self._result.created_at

    def event(self, event_input: EventInput) -> None:
        """Emit an event bound to this trace (sets trace_id automatically)."""
        event_input.trace_id = self._result.id
        self._client.event(event_input)

    async def __aenter__(self) -> _TraceContext:
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        if exc_type is not None:
            self.event(EventInput(
                trace_id=self._result.id,
                event_type=EventType.ERROR,
                error_message=str(exc_val),
            ))


def _serialize_event(ev: EventInput) -> dict[str, Any]:
    d: dict[str, Any] = {
        "traceId": ev.trace_id,
        "eventType": ev.event_type.value if isinstance(ev.event_type, EventType) else ev.event_type,
    }
    _opt = [
        ("payload", ev.payload), ("input", ev.input), ("output", ev.output),
        ("metadata", ev.metadata), ("resource", ev.resource),
        ("durationMs", ev.duration_ms), ("errorMessage", ev.error_message),
        ("estimatedCostUsd", ev.estimated_cost_usd),
        ("tokensIn", ev.tokens_in), ("tokensOut", ev.tokens_out),
    ]
    for k, v in _opt:
        if v is not None:
            d[k] = v
    return d


class Fluq:
    """Fluq observability client with auto-batched events."""

    def __init__(self) -> None:
        self._config: FluqConfig | None = None
        self._buffer: list[EventInput] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._http: httpx.AsyncClient | None = None

    @property
    def _initialized(self) -> bool:
        return self._config is not None

    def _require_init(self) -> FluqConfig:
        if self._config is None:
            raise RuntimeError("Fluq client is not initialized. Call init() first.")
        return self._config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, config: FluqConfig) -> None:
        """Initialize the client. Must be called before any other method."""
        if not config.api_key:
            raise ValueError("apiKey is required")
        if not config.agent_id:
            raise ValueError("agentId is required")
        if not config.capabilities:
            raise ValueError("capabilities must be a non-empty list")

        config.base_url = config.base_url.rstrip("/")
        self._config = config
        self._buffer = []
        self._http = httpx.AsyncClient()
        self._start_flush_timer()

    def _start_flush_timer(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._flush_task = loop.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        cfg = self._require_init()
        interval = cfg.flush_interval_ms / 1000.0
        while self._initialized:
            await asyncio.sleep(interval)
            try:
                await self.flush()
            except Exception:
                pass

    async def destroy(self) -> None:
        """Flush remaining events and shut down."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        if self._initialized:
            await self.flush()

        if self._http is not None:
            await self._http.aclose()
            self._http = None

        self._config = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> httpx.Response:
        cfg = self._require_init()
        assert self._http is not None
        url = f"{cfg.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            if attempt > 0:
                delay = cfg.retry_base_delay_ms * (2 ** (attempt - 1)) / 1000.0
                await asyncio.sleep(delay)
            try:
                resp = await self._http.request(method, url, json=json, headers=headers)
                if resp.status_code >= 500:
                    last_exc = FluqError(resp.text, resp.status_code, _try_json(resp))
                    continue
                if resp.status_code >= 400:
                    raise FluqError(resp.text, resp.status_code, _try_json(resp))
                return resp
            except FluqError:
                raise
            except Exception as exc:
                last_exc = exc
                continue

        if isinstance(last_exc, FluqError):
            raise last_exc
        raise FluqError(str(last_exc), 0)

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    async def trace(self, trace_input: TraceInput) -> _TraceContext:
        """Create a new trace. Returns a context manager for use with `async with`."""
        cfg = self._require_init()
        body: dict[str, Any] = {
            "agentId": cfg.agent_id,
            "name": trace_input.name,
        }
        if trace_input.input is not None:
            body["input"] = trace_input.input
        if trace_input.metadata is not None:
            body["metadata"] = trace_input.metadata
        if trace_input.parent_trace_id is not None:
            body["parentTraceId"] = trace_input.parent_trace_id
        if trace_input.environment is not None:
            body["environment"] = trace_input.environment

        resp = await self._request("POST", "/api/v1/traces", json=body)
        data = resp.json()
        result = TraceResult(
            id=data["id"],
            name=data["name"],
            agent_id=data["agentId"],
            status=TraceStatus(data["status"]),
            created_at=data["createdAt"],
        )
        return _TraceContext(_client=self, _result=result)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def event(self, event_input: EventInput) -> None:
        """Buffer an event for later flushing."""
        cfg = self._require_init()
        self._buffer.append(event_input)
        if len(self._buffer) >= cfg.flush_batch_size:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.flush())
            except RuntimeError:
                pass

    async def flush(self) -> None:
        """Send all buffered events to the server."""
        if not self._initialized or not self._buffer:
            return
        cfg = self._require_init()
        events = self._buffer[:]
        self._buffer.clear()
        serialized = []
        for ev in events:
            d = _serialize_event(ev)
            d["agentId"] = cfg.agent_id
            serialized.append(d)
        await self._request("POST", "/api/v1/events", json={"events": serialized})

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    async def pull_task(self) -> Task | None:
        """Pull the next available task matching agent capabilities."""
        cfg = self._require_init()
        try:
            resp = await self._request("POST", "/api/v1/tasks/pull", json={
                "agentId": cfg.agent_id,
                "capabilities": cfg.capabilities,
            })
        except FluqError as exc:
            if exc.status == 404:
                return None
            raise
        data = resp.json()
        return Task(
            id=data["id"],
            fleet_id=data["fleetId"],
            name=data["name"],
            description=data.get("description"),
            priority=TaskPriority(data["priority"]),
            status=TaskStatus(data["status"]),
            required_capabilities=data["requiredCapabilities"],
            input=data.get("input"),
            metadata=data.get("metadata"),
            deadline=data.get("deadline"),
            retries=data.get("retries", 0),
            created_at=data["createdAt"],
        )

    async def complete_task(self, task_id: str, input: CompleteTaskInput | None = None) -> None:
        """Mark a task as completed."""
        cfg = self._require_init()
        body: dict[str, Any] = {"agentId": cfg.agent_id}
        if input is not None and input.output is not None:
            body["output"] = input.output
        await self._request("POST", f"/api/v1/tasks/{task_id}/complete", json=body)


def _try_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text


# ======================================================================
# Module-level auto-instrumentation API
# ======================================================================

@dataclass
class _ModuleState:
    api_key: str
    agent_id: str
    base_url: str
    http: httpx.Client
    buffer: list[dict[str, Any]]
    lock: threading.Lock
    flush_batch_size: int


_state: _ModuleState | None = None


def init(
    *,
    api_key: str,
    agent_id: str,
    base_url: str = "https://api.fluq.dev",
    flush_batch_size: int = 50,
) -> None:
    """Initialize the module-level auto-instrumentation.

    Must be called before ``watch()``.
    """
    global _state
    if _state is not None:
        _state.http.close()
    _state = _ModuleState(
        api_key=api_key,
        agent_id=agent_id,
        base_url=base_url.rstrip("/"),
        http=httpx.Client(),
        buffer=[],
        lock=threading.Lock(),
        flush_batch_size=flush_batch_size,
    )
    atexit.register(flush)


def flush() -> None:
    """Flush all buffered auto-instrumentation events."""
    if _state is None:
        return
    with _state.lock:
        if not _state.buffer:
            return
        events = _state.buffer[:]
        _state.buffer.clear()
    try:
        _state.http.post(
            f"{_state.base_url}/api/v1/events",
            json={"events": events},
            headers={
                "Authorization": f"Bearer {_state.api_key}",
                "Content-Type": "application/json",
            },
        )
    except Exception:
        # Best-effort: don't crash the user's app
        pass


def _emit_event(
    *,
    trace_id: str,
    event_type: str,
    resource: str | None = None,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    estimated_cost_usd: float | None = None,
    error_message: str | None = None,
) -> None:
    """Buffer an arbitrary event from integrations."""
    if _state is None:
        return
    event: dict[str, Any] = {
        "agentId": _state.agent_id,
        "traceId": trace_id,
        "eventType": event_type,
    }
    if resource is not None:
        event["resource"] = resource
    if input_data is not None:
        event["input"] = input_data
    if output_data is not None:
        event["output"] = output_data
    if payload is not None:
        event["payload"] = payload
    if metadata is not None:
        event["metadata"] = metadata
    if duration_ms is not None:
        event["durationMs"] = round(duration_ms, 2)
    if tokens_in is not None:
        event["tokensIn"] = tokens_in
    if tokens_out is not None:
        event["tokensOut"] = tokens_out
    if estimated_cost_usd is not None:
        event["estimatedCostUsd"] = estimated_cost_usd
    if error_message is not None:
        event["errorMessage"] = error_message

    with _state.lock:
        _state.buffer.append(event)
        should_flush = len(_state.buffer) >= _state.flush_batch_size
    if should_flush:
        flush()


def _emit_llm_event(
    *,
    model: str,
    provider: str,
    input_data: dict[str, Any] | None,
    output_data: dict[str, Any] | None,
    tokens_in: int,
    tokens_out: int,
    estimated_cost_usd: float | None,
    duration_ms: float,
    error_message: str | None,
) -> None:
    """Buffer an llm_call event from auto-instrumentation."""
    if _state is None:
        return
    trace_id = str(uuid.uuid4())
    event: dict[str, Any] = {
        "agentId": _state.agent_id,
        "traceId": trace_id,
        "eventType": "llm_call",
        "resource": f"{provider}.{model}",
        "durationMs": round(duration_ms, 2),
        "tokensIn": tokens_in,
        "tokensOut": tokens_out,
        "metadata": {"provider": provider, "model": model},
    }
    if input_data is not None:
        event["input"] = input_data
    if output_data is not None:
        event["output"] = output_data
    if estimated_cost_usd is not None:
        event["estimatedCostUsd"] = estimated_cost_usd
    if error_message is not None:
        event["errorMessage"] = error_message

    with _state.lock:
        _state.buffer.append(event)
        should_flush = len(_state.buffer) >= _state.flush_batch_size
    if should_flush:
        flush()


def watch(client: Any) -> Any:
    """Wrap an OpenAI or Anthropic client to auto-capture LLM calls.

    Returns an instrumented proxy that behaves identically to the original
    client but emits ``llm_call`` events for every completion request.

    Supports: ``openai.OpenAI``, ``openai.AsyncOpenAI``,
    ``anthropic.Anthropic``, ``anthropic.AsyncAnthropic``.
    """
    if _state is None:
        raise RuntimeError("fluq.init() must be called before fluq.watch()")

    cls_module = type(client).__module__ or ""
    cls_name = type(client).__name__

    if cls_module.startswith("openai"):
        if "Async" in cls_name:
            from .instruments.openai import AsyncOpenAIProxy
            return AsyncOpenAIProxy(client, _emit_llm_event)
        from .instruments.openai import OpenAIProxy
        return OpenAIProxy(client, _emit_llm_event)

    if cls_module.startswith("anthropic"):
        if "Async" in cls_name:
            from .instruments.anthropic import AsyncAnthropicProxy
            return AsyncAnthropicProxy(client, _emit_llm_event)
        from .instruments.anthropic import AnthropicProxy
        return AnthropicProxy(client, _emit_llm_event)

    raise TypeError(
        f"Unsupported client type: {cls_module}.{cls_name}. "
        "Expected an OpenAI or Anthropic client."
    )
