"""Tests for the Fluq Python SDK."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from fluq import (
    CompleteTaskInput,
    EventInput,
    EventType,
    Fluq,
    FluqConfig,
    FluqError,
    Task,
    TaskPriority,
    TaskStatus,
    TraceInput,
    TraceResult,
    TraceStatus,
)

BASE = "https://api.fluq.dev"
AGENT_ID = "550e8400-e29b-41d4-a716-446655440000"
TRACE_ID = "660e8400-e29b-41d4-a716-446655440000"


def _cfg(**overrides: object) -> FluqConfig:
    defaults = dict(
        api_key="sk-test",
        agent_id=AGENT_ID,
        capabilities=["code", "test"],
    )
    defaults.update(overrides)
    return FluqConfig(**defaults)  # type: ignore[arg-type]


def _client(**overrides: object) -> Fluq:
    c = Fluq()
    c.init(_cfg(**overrides))
    return c


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TestInit:
    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="apiKey"):
            _client(api_key="")

    def test_requires_agent_id(self) -> None:
        with pytest.raises(ValueError, match="agentId"):
            _client(agent_id="")

    def test_requires_capabilities(self) -> None:
        with pytest.raises(ValueError, match="capabilities"):
            _client(capabilities=[])

    def test_strips_trailing_slash(self) -> None:
        c = _client(base_url="https://example.com/")
        assert c._config is not None
        assert c._config.base_url == "https://example.com"

    def test_not_initialized_raises(self) -> None:
        c = Fluq()
        with pytest.raises(RuntimeError, match="not initialized"):
            c.event(EventInput(trace_id=TRACE_ID, event_type=EventType.ACTION))


# ------------------------------------------------------------------
# Traces
# ------------------------------------------------------------------

class TestTrace:
    @respx.mock
    async def test_create_trace(self) -> None:
        route = respx.post(f"{BASE}/api/v1/traces").mock(
            return_value=httpx.Response(200, json={
                "id": TRACE_ID,
                "name": "test_trace",
                "agentId": AGENT_ID,
                "status": "active",
                "createdAt": "2026-03-26T12:00:00Z",
            })
        )
        c = _client()
        try:
            ctx = await c.trace(TraceInput(name="test_trace"))
            assert ctx.id == TRACE_ID
            assert ctx.name == "test_trace"
            assert ctx.status == TraceStatus.ACTIVE

            req = route.calls.last.request
            body = json.loads(req.content)
            assert body["agentId"] == AGENT_ID
            assert body["name"] == "test_trace"
            assert req.headers["authorization"] == "Bearer sk-test"
        finally:
            await c.destroy()

    @respx.mock
    async def test_trace_with_optional_fields(self) -> None:
        respx.post(f"{BASE}/api/v1/traces").mock(
            return_value=httpx.Response(200, json={
                "id": TRACE_ID, "name": "t", "agentId": AGENT_ID,
                "status": "active", "createdAt": "2026-03-26T12:00:00Z",
            })
        )
        c = _client()
        try:
            await c.trace(TraceInput(
                name="t",
                input={"doc": 1},
                metadata={"env": "test"},
                parent_trace_id="parent-id",
                environment="staging",
            ))
            body = json.loads(respx.calls.last.request.content)
            assert body["input"] == {"doc": 1}
            assert body["metadata"] == {"env": "test"}
            assert body["parentTraceId"] == "parent-id"
            assert body["environment"] == "staging"
        finally:
            await c.destroy()

    @respx.mock
    async def test_trace_context_manager(self) -> None:
        respx.post(f"{BASE}/api/v1/traces").mock(
            return_value=httpx.Response(200, json={
                "id": TRACE_ID, "name": "t", "agentId": AGENT_ID,
                "status": "active", "createdAt": "2026-03-26T12:00:00Z",
            })
        )
        respx.post(f"{BASE}/api/v1/events").mock(
            return_value=httpx.Response(200, json={"success": True, "data": {"eventIds": [], "count": 0}})
        )
        c = _client()
        try:
            async with await c.trace(TraceInput(name="t")) as t:
                t.event(EventInput(trace_id="", event_type=EventType.ACTION))
                assert len(c._buffer) == 1
                assert c._buffer[0].trace_id == TRACE_ID
        finally:
            await c.destroy()

    @respx.mock
    async def test_trace_context_manager_on_exception(self) -> None:
        respx.post(f"{BASE}/api/v1/traces").mock(
            return_value=httpx.Response(200, json={
                "id": TRACE_ID, "name": "t", "agentId": AGENT_ID,
                "status": "active", "createdAt": "2026-03-26T12:00:00Z",
            })
        )
        respx.post(f"{BASE}/api/v1/events").mock(
            return_value=httpx.Response(200, json={"success": True, "data": {"eventIds": [], "count": 0}})
        )
        c = _client()
        try:
            with pytest.raises(ValueError, match="boom"):
                async with await c.trace(TraceInput(name="t")) as t:
                    raise ValueError("boom")
            # Should have buffered an error event
            error_events = [e for e in c._buffer if e.event_type == EventType.ERROR]
            assert len(error_events) == 1
            assert error_events[0].error_message == "boom"
        finally:
            await c.destroy()


# ------------------------------------------------------------------
# Events & Batching
# ------------------------------------------------------------------

class TestEvents:
    def test_event_buffers(self) -> None:
        c = _client()
        c.event(EventInput(trace_id=TRACE_ID, event_type=EventType.ACTION))
        c.event(EventInput(trace_id=TRACE_ID, event_type=EventType.HEARTBEAT))
        assert len(c._buffer) == 2

    @respx.mock
    async def test_flush_sends_batch(self) -> None:
        route = respx.post(f"{BASE}/api/v1/events").mock(
            return_value=httpx.Response(200, json={
                "success": True, "data": {"eventIds": ["e1", "e2"], "count": 2},
            })
        )
        c = _client()
        try:
            c.event(EventInput(
                trace_id=TRACE_ID, event_type=EventType.API_CALL,
                resource="openai.chat", duration_ms=245,
                tokens_in=150, tokens_out=87, estimated_cost_usd=0.0045,
            ))
            c.event(EventInput(trace_id=TRACE_ID, event_type=EventType.ACTION))
            await c.flush()

            assert len(c._buffer) == 0
            body = json.loads(route.calls.last.request.content)
            assert len(body["events"]) == 2
            assert body["events"][0]["agentId"] == AGENT_ID
            assert body["events"][0]["traceId"] == TRACE_ID
            assert body["events"][0]["eventType"] == "api_call"
            assert body["events"][0]["tokensIn"] == 150
        finally:
            await c.destroy()

    @respx.mock
    async def test_flush_noop_when_empty(self) -> None:
        c = _client()
        try:
            await c.flush()  # Should not raise or make requests
            assert respx.calls.call_count == 0
        finally:
            await c.destroy()

    @respx.mock
    async def test_destroy_flushes(self) -> None:
        route = respx.post(f"{BASE}/api/v1/events").mock(
            return_value=httpx.Response(200, json={"success": True, "data": {"eventIds": [], "count": 0}})
        )
        c = _client()
        c.event(EventInput(trace_id=TRACE_ID, event_type=EventType.HEARTBEAT))
        await c.destroy()
        assert route.called


# ------------------------------------------------------------------
# Error handling & retries
# ------------------------------------------------------------------

class TestErrors:
    @respx.mock
    async def test_4xx_raises_immediately(self) -> None:
        respx.post(f"{BASE}/api/v1/traces").mock(
            return_value=httpx.Response(422, json={"error": "validation failed"})
        )
        c = _client(max_retries=3)
        try:
            with pytest.raises(FluqError) as exc_info:
                await c.trace(TraceInput(name="t"))
            assert exc_info.value.status == 422
            assert exc_info.value.body == {"error": "validation failed"}
            # Should NOT have retried
            assert respx.calls.call_count == 1
        finally:
            await c.destroy()

    @respx.mock
    async def test_5xx_retries_then_raises(self) -> None:
        respx.post(f"{BASE}/api/v1/traces").mock(
            return_value=httpx.Response(503, text="Service Unavailable")
        )
        c = _client(max_retries=2, retry_base_delay_ms=1)
        try:
            with pytest.raises(FluqError) as exc_info:
                await c.trace(TraceInput(name="t"))
            assert exc_info.value.status == 503
            # 1 initial + 2 retries = 3
            assert respx.calls.call_count == 3
        finally:
            await c.destroy()

    @respx.mock
    async def test_5xx_then_success(self) -> None:
        route = respx.post(f"{BASE}/api/v1/traces")
        route.side_effect = [
            httpx.Response(500, text="err"),
            httpx.Response(200, json={
                "id": TRACE_ID, "name": "t", "agentId": AGENT_ID,
                "status": "active", "createdAt": "2026-03-26T12:00:00Z",
            }),
        ]
        c = _client(retry_base_delay_ms=1)
        try:
            ctx = await c.trace(TraceInput(name="t"))
            assert ctx.id == TRACE_ID
            assert respx.calls.call_count == 2
        finally:
            await c.destroy()


# ------------------------------------------------------------------
# Tasks
# ------------------------------------------------------------------

class TestTasks:
    @respx.mock
    async def test_pull_task(self) -> None:
        respx.post(f"{BASE}/api/v1/tasks/pull").mock(
            return_value=httpx.Response(200, json={
                "id": "task-1",
                "fleetId": "fleet-1",
                "name": "fix_bug",
                "description": "Fix pagination",
                "priority": "high",
                "status": "assigned",
                "requiredCapabilities": ["code"],
                "input": {"bug": 123},
                "metadata": None,
                "deadline": "2026-03-27T12:00:00Z",
                "retries": 0,
                "createdAt": "2026-03-26T10:00:00Z",
            })
        )
        c = _client()
        try:
            task = await c.pull_task()
            assert task is not None
            assert task.id == "task-1"
            assert task.name == "fix_bug"
            assert task.priority == TaskPriority.HIGH
            assert task.status == TaskStatus.ASSIGNED
            assert task.input == {"bug": 123}
        finally:
            await c.destroy()

    @respx.mock
    async def test_pull_task_returns_none_on_404(self) -> None:
        respx.post(f"{BASE}/api/v1/tasks/pull").mock(
            return_value=httpx.Response(404, json={"error": "no tasks"})
        )
        c = _client()
        try:
            task = await c.pull_task()
            assert task is None
        finally:
            await c.destroy()

    @respx.mock
    async def test_complete_task(self) -> None:
        route = respx.post(f"{BASE}/api/v1/tasks/task-1/complete").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        c = _client()
        try:
            await c.complete_task("task-1", CompleteTaskInput(output={"result": "done"}))
            body = json.loads(route.calls.last.request.content)
            assert body["agentId"] == AGENT_ID
            assert body["output"] == {"result": "done"}
        finally:
            await c.destroy()


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------

class TestSerialization:
    @respx.mock
    async def test_event_serialization_camel_case(self) -> None:
        route = respx.post(f"{BASE}/api/v1/events").mock(
            return_value=httpx.Response(200, json={"success": True, "data": {"eventIds": [], "count": 0}})
        )
        c = _client()
        try:
            c.event(EventInput(
                trace_id=TRACE_ID,
                event_type=EventType.ERROR,
                error_message="something broke",
                duration_ms=100,
                estimated_cost_usd=0.01,
                tokens_in=10,
                tokens_out=20,
            ))
            await c.flush()
            body = json.loads(route.calls.last.request.content)
            ev = body["events"][0]
            assert ev["traceId"] == TRACE_ID
            assert ev["eventType"] == "error"
            assert ev["errorMessage"] == "something broke"
            assert ev["durationMs"] == 100
            assert ev["estimatedCostUsd"] == 0.01
            assert ev["tokensIn"] == 10
            assert ev["tokensOut"] == 20
        finally:
            await c.destroy()

    @respx.mock
    async def test_none_fields_omitted(self) -> None:
        route = respx.post(f"{BASE}/api/v1/events").mock(
            return_value=httpx.Response(200, json={"success": True, "data": {"eventIds": [], "count": 0}})
        )
        c = _client()
        try:
            c.event(EventInput(trace_id=TRACE_ID, event_type=EventType.HEARTBEAT))
            await c.flush()
            ev = json.loads(route.calls.last.request.content)["events"][0]
            assert "errorMessage" not in ev
            assert "tokensIn" not in ev
            assert "payload" not in ev
        finally:
            await c.destroy()
