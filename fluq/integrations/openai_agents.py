"""OpenAI Agents SDK integration for Fluq.

Install: pip install fluq-sdk[openai-agents]

Usage::

    import fluq
    from fluq.integrations.openai_agents import FluqTracingProcessor

    fluq.init(api_key="fo_xxx", agent_id="support-agent")
    processor = FluqTracingProcessor()

    from agents import Agent, Runner, RunConfig
    agent = Agent(name="support", instructions="...")
    result = Runner.run_sync(agent, "Help me", run_config=RunConfig(tracing_processors=[processor]))
"""

from __future__ import annotations

import uuid
from typing import Any

import fluq


class FluqTracingProcessor:
    """Implements the OpenAI Agents SDK ``TracingProcessor`` protocol.

    Maps agent SDK spans to Fluq events:
    - ``AgentSpanData``      → ``action``
    - ``GenerationSpanData`` → ``llm_call``
    - ``FunctionSpanData``   → ``tool_use``
    - ``HandoffSpanData``    → ``spawn``
    - ``GuardrailSpanData``  → ``decision``
    """

    def on_trace_start(self, trace: Any) -> None:
        if fluq._state is None:
            return
        trace_id = _trace_id(trace)
        name = getattr(trace, "name", None) or "agent_run"
        fluq._emit_event(
            trace_id=trace_id,
            event_type="action",
            payload={"event": "trace_start", "name": name},
            metadata={"framework": "openai-agents"},
        )

    def on_trace_end(self, trace: Any) -> None:
        if fluq._state is None:
            return
        trace_id = _trace_id(trace)
        fluq._emit_event(
            trace_id=trace_id,
            event_type="action",
            payload={"event": "trace_end", "name": getattr(trace, "name", None) or "agent_run"},
            metadata={"framework": "openai-agents"},
        )

    def on_span_start(self, span: Any) -> None:
        pass  # All useful data is available at span end

    def on_span_end(self, span: Any) -> None:
        if fluq._state is None:
            return

        trace_id = _span_trace_id(span)
        span_data = getattr(span, "span_data", None)
        if span_data is None:
            return

        duration_ms = _span_duration(span)
        span_type = type(span_data).__name__

        if span_type == "GenerationSpanData":
            _handle_generation(trace_id, span_data, duration_ms)
        elif span_type == "FunctionSpanData":
            _handle_function(trace_id, span_data, duration_ms)
        elif span_type == "HandoffSpanData":
            _handle_handoff(trace_id, span_data, duration_ms)
        elif span_type == "GuardrailSpanData":
            _handle_guardrail(trace_id, span_data, duration_ms)
        elif span_type == "AgentSpanData":
            _handle_agent(trace_id, span_data, duration_ms)
        # Unknown span types are silently ignored


# ---------------------------------------------------------------------------
# Internal span handlers
# ---------------------------------------------------------------------------

def _handle_generation(trace_id: str, data: Any, duration_ms: float | None) -> None:
    model = getattr(data, "model", None) or "unknown"
    usage = getattr(data, "usage", None)
    tokens_in = tokens_out = 0
    if usage is not None:
        if isinstance(usage, dict):
            tokens_in = usage.get("input_tokens", 0) or 0
            tokens_out = usage.get("output_tokens", 0) or 0
        else:
            tokens_in = getattr(usage, "input_tokens", 0) or 0
            tokens_out = getattr(usage, "output_tokens", 0) or 0

    raw_input = getattr(data, "input", None)
    raw_output = getattr(data, "output", None)

    fluq._emit_event(
        trace_id=trace_id,
        event_type="llm_call",
        resource=f"openai.{model}",
        input_data={"messages": raw_input} if raw_input is not None else None,
        output_data={"content": raw_output} if raw_output is not None else None,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_ms=duration_ms,
        metadata={"model": model, "provider": "openai", "framework": "openai-agents"},
    )


def _handle_function(trace_id: str, data: Any, duration_ms: float | None) -> None:
    name = getattr(data, "name", None) or "unknown_tool"
    raw_input = getattr(data, "input", None)
    raw_output = getattr(data, "output", None)
    fluq._emit_event(
        trace_id=trace_id,
        event_type="tool_use",
        resource=name,
        input_data={"input": raw_input} if raw_input is not None else None,
        output_data={"output": raw_output} if raw_output is not None else None,
        duration_ms=duration_ms,
        metadata={"framework": "openai-agents"},
    )


def _handle_handoff(trace_id: str, data: Any, duration_ms: float | None) -> None:
    fluq._emit_event(
        trace_id=trace_id,
        event_type="spawn",
        payload={
            "from_agent": getattr(data, "from_agent", None),
            "to_agent": getattr(data, "to_agent", None),
        },
        duration_ms=duration_ms,
        metadata={"framework": "openai-agents"},
    )


def _handle_guardrail(trace_id: str, data: Any, duration_ms: float | None) -> None:
    triggered = getattr(data, "triggered", False)
    fluq._emit_event(
        trace_id=trace_id,
        event_type="decision",
        payload={
            "name": getattr(data, "name", None) or "guardrail",
            "triggered": triggered,
            "output": getattr(data, "output", None),
        },
        duration_ms=duration_ms,
        metadata={"framework": "openai-agents"},
    )


def _handle_agent(trace_id: str, data: Any, duration_ms: float | None) -> None:
    raw_output = getattr(data, "output", None)
    fluq._emit_event(
        trace_id=trace_id,
        event_type="action",
        payload={
            "agent": getattr(data, "name", None) or "agent",
            "handoffs": getattr(data, "handoffs", None),
            "tools": getattr(data, "tools", None),
        },
        output_data={"output": raw_output} if raw_output is not None else None,
        duration_ms=duration_ms,
        metadata={"framework": "openai-agents"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace_id(trace: Any) -> str:
    return str(getattr(trace, "trace_id", None) or uuid.uuid4())


def _span_trace_id(span: Any) -> str:
    return str(getattr(span, "trace_id", None) or uuid.uuid4())


def _span_duration(span: Any) -> float | None:
    started_at = getattr(span, "started_at", None)
    ended_at = getattr(span, "ended_at", None)
    if started_at is None or ended_at is None:
        return None
    try:
        return (ended_at - started_at) * 1000.0
    except TypeError:
        return None
