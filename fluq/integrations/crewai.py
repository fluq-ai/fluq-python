"""CrewAI integration for Fluq.

Install: pip install fluq-sdk[crewai]

Usage::

    import fluq
    from fluq.integrations.crewai import FluqCrewHandler

    fluq.init(api_key="fo_xxx", agent_id="research-crew")
    handler = FluqCrewHandler()

    from crewai import Agent, Task, Crew
    crew = Crew(agents=[...], tasks=[...], callbacks=[handler])
    crew.kickoff()

The handler uses the LangChain-style callback interface that CrewAI passes to its
underlying LLM calls.  All callback methods are duck-typed — no explicit inheritance
from ``BaseCallbackHandler`` is required, so this module has no hard dependency on
``langchain-core`` at import time.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import fluq


class FluqCrewHandler:
    """LangChain-style callback handler for CrewAI.

    Maps CrewAI / LangChain callback events to Fluq events:

    - ``on_chain_start``   → ``action`` (trace root when no parent)
    - ``on_chain_end``     → ``action``
    - ``on_llm_start``     → internal timing bookkeeping
    - ``on_llm_end``       → ``llm_call``
    - ``on_llm_error``     → ``error``
    - ``on_tool_start``    → internal timing bookkeeping
    - ``on_tool_end``      → ``tool_use``
    - ``on_tool_error``    → ``error``
    - ``on_agent_action``  → ``action``
    - ``on_agent_finish``  → ``decision``
    """

    def __init__(self) -> None:
        # Maps str(run_id) → trace_id (str)
        self._run_traces: dict[str, str] = {}
        # Maps str(run_id) → start timestamp (float)
        self._run_start_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Chain callbacks
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        # Top-level chain (no parent) is our "crew run" trace.
        # Child chains inherit the root trace_id.
        if parent_run_id is None:
            trace_id = run_key
        else:
            trace_id = self._run_traces.get(str(parent_run_id), str(parent_run_id))
        self._run_traces[run_key] = trace_id
        self._run_start_times[run_key] = time.monotonic()

        name = _chain_name(serialized)
        fluq._emit_event(
            trace_id=trace_id,
            event_type="action",
            payload={"event": "chain_start", "name": name},
            input_data=_safe_dict(inputs) if inputs else None,
            metadata={"framework": "crewai"},
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        trace_id = self._run_traces.get(run_key, run_key)
        duration_ms = _pop_duration(self._run_start_times, run_key)
        fluq._emit_event(
            trace_id=trace_id,
            event_type="action",
            payload={"event": "chain_end"},
            output_data=_safe_dict(outputs) if outputs else None,
            duration_ms=duration_ms,
            metadata={"framework": "crewai"},
        )
        self._run_traces.pop(run_key, None)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        trace_id = self._run_traces.get(run_key, run_key)
        duration_ms = _pop_duration(self._run_start_times, run_key)
        fluq._emit_event(
            trace_id=trace_id,
            event_type="error",
            error_message=str(error),
            duration_ms=duration_ms,
            metadata={"framework": "crewai"},
        )
        self._run_traces.pop(run_key, None)

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id is not None else None
        trace_id = self._run_traces.get(parent_key or "", str(parent_run_id or uuid.uuid4()))
        self._run_traces[run_key] = trace_id
        self._run_start_times[run_key] = time.monotonic()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        trace_id = self._run_traces.get(run_key, str(run_id))
        duration_ms = _pop_duration(self._run_start_times, run_key)

        model, tokens_in, tokens_out = _extract_llm_response(response)
        input_text = _extract_llm_input(kwargs.get("invocation_params"))

        fluq._emit_event(
            trace_id=trace_id,
            event_type="llm_call",
            resource=f"crewai.{model}" if model else "crewai.llm",
            input_data={"prompt": input_text} if input_text else None,
            output_data=_extract_llm_output(response),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            metadata={"model": model, "framework": "crewai"} if model else {"framework": "crewai"},
        )
        self._run_traces.pop(run_key, None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        trace_id = self._run_traces.get(run_key, str(run_id))
        duration_ms = _pop_duration(self._run_start_times, run_key)
        fluq._emit_event(
            trace_id=trace_id,
            event_type="error",
            error_message=str(error),
            duration_ms=duration_ms,
            metadata={"framework": "crewai"},
        )
        self._run_traces.pop(run_key, None)

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id is not None else None
        trace_id = self._run_traces.get(parent_key or "", str(parent_run_id or uuid.uuid4()))
        self._run_traces[run_key] = trace_id
        self._run_start_times[run_key] = time.monotonic()

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        trace_id = self._run_traces.get(run_key, str(run_id))
        duration_ms = _pop_duration(self._run_start_times, run_key)
        tool_name = kwargs.get("name") or "tool"
        fluq._emit_event(
            trace_id=trace_id,
            event_type="tool_use",
            resource=tool_name,
            output_data={"output": output} if output is not None else None,
            duration_ms=duration_ms,
            metadata={"framework": "crewai"},
        )
        self._run_traces.pop(run_key, None)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        trace_id = self._run_traces.get(run_key, str(run_id))
        duration_ms = _pop_duration(self._run_start_times, run_key)
        tool_name = kwargs.get("name") or "tool"
        fluq._emit_event(
            trace_id=trace_id,
            event_type="error",
            resource=tool_name,
            error_message=str(error),
            duration_ms=duration_ms,
            metadata={"framework": "crewai"},
        )
        self._run_traces.pop(run_key, None)

    # ------------------------------------------------------------------
    # Agent callbacks
    # ------------------------------------------------------------------

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id is not None else None
        trace_id = self._run_traces.get(run_key) or self._run_traces.get(parent_key or "", str(run_id))
        tool = getattr(action, "tool", None) or (action.get("tool") if isinstance(action, dict) else None)
        tool_input = getattr(action, "tool_input", None) or (action.get("tool_input") if isinstance(action, dict) else None)
        fluq._emit_event(
            trace_id=trace_id,
            event_type="action",
            resource=tool,
            payload={"tool": tool, "tool_input": tool_input},
            metadata={"framework": "crewai"},
        )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id is not None else None
        trace_id = self._run_traces.get(run_key) or self._run_traces.get(parent_key or "", str(run_id))
        output = getattr(finish, "return_values", None) or (finish.get("return_values") if isinstance(finish, dict) else None)
        log = getattr(finish, "log", None) or (finish.get("log") if isinstance(finish, dict) else None)
        fluq._emit_event(
            trace_id=trace_id,
            event_type="decision",
            payload={"event": "agent_finish", "log": log},
            output_data={"output": output} if output else None,
            metadata={"framework": "crewai"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain_name(serialized: dict[str, Any] | None) -> str:
    if not serialized:
        return "chain"
    return serialized.get("name") or serialized.get("id", ["chain"])[-1] or "chain"


def _pop_duration(times: dict[str, float], key: str) -> float | None:
    start = times.pop(key, None)
    if start is None:
        return None
    return (time.monotonic() - start) * 1000.0


def _safe_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return {str(k): str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                for k, v in value.items()}
    return {"value": str(value)}


def _extract_llm_response(response: Any) -> tuple[str | None, int, int]:
    """Return (model, tokens_in, tokens_out) from an LLMResult."""
    model: str | None = None
    tokens_in = 0
    tokens_out = 0

    if response is None:
        return model, tokens_in, tokens_out

    # LangChain LLMResult has llm_output dict
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        model = llm_output.get("model_name") or llm_output.get("model")
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(usage, dict):
            tokens_in = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
            tokens_out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0

    return model, tokens_in, tokens_out


def _extract_llm_output(response: Any) -> dict[str, Any] | None:
    if response is None:
        return None
    generations = getattr(response, "generations", None)
    if not generations:
        return None
    try:
        texts = []
        for gen_list in generations:
            if isinstance(gen_list, list):
                for gen in gen_list:
                    text = getattr(gen, "text", None) or getattr(gen, "message", {})
                    if text:
                        texts.append(str(text) if not isinstance(text, str) else text)
        return {"content": texts} if texts else None
    except Exception:
        return None


def _extract_llm_input(invocation_params: Any) -> str | None:
    if not isinstance(invocation_params, dict):
        return None
    prompt = invocation_params.get("prompt") or invocation_params.get("messages")
    return str(prompt) if prompt else None
