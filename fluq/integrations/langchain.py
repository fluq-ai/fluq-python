"""LangChain / LangGraph integration for Fluq.

Install: pip install fluq-sdk[langchain]

Usage::

    import fluq
    from fluq.integrations.langchain import FluqCallbackHandler

    fluq.init(api_key="fo_xxx", agent_id="research-agent")
    handler = FluqCallbackHandler()

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(callbacks=[handler])
    llm.invoke("What is observability?")

The handler works at any level — attach to an LLM, a chain, an agent executor,
or a LangGraph graph.  All callback methods are duck-typed so this module has
no hard import-time dependency on ``langchain-core``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import fluq


class FluqCallbackHandler:
    """LangChain ``BaseCallbackHandler``-compatible handler for Fluq.

    Maps LangChain callback events to Fluq events:

    - ``on_chain_start``     → ``action``
    - ``on_chain_end``       → ``action``
    - ``on_chain_error``     → ``error``
    - ``on_llm_start``       → internal timing bookkeeping
    - ``on_llm_end``         → ``llm_call``
    - ``on_llm_error``       → ``error``
    - ``on_chat_model_start``→ internal timing bookkeeping
    - ``on_tool_start``      → internal timing bookkeeping
    - ``on_tool_end``        → ``tool_use``
    - ``on_tool_error``      → ``error``
    - ``on_agent_action``    → ``action``
    - ``on_agent_finish``    → ``decision``
    """

    # LangChain inspects this attribute to decide whether to raise exceptions
    # from callbacks. Keeping it False matches the BaseCallbackHandler default.
    raise_error: bool = False

    def __init__(self) -> None:
        # Maps str(run_id) → trace_id
        self._run_traces: dict[str, str] = {}
        # Maps str(run_id) → monotonic start time
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
        # Root chain (no parent) gets its own trace_id = run_id
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
            input_data=_inputs_to_dict(inputs),
            metadata={"framework": "langchain"},
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
            output_data=_safe_outputs(outputs),
            duration_ms=duration_ms,
            metadata={"framework": "langchain"},
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
            metadata={"framework": "langchain"},
        )
        self._run_traces.pop(run_key, None)

    # ------------------------------------------------------------------
    # LLM callbacks (string-prompt models)
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
        self._register_child(run_id, parent_run_id)

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
        model, tokens_in, tokens_out = _extract_llm_response_meta(response)
        fluq._emit_event(
            trace_id=trace_id,
            event_type="llm_call",
            resource=f"langchain.{model}" if model else "langchain.llm",
            output_data=_extract_llm_output(response),
            tokens_in=tokens_in or None,
            tokens_out=tokens_out or None,
            duration_ms=duration_ms,
            metadata={"model": model, "framework": "langchain"} if model else {"framework": "langchain"},
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
            metadata={"framework": "langchain"},
        )
        self._run_traces.pop(run_key, None)

    # ------------------------------------------------------------------
    # Chat model callbacks (message-based models)
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if fluq._state is None:
            return
        self._register_child(run_id, parent_run_id)
        # Store input messages for the llm_end event
        run_key = str(run_id)
        flat: list[dict[str, str]] = []
        for msg_list in messages:
            for msg in msg_list:
                content = getattr(msg, "content", None)
                role = _msg_role(msg)
                if content is not None:
                    flat.append({"role": role, "content": str(content)})
        if flat:
            self._run_traces[f"{run_key}:input"] = flat  # type: ignore[assignment]

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
        self._register_child(run_id, parent_run_id)

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
            metadata={"framework": "langchain"},
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
            metadata={"framework": "langchain"},
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
        trace_id = (
            self._run_traces.get(run_key)
            or self._run_traces.get(parent_key or "", str(run_id))
        )
        tool = getattr(action, "tool", None) or (action.get("tool") if isinstance(action, dict) else None)
        tool_input = getattr(action, "tool_input", None) or (action.get("tool_input") if isinstance(action, dict) else None)
        fluq._emit_event(
            trace_id=trace_id,
            event_type="action",
            resource=tool,
            payload={"tool": tool, "tool_input": tool_input},
            metadata={"framework": "langchain"},
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
        trace_id = (
            self._run_traces.get(run_key)
            or self._run_traces.get(parent_key or "", str(run_id))
        )
        output = getattr(finish, "return_values", None) or (finish.get("return_values") if isinstance(finish, dict) else None)
        log = getattr(finish, "log", None) or (finish.get("log") if isinstance(finish, dict) else None)
        fluq._emit_event(
            trace_id=trace_id,
            event_type="decision",
            payload={"event": "agent_finish", "log": log},
            output_data={"output": output} if output else None,
            metadata={"framework": "langchain"},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_child(self, run_id: Any, parent_run_id: Any) -> None:
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id is not None else None
        trace_id = self._run_traces.get(parent_key or "", str(parent_run_id or uuid.uuid4()))
        self._run_traces[run_key] = trace_id
        self._run_start_times[run_key] = time.monotonic()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain_name(serialized: dict[str, Any] | None) -> str:
    if not serialized:
        return "chain"
    return serialized.get("name") or (serialized.get("id") or ["chain"])[-1] or "chain"


def _pop_duration(times: dict[str, float], key: str) -> float | None:
    start = times.pop(key, None)
    if start is None:
        return None
    return (time.monotonic() - start) * 1000.0


def _inputs_to_dict(inputs: Any) -> dict[str, Any] | None:
    if not inputs:
        return None
    if isinstance(inputs, dict):
        return {str(k): str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                for k, v in inputs.items()}
    return {"input": str(inputs)}


def _safe_outputs(outputs: Any) -> dict[str, Any] | None:
    if not outputs:
        return None
    if isinstance(outputs, dict):
        return {str(k): str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                for k, v in outputs.items()}
    return {"output": str(outputs)}


def _msg_role(msg: Any) -> str:
    role = getattr(msg, "role", None) or getattr(msg, "type", None)
    if role is None:
        name = type(msg).__name__.lower()
        if "human" in name:
            return "user"
        if "ai" in name or "assistant" in name:
            return "assistant"
        if "system" in name:
            return "system"
        if "tool" in name or "function" in name:
            return "tool"
    return str(role) if role else "unknown"


def _extract_llm_response_meta(response: Any) -> tuple[str | None, int, int]:
    """Return (model, tokens_in, tokens_out) from an LLMResult."""
    model: str | None = None
    tokens_in = 0
    tokens_out = 0
    if response is None:
        return model, tokens_in, tokens_out
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
        texts: list[str] = []
        for gen_list in generations:
            if isinstance(gen_list, list):
                for gen in gen_list:
                    text = getattr(gen, "text", None)
                    if text:
                        texts.append(text)
                    else:
                        msg = getattr(gen, "message", None)
                        content = getattr(msg, "content", None) if msg else None
                        if content:
                            texts.append(str(content))
        return {"content": texts} if texts else None
    except Exception:
        return None
