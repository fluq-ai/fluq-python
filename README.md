# fluq-sdk

Python SDK for [Fluq](https://fluq.ai) — AI agent fleet observability and control.

Fluq tells you what your agents **did**, not just what they said. Observe every action, enforce policies, and orchestrate tasks across your entire agent fleet.

## Install

```bash
pip install fluq-sdk
```

## Quick Start

```python
import asyncio
from fluq import Fluq, FluqConfig, TraceInput, EventInput, EventType

async def main():
    client = Fluq()
    client.init(FluqConfig(
        api_key="fo_your_api_key",
        agent_id="my-agent",
        capabilities=["code", "search"],
        base_url="https://fluq.ai",
    ))

    # Create a trace for a unit of work
    async with await client.trace(TraceInput(name="process-request")) as trace:
        # Log events as your agent works
        trace.event(EventInput(
            trace_id=trace.id,
            event_type=EventType.LLM_CALL,
            input={"prompt": "Analyze this data"},
            output={"response": "Here's the analysis..."},
            tokens_in=150,
            tokens_out=420,
            estimated_cost_usd=0.003,
        ))

        trace.event(EventInput(
            trace_id=trace.id,
            event_type=EventType.TOOL_USE,
            resource="database",
            metadata={"query": "SELECT * FROM users"},
            duration_ms=45.2,
        ))

    await client.destroy()

asyncio.run(main())
```

## Features

- **Auto-batched events** — events are buffered and sent in batches for performance
- **Trace context manager** — `async with` automatically captures errors
- **Task queue** — pull tasks from the fleet queue and report completion
- **Retry with backoff** — transient failures are retried automatically
- **Typed** — full type hints, `py.typed` marker included

## Event Types

| Type | Description |
|------|-------------|
| `llm_call` | LLM API calls with token counts and cost |
| `tool_use` | Tool/function invocations |
| `action` | General agent actions |
| `decision` | Decision points |
| `spawn` | Child agent/process spawns |
| `api_call` | External API calls |
| `file_read` / `file_write` | File system operations |
| `error` | Errors and exceptions |
| `cost` | Cost tracking events |
| `heartbeat` | Agent health signals |
| `conflict` | Resource conflict detection |

## Task Queue

```python
# Pull and complete tasks
task = await client.pull_task()
if task:
    print(f"Working on: {task.name}")
    # ... do the work ...
    await client.complete_task(task.id)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | required | Your Fluq API key (`fo_...`) |
| `agent_id` | required | Unique identifier for this agent |
| `capabilities` | required | List of agent capabilities |
| `base_url` | `https://api.fluq.dev` | Fluq API base URL |
| `flush_interval_ms` | `1000` | Auto-flush interval |
| `flush_batch_size` | `50` | Max events per batch |
| `max_retries` | `3` | Retry attempts for failed requests |

## Requirements

- Python 3.11+
- `httpx` for async HTTP

## Links

- [Dashboard](https://fluq.ai/dashboard)
- [Documentation](https://fluq.ai/docs)
- [TypeScript SDK](https://github.com/fluq-ai/fluq-js)

## License

MIT
