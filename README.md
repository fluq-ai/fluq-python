# fluq-sdk

Python SDK for the [Fluq](https://fluq.ai) agent observability platform.

```
pip install fluq-sdk
```

## Quick start

```python
import fluq

fluq.init(api_key="fo_xxx", agent_id="my-agent")

# Wrap an OpenAI client for automatic LLM call tracing
import openai
client = fluq.watch(openai.OpenAI())

# All calls are now captured
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

fluq.flush()
```

## Framework integrations

### OpenAI Agents SDK

```
pip install fluq-sdk[openai-agents]
```

```python
import fluq
from fluq.integrations.openai_agents import FluqTracingProcessor

fluq.init(api_key="fo_xxx", agent_id="support-agent")
processor = FluqTracingProcessor()

from agents import Agent, Runner, RunConfig

agent = Agent(name="support", instructions="You are a helpful customer support agent.")
result = Runner.run_sync(
    agent,
    "What is your return policy?",
    run_config=RunConfig(tracing_processors=[processor]),
)
print(result.final_output)
fluq.flush()
```

**Event mapping:**

| Span type | Fluq event |
|-----------|------------|
| `AgentSpanData` | `action` |
| `GenerationSpanData` | `llm_call` |
| `FunctionSpanData` | `tool_use` |
| `HandoffSpanData` | `spawn` |
| `GuardrailSpanData` | `decision` |

---

### CrewAI

```
pip install fluq-sdk[crewai]
```

```python
import fluq
from fluq.integrations.crewai import FluqCrewHandler

fluq.init(api_key="fo_xxx", agent_id="research-crew")
handler = FluqCrewHandler()

from crewai import Agent, Task, Crew, Process

researcher = Agent(role="Researcher", goal="Research AI trends", backstory="...")
writer = Agent(role="Writer", goal="Write summaries", backstory="...")

crew = Crew(
    agents=[researcher, writer],
    tasks=[...],
    process=Process.sequential,
    callbacks=[handler],
)
result = crew.kickoff()
fluq.flush()
```

**Event mapping:**

| Callback | Fluq event |
|----------|------------|
| `on_chain_start` / `on_chain_end` | `action` |
| `on_llm_end` | `llm_call` |
| `on_tool_end` | `tool_use` |
| `on_agent_action` | `action` |
| `on_agent_finish` | `decision` |
| `on_*_error` | `error` |

The handler propagates `run_id` → `parent_run_id` relationships so all events
within a single crew run share one `trace_id`.

---

### LangChain / LangGraph

```
pip install fluq-sdk[langchain]
```

```python
import fluq
from fluq.integrations.langchain import FluqCallbackHandler

fluq.init(api_key="fo_xxx", agent_id="research-agent")
handler = FluqCallbackHandler()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Attach at the LLM level
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[handler])
response = llm.invoke([HumanMessage(content="What is observability?")])
fluq.flush()
```

Attach at the chain level for full-chain traces:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_messages([("human", "{q}")]) | llm | StrOutputParser()
result = chain.invoke({"q": "Explain RAG."}, config={"callbacks": [handler]})
fluq.flush()
```

**Event mapping:**

| Callback | Fluq event |
|----------|------------|
| `on_chain_start` / `on_chain_end` | `action` |
| `on_llm_end` / `on_chat_model_start` + `on_llm_end` | `llm_call` |
| `on_tool_end` | `tool_use` |
| `on_agent_action` | `action` |
| `on_agent_finish` | `decision` |
| `on_*_error` | `error` |

---

## Manual traces and events

```python
import asyncio
import fluq
from fluq import Fluq, FluqConfig, TraceInput, EventInput, EventType

async def main():
    client = Fluq()
    client.init(FluqConfig(
        api_key="fo_xxx",
        agent_id="my-agent",
        capabilities=["research", "writing"],
    ))

    async with await client.trace(TraceInput(name="research_run")) as trace:
        client.event(EventInput(
            trace_id=trace.id,
            event_type=EventType.ACTION,
            payload={"step": "web_search", "query": "latest AI news"},
        ))
        # ... do work ...

    await client.destroy()

asyncio.run(main())
```

## Install options

| Extra | Installs |
|-------|---------|
| `fluq-sdk` | Core SDK only — no framework deps |
| `fluq-sdk[openai-agents]` | + `openai-agents>=0.1.0` |
| `fluq-sdk[crewai]` | + `crewai>=0.60.0` |
| `fluq-sdk[langchain]` | + `langchain-core>=0.3.0` |
| `fluq-sdk[all]` | All of the above |

## Links

- [Dashboard](https://fluq.ai)
- [Docs](https://docs.fluq.ai/sdk/python)
- [GitHub](https://github.com/fluq-ai/fluq-python)
