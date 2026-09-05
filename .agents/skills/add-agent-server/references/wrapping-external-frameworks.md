# Wrapping an external agent framework

`SimpleResponsesAPIAgent` is a thin interface — any framework can sit behind it. But the cost of
getting there varies enormously, and it is driven by one question:

> **Does the framework own the model-calling loop, or do you?**

Answer that first. It determines everything else.

## The four shapes, by where the model call happens

| Shape | Example | Model call happens | Cost |
|---|---|---|---|
| **You write the loop** | `simple_agent` | Your own `server_client.post()` | Lowest. No adapter needed. |
| **Library, you write the nodes** | `langgraph_agent` | Your hand-written graph nodes | Low. Use the library for control flow only; nodes speak Gym's API natively. |
| **Framework owns the loop** | `langchain_deepagents_agent` | Inside the framework, via its own model interface | High. You must implement the framework's model interface on top of `server_client`. |
| **Separate process** | `claude_code_agent`, `remote_agent`, `swe_agents` | Outside Python entirely | Medium. No adapter, but you must inject a base URL and give up in-process visibility. |

The middle two are where people get surprised.

## If the framework owns the loop

`deepagents`' `create_agent()` internally calls `model.bind_tools(...).ainvoke(...)` on whatever model
object it is handed. That loop is not yours to rewrite. So the only intervention point is the model
object itself — which means implementing the framework's model interface from scratch, backed by Gym's
`server_client`.

You cannot use the framework's own client (`langchain_openai.ChatOpenAI` and friends), because that
runs on the `openai` SDK, which runs on `httpx`, which `CLAUDE.md` forbids inside a Gym server process.

The minimum surface is usually smaller than it looks. For LangChain, a `BaseChatModel` subclass needs:

- `_llm_type` property
- `bind_tools(tools, *, tool_choice=None, **kwargs)` — convert schemas, then `self.bind(...)`
- `_agenerate(messages, ...)` — the actual call
- `_generate(...)` — raise `NotImplementedError`; Gym agent servers are async-only

Plus conversion functions in both directions. Budget for four:

```
to_langchain(input_items)              # Gym input items  -> framework messages   (once per request)
to_responses(new_messages, model_name) # framework messages -> Gym response       (once per request)
to_gym_input(messages)                 # framework messages -> Gym input items    (once per model call)
to_langchain_ai_message(gym_response)  # Gym response -> framework message         (once per model call)
```

Keep them in a separate module from the endpoint code — reviewers ask for this split, and it makes the
conversion logic independently testable. The existing convention is `app.py` for the server class and a
package-local bridge module (`responses_langchain_bridge.py`) for the model adapter and converters.

### The reuse that is safe, and the one that is not

`to_gym_input()` and `to_responses()` do nearly the same walk. Two of the three item types are literally
the same pydantic class on both the input and output sides:

- `NeMoGymResponseFunctionToolCall` — same class both directions, required fields `arguments`,
  `call_id`, `name`. Reusable as-is.
- `NeMoGymFunctionCallOutput` — same class both directions, required fields `call_id`, `output`.
  Reusable as-is.
- `message` — **not** the same. Input-side messages validate against the looser
  `NeMoGymEasyInputMessage` / `NeMoGymMessage`; output-side requires `NeMoGymResponseOutputMessage`,
  which needs an `id` and `output_text`-shaped `content`, and only accepts `role="assistant"`.

So reuse the input builder for the two tool item types and patch the message items. Do not assume the
whole thing round-trips.

### Per-request state for a singleton model

Build the graph once in `__init__`:

```python
self.agent = self.build_agent(GymResponsesChatModel(agent=self))
```

Now the model object is shared across concurrent requests but still needs per-request data. Use a
module-level `ContextVar`, set in `responses()` and reset in a `finally`:

```python
_request_context: ContextVar[dict] = ContextVar("my_agent_request_context")

token = _request_context.set(
    {"rollout_id": rollout_id, "cookies": request.cookies, "model_url_path": model_url_path}
)
try:
    final_state = await self.agent.ainvoke({"messages": input_messages})
finally:
    _request_context.reset(token)
```

Resolve the model URL path **in `responses()`**, via `self.url_path_for_request("/v1/responses",
request)`, and pass it through the context. Do not rebuild it inside the model adapter — the adapter has
no access to the inbound request, so it cannot see the capture mode, and hand-building the prefix there
is how the capture-path bug happens (checklist item 1).

Storing a mutable dict lets the adapter evolve cookies in place across the framework's internal turns
without a redundant `.set()`:

```python
ctx["cookies"] = resp.cookies  # visible to the next call in this task
```

Write a concurrency test. Two staggered concurrent requests, each asserting it sees its own rollout id.

### Tool schema conversion

`convert_to_openai_tool()` gives you `{"type": "function", "function": {...}}`. Gym wants the inner dict
un-nested at the top level, **plus** a present-but-nullable `strict` key:

```python
formatted = [{"type": "function", "strict": False, **convert_to_openai_tool(t)["function"]} for t in tools]
```

Omit `strict` and you get a 422 from the model server, not a framework-layer error.

## If the framework runs as a separate process

You are not adapting a model interface — you are injecting a base URL and an API key into the child
process's environment, then parsing whatever transcript it leaves behind.

- Use `base_url_for_run(base_url, body)` or `resolve_model_base_url(...)` to build the URL, so the
  rollout prefix and capture mode survive. `claude_code_agent` builds it at `app.py:328-332` and injects
  it at `:467` as `ANTHROPIC_BASE_URL`.
- Bound concurrency with an `asyncio.Semaphore` if you shell out.
- Decode subprocess output with `errors="replace"`.
- In-process trajectory capture is not available. The established alternative is to parse the harness's
  own transcript into an `AgentObservationBundle` with explicit `ObservationGap`s for what you could not
  recover — see `claude_code_agent/observability.py`.
- Auto-install the external tool on startup rather than requiring manual setup; see `CLAUDE.md`'s
  "External Tool Auto-Install" section and `claude_code_agent/setup_claude_code.py`.

## If you get to write the loop or the nodes

Take this path when you can. `langgraph_agent` uses real LangGraph — `StateGraph`, conditional edges,
`.compile()` — but every node is a hand-written async function calling `server_client.post()` directly.
Grep it: zero hits for `bind_tools`, `ChatOpenAI`, or `BaseChatModel`.

The result is no model adapter, no `ContextVar`, no tool-schema patching, and no dependency conflict
(its `requirements.txt` is just `langchain-core` + `langgraph`). The extra machinery in
`langchain_deepagents_agent` is not the cost of "using LangChain" — it is specifically the cost of
adopting a framework whose `create_agent()` owns the model-calling loop.

If you only need the framework for control flow, use it for control flow only.
