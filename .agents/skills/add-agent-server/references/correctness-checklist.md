# Agent server correctness checklist

Every item here is a real defect that shipped in a real agent server and was caught in review, not a
hypothetical. They share one trait: **they fail silently.** Rollouts complete, rewards get scored, no
exception is raised, and the damage only shows up later as missing training data or an unusable
trajectory.

Work through this list before opening a PR, and when reviewing someone else's agent server.

`responses_api_agents/simple_agent/app.py` is the reference implementation for items 1-4. When in
doubt, do what it does.

---

## 1. Forward the capture-mode path — never hardcode `/v1/responses`

**The bug.** The base class routes three different URLs to your single `responses()` method:

```
POST /v1/responses
POST /ng-rollout/{rollout_id}/v1/responses
POST /ng-rollout/{rollout_id}/training-token-capture/v1/responses
```

If you post to a literal `"/v1/responses"` when calling the model server, a request that arrived on the
`training-token-capture` variant gets forwarded as a plain call. Token IDs and logprobs are never
captured. Nothing errors — you just get a training run that produces data with no token IDs.

**The fix.** In `responses()`, derive the path from the inbound request:

```python
url_path=self.url_path_for_request("/v1/responses", request)
```

In `run()`, where there is no inbound path to inherit, derive it from the body and config:

```python
url_path=self.url_path_for_run("/v1/responses", body)
```

Need a base URL instead of a path (external CLI, LiteLLM client)? Use `base_url_for_run(base_url, body)`
or `resolve_model_base_url(...)`.

**Do not** call `rollout_path_prefix(rollout_id)` yourself. Its `token_capture` parameter defaults to
`False`, so it silently produces the eval-style prefix and drops capture mode — the exact bug, just
spelled differently. Same for `apply_rollout_prefix`.

**How to check.** `grep -n 'url_path="/v1/responses"' app.py` — any hit inside `responses()` or `run()`
is a finding. Then confirm every `server_client.post` to the model server routes through a helper.

**Prior art.** `simple_agent/app.py:263` and `:291` get this right. `langgraph_agent` hardcodes
`url_path="/v1/responses"` (`app.py:128`, `reflection_agent.py:82`) and uses no prefix at all, so
per-rollout capture is not wired for it.

---

## 2. Thread cookies explicitly — there is no ambient cookie handling

**The bug.** Two variants, both common:

- *Static cookies*: the original inbound cookies are resent on every internal model call across a
  multi-turn run, and the cookies each response returns are dropped. Model-server session/routing state
  is lost partway through the rollout.
- *Conflated cookies*: model-server cookies and resources-server cookies are kept in one bucket, so each
  overwrites the other. Downstream `/verify` then gets the wrong session and reads the wrong per-task
  state.

**Why it happens.** Gym's shared aiohttp client is built with `cookie_jar=DummyCookieJar()`
(`server_utils.py:172`) — it stores nothing and resends nothing, by design. And every server gets a
*distinct* cookie name, because `get_session_middleware_key()` returns
`f"{class_name}___{config.name}"` and that string is used as the `session_cookie` name. So a cookie from
the resources server's `/seed_session` is a different cookie than the one your agent server sets; none
of it rides along automatically.

**The fix.** Keep two separate, evolving cookie jars, reassign each from the response it came from, and
mirror both onto the outgoing response:

```python
model_server_cookies = None
resources_server_cookies = request.cookies

# ... each model call:
model_response = await self.server_client.post(..., cookies=model_server_cookies)
model_server_cookies = model_response.cookies

# ... each tool call:
api_response = await self.server_client.post(..., cookies=resources_server_cookies)
resources_server_cookies = api_response.cookies

# ... before returning:
for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
    response.set_cookie(k, v)
```

That last loop requires `response: Response` as a parameter on `responses()`. FastAPI injects it; an
agent that omits the parameter cannot propagate anything and will break downstream verification.

**Deliberate exception.** `remote_agent` keeps the remote service's own cookies *out* of the outgoing
response on purpose — they are that service's private session, not Gym's (`app.py:176-177, 248-251`).
If you exclude a cookie jar, say why in a comment; silence reads as an oversight.

**How to check.** Does `responses()` take a `response: Response` parameter? Is `set_cookie` called? Are
there two distinct cookie variables, and is each reassigned from a response inside the loop?

---

## 3. Convert prior `function_call` / `function_call_output` items on input

**The bug.** Filtering `body.input` down to `type == "message"` only. This looks fine for fresh prompts,
because a fresh prompt *is* all messages. It silently corrupts every replayed or continued trajectory:
Responses API inputs legitimately carry `function_call` and `function_call_output` items, and dropping
them changes the conversation the model sees.

**The fix.** Handle all three item types when translating input into your framework's message format.
Round-trip is the requirement — whatever you emit on output (item 4) must be re-readable as input.

```python
if item_type == "message":
    ...
elif item_type == "function_call":
    # -> your framework's assistant-with-tool-call representation
elif item_type == "function_call_output":
    # -> your framework's tool-result representation
```

Guard the `json.loads(item.arguments)` — malformed arguments should degrade to `{}`, not raise. A
crashed request scores zero and loses the rollout; an empty-args tool call gets a normal error result
from the tool layer and the model can recover.

**How to check.** Feed the agent an input list containing a `function_call` + `function_call_output`
pair and assert both survive into the framework's message list. Any agent presented as a generic
Responses API agent needs this test.

---

## 4. Emit the full tool-call trace on output, not just the final answer

**The bug.** Building the response from only the last assistant message — e.g. "walk backward through
the messages, take the first non-empty assistant text." Every intermediate tool call and tool result is
discarded. This is permanent, unrecoverable loss of the agent trace for every rollout, and it is
independent of whether structured `TrajectoryRecord` support exists yet.

**The fix.** `NeMoGymResponse.output` is a list of output items, not a single message. Emit one item per
step, in order:

| Step | Output item type | Class |
|---|---|---|
| Model asked for a tool | `function_call` | `NeMoGymResponseFunctionToolCall` |
| Tool returned | `function_call_output` | `NeMoGymFunctionCallOutput` |
| Model answered | `message` | `NeMoGymResponseOutputMessage` |

Two of these classes are shared between the input and output sides
(`NeMoGymResponseFunctionToolCall`, `NeMoGymFunctionCallOutput`), so if you already have an
input-item builder you can reuse it for those. `message` is the exception: the output-side
`NeMoGymResponseOutputMessage` requires an `id` and `output_text`-shaped `content`, which the looser
input-side message types do not.

**How to check.** Run a rollout that uses at least one tool, then assert `len(response.output) > 1` and
that the item types appear in call/result/answer order. A test that only asserts on
`output[0].content[0].text` will pass even when the whole trace is missing.

---

## 5. Use Gym's aiohttp client, never `httpx`

`CLAUDE.md` requires all async HTTP inside a Gym server process to go through Gym's aiohttp-backed
`server_client` / `nemo_gym.server_utils.request()`. `httpx`/`httpcore` connection pooling is O(n²) and
hangs at high concurrency (16k+ requests).

This bites hardest when the framework you are wrapping uses `httpx` internally — which most do, since
`langchain_openai.ChatOpenAI` → `openai` SDK → `httpx` is the standard chain. You cannot simply
configure it away. Either reimplement the framework's model interface on top of `server_client` (see
`references/wrapping-external-frameworks.md`) or replace the library's HTTP transport with an aiohttp
adapter (`resources_servers/tavily_search/app.py`'s `TavilySearchAIOHTTPClient` is the pattern).

**How to check.** `grep -rn "httpx\|AsyncClient" app.py` and check the transitive imports of any
framework you construct.

---

## 6. Gym's tool schema needs `strict`, which LangChain's converter omits

`langchain_core.utils.function_calling.convert_to_openai_tool()` produces
`{"type": "function", "function": {"name", "description", "parameters"}}`. Gym's `FunctionToolParam`
(a passthrough of the real OpenAI Responses API type) additionally requires a **present-but-nullable**
`"strict"` key. Omitting it does not fail at the framework layer — it fails downstream as a 422 from
the model server, which is a confusing place to debug from.

```python
formatted = [{"type": "function", "strict": False, **convert_to_openai_tool(t)["function"]} for t in tools]
```

Also note the un-nesting: Gym wants the `function` dict's contents at the top level, not nested under a
`"function"` key. Lock both in with a test.

---

## 7. Build the graph once, and use a `ContextVar` for per-request state

Build compiled graphs / agent objects in `__init__`, not per request — that is what every existing agent
does. But if the framework owns the model-calling loop, the model object becomes a singleton shared
across concurrent requests, and it still needs per-request data (rollout id, cookies, resolved model
URL path).

Use a module-level `contextvars.ContextVar`, set once per request in `responses()` and reset in a
`finally`. `ContextVar` values are asyncio-task-local, so concurrent in-flight rollouts do not leak into
each other. Write a test that proves it: two staggered concurrent calls, each asserting it sees its own
rollout id.

If your framework does *not* own the loop (you write the nodes), skip all of this and thread a plain
dict through the graph state, like `langgraph_agent` does. The `ContextVar` is a cost of frameworks that
call the model for you, not a general pattern.

---

## 8. Trajectory observability is opt-in work

Nothing in the base class gives you a `TrajectoryRecord`. Only `simple_agent` and
`simple_agent_with_compaction` build one today; `claude_code_agent` uses a parallel
`AgentObservationBundle` / `ng_agent_observations` mechanism instead.

Two things must both be true for a complete trajectory, and this is not obvious from
`+observability_enabled=true` alone:

1. **Agent-side capture** — gate on `self._model_call_capture_enabled()` and build the record, including
   `AgentInvocation.conversation`.
2. **`model_call_capture_dir` set at the model-server level** — the per-model-call detail is merged in
   downstream by `rollout_collection.py`'s `_build_trajectory_record`, matched by `model_call_id`. It is
   an absolute path and a hard requirement the moment observability is on: omit it and the model server
   crashes at startup with a Pydantic validation error, not a "you forgot a flag" message.

Note that even `simple_agent` does not populate the top-level `TrajectoryRecord.model_calls` — only the
nested `AgentInvocation`. That field is filled in downstream.

Deferring trajectory support to a follow-up PR is acceptable and has been explicitly allowed in review.
Items 1-4 are not deferrable, because they corrupt or discard data that already flows through the
system rather than declining to add new capture.

---

## 9. Relative paths in configs resolve against that server's own process cwd

A path like `exclude_domains_file_path` in a resources-server config resolves relative to that server's
own working directory when Gym spins it up (e.g. `resources_servers/tavily_search/`) — not the repo
root, and not wherever you ran `gym env start`. Wrong first guesses show up as `FileNotFoundError` at
startup.

---

## 10. Check for dependency conflicts before committing to a framework

`requirements.txt` starts with `-e nemo-gym[dev] @ ../../`, so your framework's transitive pins must
coexist with the entire monorepo's. This is not always possible at the framework's latest version.

Real example: `deepagents>=0.7.0` unconditionally pulls `langchain-anthropic`, which needs
`anthropic>=0.120.0`, while `nemo_gym` pins `anthropic<=0.109.2`. Ranges do not overlap; `uv` reports it
flatly unsatisfiable. Resolved by pinning `deepagents<0.7.0` and documenting why in a comment.

Run `uv sync` early. If you pin below latest, leave a comment explaining the conflict and what would
unblock a bump — otherwise the next person reads it as an arbitrary stale pin.
