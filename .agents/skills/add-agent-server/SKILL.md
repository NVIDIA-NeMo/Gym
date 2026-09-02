---
name: add-agent-server
description: >
  Guide for adding, reviewing, or debugging an agent server (agent harness) in NeMo Gym.
  Use when the user asks to add, create, wrap, or integrate an agent server, agent harness,
  or agent framework into NeMo Gym — including wrapping LangChain/LangGraph/deepagents, an
  external CLI, or a remote HTTP service. Also use when reviewing an existing agent server
  for correctness, or debugging silent data loss in rollouts (missing token IDs, lost tool
  calls, broken sessions, dropped trajectories). Covers the base class API, the four
  silent-failure pitfalls, external framework wrapping, directory layout, and config wiring.
  Triggered by: "add agent server", "new agent harness", "wrap agent framework",
  "responses_api_agents", "agent server review", "custom agent".
---

# Add or Review an Agent Server in NeMo Gym

An agent server is the harness half of a Gym environment: a FastAPI server under
`responses_api_agents/` that implements `responses()` (handle one turn) and `run()` (drive a rollout).
`SimpleResponsesAPIAgent` is a genuinely thin interface — any framework can sit behind it.

**The thing to know up front:** the gap between "my agent runs" and "my agent is correctly observed,
on-policy, and trainable" is where every real defect lives, and none of those defects raise an
exception. Rollouts complete, rewards get scored, and the damage surfaces later as training data with
no token IDs or a trajectory with no tool calls in it. Working code is not evidence of a correct agent
server.

Read `references/correctness-checklist.md` before writing or reviewing any agent server. That is the
core of this skill; everything else supports it.

## Step 1: Decide what shape you are building

Answer one question first — **does the framework own the model-calling loop, or do you?** It determines
how much work this is.

| Shape | Example | Cost |
|---|---|---|
| You write the loop | `simple_agent` | Lowest — no adapter |
| Library for control flow, you write the nodes | `langgraph_agent` | Low — nodes speak Gym's API natively |
| Framework owns the loop | `langchain_deepagents_agent` | High — you must reimplement its model interface on `server_client` |
| Separate process (CLI / remote HTTP) | `claude_code_agent`, `remote_agent` | Medium — inject a base URL, give up in-process visibility |

If you only need the framework for control flow, use it for control flow only. The extra machinery in
the third row is the cost of a framework calling the model for you — not the cost of "using LangChain."

See `references/wrapping-external-frameworks.md` for rows 2-4.

## Step 2: Scaffold

```
responses_api_agents/<name>_agent/
├── __init__.py
├── app.py              # server class + `if __name__ == "__main__": X.run_webserver()`
├── configs/<name>_agent.yaml
├── tests/test_app.py
├── requirements.txt    # starts with `-e nemo-gym[dev] @ ../../`
└── README.md
```

Check for dependency conflicts now, not later — run `uv sync` before you commit to a framework version.
Your framework's transitive pins must coexist with the whole monorepo's. See
`references/directory-and-config.md` for the full layout, config YAML shape, the
`<resources_server>_<agent>_model_server.yaml` combined-launch convention, and when to add extra
modules.

## Step 3: Implement `responses()`

Start from a reference implementation and keep its annotations while you edit:

- `scripts/reference/native_loop_agent_reference.py` — you write the loop.
- `scripts/reference/framework_bridge_reference.py` — the framework owns the loop.

Neither is runnable or imported; they are annotated shapes to copy. The real, complete implementation
is `responses_api_agents/simple_agent/app.py`.

`references/base-class-api.md` has the verified API surface: `url_path_for_request` vs
`url_path_for_run`, `rollout_id_from_run`, `_model_call_capture_enabled`, config fields, and how session
cookies actually work.

## Step 4: Work the correctness checklist

`references/correctness-checklist.md`. The four that block a PR, because they corrupt or discard data
that already flows through the system:

1. **Forward the capture-mode path.** Three URLs route to your one `responses()` method. Use
   `self.url_path_for_request("/v1/responses", request)`, never a hardcoded `"/v1/responses"` and never
   a hand-built `rollout_path_prefix(...)` (its `token_capture` defaults to `False`). Otherwise training
   token IDs are silently never captured.
2. **Thread cookies explicitly.** Gym's aiohttp client uses `DummyCookieJar()` — nothing is stored or
   resent, ever. Keep model-server and resources-server cookies in **separate** jars, reassign each from
   its response, and mirror both onto the outgoing `response` via `set_cookie`.
3. **Convert prior `function_call` / `function_call_output` items on input.** Filtering input to
   `type == "message"` works for fresh prompts and silently corrupts every replayed trajectory.
4. **Emit the full tool-call trace on output.** `NeMoGymResponse.output` is a list. Returning only the
   final assistant message permanently destroys the agent trace for every rollout.

Then the rest: no `httpx` (item 5), the `strict` tool-schema field (6), `ContextVar` for build-once
frameworks (7), opt-in trajectory capture (8), cwd-relative config paths (9), dependency conflicts (10).

## Step 5: Test what actually breaks

Unit tests that assert on `output[0].content[0].text` pass even when the entire tool trace is missing.
Write the tests that would have caught the checklist items:

- A rollout using at least one tool → assert `output` contains `function_call`, `function_call_output`,
  **and** `message` items, in order.
- An input list containing a `function_call` + `function_call_output` pair → assert both survive into
  the framework's message list.
- Tool-schema conversion → assert the `strict` key is present and the `function` dict is un-nested.
- If you used a `ContextVar`: two staggered concurrent requests → assert neither sees the other's
  rollout id.

Then run real rollouts. `CLAUDE.md` is explicit that green unit tests alone are not enough for agent
changes — run a model against it and inspect agent and verifier behavior.

```bash
gym env start --resources-server <resources_server>/<combined_config> --model-type <provider>
gym eval run --no-serve --agent <combined_config_name> --input <data>.jsonl --output /tmp/out.jsonl
gym env test --resources-server <name>
```

## Step 6: Before the PR

- `pre-commit run --files responses_api_agents/<name>_agent/**/*`
- `git commit -s` (DCO sign-off required)
- NVIDIA SPDX header on new source files; coverage >= 96%
- Put the working `gym env start` / `gym eval run` commands in a comment at the top of the combined
  config — every existing one does this.

## What is acceptable to defer

Full `TrajectoryRecord` support can ship as a stacked follow-up PR; this has been explicitly allowed in
review. Only `simple_agent` and `simple_agent_with_compaction` build one today.

Checklist items 1-4 are **not** deferrable. The distinction is that deferring trajectory support
declines to add new capture, while items 1-4 silently corrupt or destroy data that is already flowing.

## Reference files

| File | Contents |
|---|---|
| `references/correctness-checklist.md` | The 10 silent-failure pitfalls, with why/fix/how-to-check |
| `references/base-class-api.md` | Verified base class API: helpers, endpoints, config, cookies |
| `references/wrapping-external-frameworks.md` | The four shapes; model adapters, `ContextVar`, subprocess agents |
| `references/directory-and-config.md` | Layout, YAML wiring, launch commands, PR requirements |
| `scripts/reference/native_loop_agent_reference.py` | Annotated agent that writes its own loop |
| `scripts/reference/framework_bridge_reference.py` | Annotated adapter for a loop-owning framework |
