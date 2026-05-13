---
name: port-from-nemo-rlvr
description: >
  Guide for porting an existing environment from NeMo-RLVR
  (`nemo_rl/environments/*_environment.py`) into a NeMo-Gym resources server.
  Use whenever the user asks to port, migrate, translate, or recreate an RLVR
  environment in Gym, or references a file under `nemo-rlvr/nemo_rl/environments/`.
  Triggered by: "port from rlvr", "port nemo rlvr environment", "migrate environment to gym",
  "recreate this rlvr env in gym", "translate ifeval/math/code env to gym".
---

# Port a NeMo-RLVR Environment to NeMo-Gym

## When to use

The user wants to take an existing `nemo_rl/environments/<name>_environment.py`
file (a Ray-actor `EnvironmentInterface`) and reimplement it as a NeMo-Gym
resources server.

An RLVR environment typically bundles **two** kinds of logic in one Ray actor:

1. **Verification / scoring** — `verify_*` helpers that grade a final response.
2. **Tools / world simulation** — code that runs *during* the rollout: a
   sandbox that executes generated code, a database the model queries, a search
   API, a calculator, a game engine, etc. RLVR exposes these inside `step()`
   either as observations returned to the model or as side effects the
   environment maintains across turns.

Both pieces have to land somewhere in Gym. The split is:

- **Verification → `verify()`** on the resources server.
- **Tools → additional FastAPI routes** on the same resources server, exposed
  via `setup_webserver()` (one `app.post(...)` per tool). The agent server
  forwards model-emitted tool calls to those routes.
- **Multi-turn / stateful loops → the agent server.** A `simple_agent` only
  does one turn; if RLVR's `step()` is called repeatedly with the model
  consuming prior observations, you need a multi-turn agent (copy
  `proof_refinement_agent` or `tau2`) and possibly session state via cookies.

If the user is adding a brand-new benchmark from scratch, prefer the
`add-benchmark` skill instead. This skill is specifically about translating an
existing RLVR environment into the Gym layout.

## Mental model: what changes between RLVR and Gym

| NeMo-RLVR | NeMo-Gym |
|---|---|
| `@ray.remote` actor extending `EnvironmentInterface` | FastAPI server extending `SimpleResourcesServer` |
| `step(message_log_batch, metadata) -> EnvironmentReturn` | `async def verify(body) -> BaseVerifyResponse` (final scoring) |
| Tool / sandbox logic invoked inside `step()` (returned as observations) | **Separate `app.post("/<tool>")` routes** on the resources server, registered in `setup_webserver()`; agent forwards model tool calls to them |
| Tool schemas implicit in the env code | Tool schemas declared explicitly in each JSONL row's `responses_create_params.tools` (OpenAI function-call format) so the model knows what to emit |
| Multi-step loop owned by the env (`step()` called repeatedly, observations fed back) | Multi-step loop owned by the **agent** (`responses_api_agents/<kind>/app.py`'s `run()`); resources server stays stateless or uses cookies for session state |
| Cross-turn state held in the Ray actor's `self` | Per-session state keyed by cookie; propagate `cookies=request.cookies` through every downstream call |
| Batched: workers chunk a list of responses | Per-request: one HTTP call per rollout, framework handles concurrency |
| `metadata[i]["ground_truth"]` carries verification data | Top-level row fields on the JSONL (forwarded into the verify body), typed via your `BaseVerifyRequest` subclass — **not** nested under `verifier_metadata` |
| Reward is a `torch.Tensor` from `step` | Reward is a `float` field on `BaseVerifyResponse` |
| `global_post_process_and_metrics` aggregates pass-rate metrics | Aggregation lives outside the server (`ng_reward_profile`, `print_aggregate_results.py`) — don't port it |
| Worker pool, chunking, `ray.get`, stop strings, done flags | Drop all of these. The agent server owns orchestration. |
| Subprocess sandboxes via Ray actors | Subprocess via `asyncio` + `asyncio.Semaphore` for concurrency control |

The core invariants to preserve: **same response text + same ground truth →
same reward**, and **same tool call args → same tool response**. Everything
else is plumbing.

## Step-by-step

### 1. Read the RLVR environment

Triage what's in the `step()` (or its helpers). Categorize every line:

- **Scoring** — final-answer comparison, constraint checks, judge calls. Goes
  into `verify()`.
- **Tools** — anything the *model* calls during the rollout: code execution,
  search, DB queries, calculators, game-engine `step()`, retrievers. Each
  becomes its own FastAPI route.
- **Cross-turn state** — counters, accumulated context, simulated world state.
  Decide whether it's per-rollout (cookie-keyed session state on the resources
  server) or stateless-derivable (recomputed from the conversation each call).
- **Pre/post-processing** — `<think>` stripping, boxed-answer extraction, JSON
  parsing of the ground truth. Preserve exactly.
- **Diagnostics** — flags like `verification_failed`. Surface on the
  `VerifyResponse` so the rollouts JSONL records them.
- **External deps** — graders, sandboxes, datasets. Plan to vendor (copy into
  the server dir) or pin in `requirements.txt`.

Also count *turns*: does the RLVR env's `step()` get called multiple times
within one rollout, with new observations fed back to the model? If yes, this
is a multi-turn port and the agent matters as much as the resources server.

### 2. Scaffold the resources server

Use `example_single_tool_call` (simplest) or a topical neighbor as a template.
Required structure:

```
resources_servers/<name>/
├── __init__.py
├── app.py
├── configs/<name>.yaml
├── data/.gitignore                # gitignore train/validation jsonl
├── data/example.jsonl             # 5 rows for smoke-testing
├── requirements.txt               # `-e nemo-gym[dev] @ ../../`  + extras
├── tests/__init__.py
├── tests/test_app.py
└── README.md
```

If the RLVR env imports a sibling utility module (e.g.
`environments/ifeval_utils/if_functions.py`), copy it into the server
directory rather than depending on the rlvr package. Keep the original Apache
2.0 header.

### 3. Translate `step()` to `verify()`

Skeleton:

```python
class MyResourcesServerConfig(BaseResourcesServerConfig):
    pass

class MyVerifyRequest(BaseVerifyRequest):
    # One field per item read from RLVR's metadata["ground_truth"].
    # Use Union[str, dict, list] etc. if the RLVR side accepted multiple shapes.
    ground_truth: Union[str, dict, list]

class MyVerifyResponse(BaseVerifyResponse):
    # Surface the same diagnostic fields RLVR's metadata exposed
    # (e.g. verification_failed, follow_constraint_list).
    verification_failed: bool

class MyResourcesServer(SimpleResourcesServer):
    config: MyResourcesServerConfig

    async def verify(self, body: MyVerifyRequest) -> MyVerifyResponse:
        # Extract the assistant's final text from body.response.output.
        text = ""
        if body.response.output:
            last = body.response.output[-1]
            if hasattr(last, "content") and last.content:
                text = last.content[0].text

        # Apply the SAME pre-processing as RLVR (strip <think>, parse JSON, etc.)
        # Run the SAME scoring function.
        # Map exceptions to verification_failed=True with reward=0.0.

        return MyVerifyResponse(
            **body.model_dump(),
            reward=reward,
            verification_failed=verification_failed,
        )
```

Key adapter points:

- **Response extraction.** RLVR concatenates all assistant turns
  (`"".join(assistant_responses)`). Gym's single-turn agents put the final
  answer in `body.response.output[-1].content[0].text`. For most benchmarks,
  that's equivalent. For multi-turn benchmarks, follow what `simple_agent` /
  your agent emits.
- **`<think>` handling.** If RLVR did `response.split("</think>")[-1]`, do the
  same. If RLVR returned reward 0 with `verification_failed=False` for an
  unclosed `<think>`, preserve that exact distinction.
- **Verifier failures.** Wrap the scoring call in try/except. `JSONDecodeError`
  and other exceptions → `reward=0.0, verification_failed=True`. Do not crash.
- **Reward shape.** Gym uses `float` rewards (typically 0.0 or 1.0). Drop the
  `torch.Tensor`, `done`, and `next_stop_strings` outputs entirely.
- **No batching.** Delete the chunk-to-workers logic. Gym's runtime fans out
  HTTP calls; one `verify` handles one rollout.

### 4. Translate tools into FastAPI routes

Skip this section if the RLVR env is pure scoring (e.g. IFEval, plain
math-grader). It applies whenever the model is expected to *call* something
mid-rollout — code execution, search, calculator, DB, retriever, simulator.

In RLVR these often live as Python functions invoked inside `step()` (or
imported helpers), with their results spliced into the message log. In Gym
each one becomes a route on the resources server, and the **agent** is what
forwards model-emitted tool calls to those routes.

```python
class RunCodeRequest(BaseModel):
    code: str
    language: str = "python"

class RunCodeResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int

class MyResourcesServer(SimpleResourcesServer):
    config: MyResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/run_code")(self.run_code)        # ← one route per tool
        app.post("/search")(self.search)
        return app

    async def run_code(self, body: RunCodeRequest) -> RunCodeResponse:
        # Translate the RLVR tool implementation. Use asyncio.Semaphore to
        # bound concurrent subprocess calls. Decode subprocess output with
        # errors="replace". Never crash on malformed input — return an error
        # field, not a 500.
        ...

    async def search(self, body: SearchRequest) -> SearchResponse:
        ...

    async def verify(self, body: MyVerifyRequest) -> MyVerifyResponse:
        ...
```

Then for each row of the JSONL, declare the tool *schema* in
`responses_create_params.tools` (OpenAI function-call format) so the model
knows what it can emit. The agent server reads the tool name from the model's
output and POSTs to `/<tool_name>` on the wired resources server. Example:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "Compute fib(10) and tell me the answer."}],
    "tools": [
      {
        "type": "function",
        "name": "run_code",
        "parameters": {
          "type": "object",
          "properties": {"code": {"type": "string"}, "language": {"type": "string"}},
          "required": ["code"], "additionalProperties": false
        },
        "strict": true,
        "description": "Run a code snippet and return stdout/stderr."
      }
    ],
    "parallel_tool_calls": false
  },
  "ground_truth": {"expected": "55"}
}
```

References to copy from:

- `resources_servers/example_single_tool_call/app.py` — minimal one-tool case
  (`/get_weather`).
- `resources_servers/code_gen/app.py` — subprocess-execution tool with Ray
  fan-out for test cases. Useful template for any sandbox port.
- `resources_servers/example_session_state_mgmt/app.py` — cookie-based
  per-rollout state, for environments that need cross-turn memory.
- `resources_servers/tavily_search/app.py` — wrapping an external HTTP API
  through the aiohttp adapter pattern (do this if the RLVR helper uses
  `requests` or `httpx`).

Tool-routing rules:

- **Tool route = tool name.** The agent dispatches by name, so the route path
  must match what you declare in the `tools[*].name` field of the JSONL.
- **Stateless by default.** If the RLVR env held cross-turn state in `self`,
  port it as cookie-keyed session state (see `example_session_state_mgmt`),
  not module-level globals. Multiple rollouts share one server process.
- **Bound concurrency for subprocesses.** `asyncio.Semaphore(N)` around any
  shell-out. The server must survive 4k–65k concurrent requests.
- **Errors as data, not exceptions.** A failed tool call should return an
  error field on the response (e.g. `stderr`, `error_message`), not raise.
  Crashing the route fails the whole rollout.

### 5. Multi-turn rollouts: pick the right agent

If RLVR's `step()` is called repeatedly within a rollout (model emits action,
env returns observation, model emits next action, …), `simple_agent` is wrong
— it's single-turn. Choose:

- **`simple_agent`** — one model call, optional tool call(s) as part of that
  one call, then `verify()`. Most scoring-only ports land here, and so do
  benchmarks where the model can emit a few parallel tool calls in one shot.
- **`proof_refinement_agent`** — multi-turn correction loop: model proposes,
  resources server scores, model sees the error, retries. Copy this if the
  RLVR loop is "score → feedback → retry."
- **`tau2`, `aviary_agent`, `gymnasium_agent`** — full agentic loops with
  tool feedback fed back to the model across many turns. Copy whichever
  matches your RLVR turn structure most closely.
- **Custom agent** — write one if none of the above matches. Mandatory plumbing
  for any multi-turn agent: propagate `cookies=request.cookies` through every
  downstream call (so per-rollout session state stays isolated), and
  propagate `prompt_token_ids` / `generation_token_ids` /
  `generation_log_probs` from each model response into the next turn's input
  (RL training needs them).

The agent goes in `responses_api_agents/<name>/`. Wire it in the YAML the
same way `simple_agent` is wired — point `resources_server.name` at your
ifeval-style instance and `model_server.name` at `policy_model`.

### 6. Drop these RLVR concepts (do not port)

- `__init__` worker pool / `IFEvalVerifyWorker` / `ray.remote` actors — replaced
  by FastAPI handlers. If you legitimately need parallelism inside `verify`
  (e.g. running test cases), use `asyncio.gather` + a `Semaphore`, or Ray
  remotes awaited with `await future` (not `ray.get`).
- `EnvironmentReturn` (observations, terminateds, next_stop_strings, done) —
  the agent server controls turn flow.
- `global_post_process_and_metrics` — Gym computes pass@k externally. Anything
  useful (e.g. `verification_failed_rate`) becomes a per-request response field
  that aggregators read from the rollouts JSONL.
- `shutdown()` — FastAPI lifecycle is handled by the framework.

### 7. Wire up YAML and example data

`configs/<name>.yaml` defines two top-level keys: the resources server
instance and an agent instance (named e.g. `<name>_simple_agent`) that pairs
it with the chosen agent kind (`simple_agent`, `proof_refinement_agent`, or
your custom one) and `policy_model`. Set `verified: false` and `domain:` to a
meaningful tag (often the same as the source RLVR env name's category — e.g.
`instruction_following`, `math`, `coding`, `agent`).

`data/example.jsonl` should contain 5 rows. Each row has:

- `responses_create_params.input` — the prompt as OpenAI messages.
- `responses_create_params.tools` — tool schemas (empty list for scoring-only
  ports; populated for tool-using environments — the model needs the schema
  to emit valid tool calls).
- **Top-level fields** for whatever your `VerifyRequest` expects beyond the
  base (e.g. `ground_truth`, `expected_answer`, `test_cases`). These sit at
  the row's top level alongside `responses_create_params`, *not* nested under
  a `verifier_metadata` key — the agent forwards top-level row fields into
  the verify body, and a 422 at `/verify` almost always means you nested them.

Build these by hand or by sampling from the RLVR dataset — make sure they
exercise distinct code paths (positive, negative, multi-constraint, tool-call
success, tool-call failure, etc.).

If there's a real train/validation set, upload it to the GitLab dataset
registry per `CLAUDE.md`'s "Dataset Management" section and reference it via
`gitlab_identifier:` in the YAML. Never commit train/validation JSONL to git.

### 8. Tests

Write `tests/test_app.py` mirroring `instruction_following/tests/test_app.py`:

- A helper builds a `NeMoGymResponse` with synthetic assistant text.
- One `test_sanity` that just constructs the server.
- For each branch of the original scoring logic, one positive + one negative
  test.
- Tests for the format/parsing edge cases you preserved (e.g. JSON-string vs.
  dict ground truth, `<think>` stripping, unclosed `<think>`, malformed
  `ground_truth` → `verification_failed=True`, unknown identifier → reward 0).
- For each tool route, one happy-path test and one error-path test (malformed
  args → structured error response, not 500). Run subprocess-touching tools
  with `pytest.mark.skipif(shutil.which(...) is None)` so CI without the
  binary still passes — but if your server auto-installs the tool, register a
  `pytest_configure` hook in `conftest.py` that calls the installer before
  collection.

Run them with:

```bash
.venv/bin/python -m pytest resources_servers/<name>/tests/ -x
# or, with isolated venv:
ng_test +entrypoint=resources_servers/<name>
```

Coverage must be >= 95% (project-wide policy; see CLAUDE.md).

### 9. Reward profiling (baseline against the RLVR numbers)

The point of porting is to get the *same* reward distribution as RLVR. Run
`ng_collect_rollouts` against the example file, then a real model on the
training set, and compare to whatever the RLVR side reported. Document the
results in the README's "Reward Profiling" section.

```bash
ng_run "+config_paths=[resources_servers/<name>/configs/<name>.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
ng_collect_rollouts \
  +agent_name=simple_agent \
  +input_jsonl_fpath=resources_servers/<name>/data/example.jsonl \
  +output_jsonl_fpath=results/<name>_rollouts.jsonl \
  +num_repeats=5
ng_reward_profile \
  +input_jsonl_fpath=resources_servers/<name>/data/example.jsonl \
  +rollouts_jsonl_fpath=results/<name>_rollouts.jsonl \
  +output_jsonl_fpath=results/<name>_profiled.jsonl \
  +pass_threshold=1.0
```

If your numbers diverge from RLVR's by more than ~1%, the bug is almost
certainly in pre-processing (think-tag stripping, answer extraction, JSON
parse) — diff your `verify` against the RLVR `step` line by line.

## Common pitfalls

- **Forgetting to strip `<think>`.** RLVR envs almost always do this;
  thinking-model rollouts will fail otherwise.
- **Tuple-returning validators.** Some IFEval-style validators return
  `(bool, extras)`. Coerce to `bool` before using as the reward signal.
- **Naming collisions.** If a Gym server already exists for the same task
  (e.g. `instruction_following` already covers IFEval-style data with a
  different schema), pick a distinct name (`ifeval`, `math_rlvr`, etc.) and
  call out the difference in the README so users pick the right one.
- **Dependency drift.** RLVR may use older versions of grading libs. Pin in
  `requirements.txt` if behavior is version-sensitive.
- **`httpx` in vendored deps.** Gym requires aiohttp for high concurrency
  (see CLAUDE.md "Async Patterns"). If the RLVR helper uses `httpx`, swap it
  out per the `tavily_search` adapter pattern.
- **JSONL field placement.** Fields read by your `VerifyRequest` (the analog
  of RLVR's `metadata["ground_truth"]`) go at the **top level** of each JSONL
  row, *not* nested under a `verifier_metadata` key. The agent forwards
  top-level row fields into the verify request body. Nesting them yields a 422
  Unprocessable Entity at `/verify`. (See `instruction_following/data/example.jsonl`
  for the canonical layout: `instruction_id_list`, `kwargs`, `prompt` are all
  top-level.)
- **Don't port `global_post_process_and_metrics`.** Compute pass@k via
  `ng_reward_profile` + `scripts/print_aggregate_results.py`. Anything that
  doesn't fit those tools should be a per-rollout response field, not a server
  method.

## Worked examples

**Scoring-only port (no tools, single-turn):**
`resources_servers/ifeval/` is a port of
`nemo-rlvr/nemo_rl/environments/ifeval_environment.py`. It demonstrates:
- Vendoring `if_functions.py` from the RLVR `ifeval_utils/` package.
- Translating `verify_ifeval_sample` (which accepted `str | dict | list` for
  the constraint) into a `Union[str, dict, list]` Pydantic field.
- Preserving the `<think>` handling (closed → strip, unclosed → reward 0 with
  `verification_failed=False`).
- Surfacing `verification_failed`, `follow_all_constraints`, and
  `follow_constraint_list` on the response instead of returning them via
  `metadata`.
- Dropping the Ray worker pool, `EnvironmentReturn`, and
  `global_post_process_and_metrics`.
- Top-level `ground_truth` placement in the JSONL row (no `verifier_metadata`
  nesting — that yields a 422 at `/verify`).

**Tool-using port (single tool, single-turn):** see
`resources_servers/example_single_tool_call/` paired with
`responses_api_agents/simple_agent/`. Minimal `setup_webserver()` registering
a `/get_weather` route alongside the standard `/verify`, with the tool
schema declared in `responses_create_params.tools` per row.

**Sandbox / subprocess port:** see `resources_servers/code_gen/` for the
canonical pattern of executing model-generated code via Ray-backed
subprocesses with `asyncio.Semaphore`-bounded concurrency and structured
error responses.

**Stateful / multi-turn port:** see
`resources_servers/example_session_state_mgmt/` for cookie-keyed per-rollout
state, and `responses_api_agents/proof_refinement_agent/` for a multi-turn
correction loop that feeds verifier output back to the model.

Pick whichever reference most closely matches your RLVR env's shape.
