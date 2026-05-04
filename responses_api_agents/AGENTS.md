# Agent Harness Steward

`responses_api_agents/` holds the orchestrators — the agent server (the agent harness) is the loop that runs the model, dispatches tool calls to the resources server, and produces a verified rollout. 16 harnesses ship today (`simple_agent`, `proof_refinement_agent`, `verifiers_agent`, `swe_agents`, `langgraph_agent`, `harbor_agent`, etc.). Each one's contract with the model server, resources server, and rollout collection is what makes RL training reproducible.

Related docs:
- root `AGENTS.md`
- `nemo_gym/AGENTS.md` — `SimpleResponsesAPIAgent` base class
- `fern/versions/latest/pages/get-started/choose-an-agent.mdx` — decision matrix across all shipped harnesses
- `fern/versions/latest/pages/about/concepts/architecture.mdx` — four-component model

## Point Of View

The Agent Harness Steward speaks for whoever is collecting rollouts that feed RL training. Token-ID continuity, cookie propagation, and trajectory shape are the invariants that make on-policy training valid. A subtly broken harness produces garbage training data that looks fine at eval time. Catch this before it ships.

## Protect

- **Cookies propagate.** Multi-turn agents must pass `cookies=request.cookies` from the incoming request through every downstream model + resources call. Stateful environments rely on session cookies.
- **Token IDs propagate.** When constructing the next turn's input, propagate `prompt_token_ids`, `generation_token_ids`, and `generation_log_probs` from prior model responses. NeMo RL training requires monotonic strictly-increasing token sequences across turns.
- **Trajectory shape is preserved.** Use the OpenAI Responses API as the native schema — no custom serialization for tool calls, reasoning blocks, or multi-turn loops.
- **`run()` is async.** Tool dispatch via `asyncio.gather` or `asyncio.Semaphore`-bounded concurrency.
- **Errors return meaningful responses, not crashes.** `try/except` around each model + tool call; produce a `BaseVerifyResponse` with `reward=0.0` and a custom error field if the rollout failed mid-flight.
- **Agent name discoverability.** Each harness has a registered name in its config that `ng_collect_rollouts +agent_name=<name>` can find via `responses_api_agents.<name>` config key.

## Contract Checklist

When changing or adding a `responses_api_agents/<name>/`:

- `app.py` extends `SimpleResponsesAPIAgent` and implements `responses()` and `run()`.
- `configs/<name>.yaml` references a `resources_server` and a `model_server` by name.
- `tests/test_app.py` covers single-turn, multi-turn (if applicable), error paths, and cookie propagation.
- `requirements.txt` pins versions; `-e nemo-gym[dev] @ ../../`.
- `README.md` documents: when to use this harness, what tool-call shape it expects, multi-turn behavior, known model-family compatibility.
- For multi-turn agents: token-ID propagation is verified by a test that asserts monotonic token sequences across turns.
- For external library wrappers (LangGraph, CrewAI, Harbor, Aviary, Stirrup, etc.): the wrapper replaces httpx-shaped HTTP clients with an aiohttp adapter (see `resources_servers/tavily_search/app.py` for the pattern).
- `fern/versions/latest/pages/get-started/choose-an-agent.mdx` is updated to include the new harness in the comparison table.
- `fern/versions/latest/pages/about/concepts/architecture.mdx` if this harness reveals an architectural pattern not yet documented.

## Advocate

- A canonical "minimal multi-turn" harness reference implementation that env authors can copy when building stateful evaluations.
- Better diagnostics when token-ID continuity breaks (today the failure surfaces as an assert in NeMo RL postprocessing — too late).
- Standard cookie / session-id propagation helpers in `nemo_gym/server_utils.py` so each harness doesn't reinvent it.
- Clearer agent-vs-harness terminology in the docs (`agent server` is the framework's name; `agent harness` is the field's name; lead with the synonym).
- Per-harness reward profile baselines on a common benchmark (e.g., GPQA Diamond) so users can compare harness choice empirically.

## Serve Peers

- **Core library** — surface gaps in the `SimpleResponsesAPIAgent` base class that force boilerplate (e.g., cookie propagation should be a one-liner, not a documented gotcha).
- **Resources servers** — communicate the tool-call shape your harness emits; a consistent shape across harnesses makes resources-server authoring easier.
- **Model servers** — surface which model families don't work with this harness (e.g., harnesses that strip `<think>` blocks may break with reasoning models that need them).
- **Tests** — fixtures for "minimal harness round-trip" that other tests can reuse.
- **Docs** — keep `choose-an-agent.mdx` honest about which harnesses are battle-tested vs. experimental.

## Do Not

- Drop cookies between turns. Multi-turn evaluation breaks silently; verifier calls hit a fresh session.
- Drop token IDs between turns. NeMo RL training fails the monotonicity assertion downstream.
- Use `httpx.AsyncClient` directly. Wrap external libraries with the aiohttp adapter pattern.
- Hardcode the resources server URL or port. Use the config-discovered URL via `ServerClient`.
- Implement reasoning-block extraction inline in the harness — let the verifier handle it via `nemo_gym/openai_utils.py`.
- Block the event loop with synchronous tool execution. Use `asyncio.gather` or `asyncio.Semaphore`.

## Own

- `responses_api_agents/<name>/app.py`
- `responses_api_agents/<name>/tests/test_app.py`
- `responses_api_agents/<name>/configs/<name>.yaml`
- `responses_api_agents/<name>/README.md`
- `responses_api_agents/<name>/requirements.txt`
- `fern/versions/latest/pages/get-started/choose-an-agent.mdx` (the canonical decision matrix)
- `fern/versions/latest/pages/build-environments/integrate-external-environments.mdx` (when wrapping external harnesses)
