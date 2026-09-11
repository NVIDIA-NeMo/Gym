# Plan: run NOOA capability evaluations through Gym

## Goal

Use Gym as the reproducible evaluation envelope for NOOA's model-facing capability suite. Each Gym rollout must instantiate the same NOOA agent class, invoke the same method with the same arguments, and preserve the source scorer's meaning. This creates executable regression coverage behind the NOOA–Gym adapter before its architecture is simplified.

## Source contract

The source suite is `NVIDIA-NeMo/labs-OO-Agents@v0.0.9/tests/capability`:

- configuration selects an importable agent class and async method;
- each JSONL row contains `args`, `kwargs`, and `expected`;
- a fresh agent is constructed for every sample;
- scorers compare the typed method result and may inspect the native NOOA trace.

Gym's equivalent contract is:

- a self-contained JSONL row contains `responses_create_params` plus mapped method arguments;
- `nooa_agent` constructs a fresh agent and routes its policy calls through Gym;
- a Resources server receives the projected response and verifier-only expected value;
- Gym records reward and rollout evidence.

## First vertical slice: `calculate_simple`

The first slice intentionally has no environment tools and no judge model:

| Source | Gym representation |
|---|---|
| `CalculateSingleAgent.calculate(a, b, calculation)` | `CalculateSingleAgent` in `capabilities/calculate.py` |
| two `args`/`kwargs`/`expected` rows | `capability_calculate.jsonl` |
| `ExactMatchScorer` | dedicated `nooa_capability` verifier with parity-tested normalization |
| eval_pipeline model client | `GymResponsesLLM` routed to `policy_model` |
| typed Python return | projected final Responses assistant text |

Acceptance criteria:

1. The shipped config resolves with the NOOA agent, `nooa_capability`, and a policy model.
2. Only `a`, `b`, and `calculation` are mapped into the agent method; expected answers and provenance remain verifier-only.
3. Both source cases run through the real NOOA `CodeActStrategy` and the Gym `/run` seed → policy → verify path.
4. Correct results receive reward `1.0`; an intentionally wrong result receives `0.0`.
5. Existing NOOA adapter and `nooa_capability` verifier tests continue to pass.

## Expansion sequence

### Stage 2: richer exact-match capabilities

Add test definitions whose original contract is exact match and whose values survive JSON normalization:

- `calculate_batch`
- `sentiment_batch`
- `json_qa_lookup` and `json_qa_reasoning`
- `json_extract`
- `repl_exploration`
- router cases

Use one reusable capability verifier rather than adding one Resources server per test. Preserve source rows and immutable provenance.

### Stage 3: structured values

Add `structured_combined_extraction` and construction cases. Implement a verifier whose normalization is parity-tested against eval_pipeline's `ExactMatchScorer`, `TypeMatchScorer`, and the source custom scorer. Do not reduce these tests to string comparison.

### Stage 4: trace-sensitive capabilities

Add `calculate_complex`, `error_recovery`, and task-decomposition/methodology cases on the scoped native hook projection now used by the adapter. Extend `GymTraceHooks` only for lifecycle facts those scorers require; do not infer behavior from final text. Keep ATIF as an optional export/parity oracle rather than the live integration seam.

### Stage 5: stateful and concurrent capabilities

Add order/session tests, router fan-out, and concurrent subagents. A single Gym row must keep one fresh NOOA agent and one Resources session for all turns. Verify invocation parentage, tool ordering, cookie isolation, failure retention, and cancellation.

### Stage 6: truncation matrix

Materialize the selected truncation fixtures and preserve their formatter/truncation configuration. Keep the matrix bounded in normal CI; run the full model × format matrix as a scheduled or explicit evaluation job.

## Reuse policy

- Copy small JSONL fixtures into Gym so the benchmark is reproducible and independently runnable.
- Record the exact NOOA tag/commit and source path in every copied row or dataset manifest.
- Keep source agent behavior byte-for-byte equivalent. Move stable fixtures into an installable NOOA package later; do not depend on another repository's `tests` package at runtime.
- Reuse Gym Resources servers when their scoring semantics match exactly. Otherwise add a shared `nooa_capability` verifier and parity-test it against eval_pipeline.
- Never expose `expected` or scorer metadata through the NOOA argument mapping.

## Development and CI gates

For every added capability family:

1. config/data provenance test;
2. real NOOA execution with a deterministic fake Gym model response;
3. positive verifier case;
4. negative verifier case;
5. failure/timeout behavior where relevant;
6. native trace assertions for trace-sensitive scorers;
7. focused pytest and Ruff;
8. a real model rollout with inspected verifier and trajectory behavior before merge.

## Development prerequisites

Use the Python and uv versions declared by Gym's `pyproject.toml`, install the root `dev` extra, and install the adapter requirements from `responses_api_agents/nooa_agent/requirements.txt`. Keep uv's cache on a filesystem with enough space for native wheels/builds.

A live rollout additionally requires a configured policy endpoint and credentials. Offline config resolution and deterministic fake-model integration tests must remain runnable without them.

## Dual-run parity harness

Every capability case should have one canonical specification consumed by two runners:

1. **Native runner** — instantiate the NOOA agent with the configured NOOA LLM and invoke the method directly, matching `eval_pipeline`.
2. **Gym runner** — materialize the same case as a Gym row, execute it through the NOOA agent server, and verify the result through a capability Resources server.

Both runners must use the same scorer implementation. Extract stable capability fixtures and scorers from `tests/` into an importable NOOA capability package; keep the existing eval_pipeline config as one frontend and add a Gym materializer as the other. Until that package exists, copied fixtures must be pinned by commit, hashes, and canonical-row parity tests.

Compare at three levels:

- **Result parity:** normalized actual value, score, pass/fail, and error category.
- **Behavior parity:** model-call count, CodeAct versus direct return, tool/Python executions, retries, and subagent topology where the source scorer observes them.
- **Transport parity:** Gym response validity, reward, observation joins, cookie/session behavior, and training conversion.

## Full-stack test ladder

A full-stack Gym test means the real process and HTTP topology, not merely calling `NOOAAgent.run()` in process:

```text
Gym CLI / rollout collector
  -> NOOA FastAPI agent server (/run)
     -> Gym Responses model server
     -> capability Resources server (/seed_session, /verify)
  -> persisted rollout JSONL / ng_trajectory / aggregate metrics
```

Use four layers:

1. **Unit parity:** scorer, row materialization, config, mapping, and projection.
2. **In-process integration:** real NOOA strategy/runtime with deterministic fake `ServerClient` responses.
3. **Hermetic full stack:** real Gym CLI and server processes with a deterministic scripted model server; no external credentials. Assert output JSONL, reward, calls, observations, and cleanup.
4. **Live-model matrix:** native and Gym runs against the same model configuration, repeated enough to compare pass rates rather than requiring identical stochastic outputs. Inspect representative traces and verifier behavior.

The hermetic full-stack layer is the required CI gate. Live-model runs are scheduled/manual release gates because they cost money and are nondeterministic.

The first `calculate_simple` process-level gate is implemented by `tests/e2e/nooa_calculate_e2e_test.sh`; it uses a deterministic localhost Responses server and verifies the persisted rollout artifact.

## Capability migration matrix

Track each source test with: agent/method, row count, scorer types, required trace evidence, statefulness, subagents, expected Gym verifier, native status, Gym in-process status, Gym full-stack status, and live-model status. A capability is complete only when its original scoring semantics are preserved; unsupported evidence is an explicit blocked status, not a reduced substitute scorer.
