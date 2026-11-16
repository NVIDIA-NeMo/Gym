---
name: add-benchmark
description: >
  Guide for adding a new benchmark or training environment to NeMo-Gym.
  Use when the user asks to add, create, or integrate a benchmark, evaluation,
  training environment, or resources server into NeMo-Gym. Also use when wrapping
  an existing 3rd-party benchmark library. Covers the full workflow: data preparation,
  resources server implementation, agent wiring, YAML config, testing, and reward
  profiling (baselining). Triggered by: "add benchmark", "new resources server",
  "integrate benchmark", "wrap benchmark", "add training environment", "add eval".
---

# Add Benchmark to NeMo-Gym

The Gym page [Add a benchmark](../../../fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx) is the current command reference. When it and this skill disagree, the page wins. This skill adds the order of work and the checks a benchmark must pass before it merges (Step 7).

## Determine Integration Type

Before starting, determine which type of benchmark you're adding:

**Native benchmark** — verification logic implemented directly in a Gym resources server:
- Resources server implements `verify()` with reward logic
- Agent server orchestrates model calls (use `simple_agent` for single-turn, or custom agent for multi-turn)
- Example: `code_gen`, `instruction_following`, `math_with_judge`

**External benchmark** — wrapping a 3rd-party library that has its own orchestration:
- Integrate at the agent server level (not resources server)
- Agent's `/run` endpoint wraps the external library
- Pre-process from Gym schema to library input, post-process back to `BaseVerifyResponse`
- Reproduce publicly reported numbers with the original repo first, then reproduce again after Gym integration
- Add the dependency in `pyproject.toml` and commit its `uv.lock`

## Workflow

### Step 1: Scaffold the server

Run `gym env init` to generate the directory structure:

```bash
gym env init --resources-server my_benchmark
```

This creates:
```
resources_servers/my_benchmark/
├── app.py              # Server template
├── configs/my_benchmark.yaml
├── data/.gitignore
├── tests/test_app.py
├── pyproject.toml
├── uv.lock
└── README.md
```

For external benchmarks, create the agent server manually under `responses_api_agents/my_agent/` with the same structure.

### Step 2: Prepare data

Convert your source dataset to Gym JSONL format. Each line must have `responses_create_params.input` (OpenAI message format). Task-specific verification data goes in `verifier_metadata`.

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "System prompt"},
      {"role": "user", "content": "Problem statement"}
    ]
  },
  "verifier_metadata": {
    "test_cases": [{"input": "...", "expected_output": "..."}],
    "task_id": "unique_id"
  }
}
```

**Data conversion**: Write conversion scripts in the **source repo** (e.g. your dataset repository), not in NeMo-Gym. Prompt files also belong in the source repo. Exception: when there is no external source repo. See `references/patterns.md` § "Data Conversion Script Pattern".

**`example.jsonl`**: Generate 5 entries for smoke testing. This file is committed directly to git in `data/example.jsonl`.

**`train`/`validation` datasets**: Upload to the GitLab dataset registry — these must NOT be committed to git.

```bash
gym dataset upload --storage gitlab \
    --name my_benchmark \
    --revision 0.0.1 \
    --input resources_servers/my_benchmark/data/my_dataset.jsonl
```

Requires MLflow credentials in `env.yaml` (or passed via CLI):
```yaml
mlflow_tracking_uri: <your-gitlab-mlflow-tracking-uri>
mlflow_tracking_token: <your-gitlab-api-token>
```

**`data/.gitignore`**: The scaffold generates default patterns (`*train.jsonl`, `*validation.jsonl`, etc.). If your filename doesn't match (e.g. `my_eval.jsonl`), add a custom pattern (e.g. `*eval.jsonl`). If data was previously tracked, run `git rm --cached <file>`.

**Validate** your data:
```bash
# Validate example data (for PR submission)
gym dataset collate --config resources_servers/my_benchmark/configs/my_benchmark.yaml \
    --output-dir /tmp/prepare \
    --mode example_validation

# Download and prepare train/validation from GitLab
gym dataset collate --config resources_servers/my_benchmark/configs/my_benchmark.yaml \
    --output-dir data/my_benchmark \
    --mode train_preparation \
    --download \
    +data_source=gitlab
```

### Step 3: Implement verify()

Edit `app.py`. The `verify()` method receives model output + `verifier_metadata`, returns reward.

For code execution benchmarks, see `references/patterns.md` § "Subprocess Execution with Ray" and "Resources Server Pattern".

Critical rules:
- Return `reward` as 0.0 or 1.0 (binary)
- Handle empty/missing model output gracefully — return 0.0, don't crash
- For an environment failure (sandbox down, judge unreachable), do not score 0.0. Return `mask_sample`, `failure_kind` (as `<server>:<kind>`) and `failure_reason` so the row leaves the aggregate. Add one test case that masks a row
- Decide who records the grade before you write `verify()`: see Step 7b, "Failure-handling conformance"
- Must handle 4k-65k concurrent requests without crashing
- Use `asyncio.Semaphore` for subprocess concurrency control
- For Ray remote tasks: `result = await future` (Ray futures are directly awaitable). Never call `ray.get()` in async context.
- Decode subprocess output with `errors="replace"`
- Strip `<think>`/`<thinking>` blocks before parsing model output (thinking models emit these)
- Tests should `pytest.mark.skipif` when external tools aren't installed
- If the benchmark auto-installs its tool (see Step 3b), add a `pytest_configure` hook in `conftest.py` to run the install before test collection — `skipif` evaluates at import time, before fixtures run

If the server keeps state per session (a sandbox handle, a connection, a session dict):
- State keyed by the session ID needs no `num_workers: 1`: session routing sends every request of a session to the worker that owns it, and a lost owner answers 410. Override `multiple_workers_unsupported_reason()` only for state keyed by another ID in the request body
- Implement the state hooks `has_session_state()`, `initialize_session_state()` and `release_session_state()`, and set `seed_request_model`; the base then serves seed, close and checkpoint retirement under a per-session lock. Score inside `async with self.verifying(session_id) as verification`; mask `verification.is_lost`, and use `verification.retain(payload)` for compact retry evidence. Call `retain` before an awaited judge; use `retain(None)` explicitly only when the request contains everything needed to score a retry. Omitting retention makes the retry `session_lost`, including after a judge failure. The context releases state even when scoring raises. Cleanup failure is observed without replacing the score or original scoring exception, and ownership stays available for close or retirement to retry release. Release hooks must be async, idempotent, and retain handles on failure. Exported servers restore ownership through `restore_session_identity()` or `restore_session_identities()`. Use `verifying(session_id, allow_missing=True)` for a request-scored verifier that accepts an unseeded call. A seed override that widens the cookie arm declares it as `Annotated[CookieModel, BeforeValidator(reject_partial_typed_seed)]`. Override `seed_session()` and `close_resources_session()` only when the hooks cannot express the lifecycle, refuse an unfixable seed with `ResourcesSeedRejectedError` (422) or `ResourcesSessionConflictError` (409) and never a plain exception, and never add your own `/close_session` route. Close must succeed for a session that is already closed or never seeded, must not let a later seed recreate the session, and must raise if the release fails
- A custom lifecycle holds a per-session `asyncio.Lock` around the session dict, so a close racing a seed cannot leak a sandbox
- If seed fails after allocating a resource, free it before re-raising
- Export `VERIFIER_FIXTURE` (`gym env test` needs it). Set `REVERIFY_MODE = ReverifyMode.STATELESS` on the config only when `verify()` uses just the request payload

### Step 3b: Auto-install external tools (if applicable)

If the benchmark requires an external tool (compiler, runtime, etc.), auto-install it on server startup so users don't need manual setup. See `references/patterns.md` § "External Tool Auto-Install Pattern".

Key points:
- Create `setup_<tool>.py` with `ensure_<tool>()` — checks PATH, forks on `sys.platform` (brew on macOS, build from source on Linux)
- Call it in `model_post_init()` before semaphore init
- Build scripts should be idempotent and install into a local gitignored prefix
- Add a `pytest_configure` hook in `tests/conftest.py` that calls `ensure_<tool>()` before collection

### Step 4: Wire YAML config

Edit `configs/my_benchmark.yaml`. Define the resources server instance and agent pairing(s). See `references/patterns.md` § "YAML Config Pattern".

Key points:
- `verified: false` is auto-added by pre-commit hook (set to `true` after baselining)
- `license` is required for `train` and `validation` datasets
- Agent references resources server and model server by instance name

For multi-turn benchmarks, either use `proof_refinement_agent` or create a custom agent. See `references/patterns.md` § "Agent Patterns".

For a coding harness in a task sandbox, pair the resources server with a session harness (`pi_agent`, `codex_agent`, `opencode_agent`, `openclaw_agent`, `hermes_agent`) through an environment server; see [Integrate existing agents](../../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx). Do not wire a new benchmark to `opencode_sandboxed_agent`, `pi_sandboxed_agent` or `hermes_sandboxed_agent`, or to the `/run` path of `terminus_2_sandboxed_agent` or `miniswe_sandboxed_agent`: that legacy path is deprecated, even though older presets still use it.

For `train`/`validation` datasets, add `gitlab_identifier` alongside `jsonl_fpath`:
```yaml
datasets:
- name: my_dataset
  type: train
  jsonl_fpath: resources_servers/my_benchmark/data/my_dataset.jsonl
  gitlab_identifier:
    dataset_name: my_benchmark
    version: 0.0.1
    artifact_fpath: my_dataset.jsonl
  license: MIT
- name: example
  type: example
  jsonl_fpath: resources_servers/my_benchmark/data/example.jsonl
```

Both fields must coexist: `jsonl_fpath` is the local download destination, `gitlab_identifier` tells the system where to fetch from. `example` datasets don't need `gitlab_identifier` — they're committed to git directly.

### Step 5: Test

```bash
# Run server tests (creates isolated .venv, slow on first run)
gym env test --resources-server my_benchmark

# Run core library tests to check nothing broke
pytest tests/unit_tests/ -x
```

Test coverage must be >= 95%. Write tests for: verify pass, verify fail (wrong output), verify fail (no code extracted), verify fail (compilation error if applicable), verify timeout.

### Step 6: Smoke test end-to-end

```bash
# Start servers
gym env start \
    --config resources_servers/my_benchmark/configs/my_benchmark.yaml \
    --model-type openai_model

# Quick test with example data
gym eval run --no-serve \
  --agent my_benchmark_simple_agent \
  --input resources_servers/my_benchmark/data/example.jsonl \
  --output results/example_rollouts.jsonl \
  --num-repeats 1 \
  --max-output-tokens 16384 \
  --temperature 1.0

# Inspect results
```

### Step 7: Exit checks

A benchmark merges only when it passes these checks. Put the evidence in the PR.

| Check | What it asks | How to check |
|---|---|---|
| Health | Very few rollouts are unhealthy. | `gym eval run` writes `quality_summary.json` into the run folder. `gym eval health-check <run_dir>` reruns the checks. Read the counts: it does not fail the run for you. `unobserved` means Gym has no model-call record to judge by. It is not unhealthy. |
| Evidence | Every rollout passes the `gym-p0/v1` evidence gate. Harness work only. | `python scripts/inspect_harness_conformance.py --bundle <run_dir>/rollouts.jsonl --output <dir>`. Exit 0 pass, 1 evidence failure, 2 checker error. |
| Score | The score matches the reference score, inside the tolerance, on one current frontier model. | `gym eval compare` |
| Config | The resolved Gym config is saved with the result. | `resolved-config.yaml` is in the run folder. |

To run the finished benchmark as a submitted evaluation on a cluster (submit, resume, recover, acceptance gates),
follow the [run-and-babysit-eval](../run-and-babysit-eval/SKILL.md) skill.

Run on one benchmark and harness pair. Read the rollouts, not the exit code: an exit of 0, an empty failures file and a rising rollout count prove nothing. List every health check you ignore (`--health-check-ignore`) in the PR, with the reason.

Reward profiling for training environments is a separate skill: `nemo-gym-reward-profiling`.

For an external benchmark: reproduce the original repo's published numbers first, then again after the Gym integration. The scores should match.

### Step 7b: Failure-handling conformance

The evidence gate in the exit checks and the session-contract checker (named at the end of this step) are necessary but not sufficient for failure handling. They do not show that the benchmark's grade is observable, that its characteristic failures are typed, or that interrupted work resumes correctly. A benchmark is conformant when a small real run shows all three. Submit, resume, recover and the acceptance gates are in [run-and-babysit-eval](../run-and-babysit-eval/SKILL.md); this step adds what a new benchmark has to provide for them to work.

The state hooks and `verifying()` manage resource ownership and compact retry evidence; they do not record a grade. Moving scoring into that context does not replace any grading path below. The registered base `/verify` route still applies `observe_verifier`; a custom route must preserve its observation wrapper. An in-memory retry receipt is neither a saved policy answer nor durable grading evidence: keep the policy answer in the rollout, declare `REVERIFY_MODE` truthfully, and test any state-dependent re-judge before claiming offline recovery. Only servers that explicitly call `verifying()` adopt its release behavior.

**1. Declare the grading path.** Name who grades, pick one path and say which in the benchmark's README. Every scored result, including a legitimate wrong-answer zero, must bind the graded response and the reward to that grader. Grading that nothing records reads `UNVERIFIED`.

| Path | What to provide |
| --- | --- |
| A. The resources server observes it | For a judge, run the grading inside `observe_llm_judge_grading` (`nemo_gym.server_runtime.verifier_observability`), make the judge calls through `call_judge`, and call `record_parsed_judge_verdict()` (`nemo_gym.verification.judge`) only after a real verdict was parsed. For a plain calculation, wrap the actual calculation in `observe_programmatic_grading`. |
| B. The shared verifier record | `observe_verifier` records a verify that completes without sending a model request: its name, score and verdict, bound to the response. Return the actual response and reward. Set `grading_may_use_uninstrumented_model = True` on the resources server class when `verify()` may ask a model through a client that notes no request (an external SDK client); it then gets no record, because a missing note does not show that grading was model-free. |
| C. A saved grading artifact | The row carries one of the artifacts in `GRADING_ARTIFACTS` (`nemo_gym.orchestration.recovery`: `judge_record`, `rubric_verdict`, `symbolic_match`, `metric_record`, `test_execution`, `environment_reward`, `repository_test_run`, `substep_tests`, `unit_test_results`, `node_set_f1`, `session_reward_file`, `guarded_objective_score`, `native_scorer_score`), with fields that map to the graded response, the reward and the trial. Test that a mismatched or stale artifact is rejected. An aggregate score file alone is not one. |
| D. The agent grades its own rollouts | With no resources server, the invocation's grading record names the agent (`agent_ref.name`) and binds the actual response and reward. Do not invent a resources-server name. This covers programmatic rubrics; a rubric that is asynchronous or judged by a hidden model needs its own observation, as in path A. Check that your pin supports the rubric you use. |

**2. Type each characteristic failure where it happens.** Keep the response and the cause. A context-window rejection is `model_context_limit` and is final. Sandbox and provider faults use the matching registered kind, such as `provider_unavailable` or `provider_credentials_rejected`. A judge outage is a typed judge failure (`judge_failed`, or `judge_unparseable` for a verdict that cannot be parsed) with `mask_sample`: the row leaves the score and is re-judged from the saved answer. It is never a zero for the model. A genuine wrong answer stays an unmasked zero. Never infer a cause from a substring of an error body.

**3. Declare the time contract and the resume identity.** In the config or README, name the per-task deadline: its config path, the function that enforces it, the clock it runs on and the outcome when it expires. If there is no episode deadline, say so and list the step and token bounds that apply; an adapter may use `agent_timeout_sec`, but there is no universal key. Keep the model-call, setup, sandbox, scheduler-segment and whole-chain limits distinct. Persist `recovery_outcome_policy` in the saved config: a model failure or an expired deadline is final, and recovery of infrastructure or judge faults never retries it. Classify the settings a resume compares. Data, repeats, model, prompts, sampling and grading are material. Connection grace, ports and placement are operational where the implementation declares them so. A new operational setting with a default must keep the fingerprint of runs started before it existed, or those runs stop resuming. Set `mlflow_tracking_required` before launch. Required delivery needs the current export receipt `rollouts_mlflow_export_done` and the terminal status of its run; run-and-babysit-eval reports it as `gate_task.mlflow_delivery`, apart from `gate_task.evaluation_health`. Set it to `false` only for a documented choice; it removes no health requirement. Prove resume compatibility before any GPU job: `gym eval submit --resolve-only` does not compare an old run's identity.

**4. Show it with three fixtures.** Run `gym eval submit --config onboarding.yaml --resolve-only`, then `--dry-run`, and keep the resolved config and rendered scripts. Run a smoke of three distinct tasks (`run: {limit: 3, num_repeats: 1}`, `gym eval submit --config onboarding.yaml --json`) and read its gate with `gym eval resubmit "$GYM_JOB_ID" "$BENCHMARK" --recover --inspect --score-agent "$AGENT" --score-key "$SCORE_KEY"`; require complete reconciled rows, grading known for every trial, health `PASS` and `gate_task.status` `SUCCESS`; a `NOOP` is not acceptance. Then submit one fixture per row and inspect each. Gym has no fault-injection switch, so the fixture causes the fault. A fixture does not have to pass health: one evaluation fault in three trials is above the 2% limit. Assert the typed verdict and the accounting, then show a healthy completion separately. Never edit saved evidence; run `gym eval health-check` on a copy.

| Fixture | What it proves |
| --- | --- |
| Induced infrastructure fault (a known sandbox or provider failure) | The typed event, the attempt, the mask, the retained response, the bounded recovery or typed stop, and the cause and count in health. |
| Judge fault (a known outage or a malformed verdict) | A typed judge failure and the retained policy answer; a re-judge uses that answer without sampling the policy again. |
| Cut and resume (one saved row and unfinished work when a short segment ends) | Settled rows byte for byte, the original identities and budgets, the operational drift the resume names, and the typed refusal of material drift. Never resubmit a live chain, and never use `allow_unsafe_resume` to pass. |

**Tests required before you call a benchmark conformant.** (1) A scored rollout carries the grading record of the path you chose, bound to its response and reward, and a changed grader or response invalidates it. (2) The benchmark's typical failure is typed and routed correctly, and a genuine wrong answer stays an unmasked zero. (3) A cut and resume keeps the settled rows, accepts the declared operational drift including a setting that was absent and now has a default, and rejects material drift. Run them with `gym env test --resources-server <server>`, or the agent's own suite for agent-side grading. If the server keeps state per session, its tests also call `check_resources_session_contract` (`nemo_gym.server_runtime.testing.session_conformance`). Keep the test receipts next to the real smoke, fixture and resume evidence: fixture tests do not replace a real run.

### Step 8: Pre-commit and PR

```bash
pre-commit run --all-files
```

First run may fail as hooks auto-modify files (`verified: false` flag, README table). Stage changes and run again.

Set `verified: true` in YAML config after the exit checks pass. Link the run folder and its `quality_summary.json` in the PR description.

CI does not start on its own for unverified commits or external contributors; a maintainer must comment `/ok to test <full-sha>`. Use the `nemo-gym-pr-checks-and-labels` skill for CI routing and labels.

To avoid committing unrelated auto-fixes from other servers, scope pre-commit to your files:
```bash
pre-commit run --files resources_servers/my_benchmark/**/*
```
If hooks modify files in other directories, discard those changes:
```bash
git checkout -- resources_servers/other_server/
```


## Adding a harness instead

Use the [`add-harness`](../add-harness/SKILL.md) skill. It covers the session contract, pairing configuration, harness conformance, real pairing runs, and the harness exit checks. The health and evidence checks above apply to harness work too, on live rollouts: the conformance probes use scripted replies and no real sandbox. An Evaluation-level adapter fails the evidence check by design; `add-harness` says what to record instead.

## Constraints

- Use NeMo Gym's OpenAI client (`nemo_gym/client/openai_client.py`), not LiteLLM/Anthropic/other
- **Use aiohttp, not httpx, for async HTTP.** All async HTTP calls must go through `nemo_gym.client.http_client.request()` (aiohttp). httpx has O(n^2) connection pooling that hangs at high concurrency. When wrapping external libraries that use httpx internally, replace their HTTP transport with an aiohttp adapter — see `resources_servers/tavily_search/app.py` (`TavilySearchAIOHTTPClient`) for the pattern and `fern/versions/latest/pages/infrastructure/engineering-notes/aiohttp-vs-httpx.mdx` for the rationale.
- Pass configuration through Gym config (YAML), not environment variables
- Code must run on Linux
- `/run` endpoint must be async
- Errors from tool execution or bad model output must return error responses, not crash
- All commits require DCO sign-off (`-s`) and cryptographic signature (`-S`)

## Reference

For detailed code patterns, schemas, and examples: see [references/patterns.md](references/patterns.md).
