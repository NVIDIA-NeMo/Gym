---
name: add-harness
description: >
  Guide for adding or updating an agent harness in NeMo-Gym: a CLI or SDK agent runtime
  (for example Hermes, Pi, Codex, OpenCode, or an agent framework) wrapped as an Agent Server
  adapter that runs through Gym's Environment Server and agent sessions. Use when the user asks
  to add, integrate, onboard, wrap, upgrade, or migrate a harness or agent runtime, pair a harness
  with a benchmark, or move an existing agent off the legacy /run path. Covers scoping, the session
  contract, pairing configuration, harness conformance, real pairing runs, and the exit checks a
  harness must pass before its PR merges. Triggered by: "add harness", "new agent harness",
  "integrate <CLI> agent", "onboard harness", "upgrade harness version", "migrate agent to sessions".
---

# Add or Update a Harness in NeMo-Gym

The Gym page [Add or Update a Harness](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx) is the contract and command reference. When it and this skill disagree, the page wins. This skill adds the order of work, the decisions to make before writing code, and the checks a harness must pass before it merges (Step 7). Read the linked sections; do not work from this summary alone.

Words used below follow the page's [Terminology](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#terminology): the *harness* is the agent runtime, the *adapter* is the Agent Server that wraps it. A *pairing* is one harness run on one benchmark: the adapter is the agent, the benchmark's Resources Server prepares and verifies the tasks, and a `single_agent_turn` Environment Server connects the two for each episode.

## Step 0: Scope before code

1. **Check what already exists.** Look for the harness in [Built-in agents](../../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx#built-in-agents), [Sandboxed harnesses](../../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx#sandboxed-harnesses), and under `responses_api_agents/`, including generic bridges such as `harbor_agent` that can already launch some harnesses on the legacy `/run` path. If an adapter exists, the work is an update or a migration, not a new adapter. Read its README and tests first.
2. **Pick the path.** Use [Choose a Path](../../../fern/versions/latest/pages/agent-server/index.mdx#choose-a-path). This skill covers an adapter that runs one agent turn per episode through `single_agent_turn`. An agent that runs its own multi-turn episode, simulated users, or several verify calls needs its own Environment Server instead.
3. **Pick the execution model** and read [Select an execution mode](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#select-an-execution-mode).
   - *Sandboxed harness*: the harness runs inside the benchmark's task sandbox, supervised by `SandboxSession`. Models: `pi_agent` and `codex_agent` (CLIs), `hermes_agent` (a Python harness started through a shim).
   - *In-process harness*: an SDK or framework that runs inside the Agent Server with its own tools. Model: `simple_agent` for the seed hook and the session check at the top of `responses()`. Its close returns no bundle when model-call capture is off, so take the close fallback (a bundle with an `observation_capture_failed` gap) from `hermes_agent`. Do not copy `langchain_deepagents_agent` until [#5003](https://github.com/NVIDIA-NeMo/Gym/pull/5003) lands: it inherits the hooks, but its `responses()` does not check the session yet.
   - Host execution is optional. If the adapter offers it, keep it working and test it separately.
4. **Pick the capability level.** Use [Choose the capabilities you need](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#choose-the-capabilities-you-need): Evaluation, Observability, Environment tools and task sandboxes, Training. The level decides which later checks apply. Write it down; it goes in the adapter README.
5. **Name the first pairing.** Choose a benchmark whose Resources Server supports the session contract, and read its README (for example `benchmarks/terminal_bench_2_1/README.md`) for its run config and the harnesses it lists. Check that the harness's real behavior fits the tasks: a harness that only edits files cannot solve tasks that need commands run. If the benchmark's Resources Server config declares `allowed_agents`, a new harness is not in it yet; a server that declares none accepts any harness. During development, pass `--allow-unsupported-pairing` to the command that starts the servers (Step 5). Add the agent to a declared `allowed_agents` in the same PR only after Step 5 passes, and ask the benchmark's owner to review that line. Do not edit the benchmark's data, prompts, or verifier to fit the harness.
6. **Ask the user for what you cannot invent:** the pinned harness version and its runtime requirements, a model endpoint and its credentials (kept in `env.yaml`, never in committed config), a sandbox provider for sandboxed runs, and approval for downloads such as task images or benchmark data.

## Step 1: Build the adapter

Follow these sections of the page, in order:

- [Code layout](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#code-layout) and [Adapter configuration](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#adapter-configuration)
- [Agent Server and Environment Server](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#agent-server-and-environment-server): the seed, activation, and close contract
- [Agent Server and task sandbox](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#agent-server-and-task-sandbox) and [Tools](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#tools)
- [Wire an adapter to `SandboxSession`](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#wire-an-adapter-to-sandboxsession) and [Installation contract](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#installation-contract), for a sandboxed harness
- [Preserve inputs, outputs, and errors](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#preserve-inputs-outputs-and-errors)
- [Make retries and close safe](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#make-retries-and-close-safe)

Before moving on, confirm each of these in the code:

- [ ] Every model call, including summarization, compaction, and subagent calls, goes through Gym's rollout-prefixed Model Server URL.
- [ ] A seed the adapter cannot serve (a sandbox it cannot use, a required tool grant it never consumes) raises `AgentSeedRejectedError`; optional grants it does not use are skipped.
- [ ] If the adapter supports MCP grants, it wires the episode URL and headers into the harness and enforces required-server readiness before inference. Follow [MCP grant integration](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#mcp-grant-integration), including runtimes that silently skip failed plugins. MCP access and sandbox ownership are independent decisions.
- [ ] Close returns an `AgentObservationBundle`, with gaps for evidence the harness cannot report, even when activation failed.
- [ ] Agent sessions run with `num_workers: 1`.
- [ ] A sandbox failure never falls back to running on the host.
- [ ] Controls the harness does not support are rejected or documented, not silently dropped.
- [ ] The harness's own files (configuration, history, caches) stay out of the task repository. Where the harness has no setting to move them, the README says what it writes under the task's home.
- [ ] A sandbox runner that needs only the standard library is uploaded as `RUNNER_FILE` from `nemo_gym.agent_utils.sandbox_session` and launched by a `SandboxCommand` with `stdlib_runner=True`, as `codex_agent` does. It then runs under a neutral process name with no `.py` in its command line, so a `pkill python3` or `pkill -f '\.py'` in the task cannot stop it.
- [ ] Only an error that ends the harness's run fails the response, by the harness's own rule at the pinned version. See [Convert outputs without silent loss](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#convert-outputs-without-silent-loss).
- [ ] Usage is reported only when every model call has a usage record, with call references collected from every record the harness writes. Otherwise usage stays unknown, with a gap.

## Step 2: Unit tests

Cover the behaviors in [Runtime and compatibility](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#runtime-and-compatibility) that apply to your execution model. At minimum:

- Each unsupported seed returns a final 422. Use `nemo_gym.testing.agent_session_conformance.check_agent_seed_rejected`, as `responses_api_agents/hermes_agent/tests/test_app.py` does.
- For MCP support, cover authenticated discovery and tool calls, required-server failure before inference, optional-server failure, name collisions, and cleanup. If enforcement depends on a plugin or hook, test a run where it does not load. Test each supported sandbox ownership path; a sandbox-only smoke does not validate MCP grants.
- An activation runs the harness once. An activation for an unknown or closed session returns 409; a request with no session cookie takes the host path, if the adapter has one.
- Close returns observations or gaps, and returns `resources_cookies` only when the session holds a Resources cookie jar.
- Retries, cancellation, and termination, with every wait bounded. For a sandboxed harness, include a tool command that leaves a background process holding stdout open: the activation must still finish.
- Output conversion: each error kind the harness tolerates leaves the response completed, and usage stays unknown when one model call has no usage record.

Check that each new test fails when you revert the behavior it covers. Run the adapter's tests in its own environment:

```bash
gym env test +entrypoint=responses_api_agents/<adapter_name>
```

## Step 3: Pair the harness with a benchmark

Wire the adapter to the benchmark you named in Step 0 so that the benchmark's tasks run through the harness. Follow [Configure an experiment](../../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx#configure-an-experiment) (separate model, benchmark, agent, and run configs) and the paired benchmark's README.

- New pairings use typed tasksets: the benchmark's dataset declares `taskset`, `environment_server_routes` maps it to a `single_agent_turn` instance, and that instance's `agent_server` selects the harness.
- Existing flat-row configs migrate to `single_agent_turn_legacy`. Name both `resources_server` and `agent_server` in its block.
- Set `resources_tool_transports` to the grants the adapter consumes, or `[]` when it uses only its own tools.
- Swapping in another compatible harness must change only agent configuration and wiring. A pairing that needs pair-specific adapter code is a design problem; stop and report it.

Check the config offline, then start it:

```bash
gym env resolve --config <run.yaml>
gym env validate --config <run.yaml>
gym env start --config <run.yaml>
```

A config that resolves and validates can still fail when its servers start. Read the server logs before claiming the pairing works.

## Step 4: Harness conformance

Read [Harness Conformance](../../../fern/versions/latest/pages/observability/harness-conformance.mdx) and the [Observability](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#observability) section.

Conformance is a merge check from the Observability level up (Step 7). At the Evaluation level the probes are optional: an adapter without invocation and turn records fails their evidence checks by design.

- **The probes apply** to a harness that offers the model a shell tool through function calling: the scripted replies call the shell tool listed in the request's `tools`. Register the harness as the [runner guide](../../../scripts/harness_conformance/README.md) describes: an entry in `registry.HARNESSES`, its runtime install in `ci.install_runtime`, and its episode configuration. The registry key is the adapter's directory name without `_agent`. The probes give the adapter its own local sandbox rather than a borrowed one. Run on Linux with a new output directory:

  ```bash
  python scripts/run_harness_conformance.py --harness <registry-key> --output <new-dir>
  ```

  Sort each failure by where it comes from, and change only that side:

  - **The harness or adapter failed** (exit code 1: a scenario or evidence check is `not_fulfilled`, and the saved output really breaks the scenario's prescribed behavior or the [Rollout Evidence](../../../fern/versions/latest/pages/observability/rollout-evidence.mdx) contract). This is the usual case. Fix the adapter or the harness settings. Never edit a check, a scenario's expectations, or the scripted replies to make a harness pass. If no adapter change can make the harness meet a check (for example, its runtime never retries a 429), leave the failure visible, report it in the PR as a known limit, and do not claim a capability level that needs it.
  - **The probes cannot drive the harness** (a `probe_protocol_error` under `violations` in the scenario's `witness.json`, such as "harness did not advertise a supported shell tool"). The scripted provider accepts only some shell-tool names and argument shapes. In the same PR, extend `provider.Probe._tool`, and `_finish` for a finish tool, to recognize the harness's tool. Change only how the tool is recognized, not what the scenarios expect.
  - **The run did not complete** (exit code 2: execution, dependency, input, or checker errors). Fix the setup: the runtime install, the episode configuration, or the Linux host. If the runner or checker crashes on valid output, treat it as the next case.
  - **The conformance code is wrong**: a check fails output that the scenario and the Rollout Evidence contract allow, or the runner or checker crashes on it. Show the exact check, the saved output, and the contract text that allows it. Fix the conformance code in a separate PR with a regression test (`python -m pytest scripts/harness_conformance/tests tests/unit_tests/harness_capabilities -q`), and never change it to suit one harness. Until it merges, the harness PR reports the failure and links that PR.

  CI reruns the probes for every registered harness when anything under `scripts/harness_conformance/` changes, so check that the other harnesses still pass after a provider change. Rerun the failed scenarios, then the full suite. The runner guide's [Results](../../../scripts/harness_conformance/README.md#results) section explains each output file, and [Read individual results](../../../fern/versions/latest/pages/observability/harness-conformance.mdx#read-individual-results) explains the check kinds.
- **The probes do not apply** to a harness without a function-calling shell tool, including one that runs commands it reads from the model's text. Say so in the PR; Steps 2 and 5 carry the evidence.

## Step 5: Real pairing runs

Run a real model on real tasks for every pairing you claim. Enable [model-call capture](../../../fern/versions/latest/pages/model-server/model-call-capture.mdx): set `observability_enabled: true` and an absolute `model_call_capture_dir` in `run.yaml`.

```bash
gym env start --config <run.yaml>
gym eval run --no-serve --config <run.yaml> \
  --input <tasks.jsonl> --output <run_dir>/rollouts.jsonl \
  --limit 5 --concurrency 1
gym eval health-check <run_dir>
python scripts/inspect_harness_conformance.py \
  --bundle <run_dir>/rollouts.jsonl --output <run_dir>/evidence --require-sandbox
```

- `--input` needs `--no-serve`. With a taskset config you can instead let `gym eval run` serve and pass `--split`, as [Configure an experiment](../../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx#configure-an-experiment) shows; the benchmark README shows how its rows are prepared.
- While the harness is not yet in a declared `allowed_agents`, start the servers with `gym env start --config <run.yaml> --allow-unsupported-pairing`. With `--no-serve`, `gym eval run` checks the pairing against the running servers' configuration, so the flag has no effect there; it works on `gym eval run` only when that command serves. Exporting `NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING=1` in the shell that runs `gym eval run --no-serve` also works, because the collecting process reads it.
- Inspector flags: drop `--require-sandbox` for an in-process harness, and pass `--no-tools` when the pair dispatches no tools.

Read the rollouts, not the exit code:

- Reward, response, and failures sidecar for each task.
- The agent closed before verification, and the sandbox owner cleaned up.
- [Rollout health](../../../fern/versions/latest/pages/evaluation/rollout-health.mdx) counts: `unhealthy` is a finding; `unobserved` means a check lacked the evidence it reads. At the Evaluation level the turn checks are unobserved by design. At any level, name what each unobserved check was missing, for example a response without `usage` leaves the token-count check unobserved.
- A 422 from the Model Server on a later model call usually means the harness sends back a field from an earlier response, which Gym's strict request validation rejects. Normalize that field in the request model in `nemo_gym/openai_utils.py`, with a unit test. Drop only values that carry nothing, and cover each empty form the harness can send back: its SDK can write `annotations: null` where the provider returned `[]`. Keep other values strict.

For a new harness or an upgrade that claims Terminal-Bench 2.1 support, complete the [TB 2.1 full evaluation](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#terminal-bench-21-full-evaluation) before merge. Run every task with 8 repeats at a pinned benchmark revision with a real model, and link the full results in the PR. This applies at every capability level; the small smoke run above and scripted probes do not satisfy it.

Then show a second compatible pairing through configuration changes alone. For an upgrade, compare before and after with the same model, tasks, prompts, sampling, limits, image, verifier, and concurrency.

## Step 6: Training (only if the capability level includes it)

Follow [Training on External Agent Harnesses](../../../fern/versions/latest/pages/training-tutorials/external-agent-harnesses.mdx). Validate token capture separately from inference. Training frameworks send sampling controls such as `temperature`, `top_p`, and `max_output_tokens` on every request; decide how the adapter maps or rejects each one, and state whether a token limit is per model call or per episode. Model output that a later call reuses as input, such as a summary, must also go through the capture route.

## Step 7: Exit checks

A harness merges only when it passes these checks. Put the evidence in the PR.

| Check | What it asks | How to check |
|---|---|---|
| Unit tests | Session contract, refusals, close, retries, termination. | `gym env test +entrypoint=responses_api_agents/<adapter_name>` |
| Conformance | From the Observability level up, every applicable P0 check passes, or the PR explains why the probes do not apply. | `scripts/run_harness_conformance.py` report |
| Real pairing | Each claimed pairing produces rewards on a real model and tasks; the agent closes before verification; cleanup runs. | `rollouts.jsonl`, failures sidecar, server logs |
| TB 2.1 full evaluation | A new harness or upgrade claiming TB 2.1 support completes the [full evaluation](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#terminal-bench-21-full-evaluation), at every capability level. | Full run artifacts, per-task results, aggregate score, coverage and failure counts; before/after results for upgrades. |
| Health | Every `unhealthy` rollout has a stated cause in the PR: a harness limit, an infrastructure failure, or a bug that was fixed and rerun. Every `unobserved` count is explained by the capability level or a named missing input. | `quality_summary.json`, `gym eval health-check` |
| Evidence | From the Observability level up, the saved rollouts pass the evidence check. At the Evaluation level they fail it by design: record the result and the gaps the close bundle declares. | `scripts/inspect_harness_conformance.py` |
| Composition | A second pairing works through configuration alone. | Its run folder |
| Pairing declared | Where a paired benchmark's Resources Server declares `allowed_agents`, the harness is in it, reviewed by the benchmark's owner. | Resources Server config diff |
| Catalog | The adapter README states the harness version, capability level, supported pairings, and limits. The [Tools](../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#tools) table lists the harness; so do the [Sandboxed harnesses](../../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx#sandboxed-harnesses) table for a sandboxed harness and each paired benchmark's README harness table, where one exists. Each entry matches the adapter: its limits, the request fields it rejects, and whether it captures token IDs. | Docs diff |

The [capability matrix](../../../fern/versions/latest/pages/reference/trajectory-capabilities.mdx) is published by CI from registered harnesses. A local `python -m scripts.harness_conformance.table` run is a preview only; do not commit its output.

## Step 8: Pre-commit and PR

```bash
pre-commit run --all-files
pytest tests/unit_tests/test_session_contract_repo.py
```

- `test_session_contract_repo.py` checks every adapter's source for the session hooks; a search for your adapter's name does not find it.
- `tests/unit_tests/test_add_legacy_agent_environment_servers.py` checks every repository config for a missing Environment Server. In `scripts/add_legacy_agent_environment_servers.py`, add the adapter's type to the unbound session templates in `needs_environment_server` when its default config sets `resources_server: null` for composition, or to the `stirrup_agent` check when the adapter has no `/run`. Otherwise the script adds a legacy relay beside the session Environment Server. Add a test like `test_default_pi_composes_with_exactly_one_session_environment`.
- Merge current `main` before the final checks and rerun the adapter's tests and conformance. The sandbox helpers an adapter calls change often, and a long-lived branch can pass on its old base and fail after the merge.
- Search `tests/` for every config file you changed (`grep -rn <config or agent name> tests/`) and run those tests too. Core tests can assert a config's shape, and they are not part of the adapter's own suite.
- Use a Conventional Commit title such as `feat(<harness>): add the <harness> harness adapter`, and DCO sign-off (`git commit -s`).
- Use the `nemo-gym-pr-checks-and-labels` skill for CI routing and labels.
- In the PR body, list the harness version, commands, run folders, health counts, conformance result or why it does not apply, and known limits.

## Constraints

- One adapter per harness. Never add a pair-specific adapter or launcher.
- Never change a benchmark's data, prompts, or verifier to fit a harness.
- Keep credentials out of committed files; reference them from `env.yaml`.
- Use aiohttp through `nemo_gym.server_utils.request()`, not httpx, for async HTTP.
- `/run` and activation code must be async; code must run on Linux.
