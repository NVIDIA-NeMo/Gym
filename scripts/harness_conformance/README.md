# Live harness conformance probes

Run scripted model replies and failures through real Gym harnesses, collect their
rollouts, and check the retained evidence. No model service or model credentials
are needed. These probes exercise local execution, not remote sandbox deployments.

## Run diagnostic probes

Run from a Gym checkout with Gym, the selected adapters' Python dependencies and
runtime prerequisites installed. `registry.HARNESSES` holds the runtime requirements;
`python -m scripts.run_harness_conformance --help` lists registered harnesses.

```bash
python -m scripts.run_harness_conformance \
    --harness opencode --output "$(mktemp -d)/probes"
```

Repeat `--harness` for a matrix; omit it to select every registered harness.
Always use a **new output directory**; existing directories are rejected.

- Simple Agent needs `sh` and uses a direct HTTP shell tool, without a sandbox.
- Other registered adapters require Linux. Their runtimes come from locked Python
  dependencies or session setup. Hermes must be importable in the same Python
  environment; sync `responses_api_agents/hermes_agent` before running it.
- Terminus requires root and `tmux` in a disposable Linux runner. Its runtime creates
  `/logs/agent`; each episode uses a separate tmux socket directory that is cleaned up.
  For example: `sudo .venv/bin/python -m scripts.run_harness_conformance
  --harness terminus_2_sandboxed --output /tmp/terminus-new-run`.
- NOOA's CI definition uses a disposable container. Mini-SWE, Terminus and NOOA
  borrow Resources-owned local workspaces; other sandboxed probes use agent-owned
  workspaces. Commands execute through Gym's local provider.

The adapter runs its real harness. The runner uses `single_agent_turn_legacy` for
adapters implementing session seed and close, otherwise `legacy_agent` forwards to
`/run`. A session failure never falls back to `/run`. `RolloutCollectionHelper`
writes the rollout; the fixture never supplies canonical steps, joins or tool
observations on the adapter's behalf.

## Scenarios

```bash
python -m scripts.run_harness_conformance --list-scenarios
python -m scripts.run_harness_conformance --harness codex \
    --scenario tool_failure --timeout 60 --output "$(mktemp -d)/probes"
```

| Scenario | Prescribed behavior |
| --- | --- |
| `tool_success` | Two successful shell calls, results carried into subsequent requests, then a final answer |
| `tool_failure` | First command exits 7; a second command succeeds |
| `usage_omitted` | Successful tool sequence with no provider usage fields; counts remain unknown |
| `retry_429` | Two 429 replies before the tool sequence; retain distinct attempts with identical request bodies |
| `retry_500` | One 500 reply before recovery; retain the error payload |
| `model_error` | Persistent 400 reported to the EnvironmentServer as a failed response with the original error |
| `judge_verifier` | A separate scripted judge grades the final answer; its call belongs to the verifier, not a policy step |
| `verifier_failure` | Completed tool sequence receives a known zero reward |
| `context_compaction` | Automatic compaction and resumed model interaction; reuse the summary and retain helpers without creating policy turns |

Chat Completions, Responses and Anthropic Messages are supported, including
streaming. Scenarios supply default replies and reported token counts. Compaction
has no default: declare its reply schedule or `unsupported_reason` in the harness
scenario settings. The Claude Code preset supplies a session name to suppress title generation.
Probes support advertised shell/finish tools, Mini-SWE's submission sentinel and
Terminus's terminal JSON protocol; unsupported protocols produce exercise failures.

The fixture witnesses shell commands printing unique markers and prescribed exit
codes. Every harness must preserve the original witnessed tool IDs and their
request/result joins; matching names and arguments alone cannot replace missing
IDs. Retries must retain every attempt. For `model_error`, the EnvironmentServer
owns verification and masking policy, so the scenario requires error evidence but
prescribes neither a grade nor verification.

### Scenario gating and check tiers

All selected scenarios run, except those explicitly declared unsupported. The
`--gate-tier` option selects **scenario priorities** that affect the exit status:
P0 by default, or P0 and P1 with `--gate-tier P1`. Currently `context_compaction` is
the only P1 scenario. Selecting it alone requires that option:

```bash
python -m scripts.run_harness_conformance --harness opencode \
    --scenario context_compaction --gate-tier P1 --output "$(mktemp -d)/probes"
```

Checks have their own tiers. The runner currently evaluates a scenario's P0 checks
and, for a P1 scenario, its P1 checks when computing its gating verdict. Thus
`--gate-tier P1` does **not** enforce every P1 check on P0 scenarios: the P1 judge
ownership check is reported, but does not change the P0 `judge_verifier` verdict.
The published table evaluates all applicable check tiers independently of this gate.
TE labels group evidence requirements; they do not determine check priority.

A scenario setting `unsupported_reason` skips execution and records a failed
`scenario.supported` check at that scenario's priority. Simple Agent and Mini-SWE
currently use this for compaction. It is shown as **not run: declared unsupported**,
not as a measured pass or an execution error.

## Results

Start with `conformance_report.md` for the check-by-scenario matrices and
`conformance_summary.json` for structured results and the runner's gating verdict.
The published table counts each scenario once and requires all its applicable
checks to pass. It also counts failures at the first failing tier, P0 before P1
before P2. Its per-harness breakdown looks like:

```text
tool_failure: ✓ (P0:✓)
context_compaction: ✗ (P0:✓, P1:✗)
```

A dash means no applicable measurements. The displayed result can differ from the
runner's exit status because table results include checks outside the selected gate.
Registration does not imply conformance.

Artifacts under the output directory:

- `suite.json`: selected harnesses/scenarios and source hashes.
- `<harness>/<scenario>/requests.jsonl`, `launch.json`, `episode.log`: input,
  resolved configuration and process diagnostics.
- `<harness>/<scenario>/runtime.json`: agent interface, runtime pins or source
  identity, and adapter source hash.
- `<harness>/<scenario>/rollouts.jsonl`, `rollouts_failures.jsonl`, `capture/`:
  collected rollouts, failure records and model-call payloads. Terminal model-error
  evidence can be in `rollouts_failures.jsonl` rather than `rollouts.jsonl`.
- `<harness>/<scenario>/witness.json`: independently observed attempts, tool
  execution markers, model-visible results and verifier outcomes.
- `<harness>/<scenario>/scenario_result.json`, `evidence/`, `health/`: execution
  status, individual checks, artifact reports and rollout-health reports.

Only available artifacts are written. Declared unsupported scenarios have a
`scenario_result.json` without episode outputs. An interrupted run has no completed
summary; a scenario subset is marked `full_suite: false`. Timeouts clean up the
episode's subprocess tree, including new process groups.

Exit codes: **0** when the selected scenario gates pass; **1** when they fail;
**2** for execution/checker errors in gated scenarios or invalid input. Failures
outside the selected scenario gate remain visible in the results.

## Pull-request feedback

[The CI workflow](../../.github/workflows/harness-conformance.yml) runs on approved
PR mirrors, merge-queue revisions and main pushes. Scheduled runs every four hours
and manual runs select all harnesses. Other runs use [ci.select_harnesses](ci.py):
registered adapter changes select that harness; shared probe dependencies and
unknown paths select all; unrelated changes select none.

PR mirrors compare the full branch with its base, main pushes with the previous
commit, and merge-queue groups with their base SHA. An unavailable diff selects all.
Each job installs locked dependencies and the pinned runtime, runs fixture/checker/
adapter tests, then runs the full scenario suite. Setup/test failures and gated
execution errors fail the job; measured conformance failures remain valid results.
Download `harness-conformance-<harness>-<run>-<attempt>` for setup/test logs and `probes/`.

The reporter posts **Harness conformance (non-blocking)**, an optional check separate
from `Nemo_CICD_Test`. It passes when all selected harnesses pass the P0 scenario gate,
or none were selected. Failures and unavailable measurements are reported with
scenario counts, the first failure, rerun commands and artifact links. Counts and
the first failure cover all scenarios, including ungated P1 ones, so a passing
harness can still list a failed scenario. **Publish check** also fails if a probe
job failed or was cancelled. Only the reporting job
has `checks: write`; probes use read-only permissions.

On partial retries, evidence is matched to each job's actual execution. Successful
jobs not rerun retain their artifacts; a rerun without an upload stays unavailable
and cannot reuse its older evidence.

## Published documentation

Scheduled/manual runs render a `harness-conformance-docs-<run>-<attempt>` artifact.
When such a main run completes, [the docs publisher](../../.github/workflows/publish-fern-docs.yml)
runs [publish.py](publish.py) to replace the generated **Results** block in the
[Harness Conformance page](../../fern/versions/latest/pages/observability/harness-conformance.mdx#results).
A red P0 result is publishable; a cancelled run or missing rendered artifact stops
that update. Ordinary docs pushes, tags and manual publications use the newest
eligible run, falling back to an unavailable note and warning if measurements
cannot be fetched.

Insert a CI-rendered block locally with Python and authenticated `gh`:

```bash
python -m scripts.harness_conformance.publish \
    --run-id <scheduled-or-manual-main-run-id> --strict
```

Omit `--run-id` to select the latest eligible run. With `--run-id`, add
`--run-attempt` to select an exact attempt. To preview fresh local tests and probes instead:

```bash
python -m scripts.harness_conformance.table --harness opencode \
    --commit <full-source-sha> --output "$(mktemp -d)/evidence"
```

Install the selected harness's dependencies first. The checkout must match the
commit throughout generation. The command retains logs and `generation_report.json`
in the output directory and replaces the page block only after validation succeeds.
The docs publisher supplies current measurements at build time; local page changes
are previews.

## Register a harness

Add a `HarnessDefinition` to [registry.HARNESSES](registry.py). The key must match
the adapter directory without `_agent`; configuration is read from
`responses_api_agents/<key>_agent/configs/<key>_agent.yaml` and CI tests that adapter.

- `agent` names the class; Pydantic metadata supplies its config type.
- `base_settings` declares harness/probe and task overrides, session/legacy
  settings, sandbox requirements and imports. Bind timeouts to the probe timeout.
- `scenario_settings` declares overrides by scenario name, including reply
  schedules or `unsupported_reason`. OmegaConf merges shared defaults, base settings,
  then the selected scenario's overrides.
- Runtime fields describe built-in/Python runtimes, pinned npm installs,
  preparation hooks and provenance. CI always syncs the locked adapter project first.
- `ci_container`, `ci_packages` and `requires_root` supply CI execution requirements.

Use `probes.Probe` when its protocols cover the adapter. Its recognised shell-tool
names are in `probes.SHELL_TOOLS`; add a name when the existing argument handling
fits, otherwise extend the protocol or select another probe through settings. Keep
expectations and witnesses out of task input and retain the normal collection path.
No runner, installer or workflow harness-name tuple is needed.

Run the shared regression checks, then the selected adapter's tests:

```bash
python -m pytest -o addopts= --import-mode=importlib -q \
    scripts/harness_conformance/tests tests/unit_tests/tooling/test_check_suite_artifacts.py \
    tests/unit_tests/rollouts/harness_capabilities
python -m pytest -o addopts= --import-mode=importlib -q responses_api_agents/<key>_agent/tests
```
