# Live harness conformance probes

## Pull-request feedback

Gym CI runs a harness probe matrix on approved PR mirrors and merge-queue
revisions. Changes to a registered adapter select that harness; shared code,
dependency files, CI configuration, and other server code conservatively select
all registered harnesses. Changes limited to `fern/`, `docs/`, or root Markdown
files are skipped. Scheduled and manual CI runs exercise all harnesses.

Each job installs the adapter's pinned runtime, validates the runner, checker,
and adapter tests, then runs every scenario. Execution errors fail the probe job;
completed measurements can succeed as jobs even when they find conformance failures.
Warnings and a detailed table remain available in the Actions summary.

After the probes finish, a separate reporting job publishes **Harness conformance P0**
in the PR checks list. It succeeds only if every selected harness passes the current
P0 suite, or no harnesses are affected. Conformance failures and unavailable
measurements both produce a failed check, with the first failure and number of
failed scenario checks per harness, local rerun commands, and direct artifact links.
Counts include execution errors and count each scenario once, rather than counting
cascading evidence failures separately. Only the
reporting job receives `checks: write`; probe jobs keep read-only permissions.
Download the `harness-conformance-*` artifacts for test/setup logs and probe evidence.
The reporter only uses artifacts from the current run attempt; rerun all jobs to
refresh results for all selected harnesses.

After publishing the result, **Publish P0 check** also fails if P0 fails or any
probe job fails or is cancelled, making the result visible in the workflow job list.

To enforce P0 before merging, add the fixed **Harness conformance P0** check name to
the repository's required checks. Until configured, it remains optional. It is
independent of the existing `Nemo_CICD_Test` gate. This workflow does not regenerate
documentation.

Use the diagnostic command below to reproduce a warning at the reported commit.
When registering another harness in `registry.HARNESSES`, provide its pinned runtime
installation in `ci.install_runtime` as well as its episode configuration.

## Regenerate the documentation table

From a checkout matching a full Gym commit SHA, rebuild the TE table in
[`reference/trajectory-capabilities`](../../fern/versions/latest/pages/reference/trajectory-capabilities.mdx):

```bash
python -m scripts.harness_conformance.table \
    --commit <40-character-gym-commit-sha> \
    --output /absolute/path/to/new-conformance-results
```

The command runs runner unit tests, each selected harness's adapter unit tests,
checker unit tests, and then the full live scenario suite. The checkout must match
the source commit before and after validation (Git and jj are supported). There is
no test-skip switch. Failed, empty, or entirely skipped test suites, dependency or
execution errors, and checker errors leave the previous table unchanged. Evidence
FAILs are valid results and appear in the generated table.

All four harnesses are selected by default. Repeat `--harness NAME` to select a
subset; omitted harnesses show **Not run**, and no previous row is reused. Runtime
versions/source hashes, test results, scenario results, and artifact hashes are
saved in a content-addressed JSON report under `fern/assets/trajectory-capabilities/`.
Test logs and full probe artifacts are retained in the new `--output` directory.
Commit the generated page and report after the source commit to keep its pin stable.

Checker and runner unit tests use the synthetic contracts in
`tests/unit_tests/harness_capabilities/synthetic.py`. Actual harness capabilities
are measured by probes; expected harness verdicts do not belong in library tests.

## Run diagnostic probes

Run predetermined model replies and failures through real Gym harnesses, collect
their rollouts, and apply the existing P0 evidence checks:

```bash
python scripts/run_harness_conformance.py \
    --harness codex --harness pi --harness opencode --harness hermes \
    --output results/conformance
```

Run from a Gym checkout with Gym and the selected agents' Python requirements
installed. Put the selected `codex`, `pi`, and `opencode` executables on `PATH`;
Hermes must be importable in the same Python environment. The versions pinned in
`responses_api_agents/*/configs/` are the recommended starting point. The runner
requires installed runtimes; missing dependencies are execution failures.
Omitting `--harness` selects all four.

No model service or model credentials are needed. Each scenario starts fresh
local model, resources, agent, and compatibility environment servers on dynamically assigned loopback ports.
The existing Gym adapter runs its real harness and normal seed/verify lifecycle;
`RolloutCollectionHelper` produces the rollout and capture payloads. The runner
never supplies canonical steps, joins, or tool observations on the harness's
behalf. All implementation remains under `scripts/` for eventual CI integration.

The OpenCode preset disables its auxiliary title and summary agents through
[agent configuration](https://opencode.ai/docs/agents/#disable), so scripted
failures reach policy requests. This configuration is saved with the run.

The suite uses each harness's local shell tool to print a unique marker, write it
inside the output directory, and return a prescribed exit code. Local execution
must be allowed (the Codex preset uses `danger-full-access` in a fresh working
directory). These probes qualify the local adapter path; they do not exercise a
remote sandbox deployment.

## Scenarios

```bash
python scripts/run_harness_conformance.py --list-scenarios
python scripts/run_harness_conformance.py --harness codex \
    --scenario tool_failure --timeout 60 --output results/codex-tool-failure
```

| Scenario | Prescribed behavior | Evidence exercised |
| --- | --- | --- |
| `tool_success` | Two shell calls, each result carried into the next model request, then a final answer | TE-1–TE-9 |
| `tool_failure` | First command exits 7; a second command succeeds | TE-1–TE-9, including failed tool evidence |
| `usage_omitted` | Successful tool sequence with no provider usage fields | TE-1–TE-9, including unknown usage |
| `retry_429` | Two 429 replies before the tool sequence | TE-1–TE-9, including distinct attempts with identical request bodies |
| `retry_500` | One 500 reply before the tool sequence | TE-1–TE-9, including retained error payloads |
| `model_error` | Persistent 400 with no model response ID | TE-1, TE-2, TE-4, TE-7, and TE-8/TE-9 |
| `verifier_failure` | Completed tool sequence graded zero | TE-1–TE-9, including a known verifier failure |

Both Chat Completions and Responses are supported, including streaming. Supplied
usage includes prompt, completion, reasoning, total, and cached counts. Chat
replies also include reasoning text. Requests adapt only the shell-tool name and
argument schema advertised by the harness. Unsupported tool protocols are
reported as exercise failures.

The runner observes actual retry behavior. A runtime that does not retry the
injected error leaves that scenario incomplete. A terminal error must still
produce a collected rollout to satisfy its evidence checks; a failure sidecar
alone cannot qualify it. Every applicable individual P0 check must pass;
TE labels group results and do not determine priority or grant exemptions.

## Results

Use a **new output directory** for each invocation. Existing directories are
rejected, so previous rollouts cannot accidentally qualify a new run.

- `suite.json`: selected scenarios and runner source hashes.
- `<harness>/<scenario>/requests.jsonl`, `launch.json`, and `episode.log`: input,
  resolved launch configuration, and process diagnostics.
- `<harness>/<scenario>/runtime.json`: executable version or Hermes source hash,
  plus the Gym adapter source hash.
- `<harness>/<scenario>/rollouts.jsonl` and `capture/`: ordinary Gym outputs,
  directly consumable by `scripts/inspect_harness_conformance.py`.
- `<harness>/<scenario>/witness.json`: independent endpoint attempts, actual tool
  execution markers, returned tool results, and verifier outcomes.
- `<harness>/<scenario>/scenario_result.json` and `evidence/`: execution gaps,
  input hashes, and the existing artifact checker reports.
- `conformance_summary.json` and `conformance_report.md`: individual results and
  one table per harness, with check rows and scenario columns. Artifact and
  behavioral results stay independent; execution status is recorded separately.

A scenario counts as passing only when its independent observations agree with
the retained rollout and the relevant evidence checks pass. Attempt comparison
preserves multiplicity, so dropping one of two identical retries fails. Tool
comparison joins each witnessed call ID to exactly one execution and its
invocation's request/result, checking the name, arguments, prescribed exit
status, and complete model-visible output. Partial logging loss or altered
tool evidence fails even when the retained artifacts are internally consistent. The
artifact reports keep their original retained-artifact scope; the runner's
separate report adds the scenario coverage.

The summary is published only after every selected episode has been processed.
An interrupted run has no completed summary. A scenario subset is explicitly
marked `full_suite: false`. Timeouts stop the episode and clean up its subprocess
tree, including harnesses that start new process groups.

Exit codes: **0** when all selected gates pass, **1** for evidence or scenario
failures, and **2** for execution, dependency, input, or checker errors. Existing
harness evidence gaps are expected to remain visible as failures. This suite
covers P0 checks, including ownership and call-to-step checks. Multimodal content,
compaction, parallelism and transport disconnects remain outside its qualification scope.

Run regression checks with:

```bash
python -m pytest scripts/harness_conformance/tests tests/unit_tests/harness_capabilities -q
```

To add another local harness, add its adapter class to `registry.HARNESSES`, supply
its launch settings in `episode._config`, and make its advertised shell-tool
schema consumable by `provider.Probe._tool`. Keep expectations and witnesses out
of the task input and preserve the normal Gym collection path.
