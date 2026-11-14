# Visual benchmark catalog

These entries share dataset preparation and a stateless verifier. They can be
used independently of OpenCode:

- [VStar](vstar/README.md)
- [HRBench8K](hrbench8k/README.md)
- [VTCBench](vtcbench/README.md)
- [ZeroBench](zerobench/README.md)

Each entry documents its dataset and measurement contract. The catalog defaults
to `opencode_agent`. Configure the model and sandbox provider separately. Each
catalog includes [`visual_eval_defaults.yaml`](visual_eval_defaults.yaml), which loads
[`opencode_sessions.yaml`](opencode_sessions.yaml); OpenCode runs
inside the sandbox through Gym's shared agent-session lifecycle. The `opencode.yaml`
recipes remain aliases for these defaults.

| Recipe | Network policy | WebFetch / WebSearch |
| --- | --- | --- |
| [VStar](vstar/opencode.yaml) | `model_only` | Disabled |
| [HRBench8K](hrbench8k/opencode.yaml) | `model_only` | Disabled |
| [VTCBench](vtcbench/opencode.yaml) | `model_only` | Disabled |
| [ZeroBench](zerobench/opencode.yaml) | `model_only` | Disabled |

This uses the same provider execution boundary as BrowseComp's OpenCode pairing.
The deployment chooses the provider, task image and network separation. It does
not replace the benchmark's tools with BrowseComp's Exa wrappers. Prompts,
sampling and scorers are preserved. The former 55+10 grace protocol is replaced
by OpenCode’s stock advisory final-answer request after 65 steps; ZeroBench still uses five
repeats and primary pass@5.

All evaluations use owned OpenSandbox sessions. Supply provider credentials through
runtime environment variables and an image with the visual tools and offline
OpenCode assets required by the chosen network policy. Evaluation and training
share the same session implementation; the local relay recipes have been removed.
See the [agent session setup](../../responses_api_agents/opencode_agent/README.md#sandbox-sessions)
and [network profiles](../../responses_api_agents/opencode_agent/README.md#sandbox-sessions).
Resume historical runs with their recorded Gym revision and configuration.

## Run configuration

Supply a policy model using Gym's [model configuration guide](../../fern/versions/latest/pages/model-server/index.mdx).
Pass the deployment configuration to both `gym eval prepare` and `gym eval run`
with `--config run.yaml`: preparation also validates the selected agent's configuration.
For example, `run.yaml` can compose the OpenAI model adapter and set the required
model and sandbox image:

```yaml
config_paths:
- responses_api_models/openai_model/configs/openai_model.yaml
policy_model_name: <vision-model-id>
policy_base_url: ${oc.env:POLICY_BASE_URL}
policy_api_key: ${oc.env:POLICY_API_KEY}
visual_agent:
  responses_api_agents:
    opencode_agent:
      sandbox_config:
        image: <image-with-visual-tools-and-offline-opencode-assets>
```

Supply the model adapter's endpoint and credentials, and the OpenSandbox
provider's `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`, through the runtime
environment. See the agent session setup linked above for image requirements.
VStar, HRBench8K and VTCBench default to `azure/openai/gpt-4o-mini` through
`visual_judge`. Supply `GYM_JUDGE_BASE_URL` and `GYM_JUDGE_API_KEY` in the runtime
environment. ZeroBench starts no judge. To select a different judge, configure
the shared resources-server instance:

```yaml
opencode_multimodal_benchmarks:
  resources_servers:
    opencode_multimodal_benchmarks:
      judge_model: <judge-model-id>
      judge_server: visual_judge
```

The configured `visual_judge` route captures judge calls with Gym's rollout
correlation. For direct HTTP judging, also disable that imported model server.
Pass these overrides to both preparation and evaluation (replace the endpoint
and key-variable name with your deployment's values):

```bash
++visual_judge=null \
++opencode_multimodal_benchmarks.resources_servers.opencode_multimodal_benchmarks.judge_server=null \
++opencode_multimodal_benchmarks.resources_servers.opencode_multimodal_benchmarks.judge_endpoint=https://judge.example/v1/chat/completions \
++opencode_multimodal_benchmarks.resources_servers.opencode_multimodal_benchmarks.judge_api_key_env=DIRECT_JUDGE_API_KEY
```

These settings can alternatively go in `env.yaml`. A plain `--config run.yaml`
has lower priority than the selected benchmark and may have its judge route
overwritten. With the overrides above, `GYM_JUDGE_BASE_URL` and
`GYM_JUDGE_API_KEY` are unnecessary; only the selected direct judge's credential
is needed. Endpoints and credentials remain deployment inputs.

The shared verifier defaults to a 120-second timeout and three transport attempts
per judge request. VTCBench additionally
uses the public five-verdict parsing schedule. Judge failures need resolution
before a complete score is reported.

For each evaluation, retain the resolved config and inspect rollout health,
coverage and judge failures. Enable model-call capture when collecting evidence.
A valid manifest and passing verifier fixtures do not establish score parity or
harness conformance.

The [2026-11-14 rollout evidence](evidence/visual-defaults-20261114.json) records
representative real Qwen requests for all four catalogs, image delivery, the
32768-token request limit, verifier outputs and a VTCBench judge call. It labels
unfinished evaluations as partial and records the limits of cleanup telemetry.
These runs use the execution settings above; they predate the static-validation
and agent-replacement binding fixes.

For vLLM-compatible model endpoints, add
`--config benchmarks/opencode_multimodal_benchmarks/default_eval.yaml` to preparation
and evaluation. This model-independent profile uses the vLLM adapter, preserves history thinking
(`truncate_history_thinking: false`), propagates context-overflow errors and pins
temperature 0.6/top-p 0.95. Reasoning serialization retains the adapter default;
no `reasoning_field` override is applied. Supply `policy_model_name` in `run.yaml`,
`POLICY_BASE_URL` and `POLICY_API_KEY` in the environment, plus the sandbox deployment settings. With this profile, omit
the OpenAI adapter from `run.yaml` and omit `--model-type openai_model` from the
run command. Other policy adapters remain supported.

## Grading and recovery

The Resources Server owns grading. Deterministic scores, including incorrect or
empty-answer zeros, use `observe_programmatic_grading`. Gym-routed judges use
`observe_llm_judge_grading`, `call_judge`, and a parsed-verdict marker. The recorded
operation binds the full saved response and final reward, including any tool
bonus or penalty, to the judge reply. No separate grading transformation is needed. Use
`judge_server` and model-call capture for qualification. Direct HTTP judging is
supported for compatibility, but its unobserved judge calls cannot establish a
fully observed grading record. It is never labeled model-free grading.

An invalid judge reply is not a model mistake. Transport or parsing failures
preserve the policy response and become masked `judge_failed` outcomes. Direct
resource-only rows can be rejudged from the saved answer. To regrade tool rewards,
set `forward_agent_evidence: true` in the reverification configuration so the
saved trajectory and observations reach the verifier. This is a separate opt-in
from the Environment Server's setting; reverification defaults to `false`, even
when saved results contain execution evidence. The grading configuration and
required execution evidence must remain available. If a parsing retry succeeds after an unparseable
reply, the score is retained but the grading operation remains `unknown`: an
invalid reply is not falsely marked as a parsed verdict. Inspect these outcomes
before claiming complete grading evidence.

Headline completeness compares valid measurements with the resolved materialized
attempt inventory, including runtime repeat overrides. An absent repeat remains
in that inventory even when its failures are stored in a separate artifact.
Without an inventory, aggregation reports partial scores rather than asserting
that the surviving rows form a complete run.

Catalog pairings have no whole-episode deadline by default. To set one, configure
`visual_environment.environment_servers.single_agent_turn.default_episode_timeout_seconds`.
`BaseEnvironmentServer.run_request` enforces it with `asyncio.timeout` on the
event loop's monotonic clock. Cleanup has a separate
`cleanup_timeout_seconds` budget and runs after the episode deadline.

The four catalogs use a 131072-token context window, a 32768-token output limit,
agent and collector concurrency of 64, judge concurrency of 16, and a two-second
sandbox polling interval. OpenCode requests a
final answer after 65 steps. This request is advisory; tools remain available. Its `timeout` setting (optionally narrowed by request metadata
`agent_timeout_sec`) is a separate harness execution deadline, enforced
by the shared OpenCode sandbox session using the monotonic clock; installation
uses `sandbox_install_timeout_seconds`, and close uses
`session_close_timeout_seconds`. The judge timeout above bounds each transport
attempt (including Gym transport), not the wait for the judge concurrency slot.
Scheduler walltime bounds one allocation; the resumable chain budget bounds all
allocations. Save each explicitly in the resolved run configuration.

For a submitted qualification run, set the existing run-level recovery and
delivery controls under your benchmark in `driver.benchmarks`:

```yaml
resumable:
  max_retries: 3
  max_walltime: "16:00:00"
  recovery_outcome_policy:
    by_failure_kind:
      agent_timeout: fail
      model_context_limit: fail
    default_action: retry
run:
  mlflow_tracking_required: false
```

This example deliberately keeps delivery local; a tracked qualification must set
`mlflow_tracking_required: true` and retain its export receipt. Neither setting
relaxes evaluation-health checks. The configured agent/model limits are final outcomes;
infrastructure faults may be retried within the saved budget, and judge faults
use saved-answer rejudging. Do not use recovery to give a settled wrong answer
another policy attempt.

Dataset bytes, repeats, model, prompt, sampling, judge and scoring settings are
material resume identity. Placement addresses and the collector's
`dispatch_connect_grace_s` are operational. Keep the original run directory:
resume validates its manifest and leaves settled rows intact. Never override a
material mismatch with `allow_unsafe_resume`; start a new run instead.

Before qualification, resolve and dry-run the submission, then run three distinct
tasks and the fault/cut-and-resume fixtures described in Gym's
[benchmark skill](../../.agents/skills/add-benchmark/SKILL.md#step-7b-failure-handling-conformance).
Unit fixtures establish code contracts; they do not replace a real-model run or
the submitted run's health and recovery gates.

## Dataset identity

VStar/HRBench8K and ZeroBench are checked against the pinned manifests or
source checksum. VTCBench TSV releases are caller-supplied: pass
`prepare_script_args.sha256` for qualification and retain the preparation's
`dataset.json` with its source and prepared-task hashes. Use a separate output
path for a different release or preprocessing policy. A caller-supplied variant
must not be reported as the pinned reference run without matching these hashes.

Prepared VStar, HRBench8K and VTCBench image inputs use embedded data URLs.
Hosted model endpoints need no access to the preparation machine’s image paths.

## Dataset licensing

Manifest `licensing` describes the dataset, separately from the integration code.
Use an applicable publisher-declared dataset license when established; `unknown`
means that this metadata has not been established, not that the publisher has
granted unrestricted use. Dataset contents are not distributed with these entries.

| Dataset | Publisher metadata and conditions |
| --- | --- |
| VStar | The [pinned TSV mirror card](https://huggingface.co/datasets/xjtupanda/VStar_Bench/blob/b1312e6fbf89ace5a50cc5b0214010aeb0f69bdc/README.md) declares no dataset license. The original project's MIT statement is not substituted for a verified license on the external dataset. |
| HRBench8K | The [pinned card](https://huggingface.co/datasets/DreamMr/HR-Bench/blob/83b9013d6293b85dc507e87199ca52517536939c/README.md) declares `other`, but its linked `LICENSE` file is empty at that revision. The applicable dataset terms remain unresolved. |
| VTCBench | The [pinned publisher repository](https://github.com/zhuzil/VTC-Bench/tree/5e0a5533f597b0c56dae7d9705771498202ab19b) does not establish a dataset license in its root documentation. Apache-2.0 notices for its evaluator subtree concern that code, not the dataset. |
| ZeroBench | The [dataset page](https://huggingface.co/datasets/jonathan-roberts1/zerobench) requires accepting its access conditions, including no training and no public redistribution of answers. The pinned dataset metadata does not identify a standard dataset license. The scoring example's MIT license is separate. |

The current manifests retain `unknown` for these unresolved dataset-license
fields. Gym's local `env publish` finalizer currently rejects `unknown`, although
existing experimental catalog entries also use it. Manifest validation and
verifier checks can pass independently; do not report the publication check as
passed or replace these values with a guessed license to bypass it.

## Training rewards and row profiles

Training profiles using tool rewards must also set `forward_agent_evidence: true`
on their `single_agent_turn` Environment Server so the verifier receives the
trajectory and tool observations. Answer-only evaluation leaves it disabled.

The benchmark verifier accepts only `tool_rewards: null` (the default) and never
adds tool rewards. Non-null settings are rejected. Tool shaping is available only
through the training adapters below; `tool_rewards: {}` selects their default schedule.
The default applies penalties only to correct answers, with no positive bonuses.
Zero to two bash attempts have no count penalty; three, four, five, and six or more
attempts incur penalties of 0.025, 0.05, 0.075, and 0.10 respectively. Failed bash
calls incur an additional 0.05 penalty; consecutive successful duplicate calls
incur a 0.02 penalty. Other tools do not incur failure penalties. Override individual
`tool_rewards` fields to change this schedule.
Scoring reads tool execution outcomes from the shared trajectory
or observation bundle; it never interprets model-visible output as an exit status.

For existing math, MCQA, and string-match training tasks, use the
`opencode_math_tool_rewards`, `opencode_mcqa_tool_rewards`, or
`opencode_string_match_tool_rewards` Resources component, respectively, each with
`entrypoint: app.py`. Each component owns its inherited grader's dataset schema,
so strict dataset collation accepts native training rows without visual benchmark
metadata. These thin adapters import the
existing graders and retain their task schemas. For example:

```yaml
training_string:
  resources_servers:
    opencode_string_match_tool_rewards:
      entrypoint: app.py
      tool_rewards: {}
```

Bind the training Environment Server's `resources_server` to `training_string`
and set `forward_agent_evidence: true` on that Environment Server.
Set the schedule on the Resources server, not the agent. Original response messages
and token arrays are preserved; reasoning cleanup affects only the grading view.
The stock `single_agent_turn` environment forwards execution evidence when opted in.

Each row can select a different environment through `task_source`. Bind restricted
rows to an OpenCode profile with `network_access: model_only`; bind web-enabled rows
to a separate profile with `network_access: inherit` and appropriate tool permissions.
The policy is fixed when the owned sandbox is created. Configure OpenSandbox in
both profiles. Native Exa activation and a configurable bash timeout are separate
stock-agent follow-ups; the default benchmark profile does not enable web tools.

Training configurations previously selecting `*_tool_rewards.py` under
`opencode_multimodal_benchmarks` must use the corresponding component above with
`entrypoint: app.py`. Keep the instance name and environment bindings unchanged.
