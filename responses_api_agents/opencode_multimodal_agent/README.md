# OpenCode multimodal agent

Run image-based tasks through OpenCode and collect their responses, tool results,
intermediate images and usage in Gym. The agent supports training with token
capture and evaluation with configurable turn limits. A Gym resources server
scores each task.

The implementation extends `opencode_agent` and uses OpenCode **1.17.11**.
Each episode stages its input images in a private workspace. Session configurations use Gym sandbox providers. Compatibility configurations
use Bubblewrap. Session artifacts are parsed into Gym responses
and trajectory records before the workspace is cleaned up.

## Runtime requirements

For session-based evaluation, use a Gym sandbox provider and a Linux sandbox
image containing Bash. Install Python, Pillow, numpy, scipy and OpenCV in the
image when tasks need those tools. Gym installs the pinned OpenCode runtime during
session seed. Enroot, local-provider and borrowed sessions also install an
adapter-owned Python interpreter for the network/procfs wrapper, including
`inherit` sessions. The wrapper never uses the task image's interpreter or Python
environment. Owned OpenSandbox sessions use provider isolation and egress instead
and do not install this interpreter. No host OpenCode installation or Bubblewrap
is required for the session path.

For compatibility configurations, use a Linux host with:

- The native OpenCode 1.17.11 executable on `PATH`.
- Bubblewrap with permission to create user namespaces.
- A Python environment containing Pillow, numpy, scipy and OpenCV for tool use.
- Preinstalled `@opencode-ai/plugin` dependencies at version 1.17.11.
- A Gym model server and a resources server implementing task verification.

Install this agent's locked Python dependencies from its directory:

```bash
uv sync --locked
```

For compatibility configurations, dependency directories are mounted read-only
inside each episode. Input
images are available as `input-0.<extension>`, `input-1.<extension>`, and so on
under `/testbed`. Training converts inputs to PNG; evaluation preserves their
format and also supports the `image_` filename prefix.
Session-based configurations use the standard OpenCode prompt-media pipeline:
shared decoding, image-input declaration, and atomic staging at
`/tmp/nemo-gym-media/<sha256>.<extension>`. The request accepts one user prompt
and an optional system or developer message; images must be inline data URIs
in the user prompt. Remote URLs and host file paths are rejected. Staging runs
under the activation deadline and is cancelled when the session closes.
The adapter adds image-count/pixel validation and optional RGB PNG conversion;
evaluation preserves the original bytes by default.

OpenCode's `read` tool displays images saved by code; printing a filename does
not display the image to the model.

For a disposable local runtime, follow the
[multimodal setup instructions](../../scripts/harness_conformance/README.md#run-diagnostic-probes).
That setup also checks that OpenCode and the image libraries work inside the sandbox.

## Configure the agent

Start with [the agent configuration](configs/opencode_multimodal_agent.yaml).
Supply the model identifier, model and resources server references, and paths to
these runtime assets:

```yaml
sandbox:
  bwrap_command: [/usr/bin/bwrap]
  python_env: /opt/visual-runtime/python
  plugin_dependencies: /opt/visual-runtime/plugins
```

`plugin_dependencies` must contain `node_modules/@opencode-ai/plugin` and an
initialized OpenCode configuration directory. The setup helper prepares this
layout. Use the native executable rather than npm's Node launcher in the sandbox.

To use the same tool libraries as a sandbox image, set `sandbox.rootfs` to a
read-only extracted image and `sandbox.python_env` to its Python prefix inside
the image. For example, an image with `/usr/bin/python3` uses
`python_env: /usr`. System libraries then come from that image; the episode's
workspace and temporary files remain private.

Choose the agent and model entrypoints together:

| Purpose | Agent entrypoint | Model entrypoint |
|---|---|---|
| Training with token capture | `app.py` | `app_opencode.py` |
| Training with a custom system prompt | `combined.py` | `app_opencode.py` |
| Evaluation | `eval.py` | `visual_eval.py` |

The base configuration supplies an 8-turn budget, a 49152-token context and
output ceiling, and a 2048-token final-answer reserve. These are configurable
values, not requirements of the harness. Set `temperature` and `top_p` explicitly
in each task's `responses_create_params`; the agent requires both. Configure the
serving model's context capacity and sampling options to agree with the task and
agent settings.

## Agent sessions

For session-based evaluation, compose a benchmark catalog configuration with
[`opencode_sessions.yaml`](../../benchmarks/opencode_multimodal_benchmarks/opencode_sessions.yaml)
and a Gym sandbox-provider configuration. Supply a Linux task image with the
libraries your tasks need, and a model endpoint reachable from that image.
The example preserves the pinned evaluation profile’s frozen OpenCode prompt, 131072 tokens,
and a 55-turn budget with ten grace turns. The agent and model gateway share the
budget; configure both when changing it.

The Environment Server seeds Resources, seeds the agent session, activates the
agent, closes it, verifies the answer and closes Resources. A supplied sandbox is
borrowed; Resources owns destruction. Otherwise `sandbox_provider` selects an
agent-owned sandbox that is stopped on close. OpenCode is installed in the sandbox
at session seed, not on the agent-server host. A verified preinstalled binary can
be supplied with `sandbox_executable_path` and `sandbox_executable_sha256`.

For OpenSandbox with `model_only` or `model_and_tools`, use
`benchmarks/opencode_multimodal_benchmarks/opencode_sessions_opensandbox.yaml`.
The provider restricts egress at creation, before seed: a fresh sandbox cannot
download OpenCode or npm dependencies. The task image or runtime snapshot must contain OpenCode,
a plugin dependency archive, and ripgrep; supply each `sandbox_*_path` and
`sandbox_*_sha256` required by that overlay. Missing pins reject seed before
allocating a sandbox. See the shared agent's
[offline bundle instructions](../opencode_agent/README.md#pinned-offline-runtime-bundle)
for archive layout and checksum generation. The paths are inside the task image,
not paths on the Gym host. Bash, tar/gzip and checksum tools
must also be available in the image.

Owned OpenSandbox uses provider container isolation and creation-time egress,
without nested namespaces or a private Python helper. Keep control services
unreachable from task containers: allowed model/MCP hosts include all ports and
paths, and `inherit` leaves provider networking unchanged. A per-request network
mode different from the creation-time mode is rejected. See the
[deployment requirements](network-policy.md#enforcement-and-runtime-requirements).

For Enroot/local and borrowed sessions, the private Python helper is prepared on
the Gym host and uploaded. Seed runs the actual network wrapper to verify namespace creation, private
procfs, proxy setup, and model/tool/control origin separation before activation. Setup failure
closes the owned sandbox and remains an infrastructure failure. This check does
not prove that an OpenSandbox deployment's outer egress policy is configured
correctly; qualify the deployment with a real rollout and network-denial probes.

The adapter reuses `opencode_agent`'s supervised execution and cleanup. Input images,
plugins and session state are staged outside the task repository; the prompt names
the staged image paths. Tool-produced images are embedded in returned tool results
before teardown. Session retries reuse the activation rather than launch it again.
One activation is supported per session; continuation inputs are rejected.

The session path implements the **Observability** capability level: evaluation,
model-call and tool observations. MCP task grants reuse the parent adapter.
`model_and_tools` permits their endpoints in owned sessions; `model_only` rejects
required MCP grants. Grants must not share an origin with a protected Gym control
service; use a separate tool-only MCP endpoint. OpenSandbox grants whole hosts;
its deployment must separate control services accordingly. With `inherit`, public
destinations are permitted subject to the provider and applicable wrapper policy. Set
`tool_rewards: null` and disable token-ID capture for these sessions; unsupported
training requests are rejected. Training capture and auxiliary rewards remain
available on the compatibility path. Qualification requires the conformance
report and real pairing evidence for the revision being used.

The compatibility `/v1/responses` and `/run` paths remain available for existing
local Bubblewrap configurations. A configuration selecting `sandbox_provider`
must use agent sessions; it cannot fall back to running on the host. Session
configuration must not also specify the local `sandbox` field.

## Evaluation tool permissions

Evaluation uses explicit permissions for image, file and shell tools. WebFetch
and WebSearch are disabled by default. To select web tools for a benchmark,
include their settings in that benchmark's run configuration
when using `opencode_sessions.yaml`:

```yaml
visual_agent:
  responses_api_agents:
    opencode_multimodal_agent:
      network_access: inherit
      opencode_config:
        tools:
          webfetch: true
          websearch: false
```

Configure each tool independently; omitted tools default to disabled. The example
enables WebFetch only when the effective `network_access` is `inherit`. The adapter derives the corresponding permissions from
these two tool settings. On the local compatibility path, use the configured
agent name (`visual_opencode` in the
example recipes) instead of `visual_agent`. Use separate agent configurations
when evaluating benchmarks with different policies. These overrides apply to the
evaluation entrypoint and leave training configuration unchanged. Enabling web
tools does not enable skills, CodeSearch or remove shell-command denies. For
OpenCode 1.17.11, enabling WebSearch also sets `OPENCODE_ENABLE_EXA=1`, which makes it
available with the configured model provider. WebSearch uses OpenCode's Exa
service; it is independent of the model provider and requires outbound access.
The flag is explicitly off when WebSearch is disabled.

The shipped benchmark pairings select `network_access: model_only` and disable
WebFetch and WebSearch. Custom tasks can explicitly enable either web tool with
`network_access: inherit`, subject to the process/network boundary below.

This is a tool-permission setting, not a network firewall: Bash and Python can
still make network requests if the sandbox permits them. Use sandbox network
controls when an evaluation requires enforced network isolation. Do not add a
top-level `permission: {"*": "allow"}` to the evaluation policy: OpenCode 1.17.11
can give that wildcard precedence over specific denies.

## Enforced network access and per-sample tools

Set `network_access` to `model_only`, `model_and_tools`, or `inherit` to control harness and tool
network access. The adapter default is `inherit`; the shipped offline training
recipe and non-search benchmark profiles explicitly select `model_only`.
On Enroot/Bubblewrap, `inherit` allows public destinations through a proxy while
excluding private/local addresses and registered Gym control services. Owned
OpenSandbox uses provider networking and requires deployment control separation. Native WebFetch/WebSearch remain
available; the provider's outer restrictions still apply.
Restricted modes keep the configured model API reachable; `model_and_tools` also permits
the accepted MCP tool-grant endpoints in owned sessions. Without such grants it allows only the model. These controls apply to owned Linux sandbox sessions and
the local Bubblewrap training path. Only wrapper-based execution requires nested Linux namespaces.

Use `allowed_tools` to select advertised tools. Structured request metadata may
restrict both settings for individual samples, within the run's configured
limits. Read [Network access and per-sample tools](network-policy.md) for
configuration examples, runtime requirements and the exact isolation boundary.

## Prompts

Two configuration fields have different roles:

- `system_prompt` is a **user-message prefix**, despite its name. The base agent
  defaults to the guidance in `prompts/visual_prefix.txt`. Set it to `null` to omit
  that prefix.
- `build_system_prompt`, available in `combined.py` and `eval.py`, replaces the
  OpenCode **system prompt**. Evaluation requires `system_prompt: null` when this
  override is set. `combined.py` defaults to `prompts/combined.txt` and always
  omits the user-message prefix.

With no system-prompt override, OpenCode selects its own prompt. Task text and
image attachments are supplied separately from the system prompt.

## Turn and token limits

For training, `max_turns` and `final_answer_reserve` control a plugin that requests
a final answer and disables tools when the budget is reached. The paired
`app_opencode.py` model adapter handles OpenCode's version-specific final-turn
instruction before token lineage resolution and capture. See the
[model adapter documentation](../../responses_api_models/vllm_model/README.md#stock-opencode-11711-training).

Exact-token reservation requires backend support for `nemo_final_answer_reserve`.
With a backend that does not implement it, set `backend_final_answer_reserve: false`
and keep `final_answer_reserve` positive. The plugin still enforces the final-answer
turn and budgets output using the previous response's reported usage. Without
backend support, that estimate cannot account exactly for new tool-result tokens.
Setting `final_answer_reserve: null` disables the entire plugin and uses OpenCode's
native advisory step behavior; it does not enforce the training turn limit.

For evaluation, configure the same `turn_budget` on the agent and model gateway
to enforce a hard limit. For example:

```yaml
turn_budget:
  regular_turns: 55
  grace_turns: 10
  grace_tool_turns: 2
  grace_tool_calls: 2
```

Set the agent's `max_turns` to the sum of regular and grace turns: 65 in this
example. The final-answer notice starts on turn 55. Turns 56 and 57 can each
execute at most one additional tool call, subject to the total grace allowance.
The remaining grace turns disable tools. Requests beyond turn 65 are rejected
before another model generation. This mode requires one model-gateway worker,
a rollout ID and the chat-completions API. Turns count assistant calls, not
individual tools; a regular turn can contain several tool calls.

Without `turn_budget`, evaluation uses OpenCode's advisory step threshold and
validates the resulting trajectory against `max_turns`. The
`final_answer_turn_headroom` setting reserves room between that threshold and
the trajectory limit; its default is 2. This mode does not provide the gateway's
hard generation stop. Set both `turn_budget` and `max_turns` to `null` for an
uncapped evaluation episode; context and timeout limits still apply.

`context_window` and `max_output_tokens` bound model requests. Also configure the
backend's matching token limits. `timeout` limits the episode and
`command_exec_timeout` limits a shell command. Evaluation's `compaction_auto`
controls automatic OpenCode compaction.

## Images and reasoning history

The agent's `max_images` limits initial attachments. Requests exceeding it fail
before generation. `max_source_pixels` limits the size of an input image.
The default acceptance limit is 150 million pixels per image (width × height).
Raise `max_source_pixels` to accept larger inputs; exceeding it rejects the image
rather than resizing it. The model endpoint has separate image limits.

`image_policy` defaults to upstream's unrestricted OpenCode attachment thresholds
(2,147,483,647 for width, height and base64 bytes), preserving accepted prompt and
tool images without automatic resizing. Lower these thresholds explicitly if an
endpoint requires resizing. OpenCode's `auto_resize: false` rejects an image that
exceeds the thresholds; it does not bypass them. Evaluation's
`use_default_image_resize: true` explicitly restores OpenCode's stock thresholds;
it is disabled by default.

The model adapter's separate `max_images` setting limits images accumulated in
model requests. Additional images are replaced with:

> Image limit reached. This additional image was omitted. You cannot receive additional images.

This guard operates on a request copy and does not modify the saved history.
The model adapter defaults to no image cap. The standalone `eval_55_grace10.yaml`
and `eval_advisory.yaml` presets explicitly select 33; the benchmark runtime profile
and hosted preset explicitly select `null`. Set the model limit to `null` to disable it.
Evaluation also accepts `null` for the agent's initial-image limit. Match these
settings to the serving model's image capacity.

Reasoning-history rendering is controlled by the model server's
`chat_template_kwargs.truncate_history_thinking`; set it to `false` to retain
thinking. Exact-token prefix restoration during training is handled by the
training backend, so message rendering alone does not guarantee identical
training and inference token sequences.

## Verification and observability

`/run` sends the final answer to the selected resources server and retains the
full response for inspection. `strip_reasoning_for_grading` removes embedded
thinking from the scoring copy without altering the saved response. Optional
`tool_rewards` adds tool-based reward shaping; set it to `null` to use only the
verifier reward.

Reward shaping is configured under `tool_rewards`:

- `correct_only` gates all auxiliary adjustments on
  `correct_answer_reward_threshold`; it defaults to `false`.
- `correct_bash_call_penalties` supplies a nonpositive schedule indexed by total
  bash attempts, including failed and duplicate attempts. Index zero must be
  zero, and the last entry applies to all larger counts. The default `[0.0]`
  disables this adjustment.
- `invalid_tool_call_names` restricts failure penalties to selected tools;
  `null` includes all tools and an empty list disables failure penalties.
  Error counts still include failures outside the selected tools.
- Correct-answer success bonuses, exploration rewards and their cap, failure
  penalties, and consecutive-duplicate penalties are separately configurable.
  The failure-tool filter does not restrict duplicate penalties.

Shaping uses native execution records from the main trajectory; replayed and
child-agent calls do not earn auxiliary rewards. A successful tool execution
is not evidence of a correct answer: the resources server supplies that score.

Training uses `token_id_capture` to retain data for optimization. Evaluation can
disable training token capture while retaining model-call observability.
`capture_message_ids` correlates OpenCode assistant messages with model calls.
`capture_failure_artifacts` preserves available session evidence on failures.
The compatibility evaluation `/run` path records recognized agent failures with
zero reward and failure metadata. In agent sessions, the adapter reports the
failure cause and the Environment Server's `agent_outcome_policy` determines
whether to verify, retry or leave the attempt ungraded. Judge and unclassified
infrastructure errors remain errors rather than fabricated answers.

For datasets and benchmark verification, see the
[visual benchmark guide](../../benchmarks/opencode_multimodal_benchmarks/ONBOARDING.md).
For scripted harness checks, run the `opencode_multimodal` conformance suite using
the [conformance instructions](../../scripts/harness_conformance/README.md).

## Private runtime installation

Enroot/local and borrowed sessions seed the network wrapper's stdlib-only Python runtime using
Gym's `SandboxPythonRuntime`. Installation probes the sandbox platform, verifies
the pinned interpreter archive, and reuses an installed runtime only after its
health check passes. The agent host needs access to the pinned Python release or
a populated Gym runtime cache; the sandbox needs Bash, tar/gzip, and supported
Linux namespaces. It does not need Python or package-index access for this wrapper.
The interpreter is launched by absolute path with `-I`; task Python libraries and
environment variables remain available to the tools. Session close leaves the
shared runtime intact. Failed installation closes the session, retaining the
handle if cleanup must be retried.
For these sessions, `inherit` has the same namespace and private-procfs requirements
as restricted modes. Namespace failure stops setup; there is no host-network fallback.
Owned OpenSandbox sessions instead use provider isolation and do not install this helper.


### Shared-code compatibility

Evaluation infrastructure failures propagate to the collector's retry/failure
sidecar instead of becoming settled wrong answers. Established model-caused
failures retain zero rewards. A failed `/run` without a verified result has no
reward, and its partial training capture is masked. Recording a failure in an
external staging ledger must succeed before collection returns a recoverable row;
a ledger-write failure stops collection so the prefix cannot appear trainable.

`run_examples(route_failures_to_sidecar=None)` reads the global configuration;
when absent, the effective default remains false. An explicit boolean overrides
that configuration. Config-driven and explicit routing both return typed failure
records when enrichment is enabled. External staging resolves the same configured,
installed, then file-backed ledger as the model server, and rejects a process-local
ledger for cross-process collection.

The corresponding NeMo-RL consumer is pinned at
`ad05579e04776fedc38c4ee53b17b645cdfa4520`. Its
`nemo_rl/experience/route_assembly.py` consumes
`routed_experts_prefix_boundary`, a one-token routed-expert envelope emitted by
`nemo_rl/models/generation/vllm/vllm_worker_async.py`. The staging adapter validates
this envelope and preserves its bytes for digest verification. The deprecated
`compute_staging_digest(rollout_id=...)` alias supports that revision's data-plane
tests; new callers use `capture_key`, and the alias emits `FutureWarning`.

The shared OpenCode runner moves filesystem copies off the event loop and bounds
process-group cleanup after timeout/cancellation. Shared string matching exposes
`strip_trailing_eos: true` without changing default grading or captured responses;
the previous `app_passrate.py` entrypoint delegates to the common verifier.

An enforced turn budget requires matching agent and visual gateway configuration
and rejects another model entrypoint at agent startup. Turns count valid returned
completions; upstream errors, cancellations and invalid completion envelopes do not
advance the turn counter or spend grace tool calls. Exhaustion returns permanent
HTTP 422. The 55+10 budget and early notice/grace thresholds are retained. This
completion-based accounting changes the protocol for runs with failed attempts;
historical evaluations retain their recorded source revision and attempt-based
accounting. Episode wall-clock limits still apply. Provider attempts remain tracked
separately from the completed-turn budget. The hosted gateway preset preserves explicit
server sampling/template settings while omitting harness sampling defaults.

Enroot/local execution and the compatibility Bubblewrap path require
private PID/procfs and network namespaces. Owned OpenSandbox uses provider isolation
and independently enforced deployment control separation. Synthetic process/control-service
denial and public-web access have been checked on Bubblewrap and Enroot/Pyxis.
These checks do not qualify a different deployment or an unlisted public control
endpoint; see [deployment qualification](network-policy.md#provider-execution-and-deployment-qualification).

Live OpenSandbox qualification passed with a pinned OpenCode 1.17.11 runtime
snapshot in all three network modes: seed, model calls through the Gym gateway,
Bash execution, image input and answer extraction, session close and sandbox deletion. Actual
head, judge, verifier, agent and environment services bound to the Gym host's
loopback were unreachable from task containers; public web access remained
available under `inherit`. This qualifies that deployment arrangement, not other
bindings, ingress aliases or images. See the deployment checks in
[network-policy.md](network-policy.md#provider-execution-and-deployment-qualification).

Qwen reasoning retention depends on the deployed template and settings. The
adapter preserves preceding reasoning, but its trailing directive normalized to a
user message can affect templates that truncate reasoning before the last user
message. Qualify with history preservation enabled: the tested local templates
use `truncate_history_thinking: false`; the tested hosted deployment supports
`preserve_thinking: true`. Recheck when the deployment/template changes.
