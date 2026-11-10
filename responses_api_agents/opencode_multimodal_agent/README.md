# OpenCode visual benchmark and training policies

This adapter extends `opencode_agent` with image guards, final-answer controls and
native tool-reward evidence. Evaluation and training use the same Environment
Server → OpenCode session → OpenSandbox execution path. Runtime installation,
model routing, supervision, media staging, observations, capture and cleanup come
from the shared OpenCode implementation.

## Agent sessions

Use an owned OpenSandbox and set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`
in the launching environment. Keep secrets outside committed configuration.
Local, Enroot, borrowed and unseeded direct executions are rejected; there is no
Bubblewrap or proxy fallback.

Supply an image with Python and the tools the task needs. Visual tasks commonly
use Pillow, numpy, scipy and OpenCV. OpenCode is pinned to 1.17.11. With restricted
networking, provide a preinstalled runtime and pinned offline assets through the
shared `sandbox_executable_*`, `sandbox_plugin_dependencies_*` and
`sandbox_ripgrep_*` options. The plugin archive must cover the session's private
OpenCode configuration. The parent adapter verifies configured digests and stages
these assets; a missing runtime never falls back to host execution.

See the [shared OpenCode adapter](../opencode_agent/README.md) for asset formats,
installation and supervision, and the
[OpenSandbox provider guide](../../fern/versions/latest/pages/infrastructure/sandbox/opensandbox.mdx)
for deployment configuration.

## Training

Compose `configs/opencode_multimodal_agent.yaml` with the training model recipe,
Resources Server and task image settings. It includes `configs/train_sessions.yaml`.
The Environment Server uses `single_agent_turn/opencode.py`, a small extension of
main's lifecycle: seed resources, create/activate/close the agent, verify, and
close resources. It sends a final-answer copy to the verifier and preserves the
complete original response for training.

Training enables token capture, an eight-turn final-answer policy and a 2048-token
final-answer reserve by default. `backend_final_answer_reserve` requires a backend
that implements that extension; disable it explicitly for hosted backends that do
not. The final-answer plugin disables tools on the terminal call. Main's capture
handling and `app_opencode.py` normalize the corresponding harness notes.

Auxiliary rewards use native exported tool states and shell exit codes. They
reward successful bash calls and can penalize failures and consecutive duplicate
calls. The verifier's base reward determines the answer-dependent schedule.
Set `tool_rewards: null` to disable shaping. Missing or unfinished evidence fails
closed instead of inventing a successful call. The full trajectory, including
child calls, remains available to capture; shaping uses the root invocation only.

For a combined system prompt, compose `configs/train_combined_noaux.yaml`. It uses
main's `system_prompt_mode: replace`, with literal prompt braces protected from
OpenCode configuration substitution. Configure a different `system_prompt` as
needed; there is no separate prompt-rewriting plugin.

## Evaluation

The four benchmark `<benchmark>/opencode.yaml` recipes use the same sessions.
`eval.py` changes policy defaults: no auxiliary rewards or training reserve, and
optional gateway-enforced turn budgets. The benchmark profile uses 55 regular
turns plus 10 grace turns, retaining reasoning history and disabling compaction.
The gateway accounts for parent and child sessions independently and cleans idle
state. The advisory preset instead sets OpenCode's soft steps threshold.

Evaluation verification and failure routing use the shared Environment Server.
Infrastructure failures remain failures/retries; they must not be silently turned
into incorrect-answer zeros. Configure the Environment Server's typed outcome
policy when intentionally changing which model failures are verified.

## Network and tool profiles

All network enforcement comes from main's OpenSandbox provider policy:

| Mode | Egress |
| --- | --- |
| `inherit` | Provider/deployment defaults. |
| `model_only` | Model host only. |
| `model_and_tools` | Model host and the granted MCP tool hosts. |

The allowlist covers whole hosts, including all ports and paths. Keep Gym control
services on loopback or on separate inaccessible hosts; a model and a control API
on the same exposed host are not isolated by this policy. The supplied profiles
bind control servers to loopback and advertise a sandbox-reachable model address.
Additional ingress aliases and deployment routing still need validation.

The policy is fixed at sandbox creation. Use separate agent/environment instances
and `task_source` routing for different network modes, tool permissions or
verifiers. Request metadata cannot change an existing sandbox's policy. Native
web tools need `inherit`; `model_and_tools` permits granted MCP hosts, not arbitrary
public web search. Tool permissions use main's shared default command policy and
can be overridden through `opencode_config.permission`.

## Images and artifacts

The adapter requires user prompt images and enforces `max_images` and
`max_source_pixels` (150 million pixels by default). Exceeding a limit rejects the
request. Valid images preserve their bytes by default. Explicit
`preserve_image_format: false` converts prompt images to RGB PNG.

Main decodes inline images, rejects URLs/file IDs and system/developer images,
and stages content-named files atomically inside the activation deadline. Images
appear in input order before the task text; `detail` is ignored. Harness resizing
is disabled by main's unrestricted attachment thresholds. Restore limits only
explicitly through `opencode_config.attachment` for endpoints that require them.

Saved transcript text follows main's parser. Tool images used for training come
from captured model requests, rather than a second attachment-restoration parser.
Inspect capture completeness and pixels before claiming training support. Harbor
uses text-only ATIF export and is not a validated pairing for this adapter.

## Validation

Run this adapter's tests and the shared OpenCode session/capture suites. Live
qualification must include real image/tool requests, provider network allow/deny
checks, cleanup, and a NeMo-RL optimizer step with complete token/logprob and image
capture. A hosted-model rollout alone does not establish optimizer compatibility.
