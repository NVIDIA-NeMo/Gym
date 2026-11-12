# Multimodal coding training

This resources server grades one complete answer for Chart2Code, MMSVG Img2SVG,
MMSVG Text2SVG, Image2Struct LaTeX and Design2Code. The single-answer catalog entry is
`environments/mm_coding_single/`, using `simple_agent`. The
`environments/mm_coding_opencode/` entry pairs the same verifier with Gym's
OpenCode session harness.

This is a training reward. It is not a benchmark metric.

## Measurement contract

The original prompt and input image go into `responses_create_params.input`.
`task` is verifier-owned metadata: task ID, family, declared source split,
reference image relative to `reference_root`, reference SHA256, semantic prompt
and optional private reference source. Reference code and judge credentials
must never appear in model input or in the candidate execution sandbox.
Task metadata is strict. The HTTP request and response envelopes preserve the
legacy agent's routing and rollout-index fields; those transport fields are
not additional task or grading inputs.

Relative `reference_root` values resolve from this component's package/artifact
root, independently of the server's working directory. Absolute overrides are
preserved. Missing, changed and out-of-root references remain masked failures.

The verifier extracts one completed final artifact, renders it in a fresh remote
sandbox, then asks the independent quality judge for task and visual fidelity.
The pinned instructions, rubrics and JSON schema are in `contract.json`.

```
Q = (task_fidelity + visual_fidelity) / 2
R = execution * (0.2 + 0.8 * Q)
```

Malformed, truncated and unrenderable model answers are eligible zero rewards.
Missing renderer dependencies, unavailable providers, killed containers,
missing/changed references and failed or malformed judge replies are **masked**,
not model zeros. A judge failure retains the exact answer for reverification.
The response carries the task identity and SHA256 of the assistant answer,
rendering and private reference. Gym's `observe_llm_judge_grading`, `call_judge`
and `record_parsed_judge_verdict` bind real grading calls to verifier observations.
Programmatic invalid-artifact/execution zeros are observed separately.

The original renderer is versioned with this server and uploaded to every fresh
sandbox. The dependency preflight runs before candidate execution. It preserves
the last saved Matplotlib figure, including save-then-close, and supplies the
LaTeX body preamble, including `tcolorbox`. Model code never runs on the Gym or
training host. The host-execution `local` provider is refused.

## Runtime

Install the component with its committed `pyproject.toml` and `uv.lock`.
Build the sandbox image from `sandbox_image/Dockerfile`. Push it to a registry
that your sandbox provider can pull from. Set `MM_CODING_SANDBOX_IMAGE` to its
digest reference, `<image>@sha256:<digest>`. The server refuses a reference
without a digest.

Set `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`, `MM_CODING_REFERENCE_ROOT`,
`MM_CODING_JUDGE_MODEL` and `MM_CODING_JUDGE_API_KEY` in the trusted service
configuration. The judge can be any OpenAI-compatible endpoint. Set
`MM_CODING_JUDGE_BASE_URL` to select it. The default is
`https://api.openai.com/v1`. The judge model is part of the measurement
contract, so a different judge model changes the reward.

The shipped profile selects `judge_api: chat_completions` and enables the model
server's `drop_output_reasoning_items` for the judge only. Its grading contract
is the structured answer, not provider-encrypted reasoning. The rubric, original
image bytes, private reference, JSON schema, output budget and reward formula
are unchanged. Custom deployments can retain `judge_api: responses` (the config
class default). Replies without one complete verdict remain masked failures.
The policy model never inherits this answer-only judge setting. The functional
acceptance operator retains complete upstream judge replies separately from
the answer-only trajectory capture; existing Responses artifacts stay intact.

Inconclusive verifier execution retries on a fresh box at most
`inconclusive_verification_retries=1` times by default, preserving the answer.
Judge failures are masked for explicit reverification, without another policy
call. The default renderer exec deadline is `execution_timeout_s=420`, enforced by
Gym's sandbox exec API. Its candidate renderer also bounds chart/TeX/browser
subprocesses. A provider timeout with no process exit is inconclusive and masked;
an observed ordinary candidate nonzero exit is an execution failure. Signal-shaped
exits (137/139/143 and negative provider receipts, including OpenSandbox's
`return_code=-1, error_type=None, stderr="signal: killed"`) have an unknown cause
and are inconclusive. They remain masked after the bounded retries. This does
not attribute the signal to infrastructure or the candidate. Setup has a separate
`sandbox_ready_timeout_s=600`; `sandbox_ttl_s=14400` is the cleanup backstop.
Single uses `single_agent_turn_legacy`, with `simple_agent.max_steps=1`, an
explicit model output budget, a 3600-second episode deadline and 300 seconds for
cleanup. Multi uses typed `single_agent_turn`, a 10800-second episode deadline
and 300 seconds for cleanup. These deadlines leave margin below the 14400-second
sandbox TTL. Turn recovery still requires an RL revision that supports the
selected environment protocol; use episode/sibling recovery otherwise.

## Data and licensing

The five checked-in examples are hand-authored synthetic interface examples.
They are not the official benchmark test sets or the training mixture. The
component has no training data and no keys. The source datasets keep their own
licenses and access controls. Register the approved training and validation
data separately.

`renderer.py`, `extraction.py` and `contract.json` come from NVIDIA's
visual-coding RL research code. `renderer.py` adapts `mmrl/render.py`.
`extraction.py` adapts `extract_answer` from `mmrl/artifacts.py`.
`contract.json` keeps the judge instructions and rubrics from
`mmrl/judge_contract.py` and the verdict schema from `mmrl/reward.py`. They have
the same Apache-2.0 license as the rest of Gym.

The NeMo RL recipe provides a format adapter for source-prepared task records. It
does not acquire data, select train/test splits or download benchmark answers.
For existing source Gym rows, select `--input-format legacy-row`; the adapter
reads their original `task` record and rebuilds routing and prompts, rather than
reusing old agent instructions. Use the default `--input-format task` for bare
prepared task records. Its output can be used by `NemoGymDataset` after source ownership, split and
licensing have been checked. Task metadata accepts only declared `train`,
`validation` or `example` splits; this declaration is not a substitute for a
source-dataset leakage audit.

## Validation

```
gym env validate mm_coding_single --json
gym env test mm_coding_single
gym env test --resources-server mm_coding_training
gym env publish mm_coding_single
```

`VERIFIER_FIXTURE` exercises the production reward calculation's endpoints and
missing-verdict refusal, without contacting a judge. The component tests cover
the full verification pipeline with isolated external transports, including
failure masks, reference/source binding, retained answers and cleanup. These
offline checks **do not replace** a real model/remote-renderer/judge smoke on
this exact revision. Record that smoke and inspect its verifier observations
before marking a PR ready for merge.


## Multi and mixed training

The environment has a single-answer route and an OpenCode route.
`mm_coding_multi.yaml` serves the OpenCode route. `mm_coding_mixed.yaml` serves
both routes in one run.

The resources server implements `seed_request_model` and the public
`has_session_state`, `initialize_session_state` and `release_session_state` hooks.
`LentSandboxes` owns each task sandbox. The stock OpenCode agent borrows it with
`resources_server: null`; the typed environment owns seeding, verification and
close. Resource `allowed_agents` explicitly declares the pairing.

Only the public reference pixels, renderer and operator-provisioned offline
runtime dependencies enter the task sandbox. Private ground-truth source,
judge configuration and credentials stay outside. The input carries the original
image as an `input_image` data URI; the matching workspace copy is available for
inspection. Reference SHA256 checks occur before the sandbox is lent. Images
are not resized or dropped by the data adapter.

Task egress is `model_only`. Only the policy model is advertised externally;
agents, resources, environments, head/config and the quality judge bind to
loopback. Since the network policy permits every port of an allowed host, keep
these cooperating control services on the same host/network namespace. A
distributed deployment requires a separate authenticated control network.

OpenCode 1.17.11 must already exist in the selected image. The image from
`sandbox_image/Dockerfile` has it at `/opt/opencode/bin/opencode`. Configure
these pins:

| Setting | Purpose |
| --- | --- |
| `MM_CODING_OPENCODE_PATH`, `MM_CODING_OPENCODE_SHA256` | Preinstalled Linux OpenCode binary and SHA256 |
| `MM_CODING_RG_SOURCE` | Trusted host path to the prepared ripgrep binary |
| `MM_CODING_RG_PATH`, `MM_CODING_RG_SHA256` | Sandbox destination `/opt/mm-coding/rg` and SHA256 |
| `MM_CODING_PLUGIN_BUNDLE_SOURCE` | Trusted host path to the prepared plugin archive |
| `MM_CODING_PLUGIN_BUNDLE_PATH`, `MM_CODING_PLUGIN_BUNDLE_SHA256` | Sandbox destination `/opt/mm-coding/plugin-dependencies.tar.gz` and SHA256 |

The owner delivers dependency bytes through `LentSandboxes.lend(files=...)`
before activation. The stock agent checks its private copies against the pins.
No npm/GitHub download is needed or allowed in a restricted task. Prepare the
plugin bundle using the stock agent's offline-runtime instructions; retain its
package lock, licenses and build receipt. A missing or mismatched dependency
fails setup rather than producing a model zero.

The harness writes `answer.py`, `answer.svg`, `answer.tex` or `answer.html`.
Verification captures a bounded UTF-8 regular file, never a symlink or prose
substitute, and grades that source in a fresh isolated verifier sandbox.
The source must be self-contained. The reference image cannot be embedded as
the answer. Missing/invalid artifacts are measured zeros; inconclusive capture,
renderer or judge failures are masked with `evaluation_completed=false`.
Completed grades and model zeros explicitly set `evaluation_completed=true`.

Verification uses the shared `verifying` context and retained evidence. Retry
returns the cached grade or regrades retained source without another capture or
policy call. Retained source SHA256 and a binding to the task, policy reply and
model request permit trusted sidecar reverification after process restart.
Changed receipts are rejected. Closed/lost sessions remain masked. A failed
sandbox stop retains the handle for close retry; shutdown releases all boxes.

The RL adapter supports `--mode single|multi|mixed`, using OpenCode for multi.
Mixed selects multi at probabilities Chart2Code=.5,
Img2SVG=.5, Text2SVG=.25, Image2Struct=.75 and Design2Code=.5. Routing uses a stable
task/seed/epoch hash before GRPO sibling expansion, independent of rewards,
completion time or queue state. It does not reproduce asynchronous learner
scheduling or balance task-family counts.

OpenCode context metadata is 8192 (`16384 - 8192`), with output metadata 8192. Compaction is
disabled. These are model-window/request controls, not one episode-wide output
counter. PR #5833 supplies the harness-note and lowercased-tool lineage handling;
module discovery alone does not qualify training token/image/logprob capture.

## Acceptance

Run the component tests, strict configuration validation, the session and
lending contract tests and the P0 conformance checks of the rollout collector.
Then run several representative tasks twice with the Super 3.5 VL policy model.
Inspect the image, source, grade and cleanup of each task. Keep the original
failures. Separate functional inference from training token capture.

Three GRPO steps for each mode, full token, image, logprob and KL validation,
and a checkpoint restore with a fourth update are separate GPU acceptance steps.
A functional smoke does not cover them.
