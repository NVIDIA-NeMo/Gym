# OpenCode Sandboxed Agent

The existing `opencode_sandboxed_agent` entrypoint supports native EnvironmentServer
sessions. OpenCode and its tools run inside the task sandbox created by Resources;
the adapter only borrows the connection. The separate local `opencode_agent` and
existing benchmark recipes keep their current behavior.

## Native EnvironmentServer sessions

Bind `single_agent_environment_server.environment_servers.single_agent.agent_server`
to this agent and `resources_server` to a Resources implementation supporting native
sessions and returning `SandboxAccess`. Bind this agent's `model_server` to a Gym
model endpoint reachable from the task sandbox. Submit episodes to **EnvironmentServer
`/run`**. Existing legacy Resources recipes do not automatically gain native lifecycle
support merely by changing their agent.

```yaml
config_paths:
  - environment_servers/single_agent/configs/single_agent.yaml
  - responses_api_agents/opencode_sandboxed_agent/configs/opencode_sandboxed_agent.yaml

single_agent_environment_server:
  environment_servers:
    single_agent:
      agent_server:
        type: responses_api_agents
        name: opencode_sandboxed_agent
      resources_server:
        type: resources_servers
        name: task_resources  # Supply a native Resources config in this run.

opencode_sandboxed_agent:
  responses_api_agents:
    opencode_sandboxed_agent:
      num_workers: 1
      model_server:
        type: responses_api_models
        name: policy_model
      opencode_version: 1.17.11
```

The agent opens `/v1/agent_sessions`, installs the pinned standalone OpenCode runtime,
and runs one `/ng-rollout/<capture_key>/v1/responses` activation. It accepts a string
or one user text message, optionally preceded by system/developer text; `instructions`
is appended through OpenCode's instruction-file configuration. Images and conversation
replay are rejected before consuming the activation. OpenCode owns tool selection.
Required Resources HTTP/MCP tools are unsupported. Native `opencode_config` accepts
only the existing `permission` and `tools` settings so provider/model overrides cannot
bypass Gym correlation.

Request `max_output_tokens`, sampling, reasoning, formatting, tool controls, and other
unused request controls are explicitly rejected. Set effective sampling and per-model-call
output limits on the Gym model server and verify its captured requests. OpenCode's model
context metadata does not establish an enforced output-token limit. Model calls use
Chat Completions through the attempt-qualified Gym URL; the adapter never rewrites the
reasoning or tool history OpenCode sends.

The supported runtime is Linux glibc x86_64 or aarch64 with Bash and Python 3.9+ (including
SQLite and `fcntl`). Setup reports failing commands, exit codes and stderr. Missing curl
and CA certificates are installed automatically only with root and apt-get; other images
must include them. Online setup also needs tar/gzip and access to GitHub releases.
`remote_opencode_binary_path` may name a pre-staged binary; the existing optional staged
installer and dual-binary configuration remain supported. Every path is inside the sandbox,
and the installed binary's version must match `opencode_version`.

The version-scoped runtime cache lives under `/tmp/nemo-gym-opencode-runtime-<version>`
and installation is serialized with a file lock. Per-session HOME, caches, instructions,
SQLite state and supervisor files live under `/tmp/nemo-gym-opencode-sessions/<id>`,
outside the task repository. Root, `/tmp`, adapter-owned workdirs, and resolved path
aliases that overlap runtime/session storage are rejected before installation. Session close removes only the session directory and leaves
the reusable runtime cache for Resources to destroy with the sandbox. No host OpenCode
installation or execution occurs in native sessions.

A Linux subreaper per activation kills and reaps tool descendants, including detached
background processes. Close requires its positive cleanup receipt, successful adapter-file
removal, and disconnect before verification can proceed. Unknown launch outcomes and
unconfirmed cleanup fail closed and retain session state for retries. Successful close
retries return the same result for `session_close_retry_window_seconds` (300 seconds by
default); expired cookies continue to reject activation. State is process-local and requires
one worker. Failed sessions remain until owner recovery/process restart; there is no
cross-worker crash recovery. Process cleanup prevents verification races and is not a
security boundary against hostile task code.

The response preserves text, reasoning, tool results and partial output from OpenCode's
SQLite state. Close returns the existing session-tree, subagent and compaction observations.
Native usage restores inclusive cache/reasoning counts from all persisted assistant turns,
including subagents. Auxiliary model requests not persisted as assistant messages may remain
unaccounted for; compare against captured Gym calls before using totals for accounting.
Missing artifacts and reconciliation are explicit observation gaps. A failed model call may
still leave a correct patch: inspect episode failures and verifier outcomes separately.

Functional smoke validation is not an accuracy baseline. Training token IDs/logprobs,
multiple image families, and supervisor overhead at production concurrency need separate
validation. Pin model, prompt, limits, image, verifier and concurrency for before/after runs.


## Prerequisites

Complete [OpenSandbox access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#setup)
for sandbox credentials, endpoint configuration, and resource limits before launching.

## Legacy agent /run evaluation

From the repository root, with Gym installed and model/sandbox access configured, use the
[SWE-bench Verified recipe](../../benchmarks/swebench/verified/opencode.yaml), which binds
the agent to its resources server.

```bash
# In terminal 1
gym env start \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config responses_api_agents/opencode_sandboxed_agent/configs/opencode_sandboxed_agent.yaml \
    --config resources_servers/swebench/configs/swebench.yaml

# In terminal 2
python responses_api_agents/opencode_sandboxed_agent/client.py \
    +benchmark_jsonl=benchmarks/swebench/data/swebench_verified_benchmark.jsonl
```

## Prefetch OpenCode binary and upload to S3
```bash
curl -L https://opencode.ai/install -o opencode_install.sh

APP=opencode
archive_ext=".tar.gz"
os=linux
arch=x64
target="$os-$arch"
requested_version=1.17.11
filename="$APP-$target$archive_ext"
url="https://github.com/anomalyco/opencode/releases/download/v${requested_version}/$filename"
curl -L $url -o $filename
tar -xzf "$filename" -C "./"

aws s3 cp opencode_install.sh /path/to/folder/opencode/install.sh

aws s3 cp opencode /path/to/folder/opencode/$APP-$target

# Double check they are uploaded properly.
aws s3 ls /path/to/folder/opencode/
```

## Offline scientific evaluation with OpenCode or Pi

The dedicated `opencode_sandboxed_agent` and `pi_sandboxed_agent` run their native
CLIs inside a provider-managed sandbox. Gym stays on the host: it provisions the
sandbox, routes model requests, seeds authenticated MCP tools, collects transcripts,
and calls the existing verifier. No Gym installation is needed inside the sandbox.

For benchmark selection, prompts, and repeat counts, see the
[HLE](../../benchmarks/hle/README.md#sandboxed-agents) and
[APEX Shortlist](../../benchmarks/apex_shortlist/README.md#sandboxed-agents) instructions.

Set `OPENCODE_SANDBOX_IMAGE` or `PI_SANDBOX_IMAGE` to your validated image digest,
plus `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for your assigned sandbox API.
Set `OPENCODE_ARTIFACTS_DIR` or `PI_ARTIFACTS_DIR` to durable host storage.
Search additionally reads `TAVILY_API_KEY`, accepting either one key or a
comma-separated pool. Provider credentials remain on the host; the sandbox receives
only per-session Gym MCP headers. See [Tavily search](../../fern/versions/latest/pages/infrastructure/tavily-search.mdx)
for exclusions, retry limits, and the source-only search response contract.

```bash
gym eval run --benchmark hle/pi_search --model-type vllm_model
```

The presets use 512 concurrent requests per agent worker, 2 CPUs and 8 GiB per
sandbox, four-hour execution/TTL limits, and 20-minute readiness limits. Collection
concurrency controls how many requests actually arrive across workers. Align proxy
and rollout timeouts with the four-hour execution limit. Native
OpenCode runs at most 400 steps and disables background titles. Pi caps each Bash
call at 120 seconds while preserving shorter positive model-specified deadlines;
invalid or nonpositive deadlines use the configured cap. Both
presets disable compaction and use `output_token_policy: remaining_context` to
omit a fixed output-token request. Configure the inference server for the intended
262,144-token context; the hook cannot recover history already exceeding it or
remove a server-side default output cap.

OpenCode also accepts `opencode_model_call_timeout` in milliseconds for each model
request. The general-purpose OpenCode config defaults to one hour; these benchmark
presets leave it unset, so the sandbox execution budget bounds the run. The stream
idle timeout follows `sandbox_timeout`.

Native agent prompts receive only a short note about network availability and the
preinstalled scientific tools at `/opt/science/README.md`. The search variants
mention Tavily explicitly. OpenCode accepts its existing single-user-turn input
contract; Pi also combines supplied system/developer messages with its native prompt.

Restricted network policies require OpenSandbox. Python-only allows the Gym model
host; search additionally allows the Gym tool host. Other destinations are denied.
The allowlist permits **all ports and paths on each allowed host**. If model, tool,
head, or verifier services share a host, the sandbox may reach those services too;
this policy alone does not isolate privileged Gym APIs. Deploy on appropriately
isolated hosts when that separation is required. An isolated gateway enforcing
per-session routes is outside this adapter's scope.

Restricted modes reject loopback and wildcard service addresses. Set
`use_absolute_ip: true` (enabled by these presets) or configure an explicit
sandbox-reachable bind address for each model/tool server before starting Gym; changing an advertised URL does
not change the server's listening interface. Inherited networking preserves
loopback addresses for host-network Docker deployments. Remote sandboxes still
require endpoints reachable from their network.
Resource-owned sandboxes cannot be used with a restricted policy because the agent
cannot verify their existing policy. Direct unseeded `/v1/responses` is unsupported
for Pi, and OpenCode requires `/run` when tool servers are configured.

Generation receipts and native logs are written before grading. OpenCode includes
native turns in top-level `ng_trajectory`; Pi preserves its native event stream,
per-event timing, response tool calls, and agent observations. Completed execution
failures receive zero reward when `execution_failure_reward_zero` is enabled.
A terminal token-limit stop is also an execution failure: the benchmark presets
score it zero even if the partial answer is correct. The returned response is
marked `incomplete` with reason `max_output_tokens`, and retains the partial output
for inspection. This changes OpenCode's previous behavior of grading truncated
answers, and makes Pi's zero-score behavior independent of observation collection.
Both adapters label provider-reported timeouts as `timeout`, including exit code
124 when no provider error type is supplied.
The adapter verifies an empty output to obtain the resource's native score fields,
so failed attempts remain in metrics such as APEX symbolic pass@1. The verifier
must score empty output as an unmasked zero; incompatible verifiers fail the
request. The returned result and generation receipt retain the original output
for inspection. This replaces the previous zero-reward shortcut, which omitted
resource-specific fields and could inflate native aggregate metrics.

Failure to initialize configured Gym MCP tools is a setup failure: neither agent
continues silently without those tools. Setup, transcript export, and judge
failures propagate as request failures; use Gym's failure sidecar to continue unrelated rows and account for those missing rows
when reporting coverage. Export failure propagation is stricter than the previous
OpenCode adapter's best-effort empty result. Configured OpenCode overrides are deep
merged; prefer `permission` over legacy `tools`, whose native precedence can
otherwise defeat permission denies.

`sandbox_config.files` maps remote paths to text contents, not local filenames.
Image, working directory, and entrypoint are configurable. Request temperature
and top-p are forwarded to OpenCode's build agent.

### Reproducible offline images

Build OpenCode's scientific base from
`responses_api_agents/opencode_sandboxed_agent/offline_science_image`:

```bash
docker build --platform linux/amd64 -t gym-opencode-science:local \
  responses_api_agents/opencode_sandboxed_agent/offline_science_image
```

It includes OpenCode 1.17.11 and its preinstalled plugin dependency, Python 3.13.14
scientific packages, SageMath 10.8 with its own Python 3.13.14 environment, and
Lean/Mathlib 4.34.0. Tool use instructions and dependency provenance are included. See the
[image README](offline_science_image/README.md)
for dependency updates and offline validation.
The Dockerfile pins downloads and image digests, and package locks pin transitive
dependencies. Builds need network access; sandbox startup performs no package installs.

Pi adds Node 22.23.2 and Pi 0.85.1 in a small layer:

```bash
docker build --platform linux/amd64 \
  --build-arg SCIENCE_IMAGE=gym-opencode-science:local \
  -t gym-pi-science:local \
  responses_api_agents/pi_sandboxed_agent/offline_science_image
```

Use an immutable base digest for published deployments. No API credentials, model
checkpoints, benchmark questions or reference answers belong in either image.
Validate real Python and MCP rollouts, network isolation, and sandbox cleanup before
promoting a new image.
