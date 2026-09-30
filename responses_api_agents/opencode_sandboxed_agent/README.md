# OpenCode Sandboxed Agent
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
