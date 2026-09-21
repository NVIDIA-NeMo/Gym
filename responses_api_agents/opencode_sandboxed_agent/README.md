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

For E2E functional testing, run as above and remove the actual opencode run command from the exec.

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

## Python and search benchmark variants

`benchmarks/apex_shortlist/opencode.yaml`, `benchmarks/hle/opencode.yaml`, and
`benchmarks/hle/opencode_search.yaml` compose this agent with the existing graders.
They use a preinstalled image, deny public network egress, disable compaction, and
use the remaining-context plugin. APEX repeats each question 16 times; HLE uses
one repeat. Standard benchmark preparation determines the question set. Keep
collection repeats at one.

Set `OPENCODE_SANDBOX_IMAGE` to a verified image digest, `OPENSANDBOX_DOMAIN` to
your assigned HTTPS API endpoint, and `OPENSANDBOX_API_KEY` in the trusted runtime.
Search additionally needs `TAVILY_API_KEY` (one key or comma-separated keys).
Use `OPENCODE_ARTIFACTS_DIR` for durable result paths. Tool calls and returned
text are captured in the normal OpenCode transcript and Gym observability.
The image Dockerfile and lock are in `offline_science_image/`.

The agent supports `preinstalled_opencode`, `output_token_policy`,
`tool_servers`, `network_access`, `artifacts_dir`, and the
existing sandbox/OpenCode config mappings. `sandbox_config.files` maps remote
paths to text contents, not local filenames. Image, working directory and
entrypoint are configurable. `remaining_context` removes the output request cap;
it cannot recover history that already exceeds the model's context window.

OpenCode configuration overlays are deep-merged so model settings do not discard
Gym's model route. Request temperature/top-p reach the build agent. Prefer `permission`; native legacy `tools` precedence can override permission denies.
Execution/proxy/rollout limits should agree at four hours for this recipe. Sandbox
creation and individual Bash command timeouts are separate operational limits.

Search uses the resource server's native authenticated MCP endpoint and a
per-rollout token. No custom SSH gateway or Tavily key is installed in the sandbox.
The Gym model and tool servers must advertise hosts reachable from Kubernetes;
loopback/wildcard addresses are rewritten to the Gym host, which must be reachable from the sandbox.
`model_only` and `model_and_tools` create new sandboxes with explicit allowlists;
externally supplied sandboxes are rejected because their policy cannot be verified.

These variants are not yet certified: run a real model canary through normal Gym
execution, inspect trajectories/grader artifacts, and verify interruption/resume and
sandbox cleanup before promotion. Unit tests and an image smoke are insufficient.

The benchmark variants set `execution_failure_reward_zero=true`: completed
execution failures (including terminal model errors recorded in the transcript) produce reward zero
and explicit failure/exit fields without calling the judge. A generation receipt
is written before verification. Transcript export failures propagate instead of returning an empty result. For request/setup/judge failures, enable Gym's
`route_failures_to_sidecar` so unrelated rollouts continue; unresolved sidecar
rows are missing from the main metric and must not be reported as full coverage.

The benchmark variants reuse standard CoT dataset preparation and graders. APEX
uses the standard math user template with its boxed-answer request. HLE places
the standard Explanation/Answer/Confidence instructions after the question in a
single user message, since OpenCode's CLI receives only the user turn. OpenCode
keeps its default agent prompt plus the short network/tool availability notice.
This adapts the current text benchmarks; a general benchmark-to-harness adapter
is not implemented.

See [offline scientific evaluation](../../fern/versions/latest/pages/evaluation/harness.mdx#offline-scientific-evaluation-with-opencode-or-pi) for both adapters and image build instructions.
