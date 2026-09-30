# OpenCode Sandboxed Agent

Resources can select the sandbox provider per task by returning `sandbox_access`
with a named provider reference and descriptor, or the legacy inline
`sandbox_provider` and `sandbox_descriptor`. Shared provider names must be defined
in the agent's Gym configuration too. When neither is supplied, the agent uses
its configured `sandbox_provider` and the resource's `sandbox_handle`.
Invalid resource-supplied providers fail without falling back.

For session-based resources such as TB4, OpenCode returns the seed's `session_id`
and execution outcome to `/verify`, after stopping its process group and
releasing its connection. Resources owns sandbox destruction. Each seed attempt
carries an `Idempotency-Key`; transport retries reuse it. The task prompt comes
from the prepared input. Set `sandbox_model_base_url` if the configured Gym model
address is not reachable from the sandbox. Resource-supplied `mcp_servers` and
`skills_dir` are applied to that run’s OpenCode configuration, including enabling
the skill tool when a task provides skills.

TB4's profile is `benchmarks/terminal_bench_4/opencode.yaml`. Prepare with
`gym eval prepare --benchmark terminal_bench_4/opencode`, then use that same
benchmark selector with `gym eval run` and your model settings.

## Prerequisites

Complete [OpenSandbox access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#setup)
for sandbox credentials, endpoint configuration, and resource limits before launching.

## First evaluation

From the repository root, with Gym installed and model/sandbox access configured, use the
[SWE-bench Verified recipe](../../benchmarks/swebench/verified/opencode.yaml), which binds
the agent to its resources server.

```bash
# Prepare the input before starting servers (downloads SWE-bench Verified).
gym eval prepare --config benchmarks/swebench/verified/opencode.yaml

# In terminal 1
gym env start \
    --model-type vllm_model \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config benchmarks/swebench/verified/opencode.yaml

# In terminal 2, with the same Gym environment activated
gym eval run --no-serve \
    --agent swebench_verified_opencode_sandboxed_agent \
    --input benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    --output results/opencode_smoke/rollouts.jsonl \
    --limit 1 \
    --num-repeats 1 \
    --concurrency 1
```

For an end-to-end evaluation, keep OpenCode execution enabled so `/run` executes
the agent and calls the SWE-bench verifier. Skipping execution limits the test to
the surrounding infrastructure.
This one-task run uses the configured timeout defaults and consumes model and sandbox resources.

## OpenCode binary: online or pre-staged

By default, the agent downloads the [OpenCode installer](https://opencode.ai/install)
and the configured version inside each task sandbox. If `curl` is absent, a Python standard-library fallback downloads and verifies
the pinned Linux binary. The agent supplies CA certificates and bootstraps a pinned,
checksum-verified Python under `/tmp` when the image lacks Python with SQLite support.
This requires Bash, tar, a writable home directory, and network access
to OpenCode and GitHub release assets.

For sandboxes without that network access, provide a compatible installer and binary
through a mount, task image, or custom resources-server upload before the agent runs.
For S3-hosted files, arrange a mount or transfer into each task sandbox.
For the SWE-bench recipe above, configure OpenSandbox
[volume options](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#sandboxspec-provider-options)
under `swebench_verified_opencode_resources_server.resources_servers.swebench.sandbox_config.provider_options`;
the resources server creates the task sandbox.

Set both paths to existing files inside that sandbox;
setting only one leaves online installation enabled.
Save this as `offline-assets.yaml` and add `--config offline-assets.yaml` to server startup:

```yaml
swebench_verified_opencode_sandboxed_agent:
  responses_api_agents:
    opencode_sandboxed_agent:
      remote_opencode_install_script_path: /opt/gym-assets/opencode/1.17.11/install.sh
      remote_opencode_binary_path: /opt/gym-assets/opencode/1.17.11/opencode-linux-x64
      remote_opencode_musl_binary_path: null
```

The staged binary determines the installed version and must match the sandbox's
architecture and libc. Keep `remote_opencode_musl_binary_path: null` with the upstream
installer; the dual-binary mode requires a custom installer supporting
`--glibc-binary` and `--musl-binary`.
