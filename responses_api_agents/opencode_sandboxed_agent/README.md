# OpenCode Sandboxed Agent

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
and the configured version inside each task sandbox. This needs installation tools
(Bash, curl and archive extraction), a writable home directory, and network access
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
