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

## Agent identity (`agent_user`)

By default (`agent_user: null`) the in-sandbox opencode process runs as the image default
user, exactly as before. Because opencode IS the agent, its identity is the identity of every
tool call the model makes (the `bash`, `edit`, `write` and other tools run inside that
process), so on a root-default image the model runs as root. `agent_user` (an account name for
`su`, or a uid; the digit-only string `"1000"` is normalized to the int `1000`) lets the agent
run unprivileged while the verifier keeps running as the image default (root on the supported
images), which is what the Terminal-Bench `*-userroot` image derivatives are built for.

Lane-level override (applies to every row with every resources server):

```yaml
opencode_sandboxed_agent:
  responses_api_agents:
    opencode_sandboxed_agent:
      agent_user: agent
```

Per-row override: a dataset row may carry `agent_user`. The resources server echoes it from
`/seed_session` (`terminal_bench_2_1` today) and this agent reads it there; the row override is
honored only when the resources server echoes it, whereas the lane default applies with every
resources server. Precedence: row value if not `null`, otherwise the lane config. A row value
of `"root"` (or `0`) beats a lane `agent` and is the explicit per-row image-default escape
hatch: it skips the identity check, and the provider passes `"root"` through as the image
default (only the int `0` forces uid 0). For mixed pools leave the lane at `null` and set
`agent_user` per row. The effective identity is posted to `/verify` as `agent_user`.

Requirements and fail-closed check: a non-root `agent_user` requires a root-default image in
which that account exists. Before the install the agent runs `id -u` as `"root"` (must print
`0`) and `id -u && id -g` as `agent_user` (must succeed and resolve to a non-root uid/gid; for
a uid it must match). Any failure raises `RuntimeError` naming the requested identity and the
observed output, the sandbox is stopped, and the rollout fails rather than silently running as
root.

What runs as `agent_user`: the install + `opencode run` command, the `opencode session list`
and `opencode export` commands, and the observation database snapshot. Under a non-root
identity the first three commands begin by resetting `$HOME` to the identity's passwd home
(the provider's uid path does not go through `su`, so it would otherwise inherit the image
default's `/root`), so opencode's install lands in `/home/<agent_user>/.opencode/bin` and
its session store in `/home/<agent_user>/.local/share/opencode` (or the per-rollout
`XDG_DATA_HOME` when observations are collected), which is why the list, export and snapshot
commands must use the same identity. The export download uses the provider's file API as the
image default (root can read agent-owned files).

Scope and residual risk:

- The verifier always runs as the image default (root on the supported images), regardless of
  `agent_user`.
- Agent processes are not terminated before verification (service tasks need them alive while
  the tests run). Under a non-root `agent_user` the opencode process and anything it spawned
  (uid 1000) therefore stay alive during root verification and can read `/tests` (0755) and
  modify the workspace, but cannot write `/tests`, `/logs/verifier` or system paths.
- Containment is only as strong as the image's sudoers/setuid configuration; a permissive
  `sudo -n` or a setuid helper in the image gives the model root back.

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
