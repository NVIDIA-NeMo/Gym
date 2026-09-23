# Pool Sandboxed Agent

Runs [Poolside's `pool` CLI](https://poolside.ai) headlessly (`pool exec`) inside the task sandbox
owned by a resources server, like the OpenCode sandboxed agent: the resources server creates the
sandbox from the task image, this agent installs `pool` into it, points it at Gym's policy model,
runs the task, and leaves the edits in the workdir for the resources server to extract and grade.

## Prerequisites

Complete [OpenSandbox access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#setup).
The sandbox must be able to reach the Gym model server (`++use_absolute_ip=true` in the launcher);
`sandbox_model_base_url` overrides the address if needed.

## First evaluation

```bash
gym eval prepare --config benchmarks/swebench/verified/pool.yaml

# In terminal 1
gym env start \
    --model-type vllm_model \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config benchmarks/swebench/verified/pool.yaml

# In terminal 2
gym eval run --no-serve \
    --agent swebench_verified_pool_sandboxed_agent \
    --input benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    --output results/pool_smoke/rollouts.jsonl \
    --limit 1 --num-repeats 1 --concurrency 1
```

## How pool is configured

The agent generates a full pool agent config (`pool exec --agent-config-file` does not merge with
pool's defaults) with an OpenAI-compatible provider pointed at Gym's per-rollout model URL,
`pool_model` as the model id (the proxy substitutes the policy model), non-streaming inference, and
compaction thresholds derived from `pool_max_context_window`. `pool_agent_config` deep-merges
overrides into it.

`HOME` and the XDG directories are redirected to `/tmp/nemo-gym-pool-<id>` so nothing pool writes
lands in the repo workdir that the resources server diffs. Pool's own container sandboxing is
disabled and tool approvals are auto-allowed since the run is non-interactive.

The `pool exec -o json` event stream is converted into Responses output items and saved under
`results/<session>/events.jsonl` with pool's stderr. Token usage is not part of the stream, so
`usage` is empty.

## Pool binary: online or pre-staged

By default the agent downloads the installer from `downloads.poolside.ai` inside the sandbox and
installs `pool_version`. For sandboxes without egress, mount a Linux binary for the sandbox
architecture and set `remote_pool_binary_path`.
For task images without `curl`, set `local_pool_binary_path` to a binary on the Gym host; the agent
uploads it into each sandbox.
