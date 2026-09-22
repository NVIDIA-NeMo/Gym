# Pool Sandboxed Agent

Runs [Poolside's `pool` CLI](https://poolside.ai) headlessly (`pool exec`) inside a task sandbox
owned by a resources server, the same way the OpenCode sandboxed agent does. The resources
server creates the sandbox from the task image, this agent installs `pool` into it, points it at
Gym's policy model through the OpenAI-compatible endpoint, runs the task, and leaves the edits in
the workdir for the resources server to extract and grade.

## Prerequisites

Complete [OpenSandbox access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#setup).
The sandbox must be able to reach the Gym model server; the launcher does this with
`++use_absolute_ip=true`. Use `sandbox_model_base_url` to override the address if needed.

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

`pool` runs in standalone mode, driven entirely by environment variables, so no login or
credentials file is needed:

| Variable | Value |
| --- | --- |
| `POOLSIDE_STANDALONE_BASE_URL` | Gym model server URL for this rollout, with `/v1` |
| `POOLSIDE_API_KEY` | dummy; the Gym proxy does not check it |
| `POOLSIDE_STANDALONE_MODEL` | `pool_model` (default `dummy_model`; the proxy substitutes the policy model) |
| `POOLSIDE_STANDALONE_CONTEXT_LENGTH` | `pool_max_context_window` |

`HOME` and the XDG directories are redirected to `/tmp/nemo-gym-pool-<id>` so pool's config,
state and trajectories never land in the repo workdir, where the resources server runs `git diff`.
Pool's own container sandboxing is disabled (`--sandbox disabled`) since the task already runs in
a sandbox, and tool approvals are auto-allowed (`--unsafe-auto-allow`) because the run is
non-interactive.

The NLJSON event stream from `pool exec -o json` is converted into Responses output items and
saved under `results/<session>/events.jsonl` along with pool's stderr. Token usage is not part of
the event stream, so `usage` is empty for now.

## Pool binary: online or pre-staged

By default the agent downloads the installer from `downloads.poolside.ai` inside the sandbox and
installs the pinned `pool_version`. For sandboxes without egress, mount a Linux binary for the
sandbox architecture and set `remote_pool_binary_path`.
