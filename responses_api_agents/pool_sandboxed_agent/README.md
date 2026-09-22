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

`pool` runs with a generated agent config file (`pool exec --agent-config-file`) that mirrors its
built-in defaults and sets an OpenAI-compatible provider pointing at Gym's per-rollout model URL
with a dummy API key and `pool_model` as the model id (the proxy substitutes the policy model).
Streaming is disabled (`use_streaming: false`) because token ids and logprobs are only returned
on non-streaming responses. Use `pool_agent_config` to deep-merge overrides into that file.
`POOLSIDE_API_KEY` is set so pool skips its login bootstrap; no credentials file is needed.

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
