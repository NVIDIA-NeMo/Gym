# Terminal-Bench 4.0 / OpenCode

This benchmark runs the 66 official TB4 task packages through `harbor_agent_general`.
Harbor owns agent setup, task MCP servers, artifact collection, separate verifier
environments, grading, and ATIF trajectories. Task instructions and tests are not
rewritten or copied into Gym. The dataset and published task images are content-pinned.
The Compose metadata file also pins the two upstream sidecar tags to resolved digests.

The default is one attempt per task, no OpenCode step limit, and the task's official
eight-hour agent timeout. Verifier timeouts and CPU, memory, disk, and GPU types come
from each task package. The manifest records 52 CPU single-container, 11 Compose,
and three H100 tasks.

Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` in the environment. Sandboxes must
be able to reach the Gym model server; `++use_absolute_ip=true` advertises the host's
address when Gym runs on a compute node.

```sh
gym eval prepare --benchmark terminal_bench_4/opencode
gym eval run --benchmark terminal_bench_4/opencode \
  --model-type vllm_model --model-url http://MODEL_HOST:8000/v1 \
  --model MODEL_NAME --output results/tb4/rollouts.jsonl --concurrency 8 \
  ++use_absolute_ip=true ++tb4_jobs_dir=results/tb4/harbor
```

For a CPU smoke run:

```sh
gym eval prepare --benchmark terminal_bench_4/opencode \
  '++prepare_script_args.task_names=[formal-crypto,ks-solver-cpp,interleaved-vigenere]'
# Add ++tb4_max_steps=3 to the eval run command.
```

Preparation also accepts `++prepare_script_args.category=cpu` (or `compose`, `gpu`).
These filters only select tasks. Always prepare again without filters for a full run.
An eight-hour task budget requires a scheduler allocation long enough for setup,
agent execution, and verification; a four-hour smoke allocation is not an official
score run. Official leaderboard submissions use five attempts per task, while this
integration defaults to one for coverage testing.

Compose services run through `AsyncSandboxCompose`, including dependency health
checks, sidecar artifact collection, and TCP forwarding for the two tasks that use
`network_mode: service:...`. The adapter checks `SYS_PTRACE` and shared-memory
requirements against the sandbox deployment and fails explicitly if they cannot
be honored. TCP forwarding provides the declared localhost ports; it does not
create a shared Linux network namespace.

`compose-images.json` records verified Linux/amd64 OCI startup metadata for all
38 Compose images. Normalization applies Harbor's standard prebuilt main command,
image entrypoints, working directories, users, and exposed ports before creating
sandboxes. Original task files remain unchanged. Resolved Compose files are saved
beside the Harbor trial artifacts.

## Provenance

- [Release announcement](https://www.tbench.ai/news/terminal-bench-4-0)
- Dataset: `terminal-bench/terminal-bench@4.0.0`
- Dataset digest: `sha256:39d9f44b40420cde8fdcc087579c0d72a7e14fa3656d603c3f0d22fb35e27732`
- [Source release](https://github.com/harbor-framework/terminal-bench/tree/452bf305c6daa62fc59061d22133a7cbc7c1572e)
- Runtime: Harbor 0.23.0; OpenCode 1.17.11.
