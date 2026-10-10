# Harbor benchmarks

Benchmarks from the [Harbor hub](https://hub.harborframework.com/datasets), run in NeMo Gym. Each benchmark folder
pins one published version of a Harbor dataset. Its tasks are served by the
[`harbor_tasks`](../../resources_servers/harbor_tasks/README.md) Resources Server, which starts each task's sandbox
and grades it with the task's own Harbor verifier. They are solved by an agent that borrows that sandbox, by default
Harbor's Terminus-2 through [`harbor_harness_agent`](../../responses_api_agents/harbor_harness_agent/README.md), and
the single-agent-turn Environment Server drives each episode.

| Benchmark | Harbor dataset | Tasks |
| --- | --- | --- |
| `harbor/hello_world` | `harbor/hello-world` | 1 |
| `harbor/swe_bench_verified` | `swe-bench/swe-bench-verified` | 500 |

## Run a benchmark

1. **Prepare rows.** Downloads the pinned dataset with Harbor and writes one row per task, carrying its instruction.
   It also reports how many environment images the tasks need.

   ```bash
   gym eval prepare --benchmark harbor/hello_world
   ```

2. **Build environment images.** Most Harbor tasks build their sandbox image from `environment/Dockerfile`. The
   server never builds; it starts the image `image_template` names for each environment's content hash. Build those
   images once per dataset version. Identical environments share an image, and existing images are skipped.

   ```bash
   # Into the local Docker daemon, for the Docker sandbox provider:
   python -m benchmarks.harbor.prepare_utils.build_images --config benchmarks/harbor/hello_world/config.yaml --load
   # Or to a registry the sandbox provider can pull from: set image_template to that registry, then --push.
   # Only some tasks: --task-names org/task-a org/task-b, or --limit N. Parallel builds: --jobs N.
   ```

   This needs the Docker CLI with `buildx`. `--push` can use a remote builder
   (`docker buildx create --driver remote|kubernetes`). Per-instance benchmarks need one image per task:
   SWE-bench Verified needs 500, of several GB each.

3. **Run.** Add a sandbox provider and a model server.

   ```bash
   gym eval run --benchmark harbor/hello_world --split benchmark \
       --config nemo_gym/sandbox/providers/docker/configs/docker.yaml --model-type vllm_model
   ```

   Pick another agent with `--agent-type`, for example `--agent-type harbor_harness_agent/opencode`. Agents that run
   inside the sandbox, such as OpenCode, must reach the model server from there. With the local Docker provider, run
   sandboxes on the host network (`create.network: host` in the provider config).

Rewards are the verifier's `reward`; every key of its `reward.json` is also reported in `reward_components`, and
aggregate metrics report pass@k. Each rollout's verifier logs land under `results/harbor/<benchmark>/harbor_tasks/`,
and the agent's logs and ATIF trajectory under the agent's `logs_dir`.

## Add a Harbor benchmark

Copy `hello_world/` and change, in `config.yaml`, the `harbor_datasets` entry (the hub `name` and the `ref` to pin,
for example `sha256:<content hash>` of the version you validated) and the instance names; in `manifest.yaml`, the
names and metadata. `prepare.py` stays a few lines that call `prepare_utils.provisioning.prepare_rows`. Provisioning
rejects tasks harbor_tasks cannot run faithfully (multi-step tasks, docker-compose environments, separate verifier
environments, task-declared MCP servers, restricted networks); see the harbor_tasks README.

## How provisioning runs Harbor

`prepare_utils/harbor_side.py` downloads and inspects tasks with Harbor in an isolated environment
(`uv run --isolated --with harbor==<the harbor_tasks pin>`), because Harbor's dependencies conflict with Gym's. It
uses `resources_servers/harbor_tasks/tasks.py`, the rules the server applies at seed time, so provisioning and the
server agree on image tags and supported tasks.
