# Harbor tasks

This Resources Server runs [Harbor](https://www.harborframework.com/)-format tasks in NeMo Gym. It owns the parts of a
Harbor trial that belong to the task:

1. **Seed**: resolve the task from a configured Harbor dataset, start a Gym sandbox from the task's image, prepare
   Harbor's `/logs` layout, upload `environment/` for prebuilt-image tasks, and return `SandboxAccess` for the agent.
2. **Verify**: after the agent closes, run the task's own Harbor verifier (`tests/test.sh`, which writes
   `/logs/verifier/reward.txt` or `reward.json`) in the same sandbox. The agent never sees `tests/`.
3. **Close**: stop the sandbox.

A task's image is its `[environment].docker_image` when set. Otherwise the task builds from `environment/Dockerfile`,
and the server starts `image_template` formatted with Harbor's content hash of `environment/` (`{environment_hash}`).
The server never builds images; `benchmarks/harbor/prepare_utils/build_images.py` builds them ahead of a run, and a
missing image fails the seed with that hint. `tasks.py` holds these rules, shared with the provisioning code so the
two cannot drift; it imports only Harbor.

The agent harness runs elsewhere and borrows the sandbox. Use
[`harbor_harness_agent`](../../responses_api_agents/harbor_harness_agent/README.md) to run any Harbor agent
(Terminus-2, OpenCode, Claude Code, or a custom `import_path`), or any other session-capable agent that accepts
`SandboxAccess`, such as `codex_agent`, `hermes_agent` or `pi_agent`. The
[single-agent-turn Environment Server](../../environment_servers/single_agent_turn/README.md) drives the episode.

## Configuration

`harbor_tasks.yaml` maps the parts of a `harbor run` job to this server:

| Harbor job field | Here |
| --- | --- |
| `datasets` | `harbor_datasets`: alias -> `harbor.models.job.config:DatasetConfig` (local `path`, or registry `name` and `version`) |
| `environment` | `sandbox_provider` and `sandbox_config`: any Gym sandbox provider that can hand off a sandbox (Docker, OpenSandbox, E2B, Apptainer on one host); `image_template` for tasks built from a Dockerfile |
| `verifier` | `harbor_verifier`: `harbor.models.trial.config:VerifierConfig` (timeout override and cap, env, custom import path); `verifier_timeout_multiplier` |
| reward | `reward_key`; every `reward.json` key is also reported in `reward_components` |
| `jobs_dir` | `artifacts_dir`: each episode's downloaded verifier logs, keyed by rollout capture key |

The `agents` part of a Harbor job belongs to the agent's config, and concurrency to the Environment Server's
`max_concurrent_episodes`. Retries are rollout-collection attempts.

`aggregate_metrics` reports pass@k over the verifier reward.

## Rows and benchmarks

A row names one task and carries its instruction:

```json
{"task_id": "hello-world", "harbor_dataset": "example", "task_name": "hello-world",
 "responses_create_params": {"input": [{"role": "user", "content": "Create a file named `hello.txt` ..."}],
                             "metadata": {"harbor_agent_timeout_sec": "600.0"}}}
```

The `metadata` keys carry the task's `[agent]` timeout and user to `harbor_harness_agent`, which never reads the task.
Harbor hub datasets are packaged as benchmarks under [`benchmarks/harbor/`](../../benchmarks/harbor/README.md), which
prepare these rows and build environment images from the same config the server runs with.

## Example

`harbor_tasks.yaml` runs the bundled example task with Harbor's Terminus-2:

```bash
gym env start --resources-server harbor_tasks \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml --model-type vllm_model
gym eval run --no-serve --resources-server harbor_tasks --agent harbor_tasks_harbor_harness_agent \
    --input resources_servers/harbor_tasks/data/example.jsonl --output results/harbor_tasks/rollouts.jsonl
```

Start with `--agent-type harbor_harness_agent/opencode` to run another Harbor agent.

## Unsupported tasks

Seeding rejects, and benchmark preparation refuses unless told to skip them, tasks that this server cannot reproduce
faithfully:

- multi-step tasks (`[[steps]]`);
- separate verifier environments (`verifier.environment_mode = "separate"`);
- docker-compose environments, and tasks with neither a prebuilt `docker_image` nor an `environment/Dockerfile`
  (or a Dockerfile but no `image_template`);
- task-declared MCP servers, because the agent never sees the task and would run without them;
- Windows tasks;
- restricted network policies, because Gym sandboxes do not enforce them. Set `allow_unenforced_network_policy` to run
  such tasks without enforcement.

## Licensing information

Code: Apache 2.0. The example task is Apache 2.0.

Dependencies
- nemo_gym: Apache 2.0
- harbor: Apache 2.0
