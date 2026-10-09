# Harbor tasks

This Resources Server runs [Harbor](https://www.harborframework.com/)-format tasks in NeMo Gym. It owns the parts of a
Harbor trial that belong to the task:

1. **Seed**: resolve the task from a configured Harbor dataset, start a Gym sandbox from the task's
   `[environment].docker_image`, prepare Harbor's `/logs` layout, upload `environment/` for prebuilt-image tasks, and
   return `SandboxAccess` for the agent.
2. **Verify**: after the agent closes, run the task's own Harbor verifier (`tests/test.sh`, which writes
   `/logs/verifier/reward.txt` or `reward.json`) in the same sandbox. The agent never sees `tests/`.
3. **Close**: stop the sandbox.

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
| `environment` | `sandbox_provider` and `sandbox_config`: any Gym sandbox provider that can hand off a sandbox (Docker, OpenSandbox, E2B, Apptainer on one host) |
| `verifier` | `harbor_verifier`: `harbor.models.trial.config:VerifierConfig` (timeout override and cap, env, custom import path); `verifier_timeout_multiplier` |
| reward | `reward_key`; every `reward.json` key is also reported in `reward_components` |
| `jobs_dir` | `artifacts_dir`: each episode's downloaded verifier logs, keyed by rollout capture key |

The `agents` part of a Harbor job belongs to the agent's config, and concurrency to the Environment Server's
`max_concurrent_episodes`. Retries are rollout-collection attempts.

`aggregate_metrics` reports pass@k over the verifier reward.

## Rows

`prepare.py` writes one row per task. The row names the task and carries its instruction:

```json
{"task_id": "hello-world", "harbor_dataset": "example", "task_name": "hello-world",
 "responses_create_params": {"input": [{"role": "user", "content": "Create a file named `hello.txt` ..."}],
                             "metadata": {"harbor_agent_timeout_sec": "600.0"}}}
```

```bash
# Run with this server's venv, which has Harbor installed.
python resources_servers/harbor_tasks/prepare.py --alias example \
    --path resources_servers/harbor_tasks/data/tasks --output resources_servers/harbor_tasks/data/example.jsonl
python resources_servers/harbor_tasks/prepare.py --alias terminal_bench \
    --name terminal-bench --version 2.0 --output data/terminal_bench.jsonl --skip-unsupported
```

The `metadata` keys carry the task's `[agent]` timeout and user to `harbor_harness_agent`, which never reads the task.

## Example

```bash
gym env start \
    --config resources_servers/harbor_tasks/configs/harbor_tasks_terminus_2.yaml \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
    --config responses_api_models/openai_model/configs/openai_model.yaml

gym eval run --no-serve \
    --config resources_servers/harbor_tasks/configs/harbor_tasks_terminus_2.yaml \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
    --config responses_api_models/openai_model/configs/openai_model.yaml \
    --agent harbor_harness_agent \
    --input resources_servers/harbor_tasks/data/example.jsonl \
    --output results/harbor_tasks/rollouts.jsonl
```

## Unsupported tasks

Seeding rejects, and `prepare.py` refuses unless `--skip-unsupported`, tasks that this server cannot reproduce
faithfully:

- multi-step tasks (`[[steps]]`);
- separate verifier environments (`verifier.environment_mode = "separate"`);
- tasks without a prebuilt `docker_image` (build and push the image first), and docker-compose environments;
- Windows tasks;
- restricted network policies, because Gym sandboxes do not enforce them. Set `allow_unenforced_network_policy` to run
  such tasks without enforcement.

## Licensing information

Code: Apache 2.0. The example task is Apache 2.0.

Dependencies
- nemo_gym: Apache 2.0
- harbor: Apache 2.0
