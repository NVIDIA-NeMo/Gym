# Harbor Agent for NeMo Gym

This agent integrates the [Harbor Framework](https://www.harborframework.com/) into NeMo Gym.
The rollouts are compeletely owned by Harbor and NeMo Gym acts as the orchestrator,
converting the [Agent Trajectory Format](https://www.harborframework.com/docs/agents/trajectory-format) (ATIF)
to NeMo Gym-compatible outputs. Harbor-related configuration is transparently exposed for use in a NeMo Gym configuration 
file, allowing easy translation from `harbor run` commands.

> [!caution]
> NeMo Gym provides an incompatible [older implementation](../harbor_agent/README.md). 
> See [Implementation Notes](#implementation-notes) below for more details.

## Configuration

The general configuration structure relies on `HarborAgentConfig` in [app.py](./app.py).

```yaml
harbor_agent_general:
  responses_api_agents:
    harbor_agent_general:
      entrypoint: app.py
      domain: agent

      harbor_jobs_dir: ${harbor_jobs_dir}

      ## Follows harbor.models.job.config:DatasetConfig spec.
      harbor_dataset:
        ...

      ## Follows harbor.models.trial.config:EnvironmentConfig spec.
      harbor_environment:
        ...

      ## Follows harbor.models.trial.config:AgentConfig spec.
      harbor_agent:
        ...

      ## Follows harbor.models.trial.config:VerifierConfig spec.
      harbor_verifier:
        ...
```

See [configs](./configs) for specific examples.

## Example

This example uses the Scientific Computing subset of [nvidia/Nemotron-Terminal-Synthetic-Tasks](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Synthetic-Tasks) to show a fully working end-to-end example.

> [!note]
> All commands below assume the environment is activated and current working directory is the root of Gym repository.

### Prepare Dataset

Download and unpack the Harbor dataset from HuggingFace,
```shell
hf download --repo-type dataset nvidia/Nemotron-Terminal-Synthetic-Tasks skill_based/mixed/scientific_computing.tar.gz --local-dir ./data
tar -xzf ./data/skill_based/mixed/scientific_computing.tar.gz -C ./data
```

The commands will create the directory `data/scientific_computing` which contains Harbor task folders.

### Inputs

The Gym input JSONL file must have each row following the specification of `HarborRunRequest` in [app.py](./app.py):
```json
{"task_name":"scientific_computing_task_0001","responses_create_params":{"input":[]}}
```

We will use a single task from the benchmark for this example, with the input file [example_input.jsonl](./example/example_input.jsonl).

### Run Agent

> [!warning]
> Ensure that you have [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/index.html) installed on the system and it has credentials pre-configured for private container registries. See [docs](https://docs.sylabs.io/guides/3.0/user-guide/singularity_and_docker.html#making-use-of-private-images-from-private-registries). An example Docker environment configuration is also provided in [configs](./configs/docker_opencode.yaml)

Start the environment:
```shell
gym env start \
  --config responses_api_agents/harbor_agent_general/configs/singularity_opencode.yaml \
  --model <model> \
  ++harbor_dataset_path="$(pwd)/data/scientific_computing" \
  ++harbor_jobs_dir="$(pwd)/logs/harbor"
```

> [!tip]
> The `--model` argument is directly passed to the underlying Harbor agent and should be compatible.
> For instance, when using `opencode` agent, it must follow the [OpenCode provider specification](https://opencode.ai/docs/providers/).

Run the tasks:
```shell
gym eval run --no-serve \
  --config responses_api_agents/harbor_agent_general/configs/singularity_opencode.yaml \
  --agent harbor_agent_general \
  --input responses_api_agents/harbor_agent_general/example/example_input.jsonl \
  --output "$(pwd)/logs/rollouts.jsonl" \
  ++harbor_jobs_dir="$(pwd)/logs/harbor"
```

## Implementation Notes

A few notable details make it more general than the [older implementation](../harbor_agent/README.md):
- The Gym configurations now support the full Harbor spec same as the original Pydantic models (see `HarborAgentConfig` in [app.py](./app.py)), allowing a direct translation from a `harbor run` CLI/config to Gym config, without guesswork.
  - New Harbor datasets can be directly used by using appropriate keys in `harbor_dataset` config key, which maps directly to the `harbor.models.job.config:DatasetConfig` specification.
- All parsing of the Harbor trajectory into a Gym rollout trajectory now uses Harbor APIs instead of custom dictionary parsing.
- Reasoning text is correctly handled via `NeMoGymResponseReasoningItem` objects.
- A minimum harbor version sets to `0.20.0` in [pyproject.toml](./pyproject.toml) with a supporting lockfile.

> [!caution]
> Using this agent for training is currently not supported because underlying Harbor agents may not return
> token ID information. Adding this support is currently work in progress.
