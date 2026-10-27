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

      # Per-job worker variables; kept out of Ray runtime_env logs.
      harbor_worker_env: {}

      # Native harbor.models.job.config.JobConfig fields.
      job:
        jobs_dir: ${harbor_jobs_dir}
        datasets:
          - path: ${harbor_dataset_path}
        agents:
          - name: opencode
            model_name: ${policy_model_name}
        environment:
          type: docker
        timeout_multiplier: 2.0
        verifier_timeout_multiplier: 3.0
        retry:
          max_retries: 3
```

`job` is validated by the installed Harbor `JobConfig` model. Use Harbor's names and nested structures directly;
options such as retry policy, artifacts, extra instructions, and environment settings do not need individual Gym
forwarding fields. See [configs](./configs) for specific examples.

Gym runs one Harbor trial for each input row. These constraints are checked when the configuration loads:

- Exactly one entry in `datasets` and one in `agents`.
- `n_attempts` and `n_concurrent_trials` must be `1` (Gym's defaults). Configure overall concurrency and repeats in Gym.
- Omit `job_name` and dataset `task_names`, `exclude_task_names`, and `n_tasks`. Gym assigns a job name from the
  task/rollout indices and selects the task from the row's `task_name`.
- Leave `tasks`, `source_jobs`, and `install_only` unset; they bypass Gym's per-row task selection or execution.
- `environment.delete` must remain `true` so each rollout cleans up its environment.

All other options use Harbor's defaults, except `quiet`, which defaults to `true` and can be overridden.
`jobs_dir` is resolved to an absolute path; a `.jsonl` path still selects its sibling `harbor` directory.
The existing path-resolution and unique-job-name options described below retain their behavior.
The Gym-specific `harbor_ray_task_num_cpus`, `harbor_reward_key`, `harbor_worker_env`,
`harbor_resolve_paths_from_gym_root`, and `harbor_unique_job_names` remain outside `job`.
Worker environment variables are passed as task arguments and installed only for one Harbor Ray task, rather than
stored in Ray runtime environment setup logs.

The legacy fields listed below remain accepted for compatibility, but cannot be mixed with `job`.
To migrate, move their values under `job` using this mapping:

| Existing field | Native field under `job` |
| --- | --- |
| `harbor_jobs_dir`, `harbor_debug` | `jobs_dir`, `debug` |
| `harbor_max_retries` | `retry.max_retries` |
| `harbor_dataset`, `harbor_agent` | `datasets`, `agents` (single-element lists) |
| `harbor_environment`, `harbor_verifier` | `environment`, `verifier` |

The execution constraints above apply to both forms. Previously ignored conflicting settings now raise an error.

> [!important]
> The bridge converts Harbor's `agent/trajectory.json`, so the selected Harbor
> agent must declare `capabilities.atif=True` and actually emit that file. Harbor
> can run additional native agents, such as Pi, that do not emit ATIF; those
> trials are not currently convertible into Gym rollouts by this bridge. Harbor
> also has no generic image-capability flag, so local-image support must be
> validated per agent and model provider.

To inspect the exact agents and ATIF capabilities in the installed Harbor
version:

```bash
python - <<'PY'
from harbor.agents.factory import AgentFactory
from harbor.models.trial.config import AgentConfig

for name in AgentFactory.registered_names():
    cls = AgentFactory.get_agent_class_from_config(AgentConfig(name=name))
    print(f"{name}: atif={cls.capabilities.atif}")
PY
```

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

Relative job paths resolve from the agent process's working directory, and
dataset paths are passed to Harbor unchanged. Built-in profiles can opt into
repository-root resolution with `harbor_resolve_paths_from_gym_root: true`;
absolute paths remain unchanged.

Job names default to `t<task>-r<rollout>` for resuming trials. Set
`harbor_unique_job_names: true` to append a fresh UUID for every rollout
request, including retries, and prevent independent collections from colliding
or reusing an old result. Prior job directories are preserved. LabBench2 enables
both options; the generic Docker and Singularity configs retain their defaults.

A few notable details make it more general than the [older implementation](../harbor_agent/README.md):
- The Gym configurations now support the full Harbor spec same as the original Pydantic models (see `HarborAgentConfig` in [app.py](./app.py)), allowing a direct translation from a `harbor run` CLI/config to Gym config, without guesswork.
  - New Harbor datasets can be directly used by using appropriate keys in `job.datasets`, which maps directly to the `harbor.models.job.config:DatasetConfig` specification.
- All parsing of the Harbor trajectory into a Gym rollout trajectory now uses Harbor APIs instead of custom dictionary parsing.
- Reasoning text is correctly handled via `NeMoGymResponseReasoningItem` objects.
- The minimum Harbor version is `0.24.0` in [pyproject.toml](./pyproject.toml) with a supporting lockfile.

> [!caution]
> Using this agent for training is currently not supported because underlying Harbor agents may not return
> token ID information. Adding this support is currently work in progress.

### ATIF Conversion Contract

The source `trajectory.json` remains the lossless, authoritative ATIF artifact. The Gym rollout contains the closest
Responses-compatible representation and an `atif_conversion` object with source paths, trajectory identifiers, and
warnings for every conversion that is lossy or cannot be represented natively. Conversion limitations do not fail a
rollout.

The converter applies these rules:

- Scalar user, system, and assistant messages, function calls, and scalar tool outputs are converted directly and
  retain their trajectory order.
- User and system multimodal content is converted to native `input_text` and `input_image` parts when images use
  portable URLs.
- Assistant multimodal content has no native Responses output-message representation, so the complete ATIF content
  array is serialized as JSON text and documented with a warning.
- Multimodal tool outputs are converted to `input_text` and `input_image` parts when images use HTTP, HTTPS, or data
  URLs. Content containing local image paths is serialized as JSON because local paths are not portable Gym URLs.
- Missing tool-call IDs receive deterministic synthetic IDs and a warning because Gym requires a call ID.
- Missing tool output content is represented as empty text and documented.
- Unsupported ATIF metadata, continuation references, and subagent trajectories remain available in the source ATIF
  artifact and are listed in the conversion warnings.
- Training metadata is emitted only when prompt IDs, non-empty completion IDs, and aligned log probabilities describe
  one attributable LLM output. Copied context, multi-call steps, partial metadata, and ambiguous reasoning/tool-call
  steps use ordinary output messages and record why training metadata was omitted.

Example rollout metadata:

```json
{
  "atif_conversion": {
    "lossless": false,
    "warnings": ["step 2: multimodal message serialized as JSON because ..."],
    "source_trajectory_paths": ["/path/to/trial/agent/trajectory.json"],
    "trajectories": [
      {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-id",
        "trajectory_id": "trajectory-id",
        "agent_name": "opencode",
        "agent_version": "1.0",
        "agent_model_name": "provider/model"
      }
    ]
  }
}
```

### SkillsBench

`benchmarks/skillsbench/harbor.yaml` and `harbor_opensandbox.yaml` use
`skillsbench_app.py`, a thin subclass of this bridge. It supplies the Gym model
server URL to the worker-side OpenHands harness and passes the named sandbox
provider through `harbor_worker_env`, keeping credentials out of Harbor's
retained configuration. Scheduling, job execution, scoring, and ATIF conversion
use this general adapter. See [SkillsBench](../../benchmarks/skillsbench/README.md)
for preparation and execution.
