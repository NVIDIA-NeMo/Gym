# BenchFlow Agent for NeMo Gym

[BenchFlow](https://github.com/benchflow-ai/benchflow) is a framework for running various agentic benchmarks, most notably [SkillsBench](https://github.com/benchflow-ai/skillsbench). It supports running multiple agent harnesses via the ACP protocol, such as OpenHands and OpenCode. This environment integrates BenchFlow into NeMo-Gym with support for running per-task environments through Singularity/Apptainer.

## Overview

BenchFlow takes care of managing the task environments throughout the whole rollout lifecycle: installing the agent harness inside of the environment, running the harness, verifying the solution and obtaining the reward. This agent server exposes a single `/run` endpoint that accepts a task name, runs the entire process using the BenchFlow Python API, and returns a NeMo-Gym response.

Upstream BenchFlow supports Docker, Daytona and Modal environments. We use a [fork of BenchFlow](https://github.com/ludwig-n/benchflow/tree/gym) that adds support for Singularity/Apptainer environments. The implementation is based on Harbor's [SingularityEnvironment](https://github.com/harbor-framework/harbor/tree/main/src/harbor/environments/singularity) (launches a FastAPI server inside of the container to run commands) but with some modifications to make it work with BenchFlow. Notably, BenchFlow requires a continuous channel of communication with the agent harness process via the ACP protocol, which we implement via [`ncat`](https://nmap.org/ncat/).

## Quickstart: SkillsBench evaluation in Singularity

1. Install dependencies. To use Singularity environments, this agent requires Apptainer (recommended version: 1.5.1 or above) and the `ncat` utility. You will also need `git` to clone the SkillsBench repo.
    ```
    apt update
    apt install -y git wget ncat
    wget https://github.com/apptainer/apptainer/releases/download/v1.5.1/apptainer_1.5.1_amd64.deb
    apt install -y ./apptainer_1.5.1_amd64.deb
    ```

2. Prepare the SkillsBench dataset:
    ```
    gym eval prepare --benchmark skillsbench
    ```
    This command will prepare the following:
    - the SkillsBench tasks folder at `benchmarks/skillsbench/data/skillsbench_repo/tasks`,
    - an input .jsonl file at `benchmarks/skillsbench/data/skillsbench_benchmark.jsonl`.

3. Prepare prebuilt per-task containers in .sif format.

4. Spin up a local vLLM server. Note that the LLM must have a tool call parser configured in order to run on agentic benchmarks: `--enable-auto-tool-choice --tool-call-parser <PARSER_NAME>`. For more details, see [vLLM docs](https://docs.vllm.ai/en/stable/features/tool_calling/).

5. Run evaluation with the following command, replacing `<...>` and the `container_formatter` path as needed:
    ```
    gym eval run \
        --model-type vllm_model \
        --model-url http://127.0.0.1:<VLLM_PORT>/v1 \
        --model-api-key EMPTY \
        --model <VLLM_SERVED_MODEL_NAME> \
        --benchmark skillsbench \
        --split benchmark \
        --output results/skillsbench.jsonl \
        "++skillsbench_benchflow_agent.responses_api_agents.benchflow_agent.container_formatter='/path/to/containers/{task_name}.sif'"
    ```
    This will spin up the NeMo-Gym server and start collecting rollouts. Results will appear in the following folders:
    - `responses_api_agents/benchflow_agent/jobs/`: logs, artifacts and results written by BenchFlow.
    - `results/`: collected rollouts and rewards in NeMo-Gym format.

    Note that the path to the .sif containers is specified with a `{task_name}` placeholder. For each task, NeMo-Gym will replace it with the corresponding task name at runtime.

## Customizing the configuration

Several parameters can be customized, such as the agent harness, skill availability, etc. For a list of all available parameters, see [configs/benchflow_agent.yaml](configs/benchflow_agent.yaml). This config is then inherited and specialized for SkillsBench in [benchmarks/skillsbench/config.yaml](../../benchmarks/skillsbench/config.yaml).

To use your custom parameters, you can either edit [benchmarks/skillsbench/config.yaml](../../benchmarks/skillsbench/config.yaml) adding your overrides, or pass them as CLI arguments to `gym eval run`, like `container_formatter` in the example above.

### Task configuration

In the BenchFlow task format, each task has its own config that lives in `task.md`. You can customize these configs as follows.
- **Override a parameter for all tasks:** we expose a `task_config_overrides` option that can automatically override a particular parameter to a constant value for all tasks. Internally this is implemented by creating a temporary copy of the task folder and editing its `task.md`.
- **Override a parameter for only some tasks:** you will need to edit the respective `task.md` files yourself before running `gym eval run`. In particular, if you want to use per-task .sif paths that cannot be expressed using the `container_formatter` option, you can edit `task.md` to set `environment.docker_image` to the raw .sif path directly.
