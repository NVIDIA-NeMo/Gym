# SkillsBench (via BenchFlow)

[SkillsBench](https://github.com/benchflow-ai/skillsbench) is an agentic benchmark. This benchmark
definition runs it through the [`benchflow_agent`](../../responses_api_agents/benchflow_agent), which
wraps the [BenchFlow](https://github.com/benchflow-ai/benchflow) library: BenchFlow installs the
OpenHands harness into a per-task **Singularity/Apptainer** sandbox and lets it drive the model. One
NeMo Gym `/run` request evaluates one SkillsBench task. Evaluation-only.

## Layout

- `prepare.py` — clones SkillsBench at a pinned commit into `data/skillsbench_repo/` (the agent's
  runtime `tasks_dir`) and writes one JSONL row per task (`instance_id = skillsbench::<task>`).
- `config.yaml` — inherits the base `benchflow_agent` and sets `tasks_dir`, `images_dir`, and
  `task_config_overrides` (128 GB memory + unbounded agent timeout), plus the benchmark dataset.

## Prerequisites

- Install the `benchflow_agent` requirements (the BenchFlow fork pinned in its `requirements.txt`).
- Apptainer/Singularity on `PATH`.
- Prebuilt per-task `.sif` images named `<task>.sif`. Set `images_dir` in `config.yaml` (or override it
  at launch) to that directory. (Alternatively, set `images_dir: null` and let each task's `task.toml`
  `docker_image` be used directly.)
- A model server config (e.g. a vLLM model server) merged in at run time.

## Usage

```bash
# 1) Clone SkillsBench + write the dataset JSONL.
ng_prepare_benchmark "+config_paths=[benchmarks/skillsbench/config.yaml]"

# 2) Start the agent + model servers (point images_dir at your prebuilt .sif dir).
ng_run "+config_paths=[benchmarks/skillsbench/config.yaml, responses_api_models/<vllm>/configs/<vllm>.yaml]"

# 3) Collect rollouts (each row runs one task through BenchFlow).
ng_collect_rollouts +agent_name=skillsbench_benchflow_agent \
  +input_jsonl_fpath=benchmarks/skillsbench/data/skillsbench_benchmark.jsonl \
  +output_jsonl_fpath=results/skillsbench.jsonl

# 4) Inspect.
ng_viewer +jsonl_fpath=results/skillsbench.jsonl
```

The scalar reward per task comes from BenchFlow's verifier (`RolloutResult.rewards["reward"]`). Full
per-task artifacts (trajectory, agent/verifier logs) are written under the agent's `jobs_dir`.
