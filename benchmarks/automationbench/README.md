# AutomationBench on Verifiers V1

This integration runs AutomationBench tasks through the shared
`responses_api_agents/verifiers_agent` component on the stable Verifiers V1 task,
tool-server, trace, and scoring lifecycle. Each rollout gets an isolated
AutomationBench `WorldState`; reward is assertion-level `partial_credit`, and
`task_completed_correctly` is `1.0` only when every assertion passes.

## Public benchmark

`config.yaml` loads the six public scored domains. Materialize the full benchmark,
start the configured model server, and run an evaluation with:

```bash
python benchmarks/automationbench/prepare.py --force

gym env start \
  --config benchmarks/automationbench/config.yaml \
  --model-type vllm_model

gym eval run --no-serve \
  --agent automationbench_benchmark_agent \
  --input benchmarks/automationbench/data/automationbench_benchmark.jsonl \
  --output results/automationbench-v1.jsonl \
  --split benchmark \
  --limit 1
```

The committed `data/example.jsonl` is a five-row smoke subset of the same taskset.

## Path-provided V1 tasks

`config_training.yaml` loads AutomationBench-shaped `task_*.json` files. Stage and
export a local corpus without copying it into Git:

```bash
python benchmarks/automationbench/prepare.py \
  --tasks-dir /Users/dmolinari/Downloads/prime-forge-training-53-v1/tasks

gym env start \
  --config benchmarks/automationbench/config_training.yaml \
  --model-type vllm_model

gym eval run --no-serve \
  --agent automationbench_benchmark_agent \
  --input benchmarks/automationbench/data/v1_tasks.jsonl \
  --output results/automationbench-v1-tasks.jsonl \
  --split train \
  --limit 1
```

The prepare step creates an ignored `data/tasks` symlink and an ignored
`data/v1_tasks.jsonl`. It never replaces a real directory, and it will not replace a
symlink pointing elsewhere unless `--force` is explicit. On a cluster, stage the task
directory at the same configured path or rerun the prepare command there.

Rows refer to V1 tasks by `task_idx`. The exporter and agent must therefore use the
same task directory, file ordering, and taskset config. The loader sorts `task_*.json`
paths before assigning indices.

## Dependency override

The Gym project constrains OpenAI for its head environment, while Verifiers V1 needs
the newer Responses API `ContextManagement` type. This component's `overrides.txt`
pins `openai==2.54.0`; Gym detects that file and installs the component with
`uv pip install --override overrides.txt`. The component test suite checks both the
installed distribution version and the required import.

The AutomationBench schema and simulated API tools are pinned to commit `6f0e683` in
the component requirements. The supplied task JSON is internal data and is not tracked.

## Metrics

- `reward` / `partial_credit`: fraction of assertions satisfied.
- `task_completed_correctly`: `1.0` only when partial credit is exactly `1.0`.

Code is Apache 2.0. Public benchmark data follows the upstream AutomationBench terms;
path-provided training tasks use the license declared in `config_training.yaml`.
