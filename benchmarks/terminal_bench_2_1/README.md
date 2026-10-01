# terminal_bench_2_1

TODO: Describe this benchmark and replace the sample data.

- Integration profile: `custom-gym-verifier`
- Scorer: `terminal_bench_2_1`

## Pi in a task sandbox

Compose the independent benchmark, Pi, and single-agent Environment Server configs.
Pi borrows the task sandbox; Resources owns it. The Environment Server closes Pi before
verification, then closes Resources, including when agent setup or execution fails.
Existing OpenCode and Terminus2 recipes keep their legacy seed/verify paths.

Save as `run.yaml` in the Gym checkout:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/pi_agent/configs/pi_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml
  - nemo_gym/sandbox/providers/docker/configs/docker.yaml
  - responses_api_models/openai_model/configs/openai_model.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: terminal_bench_2_1_resources_server
      agent_server:
        name: pi_agent

pi_agent:
  responses_api_agents:
    pi_agent:
      thinking: high
      max_output_tokens: 32768
      timeout: 600

terminal_bench_2_1_resources_server:
  resources_servers:
    terminal_bench_2_1:
      evaluation_timeout: 300
```

These are single-task smoke limits, not full-benchmark settings. Put `policy_base_url`,
`policy_api_key`, and `policy_model_name` in your private `env.yaml`. The task container
must reach the Gym model proxy; use `++use_absolute_ip=true` for Docker on a reachable host.
Both servers use one worker because sessions are process-local.

Prepare the pinned benchmark checkout and select one task:

```bash
git clone https://github.com/harbor-framework/terminal-bench-2-1 \
  benchmarks/terminal_bench_2_1/terminal-bench-2-1
git -C benchmarks/terminal_bench_2_1/terminal-bench-2-1 \
  checkout --detach 7131e4375048a0e408a8fb404b5f499d726b695b
python benchmarks/terminal_bench_2_1/prepare.py
mkdir -p results
python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path("benchmarks/terminal_bench_2_1/data/benchmark.jsonl").read_text().splitlines()]
rows = [row for row in rows if row["task_name"] == "terminal-bench/regex-log"]
assert len(rows) == 1
assert (Path(rows[0]["task_folder"]) / "tests/test.sh").is_file()
Path("results/pi-tb21-input.jsonl").write_text(json.dumps(rows[0]) + "\n")
PY
gym env start --config run.yaml ++use_absolute_ip=true
```

In a second terminal, in the same checkout and Python environment:

```bash
gym eval run --no-serve --config run.yaml ++use_absolute_ip=true \
  --agent pi_agent --input results/pi-tb21-input.jsonl \
  --output results/pi-tb21-rollouts.jsonl --limit 1 --num-repeats 1 --concurrency 1
```

Keep task tests/solutions on the host; do not mount the benchmark checkout into the task
sandbox. Only verifier assets are uploaded, after Pi closes. Native seed fails early if
the host task's `tests/test.sh` is missing. Do not enable golden-patch mode for model runs.
Pi installs its private Node/Pi runtime and, if absent, Python 3 and curl using apt/apk as root;
non-root images must contain bootstrap dependencies. Existing task runtimes are not replaced.
See [Pi requirements and lifecycle](../../responses_api_agents/pi_agent/README.md).

Inspect the rollout's `response.metadata.harness_execution` (`sandbox`), nonempty output,
`ng_agent_observations`, `evaluation_completed`, reward, and verifier `test_output`.
Also inspect failure sidecars and confirm sandbox teardown. An incomplete evaluation with
reward 0 is an infrastructure failure, not a valid model score. A successful collector exit
alone does not establish a passing smoke; one task does not establish a benchmark baseline.
