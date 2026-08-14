# AutomationBench

[Zapier AutomationBench](https://github.com/zapier/AutomationBench) evaluates
agents on multi-step business workflows across CRM, email, calendar and 47
simulated SaaS tools. Run through `verifiers_agent`.

## Run

Five-row example:

```bash
gym env start --benchmark automationbench --model-type vllm_model
gym eval run --agent automationbench_benchmark_agent \
    --input benchmarks/automationbench/data/example.jsonl \
    --output results/automationbench_example.jsonl \
    --limit 5
```

Full 600-task benchmark:

```bash
python benchmarks/automationbench/prepare.py
gym eval run --benchmark automationbench --model-type vllm_model \
    --split benchmark \
    --output results/automationbench.jsonl
```

`prepare.py` materializes the task set from the installed package. Pass
`--force` to regenerate.

## Scope

`domains: all` is the six scored domains (sales, marketing, operations, support,
finance, hr) at 100 tasks each, 600 total. The 200 `simple` tasks are excluded
from the benchmark score. Pass `simple` explicitly to include them.

## Metrics

| metric | meaning |
|---|---|
| `task_completed_correctly` | 1.0 only if every assertion passes. Official pass rate. |
| `partial_credit` | Fraction of assertions satisfied. Denser training signal. |
| `num_turns`, `total_tool_calls` | Agent turns and tool invocations. |
| `api_search_calls`, `api_fetch_calls` | Breakdown for the `api` toolset. |

`reward` equals `partial_credit`, because `task_completed_correctly` carries
weight 0.0. Report `task_completed_correctly` for pass rate, not `reward`.

## Scores

600-task public set, default reasoning effort, one rollout per task, 4096 max
tokens per turn.

| model | pass rate | partial credit |
|---|---|---|
| gpt-5.5 | 31.33% | 0.705 |
| claude-opus-4-7 | 23.62% | 0.626 |
| glm-5.2 | 14.83% | 0.379 |

The leaderboard at zapier.com/benchmarks uses a separate held-out private set
and runs every model at its highest reasoning effort, so these are lower and not
directly comparable. GLM-5.2 is published there at 26.17% against 14.83% here.

## Requirements

Needs `verifiers>=0.3.0`. Published under the `zapier` hub namespace, not
`primeintellect`.

## License

AutomationBench and its task data are provided by Zapier under the MIT License.
See the upstream [copyright and license notice](https://github.com/zapier/AutomationBench/blob/main/LICENSE).
