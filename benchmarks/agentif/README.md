# AgentIF (gym-native)

[AgentIF](https://github.com/THU-KEG/AgentIF) is an **agentic
instruction-following** benchmark: 707 real-world scenarios, each carrying a
list of natural-language constraints the model's response must satisfy.

This entry runs AgentIF through the **gym-native** eval path against the
**whole** dataset. Scoring, aggregation, and the constraint checkers all live in
the `agentif` resources server (`resources_servers/agentif/`); this benchmark
only supplies data and wiring, chaining to
`resources_servers/agentif/configs/agentif.yaml`.

## Metrics

- **ISR** (Instruction Success Rate) — fraction of rows where *every* scored
  constraint passed (all-or-nothing per row); returned per row as the reward,
  so the report's final reward is ISR.
- **CSR** (Constraint Success Rate) — fraction of scored constraints that
  passed; reported corpus-wide.

Per-dimension (`vanilla` / `condition` / `example`) and per-type (`formatting` /
`semantic` / `tool`) accuracy breakdowns are computed by the resources server's
`compute_metrics`; `isr`, `csr`, and `mean_reward` are the headline keys.

## Prepare data

```bash
gym eval prepare --benchmark agentif
```

Reuses the resources server's `prepare_agentif.build_row` to write the whole
707-row dataset to `data/agentif_benchmark.jsonl`.

## Running servers

Constraint scoring uses an LLM judge (recommended: `gpt-4o-mini`); supply its
credentials at launch.

```bash
gym env start \
    --model-type vllm_model \
    --benchmark agentif \
    +judge_base_url=... +judge_api_key=... +judge_model_name=...
```

## Collecting rollouts and scoring

```bash
gym eval run --no-serve \
    --agent agentif_benchmark_simple_agent \
    --input benchmarks/agentif/data/agentif_benchmark.jsonl \
    --output results/agentif_rollouts.jsonl \
    --num-repeats 1
```
