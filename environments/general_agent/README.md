# general-agent

Prime Intellect Environments Hub `general-agent` solver environment, run through
`verifiers_agent`. Multi-turn tool use over a local task database.

## Run

```bash
gym env start --environment general_agent --model-type vllm_model
gym eval run --agent general_agent \
    --input environments/general_agent/data/example.jsonl \
    --output results/general_agent.jsonl \
    --limit 5
```

No prepare step. Tasks ship inside the installed package.

## Notes

First load maps roughly 4400 tasks and takes a few minutes.

Requirements pin `general-agent==0.1.3`. Version 0.1.4 ships no `tasks/` data,
so it fails with `Tasks directory not found`.

`backend: local` needs no sandbox. The `opencode` and `rlm` backends need Prime
sandbox provisioning and are not covered here.

## Metrics

`score`, `verify`, `db_hash`, `num_turns`, `total_tool_calls`.
