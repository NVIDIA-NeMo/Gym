# ToolSandbox

[ToolSandbox](https://github.com/apple/ToolSandbox) is a multi-turn, stateful tool-use benchmark from Apple that evaluates how well a model interacts with a simulated user while issuing Python tool calls against a stateful environment (contacts, messaging, reminders, device settings). Each scenario is scored against milestone / minefield snapshots; the reward is milestone **similarity** in `[0, 1]` (0 if a minefield is hit).

## Configuration

- **Grading mode**: similarity — reward is a float in `[0, 1]`
- **Resources server**: `toolsandbox` (multi-turn stateful; runs user simulator and Python execution environment)
- **Agent**: `toolsandbox_agent` (multi-turn harness that drives the policy model)

## Prepare data

```bash
gym eval prepare --benchmark toolsandbox
```

## Run

```bash
gym eval run \
    --benchmark toolsandbox \
    --model-type openai_model \
    --output results/benchmarks/toolsandbox.jsonl \
    --split benchmark \
    --model-url <> \
    --model-api-key <> \
    --model <> \
    --resume \
    ++overwrite_metrics_conflicts=true \
    ++reuse_existing_data_preparation=true
```

## Scoring caveat: Responses API vs Chat Completions

See `resources_servers/toolsandbox/README.md` for a detailed explanation of why
gym scores are ~0.08 lower than Apple's published baseline. All models are
scored through the same Responses-API harness, so cross-model comparisons are
apples-to-apples — only the absolute number differs from Apple's reference.
