# Blackjack

Multi-step gymnasium-style environment. 

Model hits or stands using `<action>` tags until the hand ends. Game state managed per session.

Example data provided in `data/example.jsonl` (system prompt only, no verifier_metadata needed). No train/validation data.

## Run

```bash
gym env start --environment blackjack --model-type vllm_model
```

## Data

Each game is generated on the fly during `reset()`, so every row in `example.jsonl` is identical. To create more data, duplicate the row. Each rollout gets a fresh random deal. Use `num_repeats` in the YAML config or the `+num_repeats` CLI flag to control how many games per row.

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent blackjack_gymnasium_agent \
    --input environments/blackjack/data/example.jsonl \
    --output results/blackjack_rollouts.jsonl
```


## Prepare training data

```bash
python environments/blackjack/prepare.py --size 1000
```

Each row in the generated JSONL is identical (same as `example.jsonl`) — a fresh game is dealt on the resources server side per rollout. Use `--size` to control how many rows.
