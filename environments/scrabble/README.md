# Scrabble

Multi-step gymnasium-style environment. The model finds the highest-scoring legal move for a pinned board and rack and plays it with `play_move`; illegal moves are rejected with the reason and may be retried. Reward is the move's score over the best possible score. See [`resources_servers/scrabble`](../../resources_servers/scrabble/README.md) for the rules, reward, and data generation.

## Prepare data

Downloads the pinned ENABLE word list (needed once before the server starts) and writes `train.jsonl`, `validation.jsonl`, and `example.jsonl` here:

```bash
python environments/scrabble/prepare.py
```

## Run

```bash
gym env start --environment scrabble --model-type vllm_model
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent scrabble_gymnasium_agent \
    --input environments/scrabble/data/example.jsonl \
    --output results/scrabble_rollouts.jsonl
```
