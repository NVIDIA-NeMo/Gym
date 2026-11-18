# Wordle

Multi-step gymnasium-style environment. The model guesses a secret 5-letter word in 6 attempts using `submit_guess`, `check_word_validity`, and `get_game_state`. See [`resources_servers/wordle`](../../resources_servers/wordle/README.md) for the reward and word lists.

## Prepare data

Downloads the pinned word lists (needed once before the server starts) and writes `train.jsonl`, `validation.jsonl`, and `example.jsonl` here:

```bash
python environments/wordle/prepare.py
```

## Run

```bash
gym env start --environment wordle --model-type vllm_model
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent wordle_gymnasium_agent \
    --input environments/wordle/data/example.jsonl \
    --output results/wordle_rollouts.jsonl
```
