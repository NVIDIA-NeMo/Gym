# Chess

Multi-step chess against a built-in `random`, `greedy`, or `lookahead` opponent, with UCI moves through a `make_move`
tool. Reward: +1 win, 0 draw, -1 loss or forfeit. See [`resources_servers/chess`](../../resources_servers/chess/README.md)
for the rules, observation, rewards, and task fields.

## Data

`data/example.jsonl` has five tasks: full games as White and as Black, a mate in one, a pawn promotion, and a
queen-versus-king ending. Train and validation games start from the initial position, cycle through both colors and
all three opponents, and use disjoint seeds:

```bash
python environments/chess/prepare.py --split train --size 1000
python environments/chess/prepare.py --split validation --size 100
```

For other positions (openings, endgames, puzzles), write rows with a `fen` using `build_row` from `prepare.py`.

## Run

```bash
gym env start --environment chess --model-type vllm_model
```

```bash
gym eval run --no-serve --environment chess --model-type vllm_model \
    --agent chess_gymnasium_agent \
    --input environments/chess/data/example.jsonl \
    --output results/chess_rollouts.jsonl
```

`max_steps: 100` in `config.yaml` caps model calls (moves plus invalid actions) per game. Long games resend the whole
game each turn, so keep `--concurrency` low enough for the conversations to fit in the model server's KV cache; with
vLLM on one H100 and a 4B model, about 10 kept the prefix-cache hit rate near 98%.

## Training

`--tasks mate1` writes positions where the side to move can checkmate at once, a short task that trains quickly:

```bash
python environments/chess/prepare.py --split train --tasks mate1 --size 3000
python environments/chess/prepare.py --split validation --tasks mate1 --size 256
```

With NeMo RL's `examples/nemo_gym/run_grpo_nemo_gym.py`, Qwen3-4B-Instruct-2507 with LoRA (rank 32, learning rate
2e-5), 16 tasks × 8 rollouts per step, and the agent's `max_steps` set to 4, the validation mate rate rose from 3.9% to
13.7% over 100 steps on one H100.
