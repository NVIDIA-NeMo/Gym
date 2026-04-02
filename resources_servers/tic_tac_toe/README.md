# Tic-Tac-Toe

A tic-tac-toe game environment designed for use with the [Multi-Turn Agent](../../responses_api_agents/multi_turn_agent/README.md).

## How It Works

The policy model plays as **X** and the user model plays as **O**, both using the `make_move` tool. The server tracks whose turn it is and checks for a winner or draw after each move.

By default X goes first, but the JSONL data can specify `initial_moves` in `verifier_metadata` to pre-populate the board — for example, having O open with a move so the policy must respond to the opponent's opening. This lets training data mix both first-mover and second-mover scenarios without any server or agent changes.

This demonstrates the multi-turn agent's key feature: two LLMs interacting through tool calls and natural language, with the resources server managing shared game state.

### Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `make_move` | `position: int` (0-8) | Place the current player's mark at the given position |
| `get_board` | none | Return the current board state |

Both the policy model and user model call the same `make_move` endpoint. The server determines which mark (X or O) to place based on turn order.

### Board Layout

Positions are numbered 0-8, left to right, top to bottom:

```
0 | 1 | 2
-----------
3 | 4 | 5
-----------
6 | 7 | 8
```

### Prompts

The **policy model's** system prompt lives in the JSONL data as the first message in `responses_create_params.input` (role `developer`). It tells the policy to play as X and use `make_move`. Each JSONL row can have a different prompt.

The **user model's** system prompt lives in the YAML config as `user_model_system_prompt`. It tells the user LLM to play as O and think strategically. This applies to all episodes.

### Rewards

| Outcome | Reward |
|---------|--------|
| X wins | 1.0 |
| Draw | 0.5 |
| O wins | 0.0 |
| Incomplete | 0.0 |

## Running

```bash
ng_run "+config_paths=[resources_servers/tic_tac_toe/configs/tic_tac_toe.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```
