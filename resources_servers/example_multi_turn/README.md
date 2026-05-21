# Example Multi-Turn (Tic-Tac-Toe)

A tic-tac-toe game environment designed for use with the [Multi-Turn Agent](../../responses_api_agents/multi_turn_agent/README.md). Demonstrates multi-turn interaction between a policy model and an LLM user model through a shared game environment.

## How It Works

The policy model plays as **X** and the user model plays as **O**, both using the `make_move` tool. The server tracks whose turn it is and checks for a winner or draw after each move.

By default X goes first, but the JSONL data can specify `initial_moves` in `verifier_metadata` to pre-populate the board — for example, having O open with a move so the policy must respond to the opponent's opening. This lets training data mix both first-mover and second-mover scenarios without any server or agent changes.

### Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `make_move` | `position: int` (0-8) | Place the current player's mark at the given position |

Both the policy model and user model call the same `make_move` endpoint. The server determines which mark (X or O) to place based on turn order. The response includes the full board state, game-over status, and winner, so no separate observation tool is needed.

The config sets `max_steps_per_turn: 1` so each player makes exactly one move per turn. This prevents either model from making multiple `make_move` calls in a single turn (which would inadvertently place marks for both sides, since the server alternates marks).

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

The **user model's** system prompt lives in the YAML config as `user_model_system_prompt`. It tells the user LLM to play as O and think strategically. This applies to all episodes. The config also sets `user_model_tool_choice: required` to ensure the user model calls `make_move` rather than just describing its move in text, and `user_model_stop_token: '"game_over":true'` to stop the conversation when the game ends (the `make_move` response JSON contains this substring when the game is over).

### Rewards

| Outcome | Reward |
|---------|--------|
| X wins | 1.0 |
| Draw | 0.5 |
| O wins | 0.0 |
| Incomplete | 0.0 |

## Usage

```bash
ng_prepare_data "+config_paths=[resources_servers/example_multi_turn/configs/example_multi_turn.yaml]" \
    +output_dirpath=resources_servers/example_multi_turn/data \
    +mode=example_validation \
    "+user_model={responses_api_models: {dummy_model: {entrypoint: app.py}}}" \
    +user_base_url="" +user_api_key="" +user_model_name=""
```

```bash
ng_run "+config_paths=[resources_servers/example_multi_turn/configs/example_multi_turn.yaml,responses_api_models/openai_model/configs/openai_model_with_user.yaml]"
```

```bash
ng_collect_rollouts +agent_name=example_multi_turn_agent \
    +input_jsonl_fpath=resources_servers/example_multi_turn/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/example_multi_turn/data/example_rollouts.jsonl \
    +num_repeats=1 \
    "+responses_create_params={max_output_tokens: 4096, temperature: 0.0}"
```
