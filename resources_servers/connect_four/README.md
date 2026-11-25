# Connect Four Env

Multi-step gymnasium-style environment. The model plays X on a 6×7 board and drops one disc per turn with an `<action>N</action>` tag (N = column 1–7). After every model move, a scripted opponent replies.

Game state is managed per session.

## Opponent

The opponent is a depth-limited negamax search with alpha-beta pruning and a simple line-counting heuristic. Ties between equally good moves are broken by the row's seeded RNG.

| `opponent_depth` | Strength |
|---|---|
| 0 | Uniformly random legal moves |
| 1 | Takes an immediate win when one exists |
| 2 (default) | Also blocks the model's immediate wins |
| 3–6 | Progressively stronger lookahead (depth 6 takes ~0.1 s per move) |

## Task data

| Field | Default | Meaning |
|---|---|---|
| `seed` | unseeded | Seeds the opponent's random moves and tie-breaks. A fixed seed makes the opponent deterministic. |
| `opponent_depth` | 2 | Search depth in plies (0–6). Other values are rejected with HTTP 400 at reset. |
| `model_first` | `true` | Whether the model opens. When `false`, the first observation already shows the opponent's opening disc. |
| `game_id` | — | Provenance only. |

`data/example.jsonl` holds five games covering depths 0–4 with both move orders. Use `num_repeats` to play each configuration several times.

## Reward

| Outcome | Reward | `info.result` |
|---|---|---|
| Model connects four | +1 | `win` |
| Board fills up | 0 | `draw` |
| Opponent connects four | -1 | `loss` |
| Illegal move | -1 | `invalid_move` |

An illegal move ends the game immediately. The reason is recorded in `info.invalid_reason`:

- `no_action_tag`: no `<action>` tag in the reply.
- `multiple_actions`: tags naming different columns. Ambiguous replies are never treated as a move.
- `not_a_column`: not an integer from 1 to 7.
- `column_full`: the column has no empty cell.

Terminal `info` also includes `moves` (all columns played, 1-based, in order), `model_first`, `opponent_depth` and the final `board`.

## Run

```bash
gym env start \
    --resources-server connect_four \
    --model-type vllm_model
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent connect_four_gymnasium_agent \
    --input resources_servers/connect_four/data/example.jsonl \
    --output results/connect_four_rollouts.jsonl
```

## Smoke test

`data/example_rollouts.jsonl` holds one pass over the five example games played by `gemma3:12b` (served locally through Ollama, temperature 1.0, 1024 max output tokens). Every game ran to a terminal state with a well-formed `<action>` tag on every turn; no game ended in `invalid_move`.

| Game | `opponent_depth` | `model_first` | Result |
|---|---|---|---|
| 1 | 3 | false | loss |
| 2 | 1 | false | loss |
| 3 | 4 | true | loss |
| 4 | 2 | true | loss |
| 5 | 0 | true | loss |

Mean reward -1.0. A 12B model with no reasoning budget does not beat even the random opponent, so the environment has headroom for stronger policies.
