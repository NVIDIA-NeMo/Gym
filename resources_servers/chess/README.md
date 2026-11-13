# Chess resources server

Gymnasium-style chess, driven by `gymnasium_agent`. The model plays one side against a built-in opponent, one
`make_move` call per turn, and is scored on the result: +1 win, 0 draw, -1 loss.

The runnable environment (agent wiring, data, `prepare.py`) is [`environments/chess`](../../environments/chess/README.md).

## Rules

`chess_rules.py` implements standard chess: castling, en passant, promotion, check, checkmate, stalemate, threefold
repetition, the fifty-move rule, and insufficient material. Positions are FEN and moves are UCI. Threefold repetition
and the fifty-move rule end the game without a claim.

It is written here instead of depending on [python-chess](https://github.com/niklasf/python-chess), which is GPL-3.0.
The tests check move generation against the published [perft counts](https://www.chessprogramming.org/Perft_Results)
for six reference positions.

## Episode

- `/reset` sets up the task's position; if the opponent is to move, it moves first. The response advertises `/close`,
  which `gymnasium_agent` calls to release games it stops early.
- Each `/step` plays the first function call of the response. Any further calls get an error and are ignored.
- The step that ends the game returns `terminated: true` and the reward. The session is then released, so a later
  `/step` returns 404.

**Action**: `make_move` with `{"move": "<uci>"}`, such as `e2e4`, `e1g1` (castling), or `e7e8q` (promotion).

**Invalid actions**: a malformed or illegal move, another tool, bad arguments, or a reply with no tool call. The
position does not change, and the model is told why (for example `e1g1 is illegal: cannot castle because the king
would pass through or land on an attacked square`). A legal move resets the count; more than
`max_consecutive_invalid_actions` in a row forfeits the game.

## Observation

The reset observation and every tool output are one JSON object:

| Field | Meaning |
| --- | --- |
| `your_move`, `opponent_move` | The moves just played. |
| `error` | Why the last action was invalid, and how many more in a row are allowed. |
| `status` | `ongoing`, `checkmate`, `stalemate`, `insufficient_material`, `fifty_moves`, `threefold_repetition`, or `forfeit`. |
| `you_play`, `turn` | The model's color and the side to move. |
| `in_check` | Whether the side to move is in check. |
| `fen`, `board` | The position as FEN and as text, White at the bottom (uppercase White, lowercase Black, `.` empty). |
| `castling_rights`, `en_passant` | `KQkq` subset or `-`; the en passant square or `null`. |
| `halfmove_clock`, `fullmove_number` | The FEN move counters. |
| `recent_moves` | The last 10 moves, numbered (`12. e2e4`, `12... e7e5`). |
| `legal_moves` | The model's legal moves, sorted; empty once the game is over. |

## Reward

Only the final step has a reward, from the model's side: 1.0 for checkmating the opponent; 0.0 for stalemate,
insufficient material, fifty moves, or threefold repetition; -1.0 for being checkmated or forfeiting. Its `info` holds
`result`, `termination`, `agent_moves`, `invalid_actions`, and `final_fen`. A game cut off at the agent's `max_steps`
is truncated with reward 0.

## Opponents

| `opponent` | Plays |
| --- | --- |
| `random` | A random legal move. |
| `greedy` | A mate in one, else the most valuable capture, else a random move. |
| `lookahead` | A two-ply material search: a mate in one, else the move whose worst-case reply leaves it the most material. |

`lookahead` beats `greedy` in about 95% of games and takes about 5 ms per move. All three break ties with a
`random.Random(seed)` owned by the episode, so the same task and the same model moves always get the same replies.

## Task fields

All optional; `task_data.py` is the schema, and `/reset` returns HTTP 400 for rows that do not match it.

| Field | Default | Meaning |
| --- | --- | --- |
| `fen` | initial position | Starting position (six-field FEN). Rejected if malformed, illegal, or already over. |
| `agent_color` | `white` | The model's side. |
| `opponent` | `random` | `random`, `greedy`, or `lookahead`. |
| `seed` | `0` | Seed for the opponent's choices. |
| `max_consecutive_invalid_actions` | `3` | Invalid actions allowed in a row before the next one forfeits. |

## Example

The model plays Black against `random` with seed 0. `/reset` returns:

```json
{"opponent_move": "e2e3", "status": "ongoing", "you_play": "black", "turn": "black", "in_check": false,
 "fen": "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
 "board": ["8 r n b q k b n r", "7 p p p p p p p p", "6 . . . . . . . .", "5 . . . . . . . .",
           "4 . . . . . . . .", "3 . . . . P . . .", "2 P P P P . P P P", "1 R N B Q K B N R", "  a b c d e f g h"],
 "castling_rights": "KQkq", "en_passant": null, "halfmove_clock": 0, "fullmove_number": 1,
 "recent_moves": ["1. e2e3"], "legal_moves": ["a7a5", "a7a6", "b7b5", "...", "h7h6"]}
```

`make_move` with `e7e5` returns reward 0 and a tool output with `"your_move": "e7e5", "opponent_move": "f2f4"` and the
new position. An illegal `e5e3` returns `"error": "e5e3 is illegal: the pawn on e5 cannot move to e3. Invalid actions
in a row left before you forfeit: 2."` and the unchanged position. From `6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1`, the
mate `d1d8` ends the episode:

```json
{"observation": null, "reward": 1.0, "terminated": true, "truncated": false,
 "info": {"result": "win", "termination": "checkmate", "agent_moves": 1, "invalid_actions": 0,
          "final_fen": "3R2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1"}}
```

## Tests

```bash
gym env test --resources-server chess
```

## Licensing

Code: Apache-2.0. Data: generated by `environments/chess/prepare.py`; no external dataset.
