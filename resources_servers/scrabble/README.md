# Scrabble Env

Multi-step, tool-calling Scrabble move-finding environment built on `GymnasiumServer`. Use with `gymnasium_agent`.

Each row pins one position: a board reached by simulated play and the rack of the player to move. The model must find the highest-scoring legal move and play it with the `play_move` tool (`word`, `row`, `col`, `direction`). An illegal move is rejected with the reason (word not in the dictionary, does not connect, tiles not on the rack, and so on) and the model may try again, up to the row's `max_attempts` (3). The episode ends at the first legal move, after the last attempt, or when the model replies without a tool call.

Scrabble is a trademark of Hasbro, Inc. in the US and Canada and of Mattel elsewhere. This environment is an independent implementation of the game's publicly known rules and is not affiliated with or endorsed by either company.

## Reward

- Legal move: `score / best_score - 0.1 * illegal_moves`, clipped to [0, 1]. `best_score` is the row's highest achievable score, found by exhaustive search, so the optimal move scores 1.0.
- No legal move (out of attempts, or no tool call): 0.0.

The penalty per illegal move is the `invalid_attempt_penalty` server config field. Each terminal step reports `score`, `best_score`, `score_ratio`, `optimal`, `legal_move`, `illegal_moves`, `bingo`, and `outcome` in `info`.

Scoring follows the standard rules: letter and word premiums apply only under tiles placed this turn, every word formed (main and cross words) is scored, and placing all 7 tiles adds 50. Blanks (`?`) score 0 and are assigned automatically: the server uses real rack tiles on the squares that count most, which gives the highest score for the placement.

## Rules implementation

`game.py` holds the rules with no dependencies beyond the standard library: board layout, move checking and scoring (`evaluate_move`), and exhaustive move generation (`generate_moves`, the anchor and cross-check algorithm from Appel and Jacobson, "The World's Fastest Scrabble Program", 1988). The server uses only `evaluate_move` and a set lookup, so it is cheap per step. Move generation runs offline in `generate_data.py`.

Tests check move generation against `evaluate_move` and against brute-force enumeration of every placement.

## Data

`generate_data.py` downloads the word list, writes it to `data/words.txt` (which the server loads at startup), and generates the puzzles:

```bash
python resources_servers/scrabble/generate_data.py               # 10,000 train, 500 validation
python resources_servers/scrabble/generate_data.py --train_samples 50000 --workers 64
```

Positions come from seeded self-play: two simulated players alternate for 1-16 moves (5% of rows are empty-board openings), each move chosen from the top-k scoring moves with k drawn from {1, 1, 2, 3, 5, 10}, so boards range from greedy to loosely played. Each row records `board`, `rack`, `best_score`, and one `best_move`. Train and validation use disjoint seed ranges and duplicate positions are dropped. `example.jsonl` holds the first 5 validation rows. Generation is deterministic for a given worker count and takes about 2 minutes for the default sizes on 64 cores.

In the default train split, the best score has median 30 and mean 35; 12% of rows have a best move of 60 or more (usually a bingo) and 9% of racks hold a blank.

### Word list

The dictionary is ENABLE (`enable1.txt`, [dolph/dictionary mirror](https://github.com/dolph/dictionary) at a pinned commit), which is in the public domain: its 168,551 words of 2-15 letters. It is the same source as `resources_servers/wordle`. Set `dictionary_path` to use a different list, and regenerate the data so `best_score` matches it.

## Run

```bash
python resources_servers/scrabble/generate_data.py
gym env start \
    --resources-server scrabble \
    --model-type vllm_model
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent scrabble_gymnasium_agent \
    --input resources_servers/scrabble/data/example.jsonl \
    --output resources_servers/scrabble/data/example_rollouts.jsonl
```

## Licensing

Code: Apache 2.0. Generated puzzle data: Apache 2.0. Word list: ENABLE, public domain.
