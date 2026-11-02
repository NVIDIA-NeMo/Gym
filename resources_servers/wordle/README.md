# Wordle Env

Multi-step, tool-calling Wordle environment built on `GymnasiumServer`. Use with `gymnasium_agent`.

The model guesses a secret 5-letter word in 6 attempts using three tools: `submit_guess`, `check_word_validity`, and `get_game_state`. The episode ends as soon as the game is won or lost, or when the model replies without a tool call.

## Reward

- Win: `2.0 - 0.2 * (turns - 1)`, so turn 2 scores 1.8 and turn 6 scores 1.0. A turn-1 win is scored like turn 3 (1.6) so lucky openers are not over-rewarded. Floor of 0.1.
- Loss or incomplete game: 0.0.
- Penalties accumulate over the game and only reduce a win: repeated guess (-0.2), ignoring a known green (-0.05 per position), ignoring all known yellows (-0.03), reusing an eliminated letter (-0.02 per letter), wrong length or unknown word (-0.02). Invalid guesses still use a turn.

## Data

Every row pins its target word in `custom_target`, so all rollouts of a row play the same word. `reset()` rejects rows without a valid target.

`generate_data.py` downloads the word lists from pinned ENABLE and SCOWL/ESDB URLs, writes them to `data/targets.txt` and `data/guesses.txt`, and generates `train.jsonl`, `validation.jsonl`, and `example.jsonl`. Run it once before starting the server, which loads `data/guesses.txt`:

```bash
python resources_servers/wordle/generate_data.py
```

Train and validation targets are disjoint splits of the 3,088 targets (2,625 train, 463 validation). `validation.jsonl` uses the first 100 validation words.

### Word lists

- Valid guesses (8,636): 5-letter words from ENABLE (`enable1.txt`, [dolph/dictionary mirror](https://github.com/dolph/dictionary)), public domain.
- Targets (3,088): base forms from the SCOWL/ESDB hunspell `en_US` dictionary ([en-wl/wordlist](https://github.com/en-wl/wordlist) release 2026.02.25) that are also in ENABLE, minus one word the dictionary flags as taboo. Used under this notice, which `generate_data.py` also writes at the top of `targets.txt`:

```
Copyright 2000-2026 by Kevin Atkinson

Permission to use, copy, modify, distribute, and sell any part of the English
Speller Database (ESDB, previously known as SCOWLv2), or word lists
created from it, is hereby granted without fee, provided that the above
copyright notice appears in all copies and that both the above copyright
notice and this notice appear in supporting documentation.  Kevin Atkinson
makes no representations about the suitability of this database for any
purpose.  It is provided "as is" without express or implied warranty.
```

## Run

```bash
gym env start \
    --resources-server wordle \
    --model-type vllm_model
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent wordle_gymnasium_agent \
    --input resources_servers/wordle/data/example.jsonl \
    --output resources_servers/wordle/data/example_rollouts.jsonl
```
