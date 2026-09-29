# Wordle Env

Multi-step, tool-calling Wordle environment built on `GymnasiumServer`. Use with `gymnasium_agent`.

The model guesses a secret 5-letter word in 6 attempts using three tools: `submit_guess`, `check_word_validity`, and `get_game_state`. The episode ends as soon as the game is won or lost, or when the model replies without a tool call.

## Reward

- Win: `2.0 - 0.2 * (turns - 1)`, so turn 2 scores 1.8 and turn 6 scores 1.0. A turn-1 win is scored like turn 3 (1.6) so lucky openers are not over-rewarded. Floor of 0.1.
- Loss or incomplete game: 0.0.
- Penalties accumulate over the game and only reduce a win: repeated guess (-0.2), ignoring a known green (-0.05 per position), ignoring all known yellows (-0.03), reusing an eliminated letter (-0.02 per letter), wrong length or unknown word (-0.02). Invalid guesses still use a turn.

## Data

Every row pins its target word in `custom_target`, so all rollouts of a row play the same word. `reset()` rejects rows without a valid target.

Train and validation targets come from disjoint splits of the 3,088 target words in `wordle_words.py` (2,625 train, 463 validation). `validation.jsonl` uses the first 100 validation words. Generate `train.jsonl`, `validation.jsonl`, and `example.jsonl`:

```bash
python resources_servers/wordle/generate_data.py --output_dir resources_servers/wordle/data
```

### Word lists

- Valid guesses (8,636): 5-letter words from [ENABLE](https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt) (`enable1.txt`), public domain.
- Targets (3,088): common words from the SCOWL/ESDB hunspell `en_US` dictionary ([en-wl/wordlist](https://github.com/en-wl/wordlist) release 2026.02.25) that are also in ENABLE, minus one word the dictionary flags as taboo. Copyright 2000-2026 by Kevin Atkinson, used under the SCOWL/ESDB permissive license. The full notice is in `wordle_words.py`.

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
