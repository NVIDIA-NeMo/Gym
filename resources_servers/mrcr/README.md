# MRCR

OpenAI's Multi-Round Coreference Resolution benchmark. Each task is a
multi-turn conversation where the model has produced several outputs of
the same kind (e.g. multiple poems); the final turn asks the model to
reproduce the Nth occurrence exactly, prefixed by a random token.

## Scoring

1. The response must start with a `random_string_to_prepend` prefix
   (reward = 0.0 if missing).
2. Otherwise the prefix is stripped from both response and expected
   answer, and `difflib.SequenceMatcher(...).ratio()` becomes the reward
   (continuous similarity in [0, 1]).

Ported from
[NeMo Skills MRCR evaluator](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/mrcr.py)
which follows the
[official MRCR grading](https://huggingface.co/datasets/openai/mrcr).

## Data

Upstream: [openai/mrcr](https://huggingface.co/datasets/openai/mrcr).
`example.jsonl` contains 5 small synthetic tasks for smoke testing.

## Licensing

- Code: Apache 2.0
- Data (openai/mrcr): see upstream license
