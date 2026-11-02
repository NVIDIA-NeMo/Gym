# Description

Resources server for short-answer question-answering benchmarks (HotpotQA-style).
Verification is fully deterministic — there is no LLM judge — and is a faithful
port of the scoring used by NeMo Skills' `hotpotqa_closedbook` benchmark:

* **JSON answer extraction** — pulls the predicted answer from the last valid
  JSON object in the model output (`{"answer": "..."}`).
* **SQuAD-style EM and F1** — official HotpotQA answer normalization (lowercase,
  strip articles, strip punctuation, collapse whitespace) followed by
  token-overlap F1 / exact-match.
* **Alternative-aware substring matching** — generates surface-form
  alternatives of the ground truth (article stripping, parens normalization,
  number-word ↔ digit, ampersand ↔ "and", hyphen handling, etc.) and checks
  whether any alternative appears as a substring of the model output. Two
  variants are emitted: `is_correct` (lenient) and `is_correct_strict`
  (word-boundary + position guards for short alternatives / long answers).
* **Unreliable-GT filtering** — flags ground-truth answers that are too long
  (`>40` chars) or look like multi-word proper names (3-6 word, mostly
  capitalized, mostly non-stopword) so the benchmark can report both
  `unfiltered` and `filtered_*` metrics side by side.

The reward emitted on `/verify` is set by `reward_metric` (below). The full
set of scores (`answer_em`, `answer_f1`, `is_correct`, `is_correct_strict`) is
returned in the verify response so downstream metric aggregation can compute
pass@k for each channel.

## Reward metric

| `reward_metric` | Reward | Use |
|---|---|---|
| `is_correct_strict` (default) | Alternative-aware substring match, as in NeMo Skills | Evaluation; results stay comparable with Skills |
| `answer_em` | SQuAD-normalized exact match | Recommended for training |

`is_correct_strict` accepts the ground truth inside a longer answer, so for
ground truth `blue` the answers `"blue or red"` and
`"blue is wrong; red is correct"` get reward 1.0, as does `"yes or no"` for
`yes`. Used as a training reward, that pays for hedged answers. `answer_em`
scores them 0.0. It also scores 0.0 for a correct answer wrapped in a sentence
(`"The capital is Paris"`) or written as a surface-form variant (`5` for
`five`), so it relies on the prompt asking for the bare answer.

To train with exact match, set `reward_metric: answer_em` in the resources
server config or override it on the command line:

```bash
gym env start \
    --model-type openai_model \
    --resources-server hotpotqa_qa \
    ++hotpotqa_qa.resources_servers.hotpotqa_qa.reward_metric=answer_em
```

# Example usage

## Running servers

```bash
gym env start \
    --model-type openai_model \
    --resources-server hotpotqa_qa
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent hotpotqa_qa_simple_agent \
    --input resources_servers/hotpotqa_qa/data/example.jsonl \
    --output results/hotpotqa_qa_rollouts.jsonl \
    --num-repeats 1
```

# Licensing information

Code: Apache 2.0

Dependencies:
- nemo_gym: Apache 2.0

Verification logic adapted from NeMo Skills (Apache 2.0) and from the
[official HotpotQA evaluation script](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
The unreliable-GT filtering rules are adapted from internal NVIDIA hallucination-detection
research.
