# LongBench v2

Long-context multiple-choice QA. Each task shows one long document (or dialogue,
or code repository) followed by a question and four choices; the model answers
with a single letter.

- Paper: [LongBench v2](https://arxiv.org/abs/2412.15204)
- Dataset: [THUDM/LongBench-v2](https://huggingface.co/datasets/THUDM/LongBench-v2) (Apache 2.0), 503 questions

## Scoring

`verify()` is rule-based and needs no model:

1. Asterisks are stripped from the model's answer, so bold markdown still matches.
2. `The correct answer is (X)` is searched for across the whole answer first, and
   only if no parenthesised form is found anywhere is `The correct answer is X`
   tried, with `X` in `A`-`D`. So a parenthesised statement beats a bare one even
   when the bare one comes earlier; within a single form the leftmost match wins.
3. Reward is `1.0` when the extracted letter equals the gold letter, otherwise `0.0`.

Empty, whitespace-only or unparseable output scores `0.0` with
`extracted_answer = null`; it never raises.

## Data

`data/example.jsonl` holds five hand-built rows with synthetic contexts, so tests
and a smoke run work without network access. Their prompts are rendered by the
same code path as generated rows.

Row schema:

```json
{
  "responses_create_params": {"input": [{"role": "user", "content": "<rendered prompt>"}]},
  "expected_answer": "B",
  "_id": "...",
  "domain": "Single-Document QA",
  "sub_domain": "...",
  "difficulty": "hard",
  "length": "medium",
  "verifier_metadata": {"expected_answer": "B", "choices": [{"A": "..."}, {"B": "..."}, {"C": "..."}, {"D": "..."}]}
}
```

The context is baked into the rendered prompt; there is no separate `context`
field. The gold letter is also stored under `verifier_metadata`, and `verify()`
falls back to it when the top-level `expected_answer` is absent. The row id is
`_id` on the wire.

## Building the full dataset

```bash
python resources_servers/longbench/prepare_longbench.py --split train
```

One invocation writes two files under `data/` (both gitignored):

| File | Contents |
|------|----------|
| `longbench_full.jsonl` | every row of the split |
| `longbench_48k.jsonl` | rows whose rendered prompt is under `--max-tokens` (default 48000) |

Requires `datasets` and `transformers` at prep time only.

### Two tokenizers, two jobs

`--tokenizer` measures prompt length to pick the budgeted subset. Which rows land
in that file depends on it, so subsets built with different values are not
comparable and it must be held fixed across models. The default,
`google/gemma-4-E4B-it`, keeps 151 of the 503 rows; it is gated on the Hub, so
building a split needs an access token.

`--truncate-tokenizer` measures prompt length for truncation and should be the
tokenizer of the model that will answer. It defaults to `--tokenizer`.

The subset is chosen from untruncated lengths, so `--truncate-tokenizer` never
changes which rows the budgeted file holds.

### Truncation

A prompt longer than `--max-prompt-tokens` (default 119800) keeps its first and
last 59900 tokens and drops the middle. Contexts reach 5.1M tokens, so without
this 237 of the 503 rows would exceed any current context window. The default
leaves room for an 11144-token answer inside a 131072-token window. No row of the
budgeted file is long enough to be truncated.

Flags: `--split`, `--output-full`, `--output-48k`, `--tokenizer`,
`--truncate-tokenizer`, `--max-tokens`, `--max-prompt-tokens`.

## Configs

- `configs/longbench.yaml`: the resources server and `longbench_simple_agent`
  with the example and both prepared datasets.
- `configs/longbench_serve.yaml`: the resources server alone on port 8071, for an
  external runner that owns generation and POSTs each answer to `/verify`.

The [`longbench` benchmark](../../benchmarks/longbench_v2_48k/README.md) wires this
server into `gym eval`.

## Example rollouts

`data/example_rollouts.jsonl` is synthetic: the answers are hand-written and
scored by `verify()`. Three are correct (parenthesised, bold and bare forms), one
names the wrong letter and one never states an answer.

## Tests

```bash
gym env test --resources-server longbench
```
