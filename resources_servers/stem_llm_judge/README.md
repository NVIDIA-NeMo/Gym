# `stem_llm_judge` resources server

LLM-as-judge for **broad STEM** open-answer grading — physics, chemistry and
biology. Binary reward: the judge is shown the problem, the reference answer and
the student's post-`</think>` solution, and answers `Judgement: 'yes'` or
`Judgement: 'no'`.

Built for RL training on open-answer STEM data, where the answer is a quantity,
an expression or a short explanation that a string match cannot score.

## Prompt

`prompt_templates/stem_llm_judge.txt`. Physics rules (physical / numerical /
symbolic equivalence, unit-system conversion, acceptable numeric substitution,
unresolved placeholders, generality, required-method problems, open-ended
problems, sign conventions) plus rules for

- multi-part questions and item counts (all requested parts required; `OR`
  reference answers are alternatives, `AND` are joint requirements),
- qualitative explanations and mechanisms (core cause required, extra reference
  detail optional, alternative sound mechanisms accepted),
- chemistry and biology specifics (a concentration is not a total amount, a rate
  is not a rate constant, allele/genotype/carrier frequency are distinct),
- open-ended lists (category and requested count govern).

Placeholders: `{question}`, `{expected_answer}`, `{generated_answer}`, filled
with `str.format` — so **every literal brace in the rubric must be doubled**
(`{{` / `}}`), or the first `verify()` raises `KeyError`.
`tests/test_app.py::test_prompt_has_all_three_placeholders` catches that.

Point at a different rubric without editing the config:

```
++env.nemo_gym.stem_llm_judge.resources_servers.stem_llm_judge.judge_prompt_template_fpath=prompt_templates/my_rubric.txt
```

A bare filename is resolved relative to **this directory** — Gym runs the
entrypoint with `cwd=resources_servers/stem_llm_judge`.

## Per-row schema

`verify` reads:

| Field                                   | Required | Notes                                                                                    |
| --------------------------------------- | -------- | ---------------------------------------------------------------------------------------- |
| `responses_create_params.input`         | yes      | The last `user` message is the question shown to the judge.                              |
| `expected_answer`                       | yes      | Also accepted as `metadata.expected_answer`.                                             |
| `template_metadata.output_regex`        | no       | Per-row override of the student-answer extractor. Training data uses `</think>(.*)`.     |
| `responses_create_params.metadata.step` | no       | Injected by the trainer; recorded in the generation log so a bad verdict maps to a step. |

Extra fields are allowed and echoed back on the response.

## What the judge is shown

**Question** — the last `user` message. `extract_problem_from_prompt` is **on by
default**, so the fixed instruction preamble

```
Answer the following problem step by step.
Please use LaTeX format to represent the variables and formulas ...
Your response should be in the following format:
Explanation: {your explanation for your final answer}
Answer: {your final answer}
<problem>
```

is stripped and the judge sees only `<problem>` — it grades the answer, not the
formatting instructions the policy was handed. A prompt without that marker is
passed through unchanged, so the default is safe for any data. Set
`extract_problem_from_prompt=false` to show the whole user message, or set
`question_extract_regex` to a pattern of your own (it always wins over the
toggle).

**Student solution** — everything after the policy's last `</think>`
(`response_extract_regex: '</think>(.*)'`), or the row's
`template_metadata.output_regex` when present. A regex miss falls back to the
full generation rather than scoring 0.

## Reward

| Case                                                         | Reward                 |
| ------------------------------------------------------------ | ---------------------- |
| Judge says `yes`                                             | 1.0                    |
| Judge says `no`                                              | 0.0                    |
| No parseable verdict                                         | 0.0                    |
| Blank assistant message                                      | 0.0 (judge not called) |
| Policy hit its length cap, `reward_zero_on_truncation: true` | 0.0 (judge not called) |

Pass/fail only — one judge call per rollout, no partial credit, no swap check
and no second pass on the full generation.

**Verdict parsing** (`_parse_verdict`):

1. Only the region after the judge's own last `</think>` is searched, so a
   verdict inside the judge's scratch reasoning does not count.
2. The **last** `Judgement: yes|no` wins — a judge that revises its decision is
   scored on its final one. Markdown bold and quotes are tolerated.
3. Only if no `Judgement:` line exists, a fallback matches
   `(Final )?(Judgement|Judgment|Answer|Verdict|Conclusion): yes|no`, including
   `\boxed{...}` / `\text{...}` wrappers. This recovers ~20–27% of otherwise
   reward-0 judgements. The token stays strictly `yes`/`no`, so an answer
   restatement like `Answer: 2/3` is deliberately not matched.

## Run it

From the Gym root:

```bash
# Serve (judge = the policy model; see below for a separate judge)
gym env start --model-type vllm_model --resources-server stem_llm_judge

# Smoke-test the 5 example rollouts
gym eval run --no-serve \
    --agent stem_llm_judge_simple_agent \
    --input resources_servers/stem_llm_judge/data/example.jsonl \
    --output results/stem_llm_judge_rollouts.jsonl \
    --num-repeats 1
```

`data/example.jsonl` holds 5 hand-written rows (physics, chemistry, biology) in
the exact shape training data uses — instruction preamble, `expected_answer` and
a per-row `output_regex`. They are smoke-test fixtures, not a benchmark.

Under NeMo-RL GRPO, add the config to `env.nemo_gym.config_paths`:

```
++env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/stem_llm_judge/configs/stem_llm_judge.yaml]
```

## Per-run overrides

All under `++env.nemo_gym.stem_llm_judge.resources_servers.stem_llm_judge.`
(getting that key wrong is the common mistake — Hydra silently creates a new
node instead of erroring):

| Override                                                                     | Effect                                                                                   |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `judge_model_server.name=judge_model`                                        | Judge with a separate (usually larger) model. Must name a server in the composed config.  |
| `judge_responses_create_params.temperature` / `.top_p` / `.max_output_tokens` | Judge sampling. A judge truncated mid-reasoning never reaches its verdict.                |
| `judge_prompt_template_fpath=…`                                              | Pick the rubric (see above).                                                              |
| `extract_problem_from_prompt=false`                                          | Show the judge the whole user message, preamble included.                                 |
| `reward_zero_on_truncation=true`                                             | Score 0 and skip the judge when the policy hit `max_output_tokens`.                       |
| `generation_log_dir=/experiments/…`                                          | Append a JSONL debug record per `verify()` (see below).                                   |
| `judge_endpoint_max_concurrency=N`                                           | Cap in-flight judge requests (default 64).                                                |

## Debugging a reward

Set `generation_log_dir` to a mounted, writable path. Each `verify()` appends one
JSONL line (one file per pid) with `step`, `id`, `question`, `expected_answer`,
`generation` (full), `judged_generation` (what the judge actually saw),
`judge_generations`, `verdicts` and `reward`. Logging failures never propagate
into the verify path.

| Symptom                                             | Look at                                                                                                             |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Reward 0 everywhere, `verdicts` all `null`          | The judge is not emitting `Judgement:` — check the loaded prompt and the judge's `max_output_tokens`.                |
| `question` still contains the instruction preamble  | `extract_problem_from_prompt` was turned off, or the prompt does not carry the `Answer: {your final answer}` marker. |
| `judged_generation` is the full chain of thought    | The generation had no `</think>`, or the row's `output_regex` disagrees with the data.                              |
| `judged_generation` empty, no `judge_generations`   | Blank policy output, or the truncation guard fired.                                                                 |
| Reward 0 but `judge_generations` reads as agreement | The verdict sits outside the judge's post-`</think>` region, or is worded so neither pattern matches.                |

## Tests

`tests/test_app.py` loads `configs/stem_llm_judge.yaml` directly, so a change to
the shipped defaults fails a test rather than drifting silently.

```bash
gym env test --resources-server stem_llm_judge
# or, from the Gym root in an env that already has nemo-gym installed:
uv run pytest resources_servers/stem_llm_judge/tests/test_app.py
```
