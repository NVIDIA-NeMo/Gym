# stem_judge resources server

LLM-as-judge rewards for STEM RL.

The answer-row judge prompt (`prompt_templates/stem_v11.txt`) is the prompt of
Grigor Nalbandyan's `stem_llm_judge` resources server, branch
`gnalbandyan/stem_llm_judge` at
[d002fd98a175865e465699bb434ddcf691bffbb6](https://github.com/NVIDIA-NeMo/Gym/tree/d002fd98a175865e465699bb434ddcf691bffbb6/resources_servers/stem_llm_judge)
in this repository (`prompt_templates/stem_llm_judge.txt`, byte for byte);
`stem_v11` is the name the offline judge study below used for it. The verdict
patterns, the problem-extraction regex and the question, truncation and step
helpers are adapted from that server's `app.py`. The verdict patterns capture
the same verdicts but are rewritten so degenerate judge text cannot make them
backtrack. On top of that server, this one adds:

- **Length guards.** A rollout that hit the policy's length cap scores 0
  (optional in `stem_llm_judge`, always on here), and so does a final answer
  too long for the judge.
- **A per-row answer contract.** `boxed` rows must answer in a `\boxed{}`.
  `free` rows (proofs, explanations) are judged on the whole final answer.
- **Rubric rows.** The judge marks each criterion, and the server adds up the
  points.

## Grading one rollout

The policy's and the judge's model servers must run with a reasoning parser.
The server then takes the assistant message as the final answer, with no
special handling of `<think>` tags.

Each gate below scores 0 without calling the judge, and the response records
the reason in `grading_status`.

| `grading_status` | When |
|---|---|
| `truncated` | The policy hit its length cap (`incomplete_details.reason == "max_output_tokens"`). |
| `empty_final_answer` | There is no assistant text, for example when the reasoning never closed. |
| `no_boxed` | A boxed row whose last `\boxed{` is missing, unclosed or blank. Only the last box counts. |
| `box_too_long` | The box is longer than `max_box_chars` (off by default). |
| `answer_too_long` | The judge would see the whole final answer, and it is longer than `max_answer_chars`. |

After the gates, the base configuration sends the row to the judge. The
[question-bank preset](#question-bank-grading-options) also supports masked abstentions and JSON reasons.

- **Answer rows:** the stem_v11 prompt. The judge's last `Judgement: 'yes'|'no'`
  gives reward 1 or 0.
- **Rubric rows:** the `stem_rubric` prompt. The judge ends with a JSON object
  that marks every criterion id true or false. The score is earned points over
  total positive points, clipped to [0, 1].
  - `rubric_reward: fraction` returns that score.
  - `threshold` returns 1 when the score is at least `rubric_pass_fraction`.

Only the judge's text after its own last `</think>` is parsed. A reply with no
verdict, or one cut off at the judge's token limit, scores 0 (`judge_unparsed`).
Answer rows save the verdict as `verdict_label` (`Judgement: yes` or
`Judgement: no`); rubric rows save `criteria_met`. A failed judge call is retried by
`call_judge` when another attempt can succeed (408, 429, 5xx, timeout, lost
connection) and otherwise fails at once; either way it is reported as a judge
failure and the row is masked (`mask_sample`) instead of scored.

### What the judge sees

- **Question:** the last user message. A content list contributes its text
  blocks. `question_extract_regex` (or `extract_problem_from_prompt`, which
  strips a fixed instruction preamble ending in `Answer: {your final answer}`)
  narrows it to the problem.
- **Reference:** `expected_answer`. The prompt asks the judge to compare with
  it, but a judge can still overrule a reference that contradicts the question
  itself (in a smoke test, an inverted explanation reference was judged a typo). The prompts put it on a line of its own, so
  a reference that starts with `- ` would read as a markdown list item and the
  judge would drop the sign; the server shows `- 13.6 eV` as `-13.6 eV`. A
  reference with two or more `- ` items is a list and is left alone.
- **Student solution:** the whole final answer (boxed rows under the default
  `boxed_judge_input: final_answer`, and free rows), or only the box's content
  (`boxed_judge_input: box`). Under `final_answer` the box is a format gate: the
  judge grades the answer as written, not the box alone.

## Row fields

| Field | Meaning |
|---|---|
| `expected_answer` | The reference answer. When it is absent or null, a scalar `metadata.expected_answer` is used. Required on answer rows; a rubric row without one shows the judge `(none)`. |
| `answer_format` | `boxed` or `free`. Defaults to `default_answer_format`. |
| `rubric` | Optional list of `{"id", "points", "criterion"}`. Negative points are penalties. Points must be finite and non-zero, the ids unique, and at least one criterion must have positive points; a malformed rubric fails request validation. |

A rubric row follows its `answer_format` too, so proof or explanation rubrics
should usually be `free`. Other row fields are kept in the verify response. See
`data/example.jsonl` for one row of each kind.

## Config

| Option | Default | |
|---|---|---|
| `judge_model_server` | `policy_model` | The judge's model server. The policy grades itself unless you point this at a separate judge (see Running). |
| `judge_responses_create_params` | `input: []` | Sampling parameters for judge calls; the server fills `input`. |
| `judge_endpoint_max_concurrency` | `64` | Concurrent judge calls per server process; `null` for no limit. |
| `judge_system_message` | `null` | Optional system message before the judge prompt. |
| `answer_prompt_template_fpath`, `rubric_prompt_template_fpath` | `prompt_templates/stem_v11.txt`, `prompt_templates/stem_rubric.txt` | Relative to this folder. |
| `default_answer_format` | `boxed` | |
| `boxed_judge_input` | `final_answer` | Boxed rows: the judge sees the whole final answer (`final_answer`) or only the box (`box`). |
| `max_box_chars` | `null` | Some chemistry references run to 4K characters and correct boxes to 5K. A fixed 2K cap would zero correct answers. |
| `max_answer_chars` | `16000` | Applies whenever the judge sees the whole final answer. |
| `rubric_reward`, `rubric_pass_fraction` | `fraction`, `1.0` | |
| `extract_problem_from_prompt`, `question_extract_regex` | off | See "What the judge sees". A regex that does not compile fails at startup. |
| `generation_log_dir`, `generation_log_filename` | `null`, `stem_judge_generations.jsonl` | One JSONL record per rollout, with the status, what the judge saw, and its output. Each server process writes `<name>.<host>.<pid>.jsonl`; a directory that cannot be written fails startup. |

The shipped defaults are the setup the offline STEM judge study measured:
stem_v11 on the whole final answer, with reward 0 when there is no box. The
study used 2,000 STEM rollouts from earlier RL runs and judged 1,950 of them:

- On 1,947, the server sends the same judge prompt as the study.
- On 1, the prompt differs only by a leading newline the server strips.
- The other 2 are degenerate answers of 291K and 321K characters. The study
  judged them; the server scores them 0 (`answer_too_long`).

On 26,880 sampled training rollouts, the gates fire as follows:

- 229 `truncated`
- 130 `empty_final_answer`
- 44 `no_boxed`
- 7 `answer_too_long`. Each message runs 19K to 237K characters and holds a
  second reasoning block that ends in `</think>`.

## Running

Add a judge model server and route `judge_model_server` to it. For example,
with both models on the NVIDIA API, put this in `judge.yaml`:

```yaml
judge_model:
  responses_api_models:
    vllm_model:
      # Copy the remaining fields of policy_model from
      # responses_api_models/vllm_model/configs/vllm_model.yaml.
      entrypoint: app.py
      base_url: https://integrate.api.nvidia.com/v1
      api_key: ${oc.env:NVIDIA_API_KEY}
      model: nvidia/nemotron-3-ultra-550b-a55b
      uses_reasoning_parser: true
```

```bash
gym env start --resources-server stem_judge --model-type vllm_model \
  --model nvidia/nemotron-3.5-lightning-30b-a3b \
  --model-url https://integrate.api.nvidia.com/v1 --model-api-key "$NVIDIA_API_KEY" \
  --config judge.yaml \
  ++stem_judge.resources_servers.stem_judge.judge_model_server.name=judge_model
gym eval run --no-serve --agent stem_judge_simple_agent \
  --input resources_servers/stem_judge/data/example.jsonl --output rollouts.jsonl \
  --temperature 0.6 --max-output-tokens 16384
```

Set the route with the `++` override: the same key in a `--config` file loses
to the shipped config's `policy_model`.

`data/example_rollouts.jsonl` holds one rollout per example row from these
commands: the policy `nvidia/nemotron-3.5-lightning-30b-a3b` and the judge
`nvidia/nemotron-3-ultra-550b-a55b`.

## Question-bank grading options

The question-bank preset uses the same resources server, agent and rubric grader.
Select it with `gym env start --resources-server stem_judge/question_bank` and the
usual model arguments. It includes the base config and overrides its answer options.

| Answer option | Base config | Question-bank preset |
| --- | --- | --- |
| `guideline_scope` | `stem` | `cross_domain` treats v11 examples as guidance for reference comparisons in any subject. |
| `extra_answer_rules` | `[]` | A list of additional instructions for model-judged answer rows. |
| `verdict_mode` | `binary` | `yes_no_unresolved` adds abstention. An explicit no is a measured zero. An abstention or unparseable verdict has unknown correctness and sets `mask_sample=true`. |
| `verdict_format` | `line` | `json` saves the verdict, reason code and supporting evidence on model-judged answer rows. |
| `box_parser` | `exact` | `last_command` recognizes whitespace before the opening brace, skips escaped commands, and rejects a malformed final box even when an earlier one is valid. |
| `boxed_judge_input` | `final_answer` | `box` grades the extracted answer. Free-answer rows use their full final message. |

Set each option independently. The prompt options reuse the sections of the shipped
`stem_v11.txt` template. Custom templates used with those options must retain its section
markers. Gym marks the individual sample. On NeMo RL branches that provide SingleController's
`grpo.exclude_failed_judge_groups`, enable it with NeMo Gym, `grpo.use_dynamic_sampling=true`
and `env.should_mask_flagged_samples=true` to exclude and refill the entire group.

If recovery rejudges a saved answer, a parsed `unresolved` verdict completes the
regrade once. The saved answer stays masked with unknown correctness. A malformed
judge reply leaves the original answer eligible for another grading attempt.

The question-bank preset requests a structured explanation. Select `verdict_format: line`
to use the text verdict instead.
The model returns one JSON object, for example:

```json
{"verdict":"incorrect","reason_code":"wrong_denominator","evidence":"The submitted 1/3 differs from the reference 1/2."}
```

The saved rollout keeps that object in `judge_evaluations[0].judgment`, alongside
the usual `verdict_label`. The optional generation log records it in `judgments`.
Binary mode accepts `correct` and `incorrect`. Three-way mode also accepts
`unresolved`. A missing reason, blank evidence, duplicate field, or truncated
reply produces `judge_unparsed`. Binary mode gives an unparsed reply reward zero.
Three-way mode masks it as unknown.

The solver instruction is a separate generation option. For a dataset of **boxed**
questions with a raw `question` field, render it once before generating rollouts:

```bash
gym dataset render --input boxed_questions.jsonl \
  --prompt-config resources_servers/stem_judge/prompt_templates/boxed_final_answer.yaml \
  --output boxed_prompts.jsonl
```

The template asks the solver to put its final answer, and only its final answer,
inside `\boxed{}`. Use the original prompts for free-response tasks or historical
rollouts. Gym's renderer rejects already materialized input.

### Adding rules to the answer judge

Add `--config rules.yaml` to the launch command. Put the rules under the STEM
instance they apply to. Gym includes this list in the resolved run configuration.

```yaml
stem_judge:
  resources_servers:
    stem_judge:
      extra_answer_rules:
        - For archive classes, accept K-17 as an alias for Blue Ledger.
```

For a working example, pass `--config resources_servers/stem_judge/examples/archive_rules.yaml`.
Each string becomes a `<rule>` block immediately after the last built-in rule,
before the verdict instructions. Braces in LaTeX, JSON and examples are literal.
Custom answer templates using this option need a closing `</rule>` marker.
Additional rules take precedence over the default exact-match shortcut. A STEM
instance with extra rules sends answer comparisons to the model, including exact
math matches. Rubric rows use their own criteria.

## Test

```bash
gym env test --resources-server stem_judge +should_validate_data=true
```

## License

The code and the example data are Apache 2.0. The example rows were written for
this server.
