# stem_judge resources server

LLM-as-judge rewards for STEM RL. It is built from Grigor Nalbandyan's
`stem_v11` server and keeps its judge prompt (`prompt_templates/stem_v11.txt`, byte-identical) and
verdict parser. It drops the inherited `equivalence_llm_judge` options (A/B
swap, full-generation rescue, per-record extraction regexes) and adds:

- **Reasoning guards.** The judge sees only the policy's final answer. This
  holds whether the model server splits reasoning into its own output item or
  leaves `<think>...</think>` in the message.
- **A per-row answer contract.** `boxed` rows must answer in a `\boxed{}`.
  `free` rows (proofs, explanations) are judged on the whole final answer.
- **Rubric rows.** The judge marks each criterion, and the server adds up the
  points.

## Grading one rollout

Each gate below scores 0 without calling the judge, and the response records
the reason in `grading_status`.

| `grading_status` | When |
|---|---|
| `truncated` | The policy hit its length cap (`incomplete_details.reason == "max_output_tokens"`). |
| `empty_final_answer` | There is no assistant text, or none after the reasoning. |
| `think_not_closed` | There is no reasoning item and no `</think>` (only when `require_think_end` is on). |
| `think_reopened` | A `<think>` appears after the last `</think>`. |
| `no_boxed` | A boxed row whose final answer has no non-empty, closed `\boxed{}`. Boxes inside the reasoning do not count. |
| `box_too_long` | The box is longer than `max_box_chars` (off by default). |
| `answer_too_long` | The judge would see the whole final answer, and it is longer than `max_answer_chars`. |

Text before the last `</think>` in the message is reasoning that leaked into it,
and is dropped. After the gates, the server makes one judge call:

- **Answer rows:** the stem_v11 prompt. The judge's last `Judgement: 'yes'|'no'`
  after its own `</think>` gives reward 1 or 0.
- **Rubric rows:** the `stem_rubric` prompt. The judge ends with a JSON object
  that marks every criterion id true or false. The score is earned points over
  total positive points, clipped to [0, 1].
  - `rubric_reward: fraction` returns that score.
  - `threshold` returns 1 when the score is at least `rubric_pass_fraction`.

An output that cannot be parsed scores 0 (`judge_unparsed`).

## Row fields

| Field | Meaning |
|---|---|
| `expected_answer` | The reference answer. |
| `answer_format` | `boxed` or `free`. Defaults to `default_answer_format`. |
| `require_think_end` | Overrides the server setting. Use it for rows sampled with thinking off. |
| `rubric` | Optional list of `{"id", "points", "criterion"}`. Negative points are penalties. The ids must be unique, at least one criterion must have positive points, and a malformed rubric fails request validation. |

A rubric row follows its `answer_format` too, so proof or explanation rubrics
should usually be `free`. See `data/example.jsonl` for one row of each kind.

## Config

| Option | Default | |
|---|---|---|
| `default_answer_format` | `boxed` | |
| `boxed_judge_input` | `final_answer` | Boxed rows: the judge sees the whole final answer (`final_answer`) or only the box (`box`). |
| `require_think_end` | `true` | Set to `false` for a thinking-off policy. |
| `max_box_chars` | `null` | Some chemistry references run to 4K characters and correct boxes to 5K. A fixed 2K cap would zero correct answers. |
| `max_answer_chars` | `16000` | Applies whenever the judge sees the whole final answer. |
| `rubric_reward`, `rubric_pass_fraction` | `fraction`, `1.0` | |
| `extract_problem_from_prompt`, `question_extract_regex` | off | Same as in stem_v11. |
| `generation_log_dir` | `null` | One JSONL record per rollout, with the status, what the judge saw, and its output. |

The shipped defaults are the setup the offline STEM judge study measured:
stem_v11 on the whole post-`</think>` answer, with reward 0 when there is no
box. On the study's 2,000 STEM rollouts from earlier RL runs, the server matches
the study's box gate on every row. The judge prompt is identical on 1,947 of
the 1,950 judged rows. The other 3 contain a stray `</think>` in the answer:
the server judges the text after it, while the study judged the whole message.
On 26,880 sampled training rollouts, the gates fire as follows:

- 229 `truncated`
- 130 `empty_final_answer`
- 44 `no_boxed`
- no `think_not_closed`, `think_reopened` or `answer_too_long`

## Install and test

The server uses the pre-reorganization import paths
(`nemo_gym.base_resources_server`, `nemo_gym.openai_utils`, ...). Older Gym
checkouts, such as the ones the STEM RL runs use, have only those paths, and
current Gym still serves them as aliases. To install, copy this folder into a
Gym checkout as `resources_servers/stem_judge`, then route rows to
`stem_judge_simple_agent`.

```bash
pytest -o asyncio_mode=auto resources_servers/stem_judge/tests/test_app.py
```

The tests pass both on Gym `main` (824f2b3) and on a June 2026 checkout
(ea3eac9).
