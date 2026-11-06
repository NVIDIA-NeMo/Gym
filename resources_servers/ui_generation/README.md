# ui_generation resources server

Grades a generated UI answer with an LLM judge. The judge reads the user prompt, the
task's reference answer and the policy's final answer, and scores only the code inside
fenced code blocks of the policy's answer. Nothing is executed or rendered.

## Input schema

- `responses_create_params.input`: the task's prompt messages. The reference answer is
  not in the input.
- `verifier_metadata` (required):
  - `reference_answer` (required): a good answer to the task. The judge treats it as a
    guideline, not a required target.
  - `category` (optional): a task label, such as the target stack; metrics report mean
    scores per category.
  - `id` (optional): the task identifier.

`environments/ui_generation/prepare.py` writes these rows from a simple task file; see
the [environment README](../../environments/ui_generation/README.md).

## Verification

1. A cut-off policy response (`status: incomplete`) or an empty final answer scores 0
   without a judge call (`failure_reason` `policy_response_truncated` or
   `empty_final_answer`).
2. Otherwise the judge gets the rubric in `prompts/judge.yaml` as the system message and
   one user message holding
   `json.dumps({"prompt": ..., "original_guideline": ..., "candidate_answer": ...}, ensure_ascii=False)`
   (the keys the rubric names). The prompt joins every user message with a blank line;
   `original_guideline` is the reference answer; the candidate is the last assistant
   message without its reasoning.
3. The verdict counts only when the judge finished (`finish_reason == "stop"`) and its
   reply, after an optional leading `<think>` block and an optional ```` ```json ```` fence,
   is one JSON object matching the rubric's schema exactly (`schema_version`
   `ui-code-judge-v1`, integer scores 0-4, `execution_verified: false`).
4. Any other reply (invalid JSON, invalid schema, a judge cut off at its token cap) or a
   failed judge call is a judge failure: the row is masked (`mask_sample: true`,
   `failure_kind: judge_failed`, the cause in `failure_reason`) and keeps its answer for
   re-judging. It is never a 0 for the policy. Judge calls that get a 408, 429 or 5xx
   reply, time out or lose the connection are retried (`judge_max_attempts`, default 4).

## Reward

The verdict has two scores from 0 to 4: `prompt_adherence` and `quality`
(0 unacceptable, 1 major defects, 2 mixed, 3 good with minor issues, 4 strong).

- `reward` is one score divided by 4, chosen by `reward_score`: `quality` (default),
  `prompt_adherence`, `mean` of both, or `min` of both.
- `reward_components` holds both scores divided by 4.
- `prompt_adherence_score` and `quality_score` hold the raw scores, so aggregate metrics
  report their means, overall and per `category`; `judge_verdict` and `judge_reasoning`
  keep the full verdict.

## Grading path and failures

Path A: grading runs inside `observe_llm_judge_grading`, judge calls go through
`call_judge`, and `record_parsed_judge_verdict()` is called only after a schema-valid
verdict was parsed. Judge failures raise `JudgeError` (or the `judge_failed` event of
`call_judge`), which the verify endpoint's judge failsafe turns into a masked row.

No episode deadline is enforced by this server; the policy is bounded by its
`max_output_tokens` and the judge by `judge_chat_completions_create_params.max_tokens`.
`verify()` reads only its request, so reverification is stateless.

## Configuration

`configs/ui_generation.yaml` defines the resources server and a separate judge model
server, `ui_generation_judge_model` (`vllm_model`, chat completions). Supply the judge
endpoint in `env.yaml` with `judge_base_url`, `judge_api_key` and `judge_model_name`.
Use a judge from a different model family than the policy where possible.

| Field | Default | Notes |
|---|---|---|
| `judge_chat_completions_create_params` | temperature 1.0, top-p 1.0, max_tokens 65536 | Sampling of each judge request; `chat_template_kwargs` can set a reasoning effort |
| `judge_max_concurrency` | 64 | Judge requests in flight per server |
| `judge_max_attempts` | 4 | Attempts per judge request for retryable failures |
| `reward_score` | `quality` | `quality`, `prompt_adherence`, `mean` or `min` |

The judge needs a context long enough for the prompt, the reference answer and the
candidate answer together; full single-file apps can run to tens of thousands of tokens.

## Tests

```bash
gym env test +entrypoint=resources_servers/ui_generation +should_validate_data=true
```

## Licensing information

Apache 2.0, including the synthetic example tasks in `data/`.
