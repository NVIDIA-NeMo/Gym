# perceptionbench resources server

Verifies free-form visual perception VQA responses for [PerceptionBench](https://github.com/MoonshotAI/PerceptionBench)
using an LLM judge, matching the upstream evaluator (`eval/eval.py` and `eval/judge_prompt.txt` at commit
`ba032c06e9b6ee3679171ff6ba643b7a0cfebe2e`).

## Input schema

`task_data.py` defines the row schema:

- `task_id` (optional, top-level): stable task identifier (`perceptionbench_<index>`).
- `responses_create_params.input`: one user message interleaving text and `input_image` blocks at the `<|image_N|>`
  placeholders, without reference answers or hints.
- `verifier_metadata` (required):
  - `question` (required): raw upstream `problem` string (including `<|image_N|>` placeholders), passed to the judge.
  - `answer` (required): reference answer string.
  - `category` (required): one of the ten atomic visual-perception error categories.
  - `source_bmk` (optional): source benchmark name for decomposed items, or `"NA"` for newly authored items.
  - `index`, `source_idx` (optional): upstream item provenance.

## Grading path and failure handling

- **Grading path A (`observe_llm_judge_grading`):** judge calls run inside `observe_llm_judge_grading` via
  `call_judge` (`/v1/chat/completions`), `record_rejected_judge_reply()` records unparseable retries, and
  `record_parsed_judge_verdict()` is called only after an unambiguous `True` or `False` verdict is parsed after
  `[judge]`.
- **Empty / truncated policy replies:** an empty assistant answer, an `[error]` prefix, or an unclosed `<think>`
  block scores `0.0` (`no_answer=1.0`, `evaluation_completed=True`) without calling the judge, matching upstream
  `is_failed`. Context-window rejections (`nemo_gym_context_window_exceeded == "true"`) record
  `failure_kind: model_context_limit` and score `0.0`.
- **Judge failures:** transport/HTTP failures in `call_judge` and judge replies without a valid
  `[reason] ... [judge] True|False` verdict after `judge_max_attempts` tries (default `3`) raise `JudgeError`.
  `judge_failsafe` masks the row (`mask_sample: true`, `failure_kind: judge_failed`) so judge outages never count as
  wrong model answers.
- **Time contract and resume:** no episode deadline is enforced by this server; `simple_agent` runs one step bounded
  by the policy's `max_output_tokens`, and each judge call is bounded by `judge_chat_create_params.max_tokens` (4096).
  `REVERIFY_MODE = ReverifyMode.STATELESS` because `verify()` reads only the request payload.

## Metrics

- `mean/reward`: micro-average accuracy across all scored items (upstream `Overall`).
- `mean/no_answer`: share of replies with no extractable final answer.
- `by_category/<category>/accuracy` and `macro_category_accuracy`: accuracy across each of the ten perception
  capabilities and their unweighted mean.
- `by_source/<source_bmk>/accuracy`: accuracy grouped by decomposed source benchmark (for example `DA-2K`).

## Configuration

`configs/perceptionbench.yaml` declares the resources server, `simple_agent`, and `single_agent_turn_legacy`
environment server. `configs/judge_model_openai.yaml` wires `perceptionbench_judge_model` to an OpenAI-compatible
endpoint (`judge_base_url` / `judge_api_key` / `judge_model_name`; default model `openai/gpt-oss-120b` at temperature
`0.3`).

## Tests

```bash
gym env test perceptionbench
gym env test --resources-server perceptionbench +should_validate_data=true
```

## Licensing

- Server code and `tests/verifier_cases.jsonl`: Apache-2.0.
- `prompt_templates/judge_prompt.txt`: Apache-2.0, copied unchanged from
  [MoonshotAI/PerceptionBench](https://github.com/MoonshotAI/PerceptionBench) (`eval/judge_prompt.txt` at commit
  `ba032c06e9b6ee3679171ff6ba643b7a0cfebe2e`).
