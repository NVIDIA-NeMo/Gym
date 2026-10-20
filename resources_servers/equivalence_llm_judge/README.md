# Equivalence LLM-as-judge Resources Server

### Overview
Uses an LLM as a judge to compare a model’s generated answer against the expected answer.

Prompt and labels are configurable via config.

An example dataset for this resources server can be found at:
https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-openqa 

### Key config fields
- `judge_system_message`: optional system message. If omitted, no system message is added.
- `judge_prompt_template` (required): user prompt template. Placeholders: `{question}`, `{expected_answer}`, `{generated_answer}`.
- `judge_equal_label` / `judge_not_equal_label`: labels the judge must output. Defaults to `[[A=B]]` and `[[A!=B]]`.
- `check_twice_swap` (bool, default false): if true, after an initial equal verdict, performs a second judge call swapping expected and generated answers to reduce bias.
- `reward_if_swap_fails` (float, default 0.0): reward to assign if the second (swap) pass disagrees (i.e., fails). You may set this to -1.0 or other value to catch it on the training side..
- `use_per_record_regex` (bool, default true): enable independent per-problem overrides from `template_metadata.question_regex` for questions and `template_metadata.output_regex` for answers. Missing overrides fall back to the corresponding global regex.
- `check_full_generation_on_fail` (bool, default true): if true, when the first judge pass fails, retry with the full generation (no answer regex) for partial credit. Requires `use_per_record_regex=true` and `template_metadata.output_regex`.
- `reward_if_full_generation_succeeds` (float, default 0.5): reward when full generation check succeeds after extraction failure. Set to 1.0 for full credit.
- `extraction_length_threshold` (int, default 120): skip answer extraction when the expected answer exceeds this length. Use the full generation instead. Requires `use_per_record_regex=true` and `template_metadata.output_regex`. Set to null to disable.

### Empty answers, answer-length guard, and judge recovery

Missing assistant messages and empty or whitespace-only final assistant text
score zero without a judge call, with `failure_reason: empty_final_answer`.
The full response is retained, including any reasoning. This check applies before
regex extraction, so nonempty answers still use the existing extraction and
full-generation fallback logic even when a regex produces an empty match.

`max_answer_chars` is optional and defaults to `null` (no limit) for every dataset.
To protect a judge endpoint from oversized answers, set it on the resources server:

```yaml
equivalence_llm_judge:
  resources_servers:
    equivalence_llm_judge:
      max_answer_chars: 100000
```

When enabled, longer final assistant answers score zero without a judge call.
The full response is retained, with `failure_reason` starting with
`final_answer_too_long`. The check counts characters in the last assistant message
before regex extraction, excludes reasoning and earlier messages, and does not
stop generation early. Enabling the limit changes reward assignment, so choose it
explicitly in the configuration for your workload (evaluation, synthetic data
generation, or RL).

If judge calls exhaust their retries, the shared judge failsafe saves the policy
response in the failures sidecar as `judge_failed`, excluded from scored aggregates.
Recover these rows with `gym eval reverify --judge-failed-only` and the same
verifier/judge configuration. Successful grades are preserved. This applies to
new sidecar-tagged failures; older in-band `JUDGE_ERROR` rows require migration
before using it. A run with pending grades is incomplete.

### Input schema
Accepts the same outer request structure as other resources servers:
- `responses_create_params`: the original model query (used here to extract a question/context string from user messages for the judge prompt).
- `response`: the model output in OpenAI Responses schema. The last assistant message text is used as the generated answer.
- `expected_answer`: gold answer to compare against.

### Per-problem question extraction

A single training job can mix many environments and datasets with different
prompt templates. Questions routed to the same equivalence judge may contain
different instruction prefixes or wrappers, so one server-wide extraction
pattern cannot describe every question. Keeping the extraction rule on each
problem lets a shared judge receive the relevant question text while the policy
still receives the complete original prompt.

With `use_per_record_regex: true`, set `template_metadata.question_regex` on each
problem to override the server-wide `question_extract_regex`. For example, this
dataset row keeps the full prompt for generation while passing only `What is
2+2?` as the question in the judge prompt:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "Solve the following problem.\n<question>What is 2+2?</question>\nGive your final answer as Answer: ..."}]
  },
  "expected_answer": "4",
  "template_metadata": {
    "question_regex": "<question>(.*?)</question>",
    "output_regex": "Answer: ([^\\n]+)"
  }
}
```

The question override does not require `output_regex`. Missing, empty, null, or
non-string question overrides use the global `question_extract_regex`; setting
`use_per_record_regex: false` disables both per-problem overrides. A null global
pattern uses the full text of the last user message.

Question extraction uses the last regex match with multiline and dot-all flags,
then the first non-empty capture group, or the whole match if no group is
non-empty. Invalid or unmatched patterns retain the full last user text, rather
than retrying the global pattern. Text blocks in a multimodal user message are
joined before extraction; image content is excluded.

`extraction_length_threshold` and full-generation rescue apply only to answer
extraction via `output_regex`. They do not disable question extraction. Rescue
and swap judge calls reuse the extracted question. Adding a question override
can change judge inputs and rewards; record the metadata with evaluation results.

### Example config
```yaml
equivalence_llm_judge:
  resources_servers:
    equivalence_llm_judge:
      entrypoint: app.py

equivalence_llm_judge_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: equivalence_llm_judge
      model_server:
        type: responses_api_models
        name: openai_model
      datasets:
      - name: sciq_validation
        type: example
        jsonl_fpath: resources_servers/equivalence_llm_judge/data/sciq_validation.jsonl
```

### Usage
Spin up with a judge model and prompt:
```bash
gym env start \
  --resources-server equivalence_llm_judge \
  --model-type openai_model \
  +equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.max_output_tokens=256 \
  +equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_system_message="You are a careful arbiter." \
  "+equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_prompt_template='<|Problem|>\n{question}\n\n<|Gold|>\n{expected_answer}\n\n<|Prediction|>\n{generated_answer}\n'"
```

Then query via any agent; verification happens with `/verify` on this server when evaluating rollouts.

For our tests we used Gemma3-27B-it.  
You should always check the license of your chosen judge model to make sure your use case comply with it.

### Notes
- By default (`check_twice_swap=false`), the server performs a single judge pass. If the verdict is equal, reward is 1 and one evaluation is returned; if not equal, reward is 0 and one evaluation is returned.
- If `check_twice_swap=true` and the first pass is equal, a second pass is performed with expected and generated answers swapped. Reward is 1 only if the second pass is also equal; otherwise `reward_if_swap_fails` is used (default 0.0). In the double-check case, two evaluations are returned.
- If the judge output doesn't include either label, it defaults to not-equal.
- **OpenQA support (enabled by default)**: The server enables per-record regex extraction (`use_per_record_regex=true`), full generation rescue (`check_full_generation_on_fail=true`), and length-based thresholds (`extraction_length_threshold=120`) by default. Answer rescue and length thresholds only activate when `template_metadata.output_regex` is present. The independent question override is described above.

## Licensing
Code: Apache 2.0   
Data: CC-BY-NC-3.0 (Examples from https://huggingface.co/datasets/allenai/sciq)
