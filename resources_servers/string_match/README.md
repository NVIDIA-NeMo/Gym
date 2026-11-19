# Description

General-purpose string matching verifier for free-form text answers. It pulls a candidate answer out of the model response, then compares it to `expected_answer` with the NeMo RL string-match grader (no format reward).

Extraction is controlled per row by `extraction_mode`:

- `boxed` — the last `\boxed{...}`, with `\text{...}` wrappers stripped
- `final_answer` (default) — the last `Final answer: ...` / `Answer: ...` line, falling back to `\boxed{...}`
- `last_line` — the last non-empty line
- `full_response` — the whole assistant text

Comparison depends on `case_sensitive`:

- `true` — exact string equality after stripping whitespace
- `false` (default) — the grader: NFKC normalization, quote/trailing-punctuation stripping, lowercasing, then a series of equivalence checks (float, comma-separated numbers, LaTeX command stripping, list reordering, US state name/abbreviation, unit stripping). An exact match under any of these scores 1.0; a numeric answer within relative error small enough to score above 0.98 gets 0.98. Everything else is 0.0.

Data links: ?

# Input format

```json
{
  "responses_create_params": {"input": [{"role": "user", "type": "message", "content": [
    {"type": "input_text", "text": "What is the capital of France? Put your final answer in \\boxed{}."}
  ]}]},
  "expected_answer": "Paris",
  "extraction_mode": "boxed",
  "case_sensitive": false
}
```

The verify response echoes `expected_answer` and the `extracted_answer` the grader actually saw, which is what to look at when a reward is unexpectedly 0.0.

# Example usage

```bash
gym env start \
    --resources-server string_match \
    --model-type vllm_model &
gym eval run --no-serve \
    --agent string_match_simple_agent \
    --input resources_servers/string_match/data/example.jsonl \
    --output resources_servers/string_match/data/example_rollouts.jsonl
```

# Licensing information

Code: Apache 2.0
Data: the example rows are hand-written prompts, Apache 2.0.

Dependencies
- nemo_gym: Apache 2.0
- fastapi: MIT
