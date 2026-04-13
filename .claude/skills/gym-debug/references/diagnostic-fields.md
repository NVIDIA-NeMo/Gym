# Diagnostic Fields by Benchmark

Custom `VerifyResponse` fields available in rollout JSONL for debugging.

---

## code_gen

| Field | Type | What it tells you |
|-------|------|-------------------|
| `extracted_model_code` | str | Code extracted from model output. Check if think blocks leaked in. |
| `reasoning_format_violation_rate` | float | Fraction of responses with malformed `<think>` tags. High value → stripping issue. |
| `compilation_error` | str | Compiler error message if code failed to compile. |
| `execution_output` | str | stdout from running the code. Compare with expected. |
| `execution_error` | str | stderr from running the code. |

---

## spider2_lite (SQL)

| Field | Type | What it tells you |
|-------|------|-------------------|
| `extracted_sql` | str | SQL extracted from model output. Empty → extraction failure. |
| `failure_reason` | str | `NO_SQL_EXTRACTED`, `EXECUTION_ERROR`, `WRONG_RESULT`, `TIMEOUT` |
| `execution_match` | bool | Whether extracted SQL produced correct result set. |
| `execution_output` | str | Query result from running the extracted SQL. |

---

## equivalence_llm_judge

| Field | Type | What it tells you |
|-------|------|-------------------|
| `judge_evaluations` | list | Judge decisions for each evaluation path (primary, swap, fallback). |
| `check_twice_swap_result` | str | `AGREE` or `DISAGREE`. Disagree → positional bias detected. |
| `extraction_result` | str | What regex extracted from model output. Empty → extraction failed. |
| `reward_path` | str | Which reward path was taken (primary, swap_fail, fallback). |

---

## tavily_search

| Field | Type | What it tells you |
|-------|------|-------------------|
| `num_tool_calls` | int | Number of search tool calls the model made. |
| `search_queries` | list[str] | Actual queries sent to the search API. |
| `tool_call_results` | list | Results returned from each tool call. |

---

## jailbreak_detection

| Field | Type | What it tells you |
|-------|------|-------------------|
| `safety_verdict` | str | `SAFE` or `UNSAFE` from the safety judge. |
| `quality_score` | float | Quality rating from the second judge call (0.0-1.0). |
| `combined_reward_breakdown` | dict | `{safety: 1.0, quality: 0.3, final: 0.3}` showing the formula. |

---

## Usage

Read these fields from the rollout JSONL:

```python
import json

with open("rollouts.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        # Access diagnostic fields
        print(entry.get("extracted_model_code", "N/A"))
        print(entry.get("failure_reason", "N/A"))
        print(entry.get("judge_evaluations", []))
```

Filter for specific failure modes:

```python
import json

failures = []
with open("rollouts.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("reward", 1.0) == 0.0:
            failures.append({
                "task_index": entry.get("task_index"),
                "failure_reason": entry.get("failure_reason", "unknown"),
                "extraction_result": entry.get("extraction_result", ""),
            })
```
