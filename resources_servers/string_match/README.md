# String Match Resource Server

General-purpose string matching verifier for free-form text answers (Yes/No, short phrases, single words, etc.).

## Extraction Modes

| Mode | Description |
|---|---|
| `final_answer` (default) | Extracts text after the last `Final answer:` or `Answer:` prefix. Falls back to `\boxed{}` if no prefix is found. |
| `boxed` | Extracts the content of the last `\boxed{...}` in the response. Strips `\text{}` wrappers. |
| `last_line` | Uses the last non-empty line of the response. |
| `full_response` | Uses the entire assistant response text as the extracted answer. |

## Matching

- **Case-insensitive** by default (set `case_sensitive: true` to override).
- Whitespace is collapsed and trimmed.
- Unicode is NFKC-normalized for robust matching.

## Data Format

Each JSONL row should contain:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "..."}]
  },
  "expected_answer": "Yes",
  "extraction_mode": "final_answer"
}
```
