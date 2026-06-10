# Python List

Rule-based verifier for Python list and tuple answers.

This server is named `python_list` to match Gym's directory/import convention. It corresponds to the dataset verifier label `python-list`.

The verifier extracts an answer from the assistant response, parses both the prediction and expected answer with `ast.literal_eval` or JSON parsing, normalizes tuples to lists, then scores items in order. It mirrors the NeMoRL `python-list` reward behavior: numeric and structured items are compared by equality, string items receive word-F1, and extra or missing items are penalized through the sequence length denominator.

## Configuration

```bash
ng_run "+config_paths=[resources_servers/python_list/configs/python_list.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Request Fields

- `expected_answer`: Python or JSON list/tuple literal, such as `(4, 3)` or `[30, 45, 60]`.
- `extraction_mode`: one of `final_answer`, `boxed`, `last_line`, `full_response`, or `auto`.

The default extraction mode is `final_answer`, which looks for `Final answer:` or `Answer:` and falls back to `\boxed{...}`.
