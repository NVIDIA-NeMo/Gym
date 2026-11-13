# BabyVision under OpenCode, question only

[`babyvision_opencode`](../babyvision_opencode/README.md) without its visual-reasoning
instruction: the user message is the image, the question and "Please answer the question and put
the final answer within \boxed{}." OpenCode's own system prompt is all the instruction the model
gets, which suits models trained on that prompt. Data, environment, harness and scoring are
otherwise identical.

## Prepare data

`prepare.py` downloads the same pinned `UnipatAI/BabyVision` file as
[`babyvision_opencode`](../babyvision_opencode/README.md#prepare-data) (or reads `BABYVISION_PARQUET`).

```bash
gym eval prepare --benchmark babyvision_opencode_question_only
```

## Run

```bash
gym eval run \
    --model-type vllm_model \
    --benchmark babyvision_opencode_question_only \
    ++output_jsonl_fpath=results/benchmarks/babyvision_opencode_question_only.jsonl \
    ++split=benchmark \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<>
```

Each task runs OpenCode in its own sandbox session, as in [`babyvision_opencode`](../babyvision_opencode/README.md#run).

Tests for both prepare scripts live in `tests/unit_tests/benchmarks/test_babyvision_opencode_prepare.py`.
