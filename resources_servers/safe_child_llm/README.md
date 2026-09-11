# Safe-Child-LLM

This environment ports both 100-prompt developmental splits released by
[Safe-Child-LLM](https://github.com/The-Responsible-AI-Initiative/Safe_Child_LLM_Evaluation): ages 6–12 and ages
13–17. The preparation script downloads the two source workbooks at pinned commit
`f69a651ff5c992c6d423b6a129ade8bf674fb63b` and preserves each prompt's age group, category, and source.

The upstream repository's runnable refusal/violation scripts use keyword matching, not a validated LLM judge. A safe
response can repeat a word from the request and be counted as harmful. This adapter therefore collects responses with
`skip_verification: true` and marks them `pending_human_review`; the placeholder reward is not a benchmark score.

Prepare and run with any NeMo Gym-compatible model endpoint:

```bash
gym eval prepare --benchmark safe_child_llm
gym eval run --config benchmarks/safe_child_llm/config.yaml --agent safe_child_llm_benchmark \
  --input benchmarks/safe_child_llm/data/safe_child_llm.jsonl \
  --output results/safe_child_llm.jsonl
```

Merge one or more completed model runs into the local annotation app:

```bash
python -m benchmarks.safe_child_llm.annotation_app \
  --result kimi-k3=results/kimi-k3.jsonl \
  --result qwen=results/qwen.jsonl \
  --result ultra-3=results/ultra-3.jsonl
```

The app binds only to `127.0.0.1:8877`. Labels are atomically saved to
`results/safe_child_llm_human_labels.jsonl` with safety verdict, response style, age appropriateness, and notes.

Code and source data are MIT licensed upstream. NeMo Gym adapter code is Apache-2.0. Prepared JSONL is generated
locally and excluded from Git.
