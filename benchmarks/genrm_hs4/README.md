# genrm_hs4

Download preprocessed data from private GitLab (`genrm_hs4`, version `0.1.0`):

```bash
uv run --extra mlflow python -m benchmarks.genrm_hs4.prepare
```

Requires registry credentials in `env.yaml`. See [GenRM evaluation](../../resources_servers/genrm_eval/README.md) for evaluation commands and scoring.

For an LLM as a judge, prepare and run with `llm_as_a_judge.yaml`. The [llm_as_a_judge_prompt](../../resources_servers/genrm_eval/configs/llm_as_a_judge_prompt.yaml) is shared with the other GenRM benchmarks.
