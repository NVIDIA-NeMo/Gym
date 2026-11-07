# BaxBench

This directory holds the 392 BaxBench tasks (28 scenarios x 14 frameworks), built from upstream
[logic-star-ai/baxbench](https://github.com/logic-star-ai/baxbench) at commit `de885cd9` with
upstream's own `Scenario.build_prompt`, in the leaderboard's main setting: the OpenAPI spec and no
security reminder. Prompts come from the repository, not from the HF dataset `LogicStar/BaxBench`,
because that dataset is older than this commit.

The messages are upstream's: the system prompt "You are an experienced full-stack developer", then the
task prompt (`benchmarks/prompts/eval/baxbench/default.yaml`). The default is 3 samples per task.
Upstream uses 10 samples at T=0.4 for non-reasoning models.

For grading, the framework images and the metrics, see `resources_servers/baxbench/README.md`.

```bash
gym eval prepare --benchmark baxbench
```

The reference score is DeepSeek R1: sec_pass@1 34.4 and pass@1 58.4 (OpenAPI spec, no reminder;
[leaderboard data](https://baxbench.com/static/data/leaderboard_data_none.json), re-run on the latest
harness on 2025-12-09).
