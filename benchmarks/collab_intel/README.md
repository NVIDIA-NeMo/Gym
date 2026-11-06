# Collaborative Intelligence

[Collaborative Intelligence](https://github.com/zou-group/collaborative-intelligence) (Zou group, arXiv 2611.02633):
the model works in a team chat with partner seats and one scripted teammate who creates a problem (pressure to cut a
safety corner, a faked result, a hard bargain, a compromised handoff, ...). 8 dimensions, 160 of 240 scenarios released.

- `prepare.py` downloads the release at commit 3de5f7d into `data/repo/` and writes one row per released scenario of the
  scored set and test seed (157 scenarios x 3 seeds = 471 rows; the 3 encrypted GPQA-based scenarios are skipped).
- `collab_intel_agent` runs the upstream harness CLI from `data/repo/` in that repo's own uv venv: the scenario alone,
  then the test version only if the policy passed alone. Reward = the test run's pass. `alone_pass=false` rows are not
  scored (upstream score = mean test pass over scenarios passed alone; upstream also averages over dimensions).
- Partners and referee judges are one self-hosted OpenAI-compatible endpoint (`COLLAB_INTEL_JUDGE_BASE_URL`,
  `COLLAB_INTEL_JUDGE_MODEL`, default deepseek-v41-flash, thinking off). The upstream leaderboard uses Claude/GPT/Qwen3
  partners and Claude judges, so scores here are not comparable to it.
- Coding scenarios run pytest on model-written files in a local temp dir: give the job a memory cap.
- License: data CC BY 4.0 (7 scenarios CC BY-NC-SA 4.0), code MIT. Evaluation only; never train on it (some scenarios
  reuse GPQA).

```bash
python benchmarks/collab_intel/prepare.py
COLLAB_INTEL_JUDGE_BASE_URL=http://<judge host>/v1 gym eval run --config benchmarks/collab_intel/config.yaml \
  --split benchmark --output out/rollouts.jsonl --concurrency 16 \
  --model-type vllm_model --model <served name> --model-url http://<policy>/v1 --model-api-key dummy
```
