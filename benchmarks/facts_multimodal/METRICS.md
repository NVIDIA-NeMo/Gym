# FACTS Multimodal metrics

FACTS Multimodal reports three metrics:

- `coverage`: the fraction of essential rubric facts supported by the response;
- `factuality`: whether the response contains no clear contradiction with the rubric, image, or established facts;
- `accuracy`: whether coverage is strictly greater than 0.5 and factuality passes.

## Protocol

- Dataset: the 711-row Apache-2.0 public CSV, pinned by SHA-256 in `prepare.py`.
- Sampling: one rollout per runnable task, matching the paper's per-question evaluation.
- Image transport: identical validated image bytes are embedded for the policy model and factuality judge.
- Coverage: one Yes/No decision per essential rubric fact, aggregated by code.
- Factuality: one image-aware contradiction verdict over all rubric facts.
- Judge: `zai-org/GLM-5.3-Flash` for both coverage and factuality across the reported runs.

Image materialization produced 682 runnable rows. The prepared JSONL has SHA-256 `37f98cc5f8ed3b5e54deb50c9d0d9ab2eec163638986a71b2392d3be8a442167`. <!-- pragma: allowlist secret -->

## Results

| Model | Accuracy | Coverage | Factuality |
|---|---:|---:|---:|
| Kimi K3 | 54.69% | 76.92% | 61.44% |
| Qwen3.5 122B-A10B | 44.43% | 70.95% | 48.97% |
| Nemotron 3.5 Super VL | 37.50% | 63.58% | 45.74% |

The Kimi artifact contains 682 completed responses. Its single no-response identity was replaced by the corresponding uncapped answer from the prior run after verifying identical prompt, rubric, and embedded image bytes and reverifying it with the same judge.

The Qwen artifact contains 651 completed rows from the full collection plus uncapped recollection of all 31 no-response identities, joined by stable task ID. Fifteen recollected rows produced final answers; sixteen reached the endpoint's provider/model ceiling without a final answer. The reconciled artifact contains all 682 prepared task IDs with no duplicates or infrastructure-failure rows.

## Comparison boundary

The public CSV is the released public portion of FACTS Multimodal; the paper's leaderboard combines public and private data. The production autorater identity and exact prompts are not released. This adapter uses prompts adapted from the public Kaggle implementation and makes the judge model explicit, so these results are public-split measurements rather than reproductions of the private leaderboard.

The committed example rollouts are five real Kimi K3 outcomes using the same agent and verifier contract. Full model artifacts are not committed to the repository.
