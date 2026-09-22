# FACTS Multimodal metrics

The headline metric is binary accuracy: essential-fact coverage must be strictly greater than 0.5 and the response must receive a no-contradiction verdict. Coverage and factuality are reported separately to show which gate failed.

## Protocol fidelity

- Dataset: the 711-row Apache-2.0 public CSV, pinned by SHA-256 in `prepare.py`.
- Sampling: one rollout per runnable task, matching the paper's per-question evaluation.
- Image transport: identical validated image bytes are embedded for the policy model and factuality judge.
- Coverage: one yes/no decision per essential rubric fact, aggregated by code.
- Factuality: one image-aware contradiction verdict over all rubric facts.
- Exclusions: only image download or validation failures; every exclusion is written to `image_download_report.csv`.

## Known divergence from the leaderboard

The paper does not release the production autorater model or exact prompts. This adapter uses prompts adapted from the public Kaggle example and makes the judge model explicit in Gym configuration. Public-split results must name that judge and the prepared-dataset hash. They are not directly comparable to the combined public-and-private leaderboard values.

The committed example rollouts are five real Kimi K3 outcomes from the recovered baseline and include both passing and failing rows. Full model results belong in external run packages rather than this source tree.

## Baselines

The public source contains 711 rows. Image materialization produced 682 runnable rows and rejected 29 unavailable or invalid third-party assets, including a 1x1 GIF whose rubric described an electric slicer. The prepared JSONL scored below has SHA-256 `37f98cc5f8ed3b5e54deb50c9d0d9ab2eec163638986a71b2392d3be8a442167`. <!-- pragma: allowlist secret -->

Kimi K3 and Qwen3.5 used `max_output_tokens: 32768`, one rollout per task, and the same `zai-org/GLM-5.3-Flash` coverage/factuality judge. Every frozen policy answer was then reverified with up to three attempts for a malformed judge verdict. Both final runs contain the exact same 682 task IDs with no duplicates or infrastructure failures.

| Model | Accuracy | Coverage | Factuality | Completed | Incomplete / empty | Judge parse failures |
|---|---:|---:|---:|---:|---:|---:|
| Kimi K3 | 54.69% | 76.92% | 61.44% | 681/682 | 1/682 | 1 coverage, 0 factuality |
| Qwen3.5 122B-A10B | 43.55% | 69.76% | 48.09% | 651/682 | 31/682 | 1 coverage, 0 factuality |

The final rollout SHA-256 values are `16a3a8321ca0717dba2df927b78fd6d24fe1be78723a3888f63b445cc408aad9` for Kimi K3 and `2720175e676a664e44cb15d37a17e3430d07a73fba8b1ae67cd0c39eff9e4414` for Qwen3.5. Their aggregate-metrics sidecars hash to `74dac00c24d957102abfa597186951810e1414ac402c3c9d30b875384e99379c` and `3ccf19c6ceba27503bd0454fc747d693624c695015adfc619c0cd4fc3186afa5`, respectively. <!-- pragma: allowlist secret -->

An earlier Nemotron 3.5 Super VL run predates the repaired materialization. It scores 679 of the current 682 task IDs: 37.56% accuracy, 63.68% coverage, and 45.80% factuality. The unmatched rows are one earlier terminal failure and two source images that became reachable later. It is a historical vision baseline, not a complete matched-cohort result. Nemotron 3 Ultra is text-only and is excluded from this image benchmark.
