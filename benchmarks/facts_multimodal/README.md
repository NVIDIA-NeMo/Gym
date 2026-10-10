# FACTS Multimodal

FACTS Multimodal evaluates factual, sufficiently complete answers to image-based questions. Each public task contains a prompt, an image URL, and a human-authored rubric of essential and non-essential atomic facts.

The benchmark follows the public FACTS scoring rule:

```text
coverage   = essential facts supported by the response / essential facts
factuality = 1 when the response has no clear contradiction with the rubric, image, or established facts
accuracy   = (coverage > 0.5) AND factuality
reward     = accuracy
```

The coverage judge receives text only. The factuality judge receives the same image bytes as the policy model. The two calls are separate so omission and contradiction remain independently auditable. The prompts are adapted from the public Kaggle implementation because the paper does not publish the production autorater prompts or model.

## Prepare the public split

Preparation downloads the Apache-2.0 public CSV from Kaggle and requires SHA-256 `140b09d46cf8907703b4c96833ea8b966db59e065c19ca18666bfb6025974e17`. It materializes every reachable image, verifies that the bytes decode as JPEG, PNG, GIF, or WebP, rejects images smaller than 16 pixels on either edge, and writes an exclusion report.

```bash
gym eval prepare --benchmark facts_multimodal
```

The release contains 711 source rows. Third-party URLs are mutable, so the runnable row count can change as images disappear or reappear. `benchmarks/facts_multimodal/data/image_download_report.csv` is the run-specific record of included and excluded item IDs. Generated data and images are gitignored; only five real example rows and their baseline rollouts are committed.

By default, preparation embeds image bytes as data URLs. This prevents the policy and judge endpoints from independently fetching different content:

```bash
gym eval prepare --benchmark facts_multimodal ++use_base64_images=false
```

The URL mode is smaller but less reproducible and should only be used when both endpoints can fetch every source URL.

## Run

The policy model and judge are supplied as ordinary `inference_provider` model servers. The judge must support image input.

```bash
gym eval run --benchmark facts_multimodal \
  --model-type inference_provider \
  --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" \
  --model-api-key "$POLICY_API_KEY" \
  --split benchmark \
  --num-repeats 1 \
  --output results/facts_multimodal.jsonl \
  ++judge_base_url="$JUDGE_BASE_URL" \
  ++judge_api_key="$JUDGE_API_KEY" \
  ++judge_model_name="$JUDGE_MODEL"
```

The paper reports one answer per question, so the benchmark default is one rollout per task. Empty or incomplete policy responses receive zero without spending judge calls. Aggregate output includes accuracy, coverage, factuality, incomplete-response rate, empty-generation rate, and both judge-parse failure rates.

To score frozen model answers with a different judge, use `resources_servers/facts_multimodal/configs/reverify.yaml` and `gym eval reverify`.

## Comparison boundary

The public CSV is only the 711-row public part of the benchmark; the paper's leaderboard combines public and private data. The production autorater identity and exact prompts are not released. Local results are therefore public-split, adapter-and-judge-qualified measurements, not submissions to or reproductions of the private Kaggle leaderboard.
