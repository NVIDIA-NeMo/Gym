# RAGTruth Resources Server

Case-level **hallucination detection** over the `qa` / `summary` / `data2txt`
slices. The model is shown a `(reference context, candidate response)` pair and must emit a
`{"hallucination list": [...]}` JSON object listing any hallucinated spans. The
per-sample reward is `1.0` when the model's binary "any hallucination?" verdict
matches the gold label, else `0.0`.

Source dataset: [`ParticleMedia/RAGTruth`](https://github.com/ParticleMedia/RAGTruth)
(`dataset/response.jsonl` + `dataset/source_info.jsonl`).

## Task slices

The three slices differ only by the prompt template (applied at prep time);
scoring is identical. `task_type` rides on each row, and `compute_metrics`
breaks results down per slice.

| Slice | Context | Split file |
|-------|---------|-----------|
| `QA` | retrieved passages | `data/test_qa.jsonl` |
| `Summary` | original news article | `data/test_summary.jsonl` |
| `Data2txt` | structured JSON record | `data/test_data2txt.jsonl` |

## Metrics

`compute_metrics` reports both the headline metric and the original paper's:

- `mean_reward` — mean per-sample accuracy (also the reward).
- `precision` / `recall` / `f1` — corpus-level over the binary halu labels,
  reconstructed from per-row `is_halu` (gold) and `pred_halu` (predicted) flags.
- `task_type/<slice>/{accuracy,precision,recall,f1,count}` — per-slice breakdown.
- `parse_fail_rate` — fraction of responses that didn't parse as JSON.

> **Reasoning models:** verify() strips a leading `<think>…</think>` block and
> any ```json fences before parsing.

## Prepare the dataset

The committed `data/example.jsonl` is a tiny synthetic sample for tests and
smoke runs. Build the full test splits:

```bash
python resources_servers/ragtruth/prepare_ragtruth.py
# -> data/test_{qa,summary,data2txt}.jsonl
```

On first use the source JSONL files are downloaded to
`$XDG_CACHE_HOME/ragtruth` (or `~/.cache/ragtruth`). Set
`RAGTRUTH_DATASET_DIR=/path/to/dir` to read a pre-staged copy, or
`RAGTRUTH_NO_FETCH=1` to disable network fetches (air-gapped clusters).

## Run

No model is needed on the Gym side for scoring — the policy model generates the
hallucination-detection JSON, and verify() scores it deterministically.

```bash
gym env start --resources-server ragtruth --model-type vllm_model
```

## Example rollouts and metrics

`data/example_rollouts.jsonl` and `data/example_metrics.json` are committed and show live examples of rollouts and metrics.

To collect rollouts from a live model:

```bash
gym eval run --no-serve \
    --agent ragtruth_simple_agent \
    --input resources_servers/ragtruth/data/example.jsonl \
    --output resources_servers/ragtruth/data/example_rollouts.jsonl

tail -n 1 resources_servers/ragtruth/data/example_rollouts.jsonl | jq | less
```

## Test

```bash
gym env test --resources-server ragtruth
```
