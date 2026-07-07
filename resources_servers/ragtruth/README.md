# RAGTruth Resources Server

Case-level **hallucination detection**, ported from the nemo-evaluator BYOB
benchmarks `ragtruth_qa` / `ragtruth_summary` / `ragtruth_data2txt`. The model
is shown a `(reference context, candidate response)` pair and must emit a
`{"hallucination list": [...]}` JSON object listing any hallucinated spans. The
per-sample reward is `1.0` when the model's binary "any hallucination?" verdict
matches the gold label, else `0.0`.

Source dataset: [`ParticleMedia/RAGTruth`](https://github.com/ParticleMedia/RAGTruth)
(`dataset/response.jsonl` + `dataset/source_info.jsonl`). Upstream eval:
`benchmarks/ragtruth/ragtruth/baseline/{dataset,prepare_dataset,predict_and_evaluate}.py`.

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

`compute_metrics` reports both the BYOB headline metric and the original paper's:

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

On first use the upstream JSONL files are downloaded to
`$XDG_CACHE_HOME/byob_ragtruth` (or `~/.cache/byob_ragtruth`). Set
`RAGTRUTH_DATASET_DIR=/path/to/dir` to read a pre-staged copy, or
`RAGTRUTH_NO_FETCH=1` to disable network fetches (air-gapped clusters).

## Run

No model is needed on the Gym side for scoring — the policy model generates the
hallucination-detection JSON, and verify() scores it deterministically.

```bash
ng_run "+config_paths=[resources_servers/ragtruth/configs/ragtruth.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Test

```bash
ng_test +entrypoint=resources_servers/ragtruth
# or, in the repo dev env:
ng_dev_test +entrypoint=resources_servers/ragtruth
```
