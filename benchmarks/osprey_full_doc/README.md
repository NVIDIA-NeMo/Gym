# Osprey `full_doc` benchmark

Osprey `full_doc` evaluates full-document insurance extraction with the original
Osprey deterministic verifier. Gym uses the standard `tool_simulation_agent`
benchmark path, so tools are not executed during rollout.

Each rollout is scored as either correct or one of these stable failure
categories:

- `API error`
- `Extraction error`
- `False positive`
- `False negative`
- `Incorrect value`

## Quick start

Everything below runs from the repo root.

### 1. Get the benchmark dataset

The canonical prepared benchmark rows are stored in the GitLab MLflow registry
as:

- dataset name: `osprey_full_doc`
- version: `0.0.1`
- artifact: `osprey_full_doc_benchmark.jsonl`

Download them into the benchmark data directory:

```bash
ng_download_dataset_from_gitlab \
    +dataset_name=osprey_full_doc \
    +version=0.0.1 \
    +artifact_fpath=osprey_full_doc_benchmark.jsonl \
    +output_fpath=benchmarks/osprey_full_doc/data/osprey_full_doc_benchmark.jsonl
```

This artifact contains the prepared `770` benchmark rows in Gym JSONL format.
The rows already contain the Responses API input, tool definitions, and the
pinned live-eval sampling contract inside `responses_create_params`:
`temperature=1.0`, `top_p=0.95`, `max_output_tokens=32000`.

`ng_download_dataset_from_gitlab` requires MLflow credentials in repo-root
`env.yaml`:

```yaml
mlflow_tracking_uri: https://<gitlab-host>/api/v4/projects/<PROJECT_ID>/ml/mlflow
mlflow_tracking_token: <your-gitlab-api-token>
```

### 2. Start the benchmark services

Point Gym at the policy model you want to evaluate, then start the benchmark
stack:

```bash
export POLICY_BASE_URL=<your-policy-base-url>
export POLICY_MODEL_NAME=<your-policy-model-name>
export POLICY_API_KEY=<your-policy-api-key>

ng_run \
    "+config_paths=[benchmarks/osprey_full_doc/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +policy_base_url="${POLICY_BASE_URL}" \
    +policy_api_key="${POLICY_API_KEY}" \
    +policy_model_name="${POLICY_MODEL_NAME}"
```

### 3. Collect rollouts

Run the prepared benchmark rows through the policy model:

```bash
ng_collect_rollouts \
    +agent_name=osprey_full_doc_benchmark_agent \
    +input_jsonl_fpath=benchmarks/osprey_full_doc/data/osprey_full_doc_benchmark.jsonl \
    +output_jsonl_fpath=results/osprey_full_doc_rollouts.jsonl
```

Each rollout row includes:

- `reward`
- `is_correct`
- `wrong_prediction_type`
- `clean_prediction`

Use `wrong_prediction_type` as the stable field for aggregate failure analysis.

### 4. Summarize the results

```bash
python3 benchmarks/osprey_full_doc/eval_rollouts.py \
    --rollouts-jsonl results/osprey_full_doc_rollouts.jsonl \
    --expected-input-jsonl benchmarks/osprey_full_doc/data/osprey_full_doc_benchmark.jsonl \
    --summary-json results/osprey_full_doc_rollouts_summary.json \
    --report-md results/osprey_full_doc_rollouts_report.md
```

If you already have Gym's aggregate metrics file, you can summarize that
directly instead:

```bash
python3 benchmarks/osprey_full_doc/eval_rollouts.py \
    --aggregate-json results/osprey_full_doc_rollouts_aggregate_metrics.json \
    --summary-json results/osprey_full_doc_rollouts_summary.json \
    --report-md results/osprey_full_doc_rollouts_report.md
```

## Regenerating the benchmark rows

Most users should download `osprey_full_doc_benchmark.jsonl` from MLflow and
skip this step. Rebuild the benchmark locally only if you need to regenerate the
prepared rows from a raw Osprey source JSON:

```bash
python3 benchmarks/osprey_full_doc/prepare.py \
    --source-json /path/to/osprey_extraction_benchmark_dataset_full_doc.json
```

This writes:

- `benchmarks/osprey_full_doc/data/osprey_full_doc_benchmark.jsonl`
- `benchmarks/osprey_full_doc/data/osprey_extraction_benchmark_dataset_full_doc.json`

If you prefer the Gym prepare entrypoint, set the source path explicitly and run:

```bash
export OSPREY_FULL_DOC_SOURCE_JSON=/path/to/osprey_extraction_benchmark_dataset_full_doc.json

ng_prepare_benchmark "+config_paths=[benchmarks/osprey_full_doc/config.yaml]"
```

## Uploading benchmark rows to MLflow

Most users should download the published dataset and skip this step. Maintainers
can upload the prepared benchmark JSONL with:

```bash
./benchmarks/osprey_full_doc/upload_to_mlflow.sh
```

The helper uploads only
`benchmarks/osprey_full_doc/data/osprey_full_doc_benchmark.jsonl` as dataset
`osprey_full_doc` version `0.0.1`. It does not upload the raw source snapshot
JSON.

## Optional one-command helper

If your policy model is already served behind an OpenAI-compatible endpoint,
`run_ultra_eval.sh` prepares the benchmark stack, collects rollouts, and writes
the summary files:

```bash
POLICY_BASE_URL=<your-policy-base-url> \
POLICY_MODEL_NAME=<your-policy-model-name> \
POLICY_API_KEY=<your-policy-api-key> \
OUTPUT_DIR=results/osprey_full_doc_eval \
bash benchmarks/osprey_full_doc/run_ultra_eval.sh
```

If the NeMo Gym CLI binaries are not already on `PATH`, set
`NG_BIN_DIR=/path/to/.venv/bin` first. For a quick smoke run instead of the full
dataset, add `LIMIT=1`.
