# Prepare data
```bash
gym eval prepare --benchmark aalcr
```

The upstream HuggingFace dataset is pinned to a fixed revision (`HF_REVISION` in `prepare.py`, currently
dataset v1.0) so that scores stay reproducible when upstream publishes a new revision. To prepare against a
different dataset version, override the revision:

```bash
gym eval prepare --benchmark aalcr ++prepare_script_args.revision=<commit-sha-or-tag>
```

Note that scores are not comparable across upstream dataset versions — v1.1 changed 16 of the 100 answer
keys and revised the judge protocol.

# Run
```bash
gym eval run \
    --model-type vllm_model \
    --benchmark aalcr \
    ++output_jsonl_fpath=results/benchmarks/aalcr.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++resume_from_cache=true \
    ++ray_head_node_address=auto \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<> \
    '++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.base_url=<>' \
    '++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.model=<>' \
    '++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.api_key=<>'
```
