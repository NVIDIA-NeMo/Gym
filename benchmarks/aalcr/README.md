# Prepare data
```bash
gym eval prepare --benchmark aalcr
```

# Dataset version

The benchmark tracks a pinned upstream version, currently **v1.1**, rather than resolving `main` — upstream
publishes breaking revisions in place, and v1.1 corrected 16 of the 100 answer keys.

Upstream versions the answer keys and the judge protocol *together*, so `resources_servers/aalcr/versions.py`
selects them together: one `version` fixes the dataset revision, the judge system prompt, the user prompt and
the verdict format. Choosing them independently grades v1.1 keys under the v1.0 protocol (or the reverse),
which matches no published version and yields a plausible but meaningless score.

To reproduce results from an older version, move both selectors together:

```bash
gym eval prepare --benchmark aalcr ++prepare_script_args.version=1.0
gym eval run --benchmark aalcr ... '++aalcr_benchmark_resources_server.resources_servers.aalcr.dataset_version=1.0'
```

Scores are **not comparable across upstream versions** — results produced under v1.0 must be re-run, not
compared.

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
