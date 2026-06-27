# swe_bench resources server

SWE-bench **Environment** resources server: `seed_session` returns a `SessionDescriptor` (topology **C**, per-instance sandbox spec); `verify` grades a model patch in a **fresh** eval sandbox (hermetic twin).

Grading harnesses, parsing, and `verify_task` live as **private modules** under this directory (relocated from `responses_api_agents/swe_env/`).

## Wiring

```yaml
responses_api_agents:
  claude_code_agent:
    resources_server:
      type: resources_servers
      name: swe_bench
```

## Tests

```bash
gym env test --resources-server swe_bench
```

Unit tests use a fake sandbox provider (no Docker required).

## Dataset

Prepare SWE-bench Verified rows with `verifier_metadata` (see `prepare.py`):

```bash
python resources_servers/swe_bench/prepare.py --limit 5 --no-images
```

Each JSONL row includes `verifier_metadata.instance_id`, `instance_dict`, `dataset_name`, and optional `container_formatter`.

## Rollouts

```bash
gym env start --resources-server swe_bench --agent claude_code_swe_bench --model-type openai_model
gym eval run --no-serve --agent claude_code_swe_bench \
  --input resources_servers/swe_bench/data/swebench_verified.jsonl \
  --output results/swe_bench_rollouts.jsonl
```

Agent servers pass `verifier_metadata.model_patch` (git unified diff) on `POST /verify`.
