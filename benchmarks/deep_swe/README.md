# DeepSWE in Gym

Runs the pinned 113-task DeepSWE v1.1 benchmark with Pier and OpenSandbox.
Agent and verifier sandboxes deny network access by default.

## Prepare

```bash
uv run python benchmarks/deep_swe/prepare.py
```

On the Linux amd64 runner, stage the offline `tmux` bundle:

```bash
bash benchmarks/deep_swe/stage_tmux_bundle.sh \
  cache/deep_swe_runtime/tmux-jammy-amd64.tar.gz
```

## Run

```bash
export OPENSANDBOX_API_KEY=...
export POLICY_API_KEY=...

ng_e2e_collect_rollouts \
  '+config_paths=[benchmarks/deep_swe/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]' \
  ++input_jsonl_fpath=benchmarks/deep_swe/data/deep_swe_benchmark.jsonl \
  ++output_jsonl_fpath=results/deep_swe-smoke.jsonl \
  ++max_samples=1 \
  ++policy_base_url=https://inference-api.nvidia.com/v1 \
  '++policy_api_key=${oc.env:POLICY_API_KEY}' \
  ++policy_model_name=openai/openai/openai/gpt-5.5
```

See <https://deepswe.datacurve.ai/run> for the upstream Pier workflow.

## Result

GPT-5.5 at `xhigh` with mini-swe-agent 2.4.6 scored 75/113, or 66.37%.
The published range is 61% to 73%. The raw run scored 74/113. One failed
infrastructure attempt was rerun once with the same setup and passed.

See the five [inputs](data/example.jsonl) and
[rollouts](data/example_rollouts.jsonl).
