# SWE external resources server

Verification for software-engineering tasks with prebuilt images and supplied
test/solution scripts. Sandboxes use `nemo_gym.sandbox`; agent runs need a provider
with reconnection support, such as OpenSandbox.

## Data

Each JSONL row contains `responses_create_params` and `verifier_metadata`: the task
ID, image reference, workdir, test/solution assets, and resource/time limits.
See `task_data.py` for the schema. The private dataset belongs in
`data/training.jsonl`, which is gitignored.

The five public examples adapt the [SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2)
tasks selected in [Gym #3327](https://github.com/NVIDIA-NeMo/Gym/pull/3327).
Dataset license: CC-BY-4.0. Their embedded [SWE-rebench parser](https://github.com/SWE-rebench/SWE-rebench-V2)
is pinned to `c71902a8cf8d2b725f63d51f199f4d3e56f68d2d` and retains its MIT
license. Adapter code is Apache-2.0.

## Grading

The current implementation lets the agent edit its sandbox, then uploads the
held-out tests to `/tests` and runs `bash /tests/test.sh` in that same sandbox.
The script writes a binary reward to `/logs/verifier/reward.txt`. Missing or
invalid output is an incomplete evaluation, not a measured task failure.
The sandbox is stopped after grading. Same-sandbox verification does not isolate
the grader from agent changes to the runtime.

Golden mode creates a fresh sandbox, runs `/solution/solve.sh`, then grades it.
It evaluates the supplied solution, not the agent, and needs no model inference.

## Running an agent

`configs/swe_external1_opencode.yaml` pairs OpenCode with the normal resources
server. Provider credentials, registry access, and model settings stay private.

```bash
gym env start \
  --config resources_servers/swe_external1/configs/swe_external1_opencode.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml \
  --config /path/to/private-runtime.yaml
```

`configs/swe_external1.yaml` defines normal and golden server instances.
`configs/swe_external1_example.yaml` pairs the public examples with the golden
verifier; use the OpenCode pairing for actual agent evaluation.

## Validation

`verified: false` remains intentional. Live golden checks, model runs, and genuine
`data/example_rollouts.jsonl` are pending. `example_metrics.json` contains dataset
metadata, not an oracle pass rate.
