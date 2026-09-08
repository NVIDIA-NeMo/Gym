# kernel_gym

`kernel_gym` runs a coding agent inside a GPU sandbox, then evaluates the edited kernel with [KernelBench](https://github.com/ScalingIntelligence/KernelBench) in the same sandbox.

Compilation failures, incorrect kernels, and correct-but-slow kernels receive 0. Sandbox and timeout failures are masked.

## Prepare

Build and push the task image:

```bash
docker build -t registry.example/kernel-gym:kernelbench -f responses_api_agents/kernel_gym/Dockerfile .
docker push registry.example/kernel-gym:kernelbench
```

Create a one-problem Level 1 dataset from a KernelBench checkout. Problem 19 (ReLU) is the default example.

```bash
python responses_api_agents/kernel_gym/prepare.py \
  --kernelbench ../KernelBench \
  --image registry.example/kernel-gym:kernelbench
```

Repeat `--problem-id` to add tasks. The preparer writes public `reference.py` and baseline `solution.py` files plus a verifier script that is hidden until the agent exits.
Five prepared rows and five completed trajectories are included in `data/example_input.jsonl` and `data/example_rollouts.jsonl`.

## Run

```bash
export OPENSANDBOX_DOMAIN=...
export OPENSANDBOX_API_KEY=...
export NEMO_GYM_SANDBOX_MODEL_BASE_URL=...

gym env start \
  --config responses_api_agents/kernel_gym/configs/kernel_gym.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model

gym eval run --no-serve \
  --agent kernel_gym \
  --input responses_api_agents/kernel_gym/data/kernelbench.jsonl \
  --output results/kernel_gym_rollouts.jsonl \
  --limit 1
```

## Result fields

Each rollout contains `reward`, `compiled`, `correctness`, `runtime`, `ref_runtime`, `speedup`, and `mask_sample` alongside the harness trajectory.
