# kernel_gym

`kernel_gym` runs a Responses API agent in an OpenSandbox GPU sandbox and scores its kernel with [KernelBench](https://github.com/ScalingIntelligence/KernelBench).

The verifier uses `eval_kernel_against_ref` with five correctness trials and 100 timing trials. Correct kernels faster than PyTorch receive reward 1; infrastructure failures are masked.

## Prepare

```bash
python responses_api_agents/kernel_gym/prepare.py \
  --kernelbench ../KernelBench \
  --image registry.example/kernelbench:latest
```

The image must be pullable by OpenSandbox. Repeat `--problem-id` to select tasks.

## Run

```bash
export OPENSANDBOX_DOMAIN=...
export OPENSANDBOX_API_KEY=...
export NEMO_GYM_SANDBOX_MODEL_BASE_URL=...

gym env start \
  --config responses_api_agents/kernel_gym/configs/kernel_gym.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type openai_model \
  --model MODEL \
  --model-url MODEL_URL \
  --model-api-key MODEL_API_KEY

gym eval run --no-serve \
  --agent kernel_gym \
  --input responses_api_agents/kernel_gym/data/kernelbench.jsonl \
  --output results/kernel_gym_rollouts.jsonl
```

Each row includes the harness trajectory, reward, compilation, correctness, runtime, speedup, and mask. Five example inputs and rollouts are in `data/`.
