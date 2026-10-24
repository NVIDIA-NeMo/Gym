# kernel_gym

`kernel_gym` runs a Responses API agent in an OpenSandbox GPU sandbox and scores its kernel with [KernelBench](https://github.com/ScalingIntelligence/KernelBench).

The verifier uses `eval_kernel_against_ref` with five correctness trials and 100 timing trials. Correct kernels faster than PyTorch receive reward 1. After the agent finishes, its sandbox is stopped. Only its submitted `solution.py` is copied into a fresh GPU sandbox, alongside the original reference and verifier from the prepared task. Agent changes to the reference, installed packages, and verifier output files cannot carry over.

KernelBench executes the submitted Python inside the evaluator process. The fresh sandbox isolates the agent's earlier filesystem changes; adversarial submissions can still interfere with the evaluator while their code runs.

Inconclusive verification is retried once in another fresh sandbox (`inconclusive_verification_retries: 1`). Exhausted infrastructure failures have `evaluation_completed: false` and `mask_sample: true`; compilation and correctness failures receive reward 0 without masking.

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

For the default Claude Code harness, `NEMO_GYM_SANDBOX_MODEL_BASE_URL` is the Anthropic API root, without `/v1` (for example, `https://inference-api.nvidia.com`). The CLI adds `/v1/messages` itself. Use the model endpoint's corresponding Responses API base URL when selecting a Responses API harness instead.

Each row includes the harness trajectory, reward, compilation, correctness, runtime, speedup, and mask. Five example inputs and rollouts are in `data/`.
