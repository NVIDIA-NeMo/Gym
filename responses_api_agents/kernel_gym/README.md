# kernel_gym

`kernel_gym` runs a single model call or agent harness in a GPU sandbox and scores generated kernel with [KernelBench](https://github.com/ScalingIntelligence/KernelBench).

The verifier uses `eval_kernel_against_ref` with five correctness trials and 100 timing trials. Correct kernels faster than PyTorch receive reward 1. After the agent finishes, its sandbox is stopped. Only its submitted `solution.py` is copied into a fresh GPU sandbox, alongside the original reference and verifier from the prepared task. Agent changes to the reference, installed packages, and verifier output files cannot carry over.

Grading starts in a fresh sandbox with the original reference and verifier. However, submitted code runs inside the verifier's Python process, so this does not fully prevent deliberate grader tampering.

Inconclusive verification is retried once in another fresh sandbox (`inconclusive_verification_retries: 1`). Exhausted infrastructure failures have `evaluation_completed: false` and `mask_sample: true`. Compilation and correctness failures receive reward 0 without masking.

## Prepare

```bash
python responses_api_agents/kernel_gym/prepare.py \
  --kernelbench ../KernelBench \
  --image registry.example/kernelbench:latest
```

The image must be pullable from sandbox provider. Repeat `--problem-id` to select tasks.

## Run

Put the model and sandbox connection settings in the repository's gitignored `env.yaml`, which Gym loads automatically:

```yaml
policy_model_name: MODEL
policy_base_url: http://model-backend:8000/v1
policy_api_key: MODEL_API_KEY

sandbox:
  opensandbox:
    connection:
      domain: https://opensandbox.example
      api_key: OPENSANDBOX_API_KEY
```

```bash
gym env start \
  --config responses_api_agents/kernel_gym/configs/kernel_gym.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type openai_model

gym eval run --no-serve \
  --agent kernel_gym \
  --input responses_api_agents/kernel_gym/data/kernelbench.jsonl \
  --output results/kernel_gym_rollouts.jsonl
```

Each row includes the harness trajectory, reward, compilation, correctness, runtime, speedup, and mask.

### Model routing

The harness calls the Gym model server named by `model_server` by default, so Gym's request handling and capture apply to its model calls.

Set `sandbox_model_base_url` only when the sandbox cannot reach that server's `host:port`. Add an override in `env.yaml` pointing to a relay, tunnel, or gateway forwarding to the same Gym model server:

```yaml
kernel_gym:
  responses_api_agents:
    kernel_gym:
      sandbox_model_base_url: https://gym-model-gateway.example
```

Use the server root. Claude Code appends `/v1/messages`, and SimpleAgent appends `/v1/responses`. The relay must also forward Gym's rollout and token-capture URL prefixes unchanged. `policy_base_url` remains the backend model API address.
