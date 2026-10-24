# Competitive Coding Resources Server

### Overview
Verifies competitive programming solutions by executing submitted code against unit tests. The server consumes agent trajectories and returns a reward based on whether the assistant's code produces the correct outputs for given test inputs.

The dataset is in preparation and the example data can be found in `data/example.jsonl`.

### Input schema
- `responses_create_params`: OpenAI Responses create params
  - Use only a user message with the problem statement and instructions (e.g., "You are an expert competitive programmer...").
  - `verifier_metadata` (required):
    - `unit_tests` (required): dict with `inputs` and `outputs` arrays containing test cases.
      - `inputs`: list of strings representing stdin input for each test case
      - `outputs`: list of strings representing expected stdout output for each test case

**Notes**
- All test cases must pass for a solution to receive a reward of 1.0
- Failed test cases result in a reward of 0.0 with detailed error information

### Test execution (for now)
We use the LiveCodeBench execution code. By default each candidate runs in a child process on a Ray
worker, limited by the harness itself: a per-test alarm of `unit_test_timeout_secs`, an overall
deadline of `(unit_test_timeout_secs + 1) * num_tests + 5` seconds, 4 GiB `RLIMIT_DATA` and
`RLIMIT_STACK`, and disabled destructive builtins.

### Running unit tests in a sandbox
Set `sandbox_provider` to run each verification in a fresh sandbox from any
[sandbox provider](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox) instead. The
server uploads `sandbox/sandbox_runner.py` together with the unchanged harness above, so the same
limits apply inside the sandbox; the sandbox only has to be large enough not to undercut them. The
exception is the stack: the harness can raise it only up to the sandbox's hard limit, which some
providers set far below 4 GiB.

Where each part runs:

| Files | Where they run |
|---|---|
| `sandbox/sandbox_runner.py` | Only in the sandbox; uploaded with every verification. |
| `lcb_integration/__init__.py`, `correctness.py`, `testing_util.py` | The grading harness: on a Ray worker by default, or uploaded next to the runner when `sandbox_provider` is set. |
| `sandbox/Dockerfile` | Builds the sandbox image; not uploaded. |
| Everything else, including `app.py` and `sandbox_execution.py` (which does the upload) | Only in the Gym server. |

1. Build and push the runtime image. It uses the same uv-managed CPython build as this server's
   environment, and the same numpy, scipy and pandas versions, which candidates can import. With
   per-test time limits, a slower interpreter build turns borderline passes into failures:

   ```bash
   docker build --platform linux/amd64 -t <registry>/code-gen-verifier:<tag> resources_servers/code_gen/sandbox
   docker push <registry>/code-gen-verifier:<tag>
   ```

2. Add a provider config and enable the sandbox path:

   ```bash
   export CODE_GEN_SANDBOX_IMAGE=<registry>/code-gen-verifier:<tag>
   gym env start \
       --model-type openai_model \
       --config resources_servers/code_gen/configs/code_gen.yaml \
       --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
       ++code_gen.resources_servers.code_gen.sandbox_provider=sandbox
   ```

   Or put the same settings in a config file passed after the server config:

   ```yaml
   # code_gen_sandbox.yaml
   # gym env start --model-type openai_model --config resources_servers/code_gen/configs/code_gen.yaml --config code_gen_sandbox.yaml
   config_paths:
     - nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

   code_gen:
     resources_servers:
       code_gen:
         sandbox_provider: sandbox
         sandbox_config:
           image: <registry>/code-gen-verifier@sha256:<digest>
   ```

   Keys left out keep the server's defaults below.

   `num_processes` then caps concurrent sandboxes, so size it to what the provider can run at once
   rather than to Ray CPUs. The default `sandbox_config` allows up to 1 CPU and 6 GiB of memory,
   above the 4 GiB `RLIMIT_DATA`, so a memory-hungry candidate fails its test instead of getting the
   whole sandbox killed. Unlike a Ray task's CPU, that CPU is a limit, not a reservation, and the
   per-test time limits are wall-clock. The default `sandbox_config.provider_options.network_policy`
   is an OpenSandbox option that denies all egress. Other providers need their own setting (Docker's
   `network: none`), and Daytona, E2B and OpenShell reject the key, so remove it for them.

A failure of the sandbox itself, not of the candidate, returns `mask_sample: true` with
`failure_kind` set to `provider_unavailable` (the sandbox could not start) or `verifier_error` (the
runner failed or produced no valid result), and `failure_reason` holding the details. Gym's
aggregate metrics leave masked samples out; whether training skips them is up to the trainer.

### Example of rollouts and usage

```bash
# Running the server
gym env start \
    --model-type openai_model \
    --resources-server code_gen

# Collect rollouts from example problems
gym eval run --no-serve \
    --agent code_gen_simple_agent \
    --input resources_servers/code_gen/data/example.jsonl \
    --output resources_servers/code_gen/data/example_rollouts.jsonl \
    --limit null
```

## Licensing information
Apache 2.0
