# Competitive Coding Resources Server

### Overview
Verifies competitive programming solutions by executing submitted code against unit tests. The server consumes agent trajectories and returns a reward based on whether the assistant's code produces the correct outputs for given test inputs.

The dataset is in preparation and the example data can be found in `data/example.jsonl`.

### Input schema
- `responses_create_params`: OpenAI Responses create params
  - Use only a user message with the problem statement and instructions (e.g., "You are an expert competitive programmer...").
  - `verifier_metadata` (required):
    - `unit_tests` (required): dict with `inputs` and `outputs` arrays containing test cases.
      - `type` (optional): `unit-tests` (the default) or `assertions`.
      - `inputs`: list of strings representing stdin input for each test case
      - `outputs`: list of strings representing expected stdout output for each test case
      - With `type: assertions`, `tests` replaces `inputs` and `outputs`: Python statements, usually asserts,
        run in order after the candidate's code.
    - `resource_limits` (optional): the task's own limits. Each applies on its own; one left out keeps the server's.
      - `time_limit_s`: seconds per test case, or for all assertions together; fractions allowed.
        Default: `unit_test_timeout_secs`.
      - `memory_limit_mb`: memory for the candidate, see [Test execution](#test-execution). Default: none
        beyond the fixed 4 GiB guard for unit tests, 4096 MB for assertions.

**Notes**
- All test cases must pass for a solution to receive a reward of 1.0
- Failed test cases result in a reward of 0.0 with detailed error information

### Test execution
Every candidate runs in a sandbox using the LiveCodeBench harness: a per-test
timer of the time limit, 4 GiB `RLIMIT_DATA` and `RLIMIT_STACK`, and disabled destructive builtins.
The stack limit cannot exceed the provider's hard limit. The overall deadline is
`(time_limit + 1) * num_tests + 5` seconds, plus one time limit for compilation when the task sets `time_limit_s`.
With the server setting `unit_test_timeout_secs: 0`, the per-case timer is disabled and the overall deadline still
applies. The response then reports `limits.time_limit_s` as null. An explicit task time limit must be positive.

A task's `memory_limit_mb` also caps address space (`RLIMIT_AS`) at the grading process's size before grading
(harness, NumPy and test inputs) plus `memory_limit_mb`. The harness's per-test stdin copies, about 11 times the
input, count against it. `metadata.memory` records the baseline and cap; without Linux's `/proc` the sample is masked.
A fresh sandbox of this server's own gets the task's `memory_limit_mb` plus 2 GiB; the verifier's group the
Environment Server lends has the memory its `verifier.sandbox.resources` sets, so set it above the largest task's.

Confirmed timeouts return `-3`. `metadata.stage` identifies `compilation` (including module initialization),
`test`, `preparation`, or the overall `grading` deadline. A worker that exits early without a result returns
`-1` with stage `worker` and its `exit_code`; it is not reported as a timeout.

All assertion statements are compiled before measuring the worker's memory baseline and held in an immutable
tuple, so fixture compilation does not consume the candidate's memory allowance. Invalid task syntax or a worker
setup failure makes the runner fail and verification mask the sample with `failure_kind: verifier_error`.
Candidate syntax errors remain ordinary compilation failures.

Assertions run in a child process the same way (`assertion_harness/`): the candidate's code, then each statement of
`tests` in one `__main__` module, under one timer and the same memory cap, stopping at the first failure. Exiting
before the last statement fails. Tests are compiled with `optimize=0`, so assertions remain active even when
the sandbox interpreter uses `-O`, `-OO`, or `PYTHONOPTIMIZE`. Results use the harness's codes (`-1` compile, `-2` assertion, `-3` time, `-4` runtime).

### Where it runs
The shipped pairing (`code_gen_environment_server`) sets `sandbox_provider` and a `verifier` block with
`mode: separate`, and no `sandbox_config`, since the task has no sandbox. The rules are in
[Where it runs](../../environment_servers/single_agent_turn/README.md#where-it-runs).

| Agent | Verifier | Why |
|---|---|---|
| Agent Server process | `separate` | No tools; the verifier runs the candidate's code (unit tests) in a group of its own. |

With a CLI harness the Agent column is `attempt group`: such a pairing adds a `sandbox_config` for the agent's group,
with `network: model_only`, `model_server`, and a model server address the sandbox reaches.

Right before `/verify`, the Environment Server starts the verifier's group from `verifier.sandbox`: the
`CODE_GEN_SANDBOX_IMAGE` image, with no network. code_gen borrows it from the `ng_sandbox_access` field of the
`/verify` body, uploads the runner and harness into a fresh directory, runs them, and releases its client; the
Environment Server stops the group.

- code_gen's `/verify` supports only `separate` (`supported_verifier_modes`), so a pairing that sets `verifier.mode`
  to `none` or `shared` fails every attempt at seed.
- The provider must apply `no_network` and lend sandboxes: Docker and OpenSandbox do. On Enroot the Environment
  Server refuses to start.
- `verifier.sandbox` and code_gen's own `sandbox_config` both take `CODE_GEN_SANDBOX_IMAGE` and the same resources;
  change both together. The thread caps of a grade in the verifier's group come from code_gen's `sandbox_config`.
- `num_processes` caps only the sandboxes code_gen starts itself. The Environment Server starts one verifier group per
  attempt that reaches `/verify`, bounded only by its `max_concurrent_episodes`, which `code_gen.yaml` leaves unset.
- A caller that lends no sandbox to `/verify`, such as a re-verification, gets a fresh sandbox from code_gen's own
  `sandbox_provider` and `sandbox_config`, as before. A sandbox that cannot be borrowed or runs no tests masks the
  sample as below; the tests never run on the server's host.
- A `single_agent_turn` pairing of code_gen without `sandbox_provider` fails every attempt at seed, since nothing
  falls back to the host.
- The LiveCodeBench benchmark configs (`benchmarks/livecodebench*/`) define their own Environment Server blocks with
  `_inherit_from: code_gen_environment_server`, so they run the same way, and an override on
  `code_gen_environment_server` reaches them too.

### Running unit tests in a sandbox
Configure a [sandbox provider](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox)
and a runtime image before starting the server. code_gen requires `sandbox_provider` and
`sandbox_config.image`, and its pairing the same image in `verifier.sandbox.image` (see
[Where it runs](#where-it-runs)). `num_processes` caps the sandboxes code_gen starts itself.

**Migration from Ray:** the Ray execution backend and dependency have been removed. Existing
`app.py` configurations must supply a sandbox provider and image; there is no local-worker fallback.
`sandbox_app.py` remains a compatibility entrypoint for existing sandbox configurations and runs
the same sandbox-only server. Other Gym environments that use Ray are unaffected.

Where each part runs:

| Files | Where they run |
|---|---|
| `sandbox/sandbox_runner.py` | Only in the sandbox; uploaded with every verification. |
| `lcb_integration/__init__.py`, `correctness.py`, `testing_util.py` | The grading harness, uploaded next to the runner in each sandbox. |
| `assertion_harness/__init__.py`, `check.py` | The assertion harness, uploaded the same way. |
| `sandbox_limits.py` | Task memory limits for both harnesses, uploaded the same way. |
| `sandbox/Dockerfile` | Builds the sandbox image; not uploaded. |
| Everything else, including the server entrypoints and `sandbox_execution.py` (which does the upload) | Only in the Gym server. |

1. Build and push the runtime image. It uses the same uv-managed CPython build as this server's
   environment, and the same numpy, scipy and pandas versions, which candidates can import. With
   per-test time limits, a slower interpreter build turns borderline passes into failures:

   ```bash
   docker build --platform linux/amd64 -t <registry>/code-gen-verifier:<tag> resources_servers/code_gen/sandbox
   docker push <registry>/code-gen-verifier:<tag>
   ```

2. Add a provider config:

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
         entrypoint: app.py
         sandbox_provider: sandbox
         sandbox_config:
           image: <registry>/code-gen-verifier@sha256:<digest>

   code_gen_environment_server:
     environment_servers:
       single_agent_turn_legacy:
         verifier:
           sandbox:
             image: <registry>/code-gen-verifier@sha256:<digest>
   ```

   The server and parent Gym CLI do not require the `ray` extra.

   Keys left out keep the server's defaults below.

   `num_processes` then caps the sandboxes code_gen starts itself, so size it to what the provider
   can run at once rather than to Ray CPUs. The default `sandbox_config` allows up to 1 CPU and 6 GiB of memory,
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
    --resources-server code_gen \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

# Collect rollouts from example problems
gym eval run --no-serve \
    --agent code_gen_simple_agent \
    --input resources_servers/code_gen/data/example.jsonl \
    --output resources_servers/code_gen/data/example_rollouts.jsonl \
    --limit null
```

## Licensing information
Apache 2.0
