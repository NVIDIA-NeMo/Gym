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

### Test execution
Every candidate runs in a sandbox using the LiveCodeBench harness: a per-test
alarm of `unit_test_timeout_secs`, an overall deadline of
`(unit_test_timeout_secs + 1) * num_tests + 5` seconds, 4 GiB `RLIMIT_DATA` and
`RLIMIT_STACK`, and disabled destructive builtins. The stack limit cannot exceed the provider's hard limit.

### A sandbox for grading each attempt
The shipped pairing (`code_gen_environment_server`) states both sandbox opt-out switches (see "A sandbox for each
attempt" in the `single_agent_turn` README). code_gen is a pure QA task whose verifier runs the candidate's code, so
the task opts out and the verifier does not:

```yaml
code_gen_environment_server:
  environment_servers:
    single_agent_turn_legacy:
      sandbox_opt_out:
        task: true
        verifier: false
```

| Step | Where it runs |
|---|---|
| Simple Agent's loop | In the Agent Server process. The model calls no tools. |
| Grading group | Right before `/verify`, the Environment Server starts a group for the verifier alone, with no egress (`no_network`), from code_gen's `sandbox_config.image`, which code_gen declares at seed. |
| Unit tests (`/verify`) | code_gen borrows the group from the `ng_sandbox_access` field of the `/verify` body, uploads the runner and harness into a fresh directory, runs them, and releases its client. |
| Stop | The Environment Server stops the group after the episode. code_gen never stops it. |

So no sandbox runs while the model generates, and each grade gets a short-lived sandbox with no egress, as code_gen's
own verifier sandbox had. The provider must apply `no_network` and lend sandboxes: OpenSandbox and Docker do.
`num_processes` caps only the sandboxes code_gen starts itself; the Environment Server starts one group per attempt
that reaches `/verify`, bounded only by its `max_concurrent_episodes` when that is set.

A caller that lends no sandbox to `/verify`, such as a re-verification, gets a fresh sandbox from code_gen's own
`sandbox_provider` and `sandbox_config`, as before; so does a configuration that sets `sandbox_opt_out.verifier:
true`, which code_gen should never get, since its verifier runs model code. A sandbox that cannot be borrowed or runs
no tests masks the sample as below; the tests never run on the server's host.

The LiveCodeBench benchmark configs (`benchmarks/livecodebench*/`) define their own Environment Server blocks with
`_inherit_from: code_gen_environment_server`, so they run the same way. Overrides are merged before `_inherit_from` is
resolved, so an override on `code_gen_environment_server`, like the one below, reaches them too. A
`single_agent_turn` pairing of code_gen without `sandbox_provider` fails every attempt at seed, because code_gen
declares a group that nothing starts; nothing falls back to the host.

#### Variant: one group shared by the agent and the verifier
A harness that must run in a container turns the task opt-out off. The group then starts at seed: the agent borrows it
for its loop, and `/verify` borrows it after the agent closes.

```yaml
use_absolute_ip: true
absolute_ip_server_types: [responses_api_models]
code_gen_environment_server:
  environment_servers:
    single_agent_turn_legacy:
      sandbox_opt_out:
        task: false
        verifier: false
      sandbox_config:
        network: model_only
        model_server: policy_model
```

This shape needs:

- a provider that applies `model_only` and lends sandboxes, such as OpenSandbox with
  `networking.enforces_network_policy: true`. Docker and Enroot refuse `model_only` at startup;
- a model server at an address the sandbox reaches. `use_absolute_ip` with
  `absolute_ip_server_types: [responses_api_models]` makes only model servers advertise this host's address, and
  every other server stays on loopback. On a host whose name resolves to a loopback address, startup asks for
  `default_host` instead;
- an image that also runs the harness. For Simple Agent: Linux on x86_64 or aarch64, glibc 2.17 or later or musl,
  with `bash`, `tar` and a writable `/tmp`. The Agent Server uploads a Python build that matches the image, so the
  image's own Python is not used. The `sandbox/Dockerfile` image qualifies;
- one sandbox per running attempt for the whole generation.

Grading in the shared group keeps the agent's `model_only` egress, because providers cannot change a running
sandbox's egress yet, so candidate code can reach the model server's address while it is graded. For a trusted local
test of this variant on Docker, `network: inherit` with `model_server: null` gives the group the host's network.

Enroot can take neither shape: it applies only `inherit`, so it refuses `no_network` (the shipped default) and
`model_only` (the variant) at startup.

### Running unit tests in a sandbox
Configure a [sandbox provider](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox)
and a runtime image before starting the server. Both `sandbox_provider` and `sandbox_config.image`
are required. `num_processes` caps concurrent sandboxes.

**Migration from Ray:** the Ray execution backend and dependency have been removed. Existing
`app.py` configurations must supply a sandbox provider and image; there is no local-worker fallback.
`sandbox_app.py` remains a compatibility entrypoint for existing sandbox configurations and runs
the same sandbox-only server. Other Gym environments that use Ray are unaffected.

Where each part runs:

| Files | Where they run |
|---|---|
| `sandbox/sandbox_runner.py` | Only in the sandbox; uploaded with every verification. |
| `lcb_integration/__init__.py`, `correctness.py`, `testing_util.py` | The grading harness, uploaded next to the runner in each sandbox. |
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
   ```

   The server and parent Gym CLI do not require the `ray` extra.

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
