# Collaborative Intelligence agent

Runs the upstream Collaborative Intelligence harness CLI for one scenario: the scenario alone, then the test version
only if the policy passed alone. It keeps the team-chat messages and final answers of both phases in the row, and
returns them as `ng_trajectory`: one invocation per phase (`solo`, `manipulation`) whose turns are the policy's
messages. Setup, data and the run command are in [benchmarks/collab_intel/README.md](../../benchmarks/collab_intel/README.md).

## Coding-scenario isolation

Set `pytest_sandbox_image` (or `COLLAB_INTEL_PYTEST_SANDBOX_IMAGE`) to a prepared Python image with pytest and the
scenario's test dependencies. Configure `sandbox_provider` as a Gym Docker or OpenSandbox provider. The agent uses
Gym's named provider configuration, but disables pytest networking, host mounts and additional Docker run arguments.
Other providers fail closed until an equivalent isolation policy is supported.

The pinned harness's `PYTEST_RUNNER` hook sends each pytest invocation through a private phase-local socket to Gym.
Gym starts a fresh sandbox, delivers only regular files from that invocation's episode directory, executes pytest and
copies back validated regular files. Neither the server filesystem, other episodes, the control socket nor provider
credentials are delivered. The trusted harness and model/judge HTTP calls stay outside that sandbox. No model-written
Python runs in the harness process. Paths that escape the phase, symlinks and special files are rejected.

A missing image, unavailable provider, unsafe file or failed transfer aborts evaluation with `agent_run_error`; it
never falls back to pytest on the server. Timeouts and cancellation stop the sandbox as well as the harness process
group. The image must contain its dependencies already because pytest has no network access.
