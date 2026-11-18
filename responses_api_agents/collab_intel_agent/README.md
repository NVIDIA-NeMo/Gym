# Collaborative Intelligence agent

Runs the upstream Collaborative Intelligence harness CLI for one scenario: the scenario alone, then the test version
only if the policy passed alone. It keeps the team-chat messages and final answers of both phases in the row, and
returns them as `ng_trajectory`: one invocation per phase (`solo`, `manipulation`) whose turns are the policy's
messages. Setup, data and the run command are in [benchmarks/collab_intel/README.md](../../benchmarks/collab_intel/README.md).

The policy's code in the coding scenarios runs in a sandbox, not on the agent's host. The agent starts the harness through
`boxed_cli.py`, which sends each python subprocess the harness makes (pytest in the policy's workspace, a few checkers'
probes) to the agent's `/ci_exec/<phase>/run` route (`boxed_exec.py`). The workspace goes into one sandbox per phase, the
command runs there with no network, and the output and the files it changed come back. Any other way of starting the
harness's interpreter is refused. `code_execution: host` restores the unsandboxed run.
