# Collaborative Intelligence agent

Runs the upstream Collaborative Intelligence harness CLI for one scenario: the scenario alone, then the test version
only if the policy passed alone. It keeps the team-chat messages and final answers of both phases in the row, and
returns them as `ng_trajectory`: one invocation per phase (`solo`, `manipulation`) whose turns are the policy's
messages. Setup, data and the run command are in [benchmarks/collab_intel/README.md](../../benchmarks/collab_intel/README.md).
