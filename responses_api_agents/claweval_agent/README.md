# Claw-Eval agent

Wraps the supplied local Claw-Eval fork's native runner as a Gym evaluation agent.
Each `/run` executes one task in its own worker process and Pyxis sandbox, grades
it with the native grader, and returns a Gym trajectory and reward.

See [the benchmark guide](../../benchmarks/claweval/README.md) for the three task
splits, setup, smoke/full evaluation commands, metrics, and supported runtime.

The required source checkout is selected by `CLAW_EVAL_ROOT`; the external worker
Python by `CLAW_EVAL_PYTHON`. The worker receives configuration over stdin and
preserves per-trial logs, native traces, and workspaces. No Claw-Eval dependency
is installed into the Gym agent's environment.
