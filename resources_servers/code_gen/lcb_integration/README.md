This is taken directly from https://github.com/LiveCodeBench/LiveCodeBench/tree/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24

Local changes:
- `testing_util.py`: timers use `signal.setitimer`, so time limits can be fractional.
- `correctness.py`: optional task memory limit (`../sandbox_limits.py`, set before `run_test`) and optional deadline.
