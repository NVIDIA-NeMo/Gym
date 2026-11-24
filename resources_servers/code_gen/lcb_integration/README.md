This is taken directly from https://github.com/LiveCodeBench/LiveCodeBench/tree/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24

Local changes:
- `testing_util.py`: timers use `signal.setitimer`, so time limits can be fractional.
- `correctness.py`: optional task memory limit (set before `run_test`) and optional deadline.
- `testing_util.py`: confirmed timeouts return `-3` with `metadata.stage` set to `compilation`,
  `test`, or `preparation` (harness work outside compilation and test execution).
- `correctness.py`: a grading deadline with no completed result returns `-3` with stage `grading`;
  an early worker exit without a result keeps `-1` with stage `worker` and `exit_code`.
  Results and metadata are published together to survive deadline kills.
