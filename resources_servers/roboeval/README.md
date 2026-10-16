# RoboEval resources server

Scores robot programs with [RoboEval](https://github.com/ut-amrl/codebotler)'s symbolic simulator and temporal checks.
See [`benchmarks/roboeval`](../../benchmarks/roboeval/README.md) for data and run commands.

## Verification

Each row carries `task` (a RoboEval task name such as `SayGoodDay`) and `prompt_index`.

1. The program is the last fenced code block in the model's answer, or the whole answer if there is none. A missing
   `def task_program():` header is added, as CodeBotler does for chat models.
2. RoboEval's `code_replace` binds the API calls to its simulated robot. The program then runs in a subprocess once per
   world state defined by the task (3 to 5 states), and the robot actions it prints form a trace.
3. Each state's temporal-logic test checks its trace. A program that crashes fails the state. A program that times out
   fails the state, unless the test expects it to run forever (for example, waiting for a person), in which case the
   partial trace is checked.
4. Reward is `1.0` if every state passes, otherwise `0.0`. The mean reward over repeated samples is RoboEval's pass@1,
   reported as `pass@1[avg-of-k]/accuracy`.

A failure of the verifier itself (a simulator process that cannot start, an unknown task, a broken import) sets
`mask_sample` with `failure_kind="verifier_error"`, so the sample is excluded from scores. A program that crashes,
loops, or fails a check is a valid measurement and is not masked, as in RoboEval.

The response includes `program` and `state_results`, with each state's `passed`, `status` (`success`, `timeout`, or
`error`), and `error` message.

**Parity.** Over CodeBotler's 3,200 saved GPT-4 completions, this verifier matches RoboEval's stored per-state verdicts
exactly: 10,400 of 10,400 states, and pass@1 0.855 vs 0.855.

## Configuration

| Field | Default | Description |
|---|---|---|
| `roboeval_root` | `$ROBOEVAL_ROOT`, else fetched | CodeBotler checkout; task tests, simulator, and tracer are imported from it. Unset: the pinned commit is cloned into `.codebotler` |
| `simulation_timeout_s` | 5.0 | Per-state simulation budget |
| `max_concurrent_simulations` | 16 | Concurrent simulator subprocesses |

## Tests

`pytest resources_servers/roboeval/tests` runs the parsing and metric tests. With `ROBOEVAL_ROOT` set, it also runs the
verifier fixture, a partial-trace timeout test, and the masked-failure test. The fixture programs are written for this repository; no CodeBotler content is included.

## Licensing

RoboEval: BSD-3-Clause with a machine-learning restriction. Its contents may not be used as training data, so this
server is for evaluation only. This server's code: Apache-2.0.
