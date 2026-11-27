# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Gym benchmark from Python: plan it in one call, then run single tasks against running servers.

```python
planned = plan_benchmark("ifeval")                        # resolve, prepare, plan and collate
runner = TaskRunner(planned.rows, registry_server=BaseServerConfig(host="127.0.0.1", port=11000))
outcome = await runner.run_task(planned.plan.tasks[0].task_id.task_id)
outcome.reward                                            # None when the rollout has no reward
```

To start the servers from Python too, serve the benchmark and give the yielded registry server to the runner:

```python
overrides = {"policy_base_url": url, "policy_api_key": key, "policy_model_name": model}
with serve_benchmark("ifeval", overrides=overrides) as registry_server:
    runner = TaskRunner(planned.rows, registry_server=registry_server)
    ...
```

`plan_benchmark` builds the plan `gym eval plan --benchmark NAME` writes and collates the benchmark's
rows. It reads no command-line arguments and leaves Gym's process-wide configuration untouched,
so a long-lived process can call it. `serve_benchmark` starts the servers `gym env start --benchmark NAME`
starts, from the same kind of config dict; each server runs in its own process and environment either way.
The registry server is the server that serves the run's config and the other servers' addresses; a `TaskRunner`
talks to it.

This is the API for callers outside Gym. They import the names `nemo_gym.environment` exports (`plan_benchmark`,
`serve_benchmark`, `TaskRunner`, `PlannedBenchmark`, `RolloutOutcome`, `TaskFailure` and `EvalPlan`) and
`ConfigError` from `nemo_gym.config`. It is experimental: signatures and result fields may change without a
deprecation period, and tests pin the current ones. `RolloutOutcome.raw` is Gym's raw `/run` response (the failure
record for a failed `/run`), whose keys can change: depend on the other fields, and use `raw` for debugging.
"""

from nemo_gym.environment.in_process.outcomes import RolloutOutcome, TaskFailure
from nemo_gym.environment.in_process.planning import PlannedBenchmark, plan_benchmark
from nemo_gym.environment.in_process.serving import serve_benchmark
from nemo_gym.environment.in_process.task_runner import TaskRunner
from nemo_gym.eval.eval_plan import EvalPlan


__all__ = [
    "EvalPlan",
    "PlannedBenchmark",
    "RolloutOutcome",
    "TaskFailure",
    "TaskRunner",
    "plan_benchmark",
    "serve_benchmark",
]
