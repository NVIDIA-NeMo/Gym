# Oracle agent

Runs a Harbor task's reference solution instead of a model. It borrows the sandbox the `harbor` environment
server seeded for it and runs `bash /solution/solve.sh` there, in the task's working directory, as the task's
user. The environment server put the solution in place: when the agent instance it serves is implemented by
`harbor_oracle_agent`, its seed uploads the task's `solution/` folder to `/solution` (readable by the task user,
as Harbor's oracle copies it) and hands the oracle the host-resolved `[solution].env` and the task's
`[agent].timeout_sec` in the session seed. No other harness receives the solution, and the oracle reads no task
folder itself. Verification then scores the result exactly as it would score a model's work.

`solve.sh` runs with `DEBIAN_FRONTEND=noninteractive` under the task's `[agent].timeout_sec`, capped by
`max_solve_timeout_s` when that is set, and its output lands in `/logs/agent/oracle.txt`. The response metadata
records `oracle=solved|failed|unvalidated` with the exit code and any error type. Tasks without
`solution/solve.sh` are reported as `unvalidated` and are never failed.

```bash
gym dataset validate harbor:hello-world --sandbox opensandbox
```
