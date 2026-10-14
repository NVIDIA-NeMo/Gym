# Oracle agent

Runs a Harbor task's reference solution instead of a model. It borrows the sandbox the `harbor` environment
server seeded, uploads the task's `solution/` folder to `/solution`, and runs `bash /solution/solve.sh` in
the task's working directory. Harbor currently rejects explicit task-user overrides before Oracle runs,
so supported tasks use the sandbox's default user. Verification then scores the result exactly as it
would score a model's work.

Tasks without `solution/solve.sh` are reported as `unvalidated` in the response metadata and are never
failed.

```bash
gym dataset validate harbor:hello-world --sandbox opensandbox
```
