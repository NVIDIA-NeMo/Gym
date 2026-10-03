# Harbor tasks

Runs tasks written in the [Harbor](https://github.com/harbor-framework/harbor) task format: a folder with
`task.toml`, `instruction.md`, `environment/`, `tests/test.sh` and an optional `solution/`. Harbor is the input
format; Gym owns execution. Hub tasks run unchanged.

## What the server does

- `/seed_session`: checks the row's folder digest, starts the task image as a sandbox, creates the working
  directory, and returns a `SandboxAccess` the agent harness borrows.
- `/verify`: uploads `tests/` into the same sandbox, runs `bash /tests/test.sh`, and reads
  `/logs/verifier/reward.json` or `reward.txt`.
- `/close_session`: stops the sandbox.

Reward rule: when `test.sh` ran, the sample counts. A missing or invalid reward file scores 0 with a
`failure_kind`. Only a Gym-side failure (sandbox lost, transfer failed) sets `mask_sample`.

Supported today: single-step tasks with a prebuilt `docker_image` or a base-image-only Dockerfile
(`FROM` plus `WORKDIR`/`ENV`/`USER`/`LABEL`), shared verifier mode. Dockerfile builds, separate verifier
containers, multi-step tasks and in-sandbox MCP tools come in later milestones.

## Run

```bash
gym eval run harbor:hello-world --agent hermes --sandbox opensandbox \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

`harbor:<dataset>[@<version>]` resolves the Harbor registry and fetches the task folders into Gym's shared
datasets folder (`$NEMO_GYM_DATASETS_DIR`, default `./datasets`). A local task folder, or a folder of task
folders, works the same way.
