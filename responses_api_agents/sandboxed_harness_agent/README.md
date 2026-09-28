# Sandboxed harness agent

Runs a coding-agent harness CLI inside the task's own sandbox. The resources server seeds the task
and hands back its sandbox; this agent attaches to it, runs the harness's command once under
`sandbox_timeout`, lets the harness collect its transcript, calls `/verify` and stops the sandbox.

It is never started on its own. Each harness is a subclass in its own directory, for example
`opencode_sandboxed_agent`.

## Adding a harness

Subclass `SandboxedHarnessAgent` and provide:

| Member | Purpose |
|---|---|
| `harness_name` | Name used in logs, e.g. `OpenCode` |
| `harness_id` | Prefix of the harness's `/run` result fields (`<id>_finished`, `<id>_run_stdout`, ...) and its observation source |
| `finished_marker` | Text the command prints last, only when the harness succeeded |
| `system_prompt` | Optional; prepended to `responses_create_params.input` in the result |
| `verify_response_class` | The harness's `/run` response model |
| `_harness_command()` | Shell command that installs and runs the harness; prints `Shell: $SHELL` first and the finished marker last |
| `_harness_collect()` | Fetches the transcript after the command: output items, usage, export path, observations |
| `run()` | Only re-declared with the harness's request and response classes, because FastAPI builds the `/run` models from its annotations |

Failure handling is shared by every harness: an exec exception, a timeout or a missing finished
marker gives `harness_finished: false`, and `/verify` still runs. Every result carries
`harness_finished`, so one mask rule covers all harnesses.
