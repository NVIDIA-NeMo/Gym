# OpenHands SDK Sandboxed Agent

> [!NOTE]
> **Experimental.** This agent serves one benchmark today, [NL2RepoBench](../../benchmarks/nl2repobench/README.md),
> and runs each episode through its own `/run` path, like `opencode_sandboxed_agent`. It has no completed run on the
> upstream 104-task NL2RepoBench set yet; see "Combinations and status" in the benchmark README.

The OpenHands SDK harness for sandboxes that a resources server seeds. The resources server creates the sandbox
(`/seed_session` returns a `sandbox_handle`); this agent reconnects to it by handle, installs the OpenHands SDK
inside it, runs `runner.py` against the policy model server, hands the response to the resources server's `/verify`,
and stops the sandbox. Nothing here modifies the OpenCode agent: it is a separate agent type.

## Usage

The agent is configured through the benchmark that uses it. For NL2RepoBench:

```bash
gym env start \
  --config benchmarks/nl2repobench/openhands.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config responses_api_models/<model>/configs/<model>.yaml
```

The shared agent config is `configs/openhands_sdk_sandboxed_agent.yaml`. It needs a `resources_server` that seeds a
sandbox and answers `/verify`, and a `model_server`.

## Installing the SDK in the sandbox

- **Default:** `pip install` of `install_packages` (`openhands-sdk==1.50.0`, `openhands-tools==1.50.0` and a pinned
  `litellm`) into a venv at `venv_dir`, optionally from a pre-staged wheelhouse (`install_find_links`).
- **Offline bundle (`bundle_path`):** for sandboxes with no internet, or that block `python -m venv` (the
  `no_internet_preinstalled` and `block_target_v2` NL2RepoBench modes). Build it once with
  `bash build_bundle.sh <outdir>` (needs `uv` and internet on an x86_64 Linux machine). It produces
  `openhands_bundle.tar.gz` with a standalone Python, the SDK and a static tmux, which is uploaded into each rollout's
  sandbox. `bundle_python_relpath` must match the Python version the bundle pins.

## Failure handling

- **Setup failures are not graded.** An install that exits non-zero, reports a provider `error_type`, or does not print
  its `OPENHANDS_INSTALLED` marker, and a failed upload, raises a typed `openhands.setup` failure. The row is never
  sent to `/verify`: a harness that never started has not attempted the task, and an untouched workspace would score
  as a 0.
- **An unconfirmed launch is not graded either.** A provider error or exception while launching the runner (or while
  running it in the foreground mode), a detached launch whose pid file never appears, or a foreground run that never
  recorded its pid, raises `openhands.execute` and skips `/verify`.
- **A run that started is graded on what it left, once it is confirmed stopped.** The runner always has a session of its
  own (the foreground mode runs it under `setsid -w`, with its pid recorded first), so its tool children share its process
  group. At the deadline, or when the runner dies or polling gives up, the group, the session and the tmux server are
  stopped, and the run is graded only if that exec succeeded and printed `QUIESCED`. A provider error, an exec that
  raised, a missing marker or a process still alive is an ungraded `openhands.execute` failure, and nothing is
  collected from a runner that may still be writing. A partial snapshot without metrics reports no usage.
- **Polling needs positive evidence.** A poll that returns a provider error, a non-zero status or a reply it does not
  recognize counts against `max_consecutive_poll_errors` and never counts as an exit. Natural completion is only an
  explicit exit code (zero or not); `RUNNING` and `DEAD` are the other valid replies.
- **Unsupported request fields are refused.** A request with `instructions` gets HTTP 422 before any sandbox is seeded or
  set up: the runner receives only the user message, or the configured `user_prompt`.
- **Usage details stay unknown when the SDK does not report them.** `cached_tokens` and `reasoning_tokens` are `None`,
  not 0, unless the runner measured them (the runners write them as `null` when the SDK has no such counter).
- **The seeded workdir and the request's reasoning effort are used.** The runner starts in the workdir the resources
  server seeded (nested workdirs included), and a `reasoning.effort` in the request wins over `reasoning_effort` in
  the config.
- **Teardown always runs.** The stop and the removal of the sandbox handle are one operation that a verify error or
  a cancelled request cannot interrupt. The handle is dropped only after the stop succeeded; a failed stop keeps it,
  and the shutdown retries every handle still held.

## Configuration

| Option | Default | Meaning |
|---|---|---|
| `sandbox_timeout` | `21600` | Seconds one agent run may take. |
| `install_timeout_s` | `900` | Seconds the SDK install may take. |
| `max_iterations` | `500` | OpenHands iteration cap. |
| `model_call_timeout_s` | `14400` | Bound on a single model call. |
| `background_exec` | `true` | Run the SDK as a detached background process and poll, instead of one long exec; the runner also writes partial results periodically. |
| `bundle_path` | `null` | Offline bundle (above). |
| `user_prompt` | NL2RepoBench's task prompt | Set to `null` to use the benchmark row's own prompt. |

## Tests

```bash
gym env test +entrypoint=responses_api_agents/openhands_sdk_sandboxed_agent
```

## Licensing information

Code: Apache 2.0.
