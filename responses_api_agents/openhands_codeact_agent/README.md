# OpenHands CodeAct harness

This adapter runs the OpenHands 0.62 CodeAct controller in the task sandbox a Resources Server lends, one activation
per agent session. It keeps CodeAct's own system prompt and tools (`execute_bash`, `str_replace_editor`, `think`,
`finish`), takes the task's tools from the Environment Server's MCP grants, and can compact its context with a
cutoff-triggered summary. It is a separate harness from the OpenHands Software Agent SDK that `openhands_agent` runs.

- **Capability level:** Evaluation. Model-call capture and compaction observations are in place for training, but
  no training run has qualified them.
- **Pairings:** GDPVal (`benchmarks/gdpval/openhands_codeact.yaml`).
- **Harness version:** [`sdevare-nv/nv-OpenHands@54d2a95edc666551b854a7e5d4205ec69e60ff23`](https://github.com/sdevare-nv/nv-OpenHands/tree/54d2a95edc666551b854a7e5d4205ec69e60ff23), from a staged runtime.

## How a session runs

1. **Seed.** The adapter connects to the lent sandbox (`sandbox_access`, a direct connection), installs the staged
   runtime under `/opt/openhands-codeact`, and uploads the runner into its own session directory under
   `/tmp/nemo-gym-openhands-codeact-sessions/`. It never prepares the task and never stops the sandbox; the
   Resources Server owns it. A seed without a lent sandbox, with a required grant that is not MCP, or with a workdir
   inside the adapter's own directories is refused with 422.
2. **Activation.** `/v1/responses` under the rollout prefix runs the controller once, under the shared supervisor,
   with the request's text as the task message. Every model call, including the condenser's summaries, goes to the
   rollout-prefixed Gym model server with `x-session-id` set to the invocation. The MCP grants are passed to
   OpenHands with their headers, and each granted tool is offered as `<grant name>_<tool>`, so a task tool such as
   `finish` cannot shadow CodeAct's own. A required server that does not connect fails the run before the first
   model call. A retry with the same body joins the running activation; another body gets 409.
3. **Close.** The adapter stops the runner, releases its connection to the sandbox, and returns the observation
   bundle: the invocation and its model calls, one `ContextCompactionObservation` per summary call, and the sandbox
   exit.

There is no `/run` path; requests outside an agent session get 422.

## Request fields

The request's `input` must be one user message, optionally after one system or developer message, as text. Its
`instructions` and the system message are placed before the task text, since CodeAct keeps its own system prompt.
`temperature`, `top_p` and `max_output_tokens` set the controller's model settings (the last as a per-call limit,
not a budget for the run); the agent configuration supplies them otherwise. Any other non-default field, such as
`tools`, `tool_choice` or images, is refused with 422.

## Outcomes

| How the run ended | Response status | `metadata.stop_reason` |
| --- | --- | --- |
| CodeAct called its `finish`, or a tool in `finish_tool_names` succeeded | `completed` | `completed` |
| Iteration limit (`max_iterations`) | `incomplete` | `max_iterations` |
| Context limit | `incomplete` | `context_limit` |
| Loop detector | `incomplete` | `stuck_in_loop` |
| Activation deadline (`timeout`, or the request's `agent_timeout_sec`) | `incomplete` | `wall_time` |
| Any other controller error | `failed`, with an `AgentOutcome` from the last model call (HTTP status or transport error) | `error` |

An incomplete run's work is judged. GDPVal scores `wall_time` as the model's `agent_timeout`. A failed run follows
the Environment Server's `agent_outcome_policy`. A sandbox failure, a runner that wrote no result, or a lost activation
fails the episode so it can be retried.

## Training data

The response is text-only: it holds the last context the controller sent, after the task message, and its reply.
Token IDs and log probabilities come from Gym's model-call capture (`token_id_capture`), never from the completion
logs, which the policy could rewrite from its shell. With `context_summarization_cutoff` set, each compaction is
reported as a `ContextCompactionObservation` naming its summary call, and rollout collection keeps that call out of
the rebuilt turns. Usage is reported only when every answered model call logged it.

## Runtime and configuration

`runtime/build_runtime.sh` builds a portable Linux x86_64 archive from the pinned revision and a Gym checkout:

```bash
bash responses_api_agents/openhands_codeact_agent/runtime/build_runtime.sh \
  /path/to/Gym /path/to/standalone-python /path/to/runtime-build
```

Make the resulting `staged/` directory readable in the task sandbox (a read-only mount or a copy in the task image)
and set `remote_openhands_install_script_path` and `remote_openhands_archive_path`
(`OPENHANDS_CODEACT_INSTALL_SCRIPT` and `OPENHANDS_CODEACT_ARCHIVE` in the shipped configs). The installer checks the
archive's `.sha256` sidecar, removes a killed attempt's leftovers, and is retried up to four times. The task image
needs `bash`, `tar`, `sha256sum` and `timeout`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `timeout` | `10800` | Seconds per activation, inside the Environment Server's episode timeout and the sandbox's TTL |
| `max_iterations` | `200` | CodeAct's iteration limit; `null` removes it |
| `max_context` / `max_output_tokens` | `196608` / `32768` | Context window and per-call output limit |
| `temperature` / `top_p` | `1.0` / `1.0` | Used when the request does not set them |
| `context_summarization_cutoff` | `null` | Fraction of `max_context` at which the condenser summarizes; off when `null` |
| `mcp_tool_timeout_seconds` | `900` | Per MCP tool call |
| `finish_tool_names` | `[]` | Granted task tools whose successful call ends the run; when set, CodeAct's own `finish` is not offered |
| `model_read_timeout_s` / `model_read_retries` | `1800` / `0` | Per model call; the Gym model server retries transient errors |

The adapter writes only under `/opt/openhands-codeact` and its session directory. OpenHands' CLI runtime runs
commands in the workdir the Resources Server names (`/root` for GDPVal).

## Limits

- Only a lent sandbox is supported; the adapter does not create sandboxes, so it has no `network_access` setting.
- The controller's tool calls are carried by the response items; their timings are not reported (an observation gap).
- The pinned controller and its LiteLLM client differ from the OpenHands SDK runtime, and benchmark-score parity
  with other harnesses is not established.

## Provenance and checks

The summary prompt and rendering in `compaction_condenser.py` derive from
[`openhands/memory/condenser/impl/llm_summarizing_condenser.py`](https://github.com/sdevare-nv/nv-OpenHands/blob/54d2a95edc666551b854a7e5d4205ec69e60ff23/openhands/memory/condenser/impl/llm_summarizing_condenser.py)
at the revision above, under MIT. The file retains the upstream copyright and license notice. The remaining adapter
code is NVIDIA Apache-2.0 code.

```bash
gym env test +entrypoint=responses_api_agents/openhands_codeact_agent
```
