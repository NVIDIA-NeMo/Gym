# GDPVal OpenHands training harness

This adapter runs the OpenHands 0.62 CodeAct controller in the task sandbox owned
by `gdpval_disaggregated`. It preserves the controller's tools, completion logs,
optional Exa web tools, and context-compaction traces. It is a separate harness
from the OpenHands Software Agent SDK used by `openhands_agent`.

The supplied `training.yaml` uses the `legacy_agent` Environment Server. `/run`
seeds the resources session, borrows its sandbox, installs the staged runtime,
runs the controller, exports its response, and asks resources to verify. Resources
owns sandbox deletion; the agent disconnects its borrowed handle after cleanup.
The resource-session cookie follows seed, verification, and cleanup.

## Runtime and configuration

Stage an installer, runtime archive and `<archive>.sha256` sidecar in the task
sandbox. Set `remote_openhands_install_script_path` and
`remote_openhands_archive_path` explicitly. The installer accepts
`--archive <path>`, checks the sidecar, installs under `/opt/gdpval-openhands`,
and writes `.installed-sha256` after success. Interrupted installations are
removed before retrying; an existing complete installation is checked by the
installer. `sandbox_python` defaults to `/opt/gdpval-openhands/venv/bin/python`.

The runtime must provide CodeAct, the CLI runtime, `openhands.agenthub.nemo_gym_client`,
FastMCP, and the current Gym public client API. The public OpenHands source used by
Gym's SWE harness is
[`sdevare-nv/nv-OpenHands@54d2a95edc666551b854a7e5d4205ec69e60ff23`](https://github.com/sdevare-nv/nv-OpenHands/tree/54d2a95edc666551b854a7e5d4205ec69e60ff23).
A stock Software Agent SDK installation does not provide that controller API.

`runtime/build_runtime.sh` builds a portable Linux x86_64 archive from that public
revision and a local Gym checkout. Supply a standalone CPython 3.13.14 root and an
empty output directory:

```bash
bash responses_api_agents/gdpval_sandboxed_agent/runtime/build_runtime.sh \
  /path/to/Gym /path/to/standalone-python /path/to/runtime-build
```

The build records resolved dependencies and both source revisions. It overrides
the older controller's OpenAI, AnyIO, Starlette and MCP requirements to work with
the current Gym client. Mount the resulting `staged/` directory read-only into the
task sandbox. The task image also needs `bash`, `git`, `python3`, `timeout`, and the
document tools advertised in the staged task prompt. A successful runtime import
does not establish benchmark or training-score parity with another runtime.

Supply these agent settings for each instance (`gdpval_train` and `gdpval_val`):

```yaml
model_server: {type: responses_api_models, name: policy_model}
resources_server: {type: resources_servers, name: gdpval_train_resources}
checkpoint: served-model-name
remote_openhands_install_script_path: /runtime/install.sh
remote_openhands_archive_path: /runtime/runtime.tar.gz
sandbox_provider: sandbox
agent_timeout: 3600
agent_max_turns: 200
```

The optional `resources_servers` list balances sessions among named resources
instances. Each instance must have one worker so its session state stays local.
Agent `concurrency` is total capacity across workers and must divide evenly by
`num_workers`. Budget the environment for installation, agent execution, judging,
and cleanup; bound episode admission before creating task sandboxes.

Set `web_tools_enabled: true` and `web_search_api_keys_file` to enable Exa search
and page fetching. The file holds one key per line. These keys enter the sandbox,
where the agent has shell access, so use keys intended for that trust boundary.

## Training traces

Without compaction, enable Gym token capture for the agent and model server when
training; the returned response is the readable trajectory and capture supplies
its tokens. Every policy call uses the run-specific Gym endpoint.

`context_summarization_cutoff` defaults to `null`. A fraction such as `0.7`
enables summarization when the last response's prompt plus completion usage
reaches that fraction of `max_context`. The next context contains the system
prompt, original task, and summary. `context_summarization_max_event_chars`
defaults to 10000.

With compaction, `/run` additionally returns `responses`: one response for each
context segment, with echoed prompt tokens, generation tokens and logprobs.
Summary calls form their own segment; previous assistant tokens are not trained
twice. Use a downstream consumer that accepts these compaction traces and a
model server that echoes token metadata. Set Gym `token_id_capture.rebuild_response`
to `false` for this mode so finalization does not replace the exported segments.
Absent token metadata produces text-only segments, which cannot be used for
token-based training. Partial or inconsistent token echo masks the row rather
than fabricating training data. Top-k score-centering metadata is unsupported and rejected.

Unfinished infrastructure attempts retain `agent_finished: false` and an explicit
reason. Policy context, iteration and loop limits keep their distinct reasons;
the resources server controls their scoring/masking policy.

## Provenance and checks

The summary prompt and rendering in `compaction_condenser.py` derive from
[`openhands/memory/condenser/impl/llm_summarizing_condenser.py`](https://github.com/sdevare-nv/nv-OpenHands/blob/54d2a95edc666551b854a7e5d4205ec69e60ff23/openhands/memory/condenser/impl/llm_summarizing_condenser.py)
at the revision above, under MIT. The file retains the upstream copyright and
license notice. The remaining adapter code is NVIDIA Apache-2.0 code.

Run `python -m pytest responses_api_agents/gdpval_sandboxed_agent/tests` in the
agent environment. Tests cover compaction/token ownership, model transport,
resource cookies and ownership, runtime retry, and web-tool errors. A real model
rollout is required in addition to these tests; unit results do not establish
benchmark score parity.
