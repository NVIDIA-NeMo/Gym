# OfficeQA

Provides the document sandbox and answer verification for
[OfficeQA Pro](../../benchmarks/officeqa_pro/README.md). Use the benchmark README
for preparation, image builds, deployment and evaluation commands.

## Task and session contract

`task_data.TaskData` defines the fields required by both catalog validation and
runtime requests: `task_id`, `workspace_id`, `prompt_version`, `uid`, `question`
and `source_revision`. Reference answers are loaded from
`<workspace_root>/tasks/<task_id>.json`; each trusted record contains task and
workspace IDs, UID, question, source revision and `expected_answer`.
Keep this bundle on the driver, outside the document mount.

`/seed_session` accepts Gym's resources-session request with these task fields
and returns borrowed `SandboxAccess`. Agents work in `/workspace/task` and read
PDFs from `/workspace/docs`. Configure the latter as a read-only mount containing
only the prepared document store.

Each worker keeps the sessions it seeded in its memory; with several workers, session
routing sends every request of a session to the worker that holds it. Repeated seeding
with the same request reuses the sandbox. `/close_session` is idempotent for the same episode;
failed cleanup retains ownership so close can retry. Shutdown stops remaining
sandboxes, and a finite provider TTL bounds leaks.

`network_access=model_only` (default) restricts egress to the model server.
`network_access=inherit` uses the provider's network policy. The benchmark's
`officeqa_network_access` setting also controls both harnesses' web tools.

## Verification

`/verify` grades text from the last assistant message after removing complete
inline `<think>...</think>` blocks. The OfficeQA reference scorer uses
`numeric_tolerance` (default 0); caller-supplied reference answers are ignored.
The response includes `reward`, `final_answer`, the trusted `uid`,
`evaluation_completed` and `mask_sample`.

Wrong, empty and incomplete answers receive reward 0 with
`evaluation_completed=true`. Unavailable or invalid trusted references return
`mask_sample=true` and `evaluation_completed=false`. Stateless reverification
requires the trusted task bundle, but no active sandbox or model.

## Dependencies and tests

`pyproject.toml` and `uv.lock` pin the Gym development and sandbox dependencies.
Document tools live in the [sandbox image](toolset/README.md).
Integration code and the reference scorer are Apache-2.0; scorer provenance and
notices are recorded in its header and [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md).

```bash
pytest resources_servers/officeqa/tests
gym env test officeqa_pro
```

The latter runs Gym's verifier fixture, including all 13 JSONL cases.
