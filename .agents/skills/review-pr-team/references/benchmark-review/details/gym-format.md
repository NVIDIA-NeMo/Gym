# Gym format: JSONL rows (item 5)

Line numbers are at main `eeccee711d`.

- `DP` = `nemo_gym/data/dataset_preparation.py`
- `TD` = `nemo_gym/data/task_data.py`
- `CT` = `nemo_gym/config/config_types.py`
- `GC` = `nemo_gym/config/global_config.py`
- `RC` = `nemo_gym/rollouts/rollout_collection.py`

Use this file when the PR adds or changes JSONL rows, a `prepare.py`, a server's `task_data.py`, or a dataset entry in a config. For dataset, schema and request-model drift, also run the audit in `../../data-contracts.md`.

## Rows

- **Every row is a valid run request:** `responses_create_params` with an `input` (`nemo_gym/server_runtime/base_resources_server.py:172-175`).
  - `gym env validate <benchmark>` rejects a row that is not (`nemo_gym/environment/validation.py:685-697`).
  - `gym dataset collate` does not: a bad row is appended to `offending_example_idxs`, which nothing reads, and is still written (DP:535, DP:806-808, DP:1281-1320). A PR cannot rely on collate to catch it.
- **Task fields.** Task-owned fields go flat in the row or inside `verifier_metadata`; `normalize_task_fields` splices `verifier_metadata` up before validation (TD:9-20). The verifier reads them from the verify body.
- **Reserved keys** a row must not use as task fields: `responses_create_params`, `turns`, `agent_ref`, `task_source`, `_ng_task_index`, `_ng_rollout_index` (TD:62-72). `_ng_*`, `agent_ref`, `task_source` and `skills_ref` are dropped before a seed or verify sees the row (`nemo_gym/data/task_materialization.py:44-47`).
- **No `agent_ref` in new rows.** Collate strips it with a `DeprecationWarning` and stamps `task_source` with the declaring instance (DP:1266-1298, DP:1328-1334).
- **A stable ID** in `task_id`, `problem_id` or `instance_id`. Without one the ID is the row's position, which shifts when rows are added or reordered, and resume and comparison break (`task_materialization.py:20, 26-43`).
- **`prompt_config`.** A row rendered from a prompt config must not already contain `input` (`nemo_gym/data/prompt.py:98-105`). A benchmark dataset with `prompt_source: template` needs a `prompt_config` or the manifest's `standard_prompt_config` (`validation.py:647-662`).

## The server's `task_data.py`

- Every Resources Server with an `app.py` ships `task_data.py` exporting `TaskData` (TD:1-8). A presence test enforces it (`tests/unit_tests/data/test_task_data.py:556-566`).
- `extra` is `allow` (default) or `forbid`; `ignore` is banned because it silently drops row fields (TD:29-32; test `:351-372`).
- Imports: standard library, pydantic, `nemo_gym.data.task_data` and other servers' `task_data` modules only (TD:24-27; test `:330-349`).
- Required fields follow the server's wire contract, not what `verify()` happens to read (TD:33-34).
- A field the wire reads only from inside `verifier_metadata` carries `json_schema_extra={"legacy_location": "verifier_metadata"}` (TD:15-20).
- **Drift gate:** every git-tracked `jsonl_fpath` declared under `resources_servers/`, `responses_api_agents/`, `benchmarks/` and `environments/` must validate clean against its server's `TaskData` (`test_task_data.py:569-673`). Unknown keys fail collation in `example_validation` mode (DP:114-130, DP:1322-1326).
- For a stateful server, `seed_request_model` should validate the same fields at seed (`session-contract.md`).

## Dataset entries in the config

- Declare datasets on the Resources Server instance (DP:613-616; `fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx:103`). Datasets on the agent block are the older layout.
- `train` and `validation` need a `license` (CT:594-599). Use `source:`; `gitlab_identifier` and the other `*_identifier` fields are deprecated (CT:570-573, CT:639-645).
- A `benchmark` dataset needs a `prepare_script`; prepared data is generated, not committed (`validation.py:647-651`). Only `data/example.jsonl` (5 rows) and the example metrics and rollouts files are committed for a Resources Server (`configs-and-evidence.md`).
- `allowed_agents` on the dataset or the Resources Server (CT:581, CT:680; `session-contract.md`).

## Typed tasksets and `environment_server_routes`

New Gym-format work runs on a session Environment Server. Two forms are accepted (decided 2026-11-11):

- **Preferred:** rows routed by taskset to `single_agent_turn`, as below.
- **Accepted:** flat rows on `single_agent_turn_legacy`, with an agent that implements sessions. Suggest the taskset; don't block on it.
- **Not accepted:** flat rows on a `legacy_agent` relay.

The taskset form:

- The dataset declares `taskset: <name>:<split>` (CT:563-567; `type: benchmark` datasets: CT:666-670).
- The config sets a top-level `environment_server_routes: {<taskset>: <environment server instance>}` (GC:2296-2300).
  - It routes to exactly one Environment Server, which binds the Resources Server that declares the dataset (GC:2301-2339).
  - The route does not pin an `agent:` (GC:2322-2326).
- Collate turns these rows into `{task_id, task_input}` (DP:1305-1319).
- At collection, a taskset with no route raises (RC:496-501). `environment_routing_mode: taskset` refuses flat rows (RC:506-509) and needs `environment_server_routes` (RC:1595-1596). The default `agent` mode routes flat rows by agent (RC:1572-1581), to `single_agent_turn_legacy` or a `legacy_agent` relay.
- **Worked example:** `benchmarks/agentic_if/config.yaml:11-12, 38-49` (`agentic_if:benchmark` routed to `agentic_if_environment_server`, OpenCode as a session harness).
- Only 11 configs on main declare `taskset`. The flat-row layout on `single_agent_turn_legacy` is still the common one and is accepted for new work.

## Images

- An `input_image` holds a base64 `data:` URI of type png, jpeg, gif or webp (`image/jpg` is an alias). The payload's leading bytes must be a supported type, and they win over the label (`nemo_gym/core/media.py:34-47, 89-117`).
- Where Gym decodes an image, URLs and file IDs are refused, never fetched (`media.py:98-99`). Not every harness decodes prompt images: `simple_agent` passes an `input_image` URL to the model server unchanged. Either way, a row with an `https://` image is a finding.
- Gym decodes media in the `pi_agent` and `opencode_agent` prompts and in tool-output images (`nemo_gym/core/tool_output_images.py:18, 37-38`, used by `responses_api_agents/simple_agent/app.py:1312-1322` and `nemo_gym/server_runtime/mcp_auto_exposure.py:781`).
- `UnsupportedContentError` is a `ValueError`; it becomes a 422 only where the harness or server maps it (`serving.md`).
- There is no size cap in `media.py`. A benchmark with large images should say what it expects the model server to accept.
- Images do not survive the Harbor path (`harbor-format.md`).
- Check the pairing: an image benchmark lists in `allowed_agents` only harnesses that pass image input through, and says so in its README.

## Review questions

- Can every committed row be parsed as a run request and validated by the server's `TaskData`?
- Does the verifier read any field that `TaskData` does not declare, or declares in the wrong place?
- Is the row ID stable across a re-run of `prepare.py`?
- New work: do the rows reach a session Environment Server, either a `taskset` routed to `single_agent_turn` or flat rows on `single_agent_turn_legacy` with a session agent, and never a `legacy_agent` relay?
