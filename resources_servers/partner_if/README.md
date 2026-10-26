# Partner IF

Scores instruction following (IF) against per-task rubrics. Each rubric item names a
verifier: the shared LLM judge (`llm_judge_v1`) or a code check. The judge cites line
ranges from the evidence it is shown; the server checks them and records the quoted text.

## Data format

Each row has `responses_create_params` and `verifier_metadata.partner_if`, a
`GradingContract` (see `task_data.py`):

- `task_id`, `origin`, `source`, `rubric_version`, `data_kind`, `audit_signoff`.
- `evidence_mode`: `text_response` (default), `frozen_tool_history` (recorded tool calls
  and results in `input`; the new reply is text), `tool_call_response` (grades the new
  response, including tool calls, without running them) or `live_agent` (needs an
  adapter, below).
- `rubrics`: items with `id`, `constraint`, `verifier` (`{"id": ..., "params": {...}}`),
  `dimension` (`if`, the default, or `completion`) and optional `depends_on`. Optional
  `category`, `guidance` and `provenance` are shown to the judge; `taxonomy` and
  `analogues` are not.
- `prerequisites`: optional unscored items that `depends_on` can reference.

`verifier_metadata.curation`, any JSON object, is optional; it is kept but never read.

Built-in verifiers: `llm_judge_v1` (optional `artifacts`: the captured artifacts an item
needs; omit it to require all), `text_contains_v1` (`source`: `answer` or
`artifact:<name>`; `literal`) and `text_no_uuid_v1` (fails if the answer or any assistant
message contains a UUID; tool calls, tool results and context are not checked).

The fixed output checks in `fixed_checks.py` take no parameters. They parse the answer as
CommonMark with `markdown-it-py`, so code blocks, code spans and quotes don't count as the
answer's own structure:

- `scratchpad_comments_v1`: at least one scratchpad wrapped exactly as
  `<!-- [scratchpad_start] … [scratchpad_end] -->`, paired, not nested and not in code.
- `review_sections_v1`: the top-level H2 sections "Automated change summary", "Changes to
  exported or public declarations" and "Dependency manifest review", in that order.
- `max_100_words_v1`: at most 100 words; citation markers such as `[web:1]` don't count.
- `short_headings_v1`: every heading has at most 5 words, counted the same way.
- `headed_section_v1`: at least one H2 or H3 heading with content under it.
- `max_five_sections_v1`: at most 5 top-level sections, counting an introduction.
- `no_plan_tool_v1`: no call to the `plan--draft` tool.
- `made_project_edits_v1`: the environment's `project_edits` artifact lists at least one
  successful edit.

## Scoring

- `reward` is 1 when every IF item passes; `if_result` reports `passed` and `total`.
- Completion items are reported in `grading_result.dimensions` and never change the reward.
- A missing or truncated answer fails every item without a judge call: reward 0, unmasked,
  with `failure_kind` `partner_if:no_answer` or `partner_if:truncated`. This covers a
  blank reply, a reply with only reasoning or no output (including Simple Agent's
  `empty_output` and `incomplete_reasoning` replies), a tool call in `text_response` or
  `frozen_tool_history` mode, a reply cut off by the output-token cap, and a mock-tool
  episode that ends without the final answer its backend requires.
- A failed judge call raises out of `/verify` into Gym's judge failsafe: the row is masked
  with `failure_kind: judge_failed`, written to the failures file
  (`<output>_failures.jsonl`) and retried on resume. `gym eval reverify
  --judge-failed-only` can recover it, except a `live_agent` row, whose evidence is not
  in the row. For the same reason, the servers that grade `live_agent` rows (the mock-tool
  and SWE servers, and this server with `accept_verification_input: true`) report reverify
  mode `unsupported`, so a full `gym eval reverify` refuses them without `--force`.
- These rollouts are masked (`mask_sample: true`), and `failure_reason` gives the reason
  of the first item that could not be scored or, for a row masked as a whole (such as an
  unsigned contract, unsupported input or output, a missing `verification_input` or a
  provider error), the row's own reason:
  - `judge_unparseable` (Gym's kind): the judge replied, but its reply is incomplete or fails
    the verdict protocol or validation for an item that gates reward. Unknown keys inside a
    reference are dropped rather than rejected;
  - `partner_if:unscorable`: an IF item, or a prerequisite an IF item depends on, is
    unscorable: a capture the item needs is
    missing or failed, a live agent's trajectory is incomplete or does not match its tool
    receipts, or an item it depends on is unscorable. On this server's `/verify`, also a
    provider error or another incomplete reply, and input or output its mode does not
    support, including hidden context (`previous_response_id`, `conversation`, `prompt`),
    and a `live_agent` row whose `verification_input` is missing or was captured from
    other input;
  - `partner_if:evidence_too_large`: an artifact over 1,000,000 characters, for the items
    that need it, or a judge input over `judge_max_input_chars` characters: the serialized
    rubrics and evidence of the items the judge grades, with only the artifacts they need;
  - `partner_if:tool_backend_failed`: a mock-tool backend failed to run a call (a backend error, or a sandbox
    or runner that stopped answering), so the environment failed rather than the model;
  - `partner_if:unsigned_contract`: a contract with `audit_signoff: false` on a server
    without `allow_provisional_rubrics: true`. With it, the contract is scored and marked
    `provisional`.
- A mock-tool backend that requires no final answer grades a reply cut off by the
  output-token cap on the tool exchanges completed before it.
- Metrics: `all_if_pass` and `if_partial` (the passed fraction of IF items), overall and
  per `origin`. Masked rollouts are left out of them and counted under `coverage/*`;
  rows in the failures file are left out entirely.

Key config fields: `judge_model_server`, `judge_responses_create_params` (8192 output
tokens, temperature 0), `judge_max_concurrency` (8), `judge_max_input_chars` (1,000,000;
size it to the judge's context window), `allow_provisional_rubrics` (false) and
`task_check_modules` ([]). The SWE server defaults to 32768 output tokens and a
concurrency of 2.

The sandboxed mock-tool server's `sandbox_config` takes Gym's `SandboxConfig` settings (`ttl_s`, 3600 here and
never unset, `ready_timeout_s`, `resources`, `env`, `metadata`, `provider_options`) plus `image` (any Python 3 image;
`python:3.13-slim`), `network`, `package_index_url` (`https://pypi.org/simple`), `package_index_egress`
(`https://pypi.org`, `https://files.pythonhosted.org`), `install_timeout_s` (300), `runner_timeout_s` (60, until
the backend has loaded) and `call_timeout_s` (60, each request to the runner). `network` is one of:

- `package_index` (default): egress only to the hosts of `package_index_egress`; the pins are installed with
  `pip --only-binary=:all:` from `package_index_url`;
- `none`: no egress; nothing is installed, so the image must already have the pinned versions, which the runner
  checks;
- `provider_default`: no policy from this server, for a provider without network policies, such as local Docker.

`package_index` and `none` need OpenSandbox's network policies and fail at startup on any other provider.
The server reaches each runner through the provider's endpoint with the provider's own TLS setting (OpenSandbox's
`connection.tls_verify`), so a deployment with a self-signed certificate works as it does for the provider.
Running the sandboxed mode outside `gym` needs Gym's `sandbox` extra (`uv sync --extra dev --extra sandbox`).

## Task-specific checks and live agents

- `task_check_modules`: importable modules whose `CHECKERS` dict maps verifier IDs to
  `checks.Checker` objects, usually `checks.fixed(fn)`. They are registered at startup
  and cannot redefine a registered ID.
- `mock_tools_app.py` serves each task's tools from a `ToolBackend` (`tool_backend.py`), in one of two modes:
  - In-process: `backend: module:Class` serves every task in `tasks_path` over `fixture_path`. The server fails
    at startup when a task advertises a tool the backend does not serve or names a verifier that does not
    resolve. `configs/mock_tools.yaml` runs the example `example_tools.NotesBackend` on
    `data/mock_tools_example.jsonl`.
  - Sandboxed: with `sandbox_provider` (a Gym sandbox provider config, such as OpenSandbox's), each row brings
    its own backend, which runs in a fresh sandbox for every rollout, so tasks with different backends share one
    server. The row's `tool_backend` holds `entrypoint` (`module:Class`), `files` (UTF-8 text by relative path),
    `fixture` (one of `files`, passed to the class) and `requirements` (exact `name==version` pins). It sits
    outside `responses_create_params`, so the actor never sees it. The sandbox gets only the bundle, a
    standard-library runner (`tool_runner.py`) and the row's `responses_create_params`, never its rubrics or
    contract. Each seed starts a sandbox, installs the pins, starts the runner and checks that the backend serves
    every tool the task advertises (HTTP 422 if not). Close, retire and shutdown stop the sandbox, and
    `sandbox_config.ttl_s` is the backstop. `configs/mock_tools_sandboxed.yaml` runs the example backend this
    way on `data/mock_tools_sandboxed_example.jsonl`.

  In both modes a backend raises `tool_backend.ToolCallError` for the model's mistakes: the model sees
  `{"error": ...}` and the call is graded. Any other failure is the environment's: the model sees only
  `Internal Server Error`, the call leaves no receipt, and the rollout is masked as
  `partner_if:tool_backend_failed`. The verify response adds `tool_calls`, `environment_receipts` (the
  backend's records) and, in sandboxed mode, `tool_sandbox_id`. Episodes live in the server's memory, so it
  requires `num_workers: 1`.
- `swe_app.py` (Scale-SWE with an OpenCode agent) also captures `live_agent` evidence.
  Set `actor_system_prompt` and `tool_description_additions` to what the agent's own
  config gives the actor; the judge reads them. OpenCode reports a status only for a run
  the output-token cap cut off, so a run counts as finished when its trajectory ends in an
  assistant message. Any other run, such as one whose transcript export failed, is masked
  as `partner_if:unscorable`. Leave OpenCode's `execution_failure_reward_zero` off: it
  requires an unmasked 0 for a failed run, which this server masks, so such a run fails its
  request instead.
  `configs/swe.yaml` validates, but it runs only with a sandbox provider and Scale-SWE
  rows that carry a `live_agent` contract, which Gym does not ship.
- An agent that runs a `live_agent` episode itself sends what it captured to this server's
  `/verify` as `verification_input` (`native_evidence.LiveAgentCapture`: the input the
  actor was given, its documents and artifact capture receipts). With
  `accept_verification_input: true`, the server grades it with the response's visible
  trajectory; rows never carry it.
- `responses_api_agents/partner_if_dispatcher` routes each task of a mixed benchmark to
  its environment; its config runs both examples below under one score.

Relative paths in these configs resolve under `--search-dir` / `NEMO_GYM_EXTRA_ROOTS`
first, then the working directory, then the Gym root.

## Usage

```bash
gym env start --resources-server partner_if \
    +policy_base_url=... +policy_api_key=... +policy_model_name=... \
    +judge_base_url=... +judge_api_key=... +judge_model_name=...

gym eval run --no-serve --agent partner_if_simple_agent \
    --input resources_servers/partner_if/data/example.jsonl \
    --output partner_if_rollouts.jsonl
```

`configs/partner_if.yaml` includes the `openai_model` policy server. `data/example.jsonl`
holds five synthetic examples. `configs/verifier.yaml` starts only the server and its
judge, for `gym eval reverify` of rows that are not `live_agent`. For the mock-tool
example, use `--resources-server partner_if/mock_tools`, `--agent
partner_if_mock_tools_simple_agent` and `data/mock_tools_example.jsonl`.
`data/example_rollouts.jsonl` and `data/mock_tools_example_rollouts.jsonl` are rollouts of
the two examples, with GLM 5.3 (`nvidia/zai-org/glm-5.3`) as the policy and the judge.

## Testing

```bash
gym env test --resources-server partner_if
```

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
