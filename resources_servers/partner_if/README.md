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

Built-in verifiers: `llm_judge_v1` (optional `artifacts`: the captured artifacts an item
needs; omit it to require all), `text_contains_v1` (`source`: `answer` or
`artifact:<name>`; `literal`) and `text_no_uuid_v1` (fails if the answer contains a UUID).

## Scoring

- `reward` is 1 when every IF item passes; `if_result` reports `passed` and `total`.
- Completion items are reported in `grading_result.dimensions` and never change the reward.
- A failed judge call raises out of `/verify` into Gym's judge failsafe: the row is masked
  with `failure_kind: judge_failed`, written to the failures file
  (`<output>_failures.jsonl`) and retried on resume, and
  `gym eval reverify --judge-failed-only` can recover it. `/grade` returns HTTP 500
  instead; the Harbor agent sends that row, like any Harbor or grading failure, to the
  failures file.
- These rollouts are masked (`mask_sample: true`, `failure_kind: partner_if:unscorable`):
  - an IF item or IF prerequisite is unscorable: the judge reply is invalid or
    incomplete, a capture the item needs is missing or failed, a live agent's trajectory
    is incomplete or does not match its tool receipts, or an item it depends on is
    unscorable;
  - on this server's `/verify`, the reply is incomplete, or, in `text_response` and
    `frozen_tool_history` modes, blank or not text-only;
  - on this server's `/verify`, the input relies on hidden context
    (`previous_response_id`, `conversation`, `prompt`) or content its mode does not
    support, including `live_agent` rows.
- A mock-tool backend that requires no final answer grades a reply cut off by the
  output-token cap on the tool exchanges completed before it.
- Contracts with `audit_signoff: false` are scored only when the server sets
  `allow_provisional_rubrics: true`, and their results are marked `provisional`.
  Otherwise `/verify` masks the rollout, while `/grade`, the mock-tool server and the SWE
  server reject the request with HTTP 422.
- Metrics: `all_if_pass` and `if_partial` (the passed fraction of IF items), overall and
  per `origin`. Masked rollouts are left out of them and counted under `coverage/*`;
  rows in the failures file are left out entirely.

Key config fields: `judge_model_server`, `judge_max_output_tokens` (8192),
`judge_temperature` (0.0), `judge_max_concurrency` (8), `allow_provisional_rubrics`
(false) and `task_check_modules` ([]). The SWE server defaults to 32768 output tokens
and a concurrency of 2.

## Task-specific checks and live agents

- `task_check_modules`: importable modules whose `CHECKERS` dict maps verifier IDs to
  `checks.Checker` objects, usually `checks.fixed(fn)`. They are registered at startup
  and cannot redefine a registered ID.
- `mock_tools_app.py` serves the `ToolBackend` named by `backend: module:Class` over
  `fixture_path` and `tasks_path`. `swe_app.py` (Scale-SWE with an OpenCode agent) and
  `responses_api_agents/partner_if_harbor_agent` also capture `live_agent` evidence.
  `responses_api_agents/partner_if_dispatcher` routes each task of a mixed benchmark to
  its environment.

Relative paths in these configs resolve under `--search-dir` / `NEMO_GYM_EXTRA_ROOTS`
first, then the working directory, then the Gym root.

## Usage

```bash
gym env start --resources-server partner_if \
    +policy_base_url=... +policy_api_key=... +policy_model_name=... \
    +partner_if_judge_base_url=... +partner_if_judge_api_key=... +partner_if_judge_model_name=...

gym eval run --no-serve --agent partner_if_simple_agent \
    --input resources_servers/partner_if/data/example.jsonl \
    --output partner_if_rollouts.jsonl
```

`configs/partner_if.yaml` includes the `openai_model` policy server. `data/example.jsonl`
holds five synthetic examples. `configs/verifier.yaml` starts only the server and its
judge, for `gym eval reverify`.

## Testing

```bash
gym env test --resources-server partner_if
```

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
- markdown-it-py: MIT
