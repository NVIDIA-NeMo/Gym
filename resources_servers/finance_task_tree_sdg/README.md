# Finance task-tree SDG

Resources server for generating expert finance research questions from a task tree. One server
implementation serves five stages; each stage has its own config in `configs/` and is selected by
`stage`.

| Stage | Tools | Submit tool | Verify |
| --- | --- | --- | --- |
| `question_gen` | `edgar_search`, `fetch_url`, `python_exec` | `submit_question` or `submit_rejection` | judge |
| `leak_proposal` | none | `submit_leak_proposal` | judge |
| `leak_removal` | none | `submit_revised_question` | judge |
| `answer_gen` | `edgar_search`, `fetch_url`, `python_exec` | `final_answer` | passthrough |
| `adjudicate` | `edgar_search`, `fetch_url`, `python_exec` | `submit_verdict` | passthrough |

## Row contract

Each row carries `responses_create_params` (prompt and the stage's tool specs) and an `sdg`
mapping:

- `sample_id`, `as_of_date` (`YYYY-MM-DD`), `task` (task-tree leaf and variant), `companies`,
  `fiscal_year`, `anchor_documents`, `reveal_level` (`company_fy`, `company_fy_form`, or
  `specific_filing`);
- `question` from `leak_proposal` on, `question_gen_trace` for `leak_proposal`, `leak_proposal`
  for `leak_removal`, and `candidates` (`[{label, answer}]`) for `adjudicate`.

Verify returns the row plus `reward` and `sdg_result`. `sdg_result.status` is one of
`judge_pass`, `judge_fail`, `judge_unparsed`, `rejected`, `no_submission`, `answered`,
`no_answer`, or `adjudicated`. Judged stages reward 1 only on `judge_pass`.

## As-of clamp

`edgar_search` lists only filings filed on or before the row's `as_of_date`, paging the full
EDGAR submission history. `fetch_url` refuses EDGAR archive documents filed after it. Other URLs,
including the XBRL API, are not blocked.

## Judge isolation

The judge returns only `{"verdict": "pass" | "fail"}`. Its full input and output are written to
`audit_dir/<stage>/<sample_id>/<rollout_id>.json` when `audit_dir` is set and never appear in
the verify response, so no judge text reaches a later stage's model. The judge defaults to
`policy_model`; NVFlow rebinds `judge_model_server` to its judge role.

## Config overrides

- `sec_edgar_user_agent` (falls back to `SEC_EDGAR_USER_AGENT`)
- `tt_sdg_use_cache`, `tt_sdg_cache_dir`: SEC response cache
- `tt_sdg_audit_dir`: judge audit directory

## Tests

```bash
pytest resources_servers/finance_task_tree_sdg/tests
```

The example rows in `data/` are illustrative; downstream-stage questions, proposals, and
candidates are not real model outputs.
