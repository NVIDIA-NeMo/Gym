# Finance task-tree SDG

Resources server for generating expert finance research questions from a task tree. One server
implementation serves nine stages; each stage has its own config in `configs/` and is selected by
`stage`.

| Stage | Tools | Submit tool | Verify |
| --- | --- | --- | --- |
| `eligibility` | none | `submit_eligibility` | passthrough |
| `leaf_profile` | none | `submit_leaf_profile` | passthrough |
| `probe` | `edgar_search`, `fetch_url`, `python_exec` | `submit_probe` | passthrough |
| `doc_match` | `fetch_url` | `submit_match` | passthrough |
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

`eligibility` runs once per task-tree leaf before sampling; its `sdg` carries `sample_id` and
`leaf_id` (both the leaf id) and `sic_codes`, the SIC codes present in the company list.
`submit_eligibility` takes `scope` (`any_industry`, `restricted`, or `unsupported`), `sic_codes`
(non-empty and drawn from the row's list only when `restricted`), and `reason`.

`leaf_profile` also runs once per leaf; its `sdg` carries `variants`
(`{variant_id: {company_scope, role_ids}}`). `submit_leaf_profile` lists every variant once with
`sec_obtainable`, `company_relation` (`none` for single-company variants, else `peer` or
`counterparty`), and, when obtainable, exactly one entry per role with `filing_kinds`, `timing`,
and `company`. `probe` rows carry `roles` (`{role_id: [candidate urls]}`); `submit_probe` takes
`feasible` and `documents` (`[{role_id, url}]` drawn from the candidates, at least one per role
when feasible). `doc_match` rows carry `choices` (`{leaf_id: [modes]}`); `submit_match` takes a
listed `leaf_id` and one of its modes, or `leaf_id` `none`.

Verify returns the row plus `reward` and `sdg_result`. `sdg_result.status` is one of `mapped`,
`profiled`, `probed`, `matched`, `judge_pass`, `judge_off_leaf`, `judge_fail`, `judge_unparsed`,
`rejected`, `no_submission`, `answered`, `no_answer`, or `adjudicated`. Judged stages reward 1
only on `judge_pass`, and `question_gen` also on `judge_off_leaf`: the judge failed only
criterion 2 (faithful to the task), so the question is a sound finance question outside its leaf.
`probe` rewards 1 when feasible and `doc_match` when a leaf was matched.

## As-of clamp

`edgar_search` lists only filings filed on or before the row's `as_of_date`, paging the full
EDGAR submission history. `fetch_url` refuses EDGAR archive documents filed after it. Other URLs,
including the XBRL API, are not blocked.

## Offline SEC corpus

With `sec_corpus_metadata_path` set to an NVFlow `download-sec` `sec_metadata.parquet`, the SEC
tools send no request to SEC. `edgar_search` lists, and resolves tickers from, only the filings
whose primary document is in the corpus. `fetch_url` serves those documents, their exhibits,
and a generated filing index page for each filing; every other SEC URL (full-text search, browse
pages, XBRL viewer pages, documents not downloaded) is refused with a tool error. Non-HTML files
and exhibits whose saved file is shared by several URLs are not served. Non-SEC URLs are fetched
as before.

## Judge isolation

The judge returns `{"verdict", "failed_criteria", "reason"}`; verify passes on only the verdict.
The criteria, reason, and the judge's full input and output are written to
`audit_dir/<stage>/<sample_id>/<rollout_id>.json` when `audit_dir` is set and never appear in
the verify response, so no judge text reaches a later stage's model. The judge defaults to
`policy_model`; NVFlow rebinds `judge_model_server` to its judge role.

## Config overrides

- `sec_edgar_user_agent` (falls back to `SEC_EDGAR_USER_AGENT`)
- `tt_sdg_use_cache`, `tt_sdg_cache_dir`: SEC response cache
- `tt_sdg_sec_corpus_metadata_path`: offline SEC corpus (see above)
- `tt_sdg_audit_dir`: judge audit directory
- `tt_sdg_question_gen_judge_prompt`: question_gen judge prompt file (default
  `prompts/judge_question_gen.txt`); keep criterion 2 as faithful to the task, which
  `judge_off_leaf` relies on

## Tests

```bash
pytest resources_servers/finance_task_tree_sdg/tests
```

The example rows in `data/` are illustrative; downstream-stage questions, proposals, and
candidates are not real model outputs.
