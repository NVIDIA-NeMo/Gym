# WANDR resources server

Tools and scoring for [WANDR](../../benchmarks/wandr/README.md). The server needs the data that
`benchmarks/wandr/prepare.py` writes (`tasks_dir`, `wandr_core_dir`).

## Tools

| Route | Does |
| --- | --- |
| `/web_search` | With `search_backend: exa` (default), calls Exa `/search` with `exa_api_key` and returns highlights as snippets. With `perplexity`, calls `search_url` (the NVIDIA inference API's `perplexity-search`) with `inference_api_key`. Provider error bodies are not passed to the model. |
| `/fetch_page` | GETs the page and returns its main text (trafilatura for HTML, pypdf for PDF) in slices of `fetch_page_chars`. Pages are cached per server process. |
| `/write_records` | Checks each record with WANDR's row rules (http(s) `url`, `item` object with the file's item fields, `answer` object, `excerpts` list of strings) and returns accepted and rejected rows. It keeps no state. |

## Verification

1. Rebuild each required output file from the `write_records` calls whose tool output accepted rows.
2. If a required file is empty, return reward 0 with `failure_kind: agent_invalid_output`, which is how WANDR
   scores such a submission, without running the verifier.
3. Otherwise run `run_verifier.py` in a subprocess. It runs the task's unchanged upstream `verify.py` with two
   client swaps: judge model names go through `judge_model_template` to `judge_base_url`, and
   `verifier_fetch_backend` picks how cited pages are fetched: `exa` (default) uses Exa `/contents` with
   `exa_api_key`, `direct` a plain HTTP GET, and `perplexity` keeps the upstream Perplexity Agent API fetch,
   which needs `perplexity_api_key` on an org with zero data retention off. If Exa refuses the key, the
   verification fails instead of scoring records against pages that were never fetched.
   WANDR's browser fallback stays off, and the `camoufox` import is stubbed when the package is absent.
4. Read `reward.json`. The reward is `reward_metric` (default `soft_f1_full`), and all twelve metrics are returned.
   A verifier error or timeout is retried `inconclusive_verification_retries` times and then masked
   (`mask_sample: true`, `failure_kind: verifier_error`).

At most `max_concurrent_verifications` verifiers run at once. Each has the concurrency limits in
`verifier_env` (the upstream full-run settings). Set `verifier_artifacts_dir` to keep each verification's
`report.txt`, `report.html`, `wandr_metrics.json` and debug replays.

## Tests

`tests/test_app.py` covers the row checks, file reconstruction, reward selection, masking and page text. Two tests run
the real upstream verifier when `benchmarks/wandr/data` is prepared: an empty submission (offline) and, with
`WANDR_LIVE_TEST_INFERENCE_API_KEY` set, one judged record (paid judge calls).
