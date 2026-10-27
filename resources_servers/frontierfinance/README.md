# FrontierFinance Resources Server

Runs [FrontierFinance](https://huggingface.co/datasets/samaya-ai/FrontierFinance)
investor-research queries on the [Finance Agent v2](../finance_agent_v2/README.md)
tools and grades the answers with the official
[FrontierFinance grader](https://github.com/samaya-ai/frontier-finance).

Each query asks an open-ended question "as of" a date and carries expert-written
rubrics: atomic criteria a good answer should satisfy. The reward is the share of a
query's rubrics the answer satisfies.

The server subclasses the `finance_agent_v2` server, so the tools, caching, local
data modes and the `finance_agent` loop behave exactly as described in that README.
This server adds two things: a per-query date cap on the tools, and the official
grading.

## The "as of" date

Every sample carries an `as_of_date` (the dataset's `query_date`). The system prompt
tells the model that this is the current date, and the server caps `end_date` (and
`start_date`) on every `web_search`, `edgar_search` and `price_history` call at that
date. A call without an `end_date` gets `as_of_date`. The tool schemas the model sees
are unchanged.

Two sources can still return later information:

- `parse_html_page` fetches the live version of a page. SEC filings do not change,
  but company websites and news pages may have been updated since the date.
- Tavily filters by its own publish date for each page, which can be earlier than the
  page's latest edit.

## Tool limits

The FrontierFinance paper's Finance Agent v2 harness allows 200 tool calls and 300
wall-clock seconds of tool calling per question, and states both in the system
prompt. The server enforces the same limits:

- `max_tool_calls: 200`. Past it, every tool except `submit_final_result` returns an
  error telling the model to submit.
- `max_tool_time_seconds: 300`, the total time spent inside tool calls. Model
  generation time does not count, so a slow model endpoint does not shrink the
  budget. Past it, tools return the same kind of error.

Neither limit stops the rollout; the model still gets to submit an answer. The
agent's `max_steps` and `max_time_seconds` are hard stops that end the rollout with
no answer, so they are set well above the server limits. The prompt values come from
`benchmarks/frontier_finance/prepare.py`; change them there and in the config
together.

## Grading

The `frontier-finance` package is installed from a pinned commit (see
`requirements.txt`) and used directly: its judge prompt, reply parser, batching,
majority vote and metrics. Only the transport differs: each judge is a Gym model
server instead of a provider SDK.

For each query, every judge in `judge_panel` receives the query, its date, the
submitted answer and up to `max_rubrics_per_call` (30) rubrics per call, and returns
a true/false label per rubric. An unparseable reply is retried once with a format
reminder; a rubric the reply omits counts as not met.

- **Panel vote:** each rubric is decided by majority vote of the judges that
  succeeded on the query. On a tie, the earliest of them in `judge_panel` wins, so
  keep the panel odd.
- **Judge failures:** a judge that errors on any batch, or returns the wrong number
  of labels, is left out of that query's vote and counted in
  `failed_checks_by_judge`. If every judge fails, the sample is masked
  (`mask_sample: true`, `failure_kind: judge_failed`) and left out of all scores.
- **No answer:** if the agent never calls `submit_final_result`, the query scores 0
  (`grading_failure: no_response`) and counts against the "all queries" metrics.

The default panel is the one the FrontierFinance authors recommend for comparable
results: Claude Sonnet 4.6, Gemini 3.1 Pro (low thinking) and GPT-5.4, at
temperature 0 with a 32k output budget.

| Field | Meaning |
|---|---|
| `reward` | Rubrics met / total rubrics for the query |
| `rubric_labels` | Final per-rubric labels, aligned with `rubrics` |
| `judge_labels` | Each judge's labels, or `null` if that judge failed |
| `judge_replies` | Raw judge replies, for auditing |
| `num_must_have`, `num_must_have_qualified` | The essential-rubric subset |
| `grading_failure` | `no_response`, `judge_error`, or `null` |

### Metrics

Aggregate metrics are computed by the package's own `MetricsReport` and reported
under `frontier_finance/`:

- macro (mean of per-query rates) and micro (total met / total rubrics) qualification
  rates, over graded queries (`_on_success_queries`) and over all queries
  (`_on_all_queries`);
- the same four for `must_have` rubrics only;
- micro rates by `rubric_type`, `data_source_type` and `use_case`;
- `num_records`, `num_failed_queries` and failed checks per judge.

With `--num-repeats`, the headline rates are also reported per repeat.

## Setup

Run with this config plus a policy model config. Judge endpoints default to each
provider's OpenAI-compatible API, with keys from `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY` and `OPENAI_API_KEY`. To route all three through one
OpenAI-compatible gateway, set the shared base URL and key, and each judge's model
name as the gateway knows it:

```yaml
# env.yaml
frontier_finance_judge_base_url: https://my-gateway.example.com/v1
frontier_finance_judge_api_key: ${oc.env:GATEWAY_API_KEY}
frontier_finance_judge_claude_model: anthropic/claude-sonnet-4-6
frontier_finance_judge_gemini_model: google/gemini-3.1-pro-preview
frontier_finance_judge_gpt_model: openai/gpt-5.4

# Tools, as for finance_agent_v2
sec_api_key: ${oc.env:SEC_API_KEY}
tavily_api_key: ${oc.env:TAVILY_API_KEY}
pricing_data_api_key: ${oc.env:TIINGO_API_KEY}
frontier_finance_cache_dir: /shared/cache/frontier_finance
```

Each judge also accepts its own `frontier_finance_judge_<claude|gemini|gpt>_base_url`
and `_api_key`. `edgar_search_mode` defaults to `live`; set it to `local` with
`local_edgar_index_path` to search a local SEC index instead (see the
`finance_agent_v2` README).

## Run

The benchmark recipe lives in [`benchmarks/frontier_finance`](../../benchmarks/frontier_finance/README.md).

```bash
gym env test --resources-server frontierfinance
```

## Licensing

Code in this directory is Apache-2.0. The `frontier-finance` grader (Apache-2.0),
`finance-agent` (MIT) and `model-library` (MIT) are installed as dependencies, not
copied. The FrontierFinance dataset is CC-BY-4.0, gated on Hugging Face, and may not
be reshared outside a gated repository; it is downloaded at prepare time and never
committed. The queries in `data/example.jsonl` are not from the dataset.

## Example data

`data/example.jsonl` holds five single-fact SEC filing lookups, each with one rubric
that restates the expected answer. They are easy to check by hand and any capable
model should score close to 1.0 on them, so a low score points to a broken setup
(tools, date cap or judges) rather than a weak model.
