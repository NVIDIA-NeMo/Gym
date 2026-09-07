# BigFinanceBench

Gym integration for the 50-item public subset of
[BigFinanceBench](https://github.com/Rogo-Technologies/big-finance-benchmark).
Dataset content is downloaded from commit
`d794a65fe583edc6852b44c817b0a2aef33ca831`; benchmark preparation does not
depend on a sibling checkout.

## Prepare and run

From the Gym repository root:

```bash
gym eval prepare --benchmark big_finance
gym eval run --benchmark big_finance \
  -c responses_api_models/openai_model/configs/openai_model.yaml
gym eval reverify --benchmark big_finance --model-type openai_model \
  --inputs results/<run>_materialized_inputs.jsonl \
  --rollouts results/<run>.jsonl \
  --output results/<run>_rejudged.jsonl --concurrency 4
```

The resource-server config also exposes a five-row example dataset at
`resources_servers/big_finance/data/example.jsonl`. That committed fixture was
converted from the first five rows of a local checkout of the same pinned
public subset; it is for configuration and schema smoke coverage, not a
replacement for preparing the full 50-item benchmark.

Set `SERP_API_KEY` (preferred) or `TAVILY_API_KEY` for web search and
`SEC_EDGAR_USER_AGENT="Name email@example.com"` for EDGAR and SEC document
requests. Configure the policy model in the normal Gym overlay. Configure the
independent judge with `big_finance_judge_base_url`,
`big_finance_judge_api_key`, and `big_finance_judge_model_name`.

`reward_mode=final_answer` is the default and matches the paper headline:
reward is the judge's binary `final_answer_correct`. Set
`big_finance_reward_mode=rubric_points` to reward the point-weighted rubric
fraction instead. Every result includes both metrics, each rubric verdict, the
judge's raw text, and any judge error. Reverification is stateless.

## Parity and safety

The server imports the upstream `WebSearchTool`, `EdgarSearchTool`,
`FetchUrlTool`, `PythonExecTool`, and `FinalAnswerTool` from a commit-pinned
Apache-2.0 fork. Its base install contains only tool dependencies; LiteLLM and
Google providers remain in an optional `eval` extra and are not installed by
Gym. Tool names, schemas, order, and returned strings are preserved. The short
system prompt, tool surface, original source commit, and fork package commit are
frozen in `upstream_spec.json`; offline parity tests check them.

The grader ports upstream trace caps, one structured JSON judge call, binary
rubric decisions, and point aggregation, but routes the judge through a Gym
model server rather than LiteLLM.

The shared Gym `finance_agent` is configured here with
`prose_only_behavior: finish`, `tool_call_execution: concurrent`, and
`done_tools: [final_answer]`. This matches the standalone harness's prose
fallback, concurrent execution of a returned tool-call batch, and terminal-tool
semantics without adding BigFinance-specific branches to the shared loop.

`python_exec` starts an isolated Python child process with a timeout, but it is
**not a security sandbox**. Only run trusted policies in an appropriately
isolated environment.

The public 50-item dataset is © 2026 Rogo Technologies and the Big Finance
authors, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preparation records
the pinned source URL, commit, attribution, evaluation-only flag, do-not-train
flag, benchmark canary, and per-item sources. Harness code is Apache-2.0.
