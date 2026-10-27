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
  --model-type openai_model \
  --config resources_servers/big_finance/configs/openai_model.yaml \
  --split benchmark \
  --output results/big_finance.jsonl
gym eval reverify --benchmark big_finance --model-type openai_model \
  --config resources_servers/big_finance/configs/openai_model.yaml \
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

## Running without web search

Web search is on by default. To run without it, switch both the rows and the
server:

```bash
python benchmarks/big_finance/prepare.py --no-web-search --input <rows>.jsonl --output <rows>.no_web.jsonl
```

and set `big_finance_web_search: false`. The rows then omit the `web_search` tool,
the system prompt lists only SEC EDGAR filings, URL fetching, Python, and the final
answer, and the `fetch_url` description no longer points to `web_search`. Every
other tool schema, the prompt text, and the grading are unchanged. The server builds
no search backend and needs no `SERP_API_KEY` or `TAVILY_API_KEY`; a `web_search`
call returns the standard "Tool 'web_search' is not configured." error observation.
The two settings are independent, so rows and server must be switched together.

The shipped `gpt-5-mini` judge is a convenient default. Judge model choice affects
scores. When comparing with published results, explicitly override
`big_finance_judge_model_name` and match the judge model snapshot and grading
settings used for those results, including the reward mode. Set
`big_finance_judge_base_url` and `big_finance_judge_api_key` as needed for the
selected endpoint.

BigFinance copies `policy_model` into its own `big_finance_policy_model`
instance. The OpenAI overlay in the commands above applies its request settings
only to that copy, leaving the shared model available with its original settings
for other benchmarks, including Vals v1/v2. The copy also inherits the policy
model's `extra_body`; BigFinance-specific overrides can be added under
`big_finance_policy_model.responses_api_models.openai_model.extra_body` in a
custom overlay. For another model type, omit this OpenAI overlay; the copy uses
that model's normal settings.

`reward_mode=final_answer` is the default and matches the paper headline:
reward is the judge's binary `final_answer_correct`. Set
`big_finance_reward_mode=rubric_points` to reward the point-weighted rubric
fraction instead. Successful grades include both metrics, each rubric verdict,
and the judge's raw text. Reverification is stateless.

Judge call failures and replies without a readable grading JSON object use Gym's
standard `judge_failed` path, matching upstream's omission of failed grades.
The full rollout and failure reason are preserved in
`<output>_failures.jsonl`, excluded from scores by default, and retryable on resume.
The failure response is marked `mask_sample: true`; its placeholder reward is
not a correctness verdict. A valid judge verdict of `final_answer_correct: false`
remains an ordinary scored zero.

For generation-only rollouts without a reference answer or rubric, set
`big_finance_reward_mode=passthrough`. The verify endpoint then preserves the
request and rollout response, extracts a final answer when possible, and returns
`reward=1.0` without calling or requiring the judge model. The default remains
`final_answer`.

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

Tool arguments are normalized for the judge's trace using upstream JSON formatting
before truncation. Arguments that exceed Python's parsing limits retain their raw
text. This rendering leaves stored rollouts and actual tool inputs/outputs unchanged.
For a prose completion, the final answer comes only from the last model turn;
an empty last turn does not reuse earlier analysis.

The shared Gym `finance_agent` is configured here with
`prose_only_behavior: finish`, `tool_call_execution: concurrent`, and
`tool_error_observation: error_prefix`, and `done_tools: [final_answer]`.
These settings implement the standalone harness's prose fallback, concurrent
execution of a returned tool-call batch, and terminal-tool and error-observation
semantics without adding BigFinance-specific branches to the shared loop.
`done_tool_requires_success: true` applies to both sequential and concurrent
execution. A turn with no tool calls ends the run even when it contains only
reasoning or no output; such a turn has no final answer.

Tool failures are signaled explicitly by the resources server rather than inferred
from an `error` field in successful content. The agent records per-call error
statuses in rollout metadata, which the verifier also uses when rejudging. Older
rollouts without this metadata retain the legacy error detection.

With the OpenAI overlay, the policy model makes at most 13 upstream attempts per
request; the independent judge makes at most 21. Each attempt has a 1,800-second
timeout, with a fixed 0.5-second delay between retries. BigFinance uses Gym's
existing outer retry policy with inner retries disabled: HTTP errors, connection
failures, timeouts, and invalid responses share one attempt budget. The configured
terminal HTTP statuses and permanent endpoint errors stop retries immediately.
These limits describe Gym's configured policy; LiteLLM's retry scheduling may
differ. HTTP and connection errors no longer have separate retry budgets.

`python_exec` starts an isolated Python child process with a timeout, but it is
**not a security sandbox**. Only run trusted policies in an appropriately
isolated environment.

The public 50-item dataset is © 2026 Rogo Technologies and the Big Finance
authors, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preparation records
the pinned source URL, commit, attribution, evaluation-only flag, do-not-train
flag, benchmark canary, and per-item sources. Harness code is Apache-2.0.
