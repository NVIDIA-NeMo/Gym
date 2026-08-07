# Finance Agent v2 (FABv2) Resource Server

A NeMo Gym integration of the official [Vals Finance Agent Benchmark v2](https://github.com/vals-ai/finance-agent-v2)
that **reuses Vals's own tool code directly** (tools-only wrap) instead of
reimplementing it. The upstream `finance_agent.tools.*` classes are imported and
exposed as HTTP endpoints; the `responses_api_agents/finance_agent_v2` loop drives
them. Scoring uses our own per-criterion rubric judge (path A), aggregated into
Vals's dealbreaker-gated, severity-weighted Partial Credit — see
[Verification](#verification).

This is the v2 counterpart to `resources_servers/finance_sec_search` (v1). The key
difference: v1 reimplements the tools; v2 imports them from upstream so the tool
descriptions, parameters, and behavior track Vals automatically via a dependency bump.

## Tools (imported from `finance_agent.tools`)

| Tool | Description | Requires |
|------|-------------|----------|
| `web_search` | Tavily web search (`TavilyWebSearch`) | `tavily_api_key` |
| `edgar_search` | sec-api.io full-text EDGAR search (`EDGARSearch`) | `sec_api_key` |
| `price_history` | Tiingo daily OHLC for equity/etf/crypto/fx (`PriceHistory`) | `pricing_data_api_key` |
| `parse_html_page` | Fetch + parse a page to text, store under a key (`ParseHtmlPage`) | — |
| `retrieve_information` | LLM over stored docs via `{{key}}` prompts (`RetrieveInformation`) | `retrieval_model_server` |
| `calculator` | Safe arithmetic via simpleeval (`Calculator`) | — |
| `submit_final_result` | Submit the final answer; ends the loop (`SubmitFinalResult`) | — |
| `sec_filing_search` | **Optional** (training/SDG): data.sec.gov ticker→CIK filing listing (local `SecFilingSearch`, not from Vals) | — |

`parse_html_page` and `retrieve_information` share a per-session **data storage**
(`state`) dict, scoped by the HTTP session cookie. A tool whose required key/model
is not configured is registered as unavailable and its endpoint returns a clear
error (the agent can route around it). New in v2 vs v1: `calculator`, `price_history`.
The upstream date clamp is `MAX_END_DATE = 2026-03-01` (enforced inside the Vals tools).

**SEC tool surface (`enabled_sec_tools`).** `edgar_search` (sec-api.io full-text
search) is byte-parity with Vals — use it for **eval** (the default). For
**training/SDG** you can additionally enable `sec_filing_search`, a cheaper
data.sec.gov ticker→CIK listing that needs no key; it changes the tool surface vs
Vals and is **not** byte-parity, so keep it out of eval. See
[Caching](#caching-pricing--sec).

## Dependencies

Both upstream pins are exact so the benchmark stays reproducible (bump
deliberately):

```
-e nemo-gym[dev] @ ../../
model-library==0.1.25
finance-agent @ git+https://github.com/vals-ai/finance-agent-v2.git@<pinned-sha>
```

`model-library` declares `openai[aiohttp]>=2.28.0`, which cannot resolve against
the vllm-compatible `openai` that Gym pins. The upstream code runs on the older
client, so only the declared floor is the problem, and `overrides.txt` drops it:

```
openai[aiohttp]<=2.7.2
```

`uv --override` replaces *every* declared constraint on a package, Gym's included,
so this mirrors the cap in the root `pyproject.toml` rather than merely widening
the floor — `openai<3.0` here would pull a far newer client than Gym is tested
against. Bump the two together; a test asserts they stay in sync. No core
`openai`/`vllm` bump is needed.

`finance-agent` publishes no tags and is not on PyPI, so it is pinned to the HEAD
commit at integration time.

nemo-gym requires Python >=3.13.14, which is the binding floor here.

## Setup (`env.yaml`)

This is a self-contained Gym environment: run it with **two configs** — this
environment config (`configs/finance_agent_v2.yaml`) plus a model config
(`responses_api_models/openai_model/configs/openai_model.yaml` for OpenAI, or
`responses_api_models/vllm_model/configs/vllm_model.yaml` for a self-hosted
vLLM endpoint).

Nothing below is required for the config to resolve — every credential and
endpoint resolves as **config key → shell environment → null-safe default**, so
`gym env resolve` and `+dry_run` work on a clean checkout with none of it set.

Secrets live in `env.yaml` at the repo root (gitignored — never commit a populated
copy). Only the **model endpoints** need to go there; the **tool API keys are
easiest to export in your shell**, and one left unset simply registers its tool as
unavailable:

```bash
export OPENAI_API_KEY=...        # policy + judge (OpenAI)
export SEC_API_KEY=...           # edgar_search (sec-api.io)
export TAVILY_API_KEY=...        # web_search (Tavily)
export TIINGO_API_KEY=...        # price_history (Tiingo)
```

```yaml
# env.yaml — model endpoints only (tool keys come from the shell, above)
# Policy model, also reused for retrieve_information. For a self-hosted model,
# point these at your vLLM endpoint and run with vllm_model.yaml.
policy_base_url: https://api.openai.com/v1
policy_api_key: ${oc.env:OPENAI_API_KEY}
policy_model_name: gpt-5-mini

# Judge model used by /verify — a separate server from the policy on purpose, so
# that models evaluated on this benchmark are all graded by the same judge and
# their scores stay comparable. Any OpenAI-compatible endpoint, local vLLM
# included. Omit to fall back to gpt-5-mini on api.openai.com.
search_judge_model_base_url: https://api.openai.com/v1
search_judge_model_api_key: ${oc.env:OPENAI_API_KEY}
search_judge_model_name: gpt-5-mini
```

> Note: "airgap-friendly" applies to **grading** only. The v2 tools call external
> APIs (Tavily, sec-api.io, Tiingo) at rollout time and need network egress + keys.
> A tool whose key is unset simply registers as unavailable (no crash).

## Run

The gym CLI (NeMo Gym >= 0.5.0) drives everything. The public benchmark recipe
(dataset + `prepare.py` + config) lives in `benchmarks/finance_agent_v2/`; the
server code and its `gym env test` fixtures (`data/example*.jsonl`,
`data/example_metrics.json`) stay here under `resources_servers/finance_agent_v2/`.
There is no `environments/finance_agent_v2/` entry: training runs this same
resources server on externally generated SDG data.

```bash
# Unit tests + example-data validation for this resources server
gym env test --resources-server finance_agent_v2

# Build the public 27Q benchmark JSONL (downloads the Vals CSV if absent)
gym eval prepare --benchmark finance_agent_v2

# End-to-end (prepare + start servers + collect rollouts) on the benchmark set
gym eval run --benchmark finance_agent_v2 \
  -c responses_api_models/openai_model/configs/openai_model.yaml
```

### Quickstart: public 27-question smoke run (OpenAI gpt-5-mini)

Run the Vals public question set end-to-end on the OpenAI API (`gpt-5-mini` for
both policy and judge), limited to 3 rollouts to confirm the agent + tools +
rubric grading path works. Two configs only: the environment config and the
OpenAI model config.

1. **Secrets** — populate `env.yaml` (model endpoints) and export the tool keys
   (`OPENAI_API_KEY`, `SEC_API_KEY`, `TAVILY_API_KEY`, `TIINGO_API_KEY`) as shown
   in [Setup](#setup-envyaml).

2. **Prepare** the benchmark JSONL. `gym eval prepare` runs
   `benchmarks/finance_agent_v2/prepare.py`, which downloads the raw Vals public
   CSV from [finance-agent-v2](https://github.com/vals-ai/finance-agent-v2) (if no
   local source is present) and converts it. The prompts and tool schemas come from
   `benchmarks/finance_agent_v2/upstream_spec.json`, a snapshot of
   `finance_agent.prompt` / `finance_agent.tools` (see
   [Upstream snapshot](#upstream-snapshot)). The CSV's `rubric` is copied through
   verbatim — those criteria are what `/verify` scores. The script also synthesizes a
   concatenated `expected_answer` from them, which is now carried as a human-readable
   reference and is **not** read by scoring. Runs in the root venv:

   ```bash
   source .venv/bin/activate
   gym eval prepare --benchmark finance_agent_v2
   ```

3. **Start the servers** — resources server + OpenAI model config:

   ```bash
   gym env start --resources-server finance_agent_v2 \
     -c responses_api_models/openai_model/configs/openai_model.yaml
   ```

4. **Collect rollouts** against the running servers, limited to 3 questions
   (`--no-serve` reuses the servers from step 3). The agent is `finance_agent_v2`
   here because step 3 started the resources-server config; starting from the
   benchmark config instead gives you `finance_agent_v2_benchmark_agent`:

   ```bash
   gym eval run --no-serve \
     --agent finance_agent_v2 \
     --input benchmarks/finance_agent_v2/data/vals_v2_public_27q.jsonl \
     --output results/finance_agent_v2_smoke.jsonl \
     --limit 3 --concurrency 3
   ```

   Rewards land in `results/finance_agent_v2_smoke.jsonl` (Partial Credit: the
   severity-weighted share of rubric criteria passed, zeroed if a `must_pass`
   criterion failed — see [Verification](#verification)). Drop `--limit` to run the
   full set. To run on a
   self-hosted model, use `-c responses_api_models/vllm_model/configs/vllm_model.yaml`
   and point `policy_*` in `env.yaml` at your vLLM endpoint.

### Committed example fixtures

`data/example.jsonl` is the first 5 questions of the public 27Q set, so it is
regenerated by rerunning `gym eval prepare` and taking the first 5 lines.
`data/example_rollouts.jsonl` holds the matching rollouts from a `gpt-5.6-luna` run
(first repeat), scored by the current per-criterion judge; their rewards span a
tripped dealbreaker (0.0 at 6/9 criteria passed), two partial scores, and a clean
1.0, so the aggregate fixture exercises gating rather than a single flat outcome.
Refresh them together — a rollout scored by an older judge will not match the
metrics this server now emits.

## Caching (pricing / SEC)

Tiingo and SEC calls are rate-limited and (within the pinned window) return
**immutable** data, so the server can cache them to disk. Caching is a thin
**wrapper around the imported upstream tools** — never a reimplementation: each
`Cached*` class subclasses the upstream tool and overrides only its network
method, stores the **raw upstream response**, and re-serializes it with the
untouched upstream code. A cache hit is therefore **byte-identical** to a live
call (and survives a future upstream formatting/SHA bump without a refetch). See
`cache.py` and `cached_tools.py`.

Caching is on by default (`use_cache: true`); point `cache_dir` at a shared
absolute path (config, or `FINANCE_AGENT_V2_CACHE_DIR`) so it survives across
jobs. When `use_cache` is on the cache both serves hits and persists misses;
set `use_cache: false` to run every tool live.

```yaml
use_cache: true                            # true = read+write, false = off
cache_dir: /shared/cache/finance_agent_v2  # null -> ~/.cache/nemo_gym/finance_agent_v2
```

`edgar_search` and `sec_filing_search` share the same on-disk cache root and
document layout: whichever tool the agent used to locate a filing, the actual
document bytes are fetched by `parse_html_page` and land at the same
`sec_filings/<cik>/<accession>/<file>` path, so the two never re-download the
same document.

What gets cached, and why it stays in sync with Vals:

| Source | Cached unit | Parity |
|--------|-------------|--------|
| `price_history` (Tiingo) | per-`(endpoint, ticker)` master of raw records; sliced on read via the upstream `_records_to_csv` | byte-identical |
| `edgar_search` (sec-api.io) | raw `filings` list in `edgar_search/<query-slug>_<hash>.json` (request stored alongside for debugging) | byte-identical |
| `parse_html_page` on **sec.gov** filing URLs | parsed text at `sec_filings/<cik>/<accession>/<primary-doc>.txt` | identical parse; general web is **not** cached |

**Price as-of / drift.** Raw prices are immutable, and returns/ratios are
invariant to the adjustment as-of date; only an *adjusted absolute level* can
drift with the fetch date. We cache Tiingo's response **verbatim** (never
re-derive adjusted), and on a window extension keep already-cached rows, so
re-runs are reproducible.

**Prefetch (offline eval).** Populate the cache sequentially ahead of an eval so
runtime price queries are served from disk instead of hitting Tiingo:

```bash
python resources_servers/finance_agent_v2/scripts/prefetch_prices.py \
  --cache-dir /shared/cache/finance_agent_v2 \
  --tickers AAPL MSFT NVDA --asset-class equity
```

## Dataset & labels (path-A scoring)

The public FABv2 release ships **only question strings** (no ground truth/grader).
`benchmarks/finance_agent_v2/prepare.py` (the `gym eval prepare` entry point)
loads input from its `data/` dir by this precedence:

1. `data/labeled.jsonl` — labeled rows (enables real scoring)
2. `data/public.jsonl` — rows with at least `{question}`
3. `data/public.txt` — one question per line (FABv2 public format)
4. `data/public.csv` — raw Vals public CSV (`question`/`prompt` column)

**Labeled JSONL schema** (one object per line):

```json
{"question": "...", "expected_answer": "...", "rubric": "[{\"operator\": \"...\", \"criteria\": \"...\", \"modifiers\": {\"severity\": 3.0, \"category\": \"must_pass\"}}]"}
```

- `rubric` is propagated from the public CSV verbatim and is **the scoring input**:
  each `criteria` string is judged on its own, and `modifiers` supplies the
  `severity` weight and `must_pass` dealbreaker flag used to aggregate the verdicts
  (see [Verification](#verification)). `modifiers` is optional — exports predating
  Aug 2026 have none, and those criteria default to severity 1.0 and non-gating.
  (The public FABv2 release has no official grader; Vals's private grader is
  licensed and is deliberately not reproduced here — the judge here is ours.)
- `expected_answer` is the rubric criteria concatenated into a readable reference.
  Nothing reads it any more; it is kept so existing prepared datasets stay valid
  and is a candidate for removal at the next dataset regeneration.
- To source labels at scale, publish a labeled set to the GitLab Model Registry
  (mirrors v1's `finance_sec_search_vals_200_eval`) and point the dataset entry in
  `benchmarks/finance_agent_v2/config.yaml` at it (`type: benchmark` +
  `gitlab_identifier`).

**Interim dry-run:** with no `rubric`, `/verify` returns `reward=0` with
`judge_error` set, so the agent + tools path can be validated before ground truth
is available without those rows looking like genuine zeros.

## Upstream snapshot

Each prepared sample also carries the upstream system/question prompts and the tool
JSON schemas. Those are derived from the `finance_agent` package rather than
retyped, but they are read at prepare time from a committed snapshot,
`benchmarks/finance_agent_v2/upstream_spec.json`.

The reason is venv scope: `finance_agent` is a dependency of **this server**, so it
is installed in `resources_servers/finance_agent_v2/.venv` only, while `gym eval
prepare` imports the benchmark's `prepare.py` into the repo-root `gym` process. A
prepare script that imported the package would work only when invoked by hand from
this venv, which is not how benchmarks are prepared; every other benchmark's prepare
script depends on root-venv packages alone.

The snapshot is generated in this venv, where the package exists:

```bash
cd resources_servers/finance_agent_v2 && source .venv/bin/activate
python scripts/export_upstream_spec.py            # regenerate
python scripts/export_upstream_spec.py --check     # exit 1 if stale
```

A file can go stale where an import cannot, so two guards stand in for the live
import — the dataset must not advertise a tool signature the agent cannot honor:

- `tests/test_upstream_spec.py` re-derives the snapshot from the installed package
  and fails on any difference, so `gym env test` catches drift.
- `prepare.py` refuses to run when its `_UPSTREAM_SHA` and the snapshot's
  `upstream_commit_id` disagree, which is what makes bumping the `finance-agent` pin in
  `requirements.txt` without re-exporting a loud failure rather than a dataset
  pairing one commit's questions with another's prompts.

**Bumping the upstream pin** is therefore: edit the pin in `requirements.txt`, edit
`_UPSTREAM_SHA` in `prepare.py` to match, `uv sync`, re-run
`export_upstream_spec.py`, re-run `gym eval prepare`, and re-baseline any scores.

## Verification

The public FABv2 release ships **no official grader**, so scoring is **our own**:
each criterion in the dataset's `rubric` is judged separately against the answer
the agent passed to `submit_final_result`.

Per criterion, the judge is asked for a binary verdict (`1` = the answer asserts
this claim, `0` = it does not) as JSON. Calls repeat until
`judge_required_successes` (default 3) replies parse into an integer 0/1, capped at
`judge_max_attempts` (default 10); API errors, timeouts, and unparseable replies
consume an attempt and are retried. Only a reply that ran out of output budget
escalates anything (the budget doubles, and the timeout with it) — an unparseable
reply is not a budget problem, and lowering reasoning effort to chase one would
just make the judge worse at the numeric comparisons this rubric is full of. The
criterion's score is the majority of those verdicts — an odd count cannot tie —
and every vote is retained in `rubric_judgements` with its evidence and reason.
Criteria are judged concurrently up to `judge_max_concurrency`.

Verdict JSON is located by a brace-balanced, string-aware scan rather than a
regex. This is load-bearing: the judge quotes the answer verbatim as evidence, and
finance answers carry LaTeX-style subscripts (`EBITDAR_{WMT}`), so a `{.*?}` regex
ends its match inside the quote and every retry fails identically. Failed replies
are sampled into `rubric_judgements[].failed_reply_samples` so the next parse bug
is diagnosable from the rollout file instead of only from live logs.

### Aggregating verdicts into a score

The verdicts are combined the way Vals combines theirs. Each criterion carries
`modifiers` from the dataset — `severity` (a weight, 1.0/2.0/3.0 across the public
set) and `category: must_pass` (a dealbreaker) — and the reward is **Partial
Credit**: the severity-weighted pass fraction, forced to `0.0` if any dealbreaker
failed.

The stricter every-criterion-passed flag is reported alongside as
`rubric_all_pass`, and `scripts/rescore_rubrics.py` recomputes both from stored
rollouts without re-judging anything.

The judge is deliberately **not** shown severity or `must_pass`. It grades each
criterion on its merits and the weighting is applied afterwards, so a criterion
cannot be graded more harshly for being expensive to fail. Criteria with no
`modifiers` at all default to severity 1.0 and non-gating, so an unlabeled dataset
scores as a plain unweighted pass fraction.

| Field | Meaning |
|---|---|
| `reward` | **Partial Credit** — the weighted pass fraction, zeroed by any failed dealbreaker |
| `rubric_partial_credit` | the same number under Vals's name for it |
| `rubric_weighted_fraction` | Partial Credit *before* gating; the gap between the two is what the dealbreakers cost |
| `rubric_all_pass` | `true` only when every criterion resolved and passed — Vals's All-Pass |
| `rubric_fraction` | criteria passed / total, unweighted |
| `rubric_weight_passed` / `rubric_weight_total` | the severity mass behind the weighted fraction |
| `rubric_dealbreakers_failed` / `rubric_dealbreakers_total` | how many `must_pass` criteria there were and how many failed |
| `rubric_passed` / `rubric_total` | pass counts unnormalized |
| `rubric_unresolved` | criteria that never got 3 parsable verdicts |
| `rubric_judgements[].severity` / `.must_pass` | the weights used, recorded per criterion so a score can be recomputed from the rollout alone |
| `rubric_judgements[].votes` / `.unanimous` | per-criterion vote record; non-unanimous means the judge contradicted itself |
| `rubric_judgements[].failed_reply_samples` | first few failed judge replies, error-tagged and clipped |
| `judge_error` | set when scoring could not complete (no rubric, or any unresolved criterion) — **filter these out rather than reading them as zeros** |

A criterion the judge never resolved is a judge failure, not a miss, so it scores
`null` rather than `0` and flags the whole row via `judge_error`. It still weighs
as not-passed (and gates, if it was a dealbreaker), because treating an absent
verdict as a pass would inflate exactly the runs whose judging broke — `judge_error`
is what keeps those rows filterable.

`/aggregate_metrics` reports `mean/rubric_partial_credit`,
`mean/rubric_weighted_fraction`, `mean/rubric_all_pass`, `mean/rubric_fraction`,
`mean/rubric_dealbreaker_tripped`, `mean/criterion_pass_rate`, and
`mean/judge_disagreement_rate` (share of resolved criteria that were not unanimous —
watch this over time), plus unresolved/judge-failure counts alongside the score so
infrastructure noise cannot hide inside it.

The agent also records why its loop ended in `response.metadata.stop_reason`
(`done_tool`, `max_turns`, `max_time`, `max_output_tokens`, `error`) with a `steps`
count. Under dealbreaker gating a truncated trajectory scores the same as a
confidently wrong one, so check this before reading a low score as a capability
result.

Because both the judge model and this prompt are ours, scores are **not**
comparable to Vals's published numbers. The judge prompt lives in
`prompt_templates/finance_agent_v2_rubric_judge.yaml`; override it inline with
`rubric_judge_prompt_template` or point `rubric_judge_prompt_template_fpath`
elsewhere.

### Reading these metrics against the Vals leaderboard

Vals reports two scores in their
[published methodology](https://www.vals.ai/benchmarks/fabv2). Their leaderboard's
**Accuracy** column is *Partial Credit*: a dealbreaker-gated, severity-weighted
average of per-check scores, where failing any check flagged as a dealbreaker
zeroes the question outright. *All-Pass* is their secondary metric — 100% only if
every check passes. Both are computed on the private 450-question Test split, as
the mean of three runs, graded by a jury of GPT-5.4, Gemini-3.1-Pro, and Claude
Sonnet 4.6.

Both metrics now have a same-definition counterpart here:

- `mean/rubric_partial_credit` (the reward) matches their **Accuracy** definition.
- `mean/rubric_all_pass` matches their **All-Pass** definition.
- `mean/rubric_fraction` is ungated and unweighted, so it has **no counterpart** on
  the leaderboard and must not be placed next to the Accuracy column. Before
  weighting existed, a 27Q `gpt-5.6-luna` run scored `rubric_fraction` 0.822 against
  a published Accuracy of 55.04% — reported side by side that reads as beating the
  leaderboard by 27 points, when the two measure different things. The same rollouts
  score 0.580 Partial Credit, which is a number you can actually compare. Keep the
  unweighted fraction as the internal regression signal it is.

Matching definitions is not the same as matching numbers. Three differences remain,
and all of them are larger than they look on 27 questions:

1. **Different questions.** We score the 27 public questions; the leaderboard uses
   the private 450-question Test split.
2. **Different judge.** One model with self-consistency voting, against their
   three-model jury, using a prompt we wrote (theirs is not published).
3. **Small-sample noise.** With 27 questions the 95% CI on Partial Credit spans
   roughly ±0.17 — wide enough to swallow most leaderboard gaps between adjacent
   models. `scripts/report_run.py` prints that interval; use it before calling any
   difference real.

Until Aug 2026 this was moot: the public CSV shipped criteria with only `operator`
and `criteria`, so the weights and dealbreaker flags Partial Credit depends on were
not in the open release. Upstream then added a `modifiers` object to all 239 public
criteria (severity 1.0/2.0/3.0 on 170/60/9 of them; 79 marked `must_pass`) without
changing a single criterion's text — which is why old runs can be rescored rather
than re-judged.

`verify` is a pure function of the request body plus this server's config, so the
server declares `REVERIFY_MODE = STATELESS`: after changing the judge prompt or
parameters you can rescore stored rollouts without re-running the policy, and
without `--force`.

```bash
gym eval reverify --benchmark finance_agent_v2 --model-type openai_model \
  --inputs results/<run>_materialized_inputs.jsonl \
  --rollouts results/<run>.jsonl \
  --output results/<run>_rejudged.jsonl --concurrency 4
```

Select the **same** config the rollouts were collected with. Reverify routes each
row by the agent name stored on it, so rescoring a benchmark run against
`--resources-server finance_agent_v2` fails with
`KeyError: 'finance_agent_v2_benchmark_agent'` — that config declares the
`finance_agent_v2` agent instead. `--model-type` is required either way, because
the resources server's `retrieval_model_server` reference must resolve even though
reverify only calls `/verify`.

## Analyzing a finished run

Reverify re-runs the judge. When only the *aggregation* changes — different severity
weights, or gating disabled — the stored verdicts are still valid and rescoring is
free:

This one imports `app.py`, so it needs **this server's** venv:

```bash
resources_servers/finance_agent_v2/.venv/bin/python \
  resources_servers/finance_agent_v2/scripts/rescore_rubrics.py \
  results/<run>.jsonl --strict --write results/<run>_weighted.jsonl
```

Weights default to the prepared dataset (`gym eval prepare` output), so a run
collected before the modifiers existed is rescored against them with no extra flags.
It joins each stored criterion to its weights by exact text, reports anything
unmatched (`--strict` turns that into a non-zero exit), and scores through the same
`aggregate_rubric_scores` the server uses live, so a rescored number cannot drift
from a freshly collected one.

`report_run.py` is standard-library only and runs anywhere:

```bash
python resources_servers/finance_agent_v2/scripts/report_run.py \
  results/<run>.jsonl --per-question
```

`report_run.py` prints what a bare mean cannot: bootstrap confidence intervals,
repeat spread, tool usage, and tool errors. Two details matter for reading it.

Means are taken over **questions**, averaging repeats first, and the CI resamples
questions rather than rollouts. Repeats of one question are not independent draws —
a question the model reliably nails contributes almost no variance while the spread
*between* questions is large — so resampling rollouts would treat 3 repeats of 27
questions as 81 independent samples and report an interval about √3 too narrow.
Averaging repeats first also stops a run that died partway through from
over-weighting whichever questions happened to finish.

The tool-error counts expose a real hazard. Upstream's calculator uses `simpleeval`,
where `^` is bitwise XOR rather than exponentiation: `(46000/34857)^(1/3)` raises and
returns the generic `Error: invalid expression '...'`, but `2^10` **succeeds** and
returns 8. The report counts `^` usage split by outcome, because the succeeding case
puts a wrong number into the answer with nothing logged anywhere. Both behaviours are
upstream's and are reproduced byte-for-byte on purpose — the error text is part of
the observation the agent is scored on, so "fixing" it would change trajectories and
void comparisons against the leaderboard.

## File structure

```
resources_servers/finance_agent_v2/         # server code + gym env test fixtures
├── app.py                         # Resource server: tool endpoints + retrieval shim + verify
├── cache.py                       # ToolCache: namespaced atomic disk cache + read/write policy
├── cached_tools.py                # Cached* wrappers (price/edgar/parse) + SecFilingSearch
├── requirements.txt               # Pins nemo-gym + Vals model-library + finance-agent
├── overrides.txt                  # Drops model-library's openai floor (see Dependencies)
├── configs/
│   └── finance_agent_v2.yaml      # Resources-server config used by gym env test / gym dataset collate
│                                  # (the benchmark recipe config_paths-chains to this; no duplication)
├── prompt_templates/              # judge / retrieval (loaded at runtime; server cwd = this dir)
├── scripts/
│   ├── prefetch_prices.py         # Sequential Tiingo prefetch into the cache (idempotent/resumable)
│   ├── compare_runs.py            # Compare rollout JSONLs by per-question Partial Credit
│   ├── rescore_rubrics.py         # Rescore finished rollouts under severity weighting (no re-judging)
│   ├── report_run.py              # Scores with bootstrap CIs, repeat spread, tool usage, tool errors
│   └── export_upstream_spec.py    # Regenerate the benchmark's upstream prompt/tool-schema snapshot
├── data/                          # gym env test fixtures: example.jsonl (first 5 of the public 27Q),
│                                  # example_metrics.json, example_rollouts.jsonl (gpt-5.6-luna),
│                                  # example_rollouts_aggregate_metrics.json
└── tests/                         # test_app.py (server), test_cache.py (cache), test_upstream_spec.py (snapshot drift)

benchmarks/finance_agent_v2/                # public eval recipe (thin: config + prepare only)
├── config.yaml                    # Thin overlay: config_paths -> resources config + _inherit_from + benchmark dataset
├── prepare.py                     # gym eval prepare entry point + CSV->JSONL converter (stdlib only; runs in the root venv)
├── upstream_spec.json             # Generated snapshot of upstream prompts + tool schemas (do not hand-edit)
└── data/                          # gitignored; prepare.py regenerates vals_v2_public_27q.jsonl from the upstream Vals export
```

There is deliberately **no `environments/finance_agent_v2/`**: training data is
produced by an external nvflow SDG pipeline rather than shipped in-repo, so an
environment entry would only duplicate the server config and drift from it. Add
one if and when a train/validation split is released with the benchmark.

Credentials and endpoints are covered in the
[benchmark README](../../benchmarks/finance_agent_v2/README.md#setup); every one
resolves as config key → shell environment → null-safe default, so this
environment resolves standalone on a clean checkout.

## Licensing

**This environment's code** (everything under `resources_servers/finance_agent_v2/`
and `benchmarks/finance_agent_v2/`) is licensed under **Apache-2.0**, consistent
with NeMo Gym and the SPDX headers in each source file (`app.py`,
`tests/test_app.py`, `benchmarks/finance_agent_v2/prepare.py`).

**Upstream dependencies** (imported, not vendored — see `requirements.txt`):

| Package | Source | License |
|---------|--------|---------|
| `finance-agent` (`finance_agent.tools` / `finance_agent.prompt`; also the source of the benchmark's `upstream_spec.json` snapshot) | [vals-ai/finance-agent-v2](https://github.com/vals-ai/finance-agent-v2) | MIT |
| `model-library` (`model_library.*`) | [vals-ai/model-library](https://github.com/vals-ai/model-library)@`0.1.25`, from PyPI | MIT |

We import these packages at install time and do not copy their source into this
repo, so their MIT terms apply to that code as distributed upstream.

**Dataset.** The `example.jsonl` fixtures (here) and
`benchmarks/finance_agent_v2/data/vals_v2_public_27q.jsonl` derive
from the **public** Vals Finance Agent Benchmark v2 release
([vals-ai/finance-agent-v2](https://github.com/vals-ai/finance-agent-v2)); use is
subject to that project's terms. The public release ships **no official grader**.

**Grading is our own.** Reward comes from our own judge prompt
(`prompt_templates/finance_agent_v2_rubric_judge.yaml`) run once per criterion of
the public `rubric` field, voted over repeated calls. Vals's private grader
(prompts + reward logic) was obtained under a separate license and is
**deliberately not reproduced** in this public code; the prompt here was written
from scratch, so scores are not directly comparable to Vals's published numbers.
