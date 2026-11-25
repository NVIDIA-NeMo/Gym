# Arena Proxy v3 validation

## Config composition and system messages (November 25, 2026)

Sampling defaults now use Gym's `responses_create_params`; model-specific top-k
and context come from the composed model profile. Context is required in both
YAML and the model-server schema, with no 163,840-token fallback. Judge credentials,
endpoint, model, and the shared prepared-population path use `judge_api_key`,
`judge_base_url`, `judge_model_name`, and `input_jsonl` config keys. The legacy
environment variables no longer configure the benchmark. The README documents
explicit CLI overrides and protected configuration for credentials.

System text messages now pass through to tokenization and generation with their
content and message fields preserved. The reference generator already forwards
these roles. Existing prepared conversations contain only user/assistant turns,
and their outgoing reference payloads are unchanged. Adding a system prompt can
change answers; the reference must receive the same prompt for comparison.

Benchmark-local `submit/` files and their orchestration tests were removed.
External deployments own scheduling, containers, model serving, and parsers;
the README uses `gym eval run` followed by `gym eval reverify`. Preparation,
source checksums, judge templates, verdict parsing, scoring, final-answer
extraction, the 120,000-token maximum, and the 64-token reserve are unchanged.

The focused benchmark/configuration/generation/metrics/judge/English suite passed
**259 tests**. The new configuration tests run Gym's actual config parser and
collector preprocessing before checking outgoing requests, including temperature
1.0 and 0.6, the Gemma profile, and context-limited budgets. They also check
environment isolation, shared generation/judge populations, protected judge-key
loading and redaction, profile ordering, and missing-context rejection. System
messages are covered through Chat Completions, Responses input, and Responses
instructions, with identical tokenization and generation messages.

Against the previous model implementation, all 12 existing reference-payload
cases still pass and all five core system-message/required-context regressions
fail. This distinguishes the compatibility fixes from the preserved reference
request behavior. The actual CLI resolves the documented explicit profile and
`++input_jsonl` override; README Bash blocks also pass `bash -n`.

A fresh five-task Hindi run used the documented `gym eval run` / `gym eval reverify`
path, an explicit Gemma profile, and config-based judge credentials. These were
five multi-turn information/advice tasks. The model,
checkpoint, vLLM, sampling, context, budget, and judge settings match the full Hindi
run below; generation and judging concurrency were both five. All five policy
answers completed and were preserved for judging, with no empty answers, policy
truncations, or leaked thinking delimiters. Nine of ten positional verdicts were
valid; one judge reply omitted the required verdict and was retained under the
unchanged scoring protocol. All 29 available scalar metrics match across server,
repeat, and offline outputs; undefined values are omitted from Gym metrics and
no official headline is emitted.
This subset checks execution and configuration, not full-population scores or
score equivalence. Its run identifier is `gemma31b-config-smoke-20261125162303Z`.

Two additional synthetic requests against the same Gemma checkpoint passed through
the actual Gym model client: a Chat Completions system message and Responses
instructions. Tokenization and generation received identical system text, including
whitespace, and both returned completed, nonempty final answers. These probes used
a 2,048-token output cap and were excluded from benchmark scoring. The initial
probe incorrectly expected raw Chat Completions output to omit reasoning markers;
it was corrected to apply the existing final-answer extraction and rerun, with no
application-code change. Probe run: `gemma31b-system-probes-20261125T162915Z`.

```bash
RAY_TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -o addopts= --no-cov -q \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_config.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_generation.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_metrics.py \
  resources_servers/arena_judge/tests/test_app.py \
  benchmarks/lmarena_v3/tests/test_lmarena_v3_subsets.py

.venv/bin/pre-commit run --all-files
git diff --check
```

All-file pre-commit passed. The private guardword catalog was unavailable, so its
content scan was skipped; the other configured checks passed.

The full Hindi score comparison below remains evidence for revision `2197986`;
it is not relabeled as a fresh full run of this configuration follow-up.
Historical sections retain the commands and deployment files used at their
named revisions, including the subsequently removed submission configs.

## Full Hindi comparison (November 25, 2026)

Validation head: `2197986` (`21979867381558c35409b3b92ee009eb595dd17c`).
The complete Hindi population from `lmarena_v3_indic/0.0.1` was generated and
judged through Gym: **400 tasks, one repeat, 320 single-turn and 80 multi-turn**.
Inputs came from the checksum-verified internal registry cache. Prompts and frozen
GPT-5.4-mini answers matched the reference, with no sample limit or exclusions.
The comparison uses the archived reference's independently generated and judged
Hindi answers; the reference pipeline was not rerun for this evaluation.

| Setting | Value |
| --- | --- |
| Policy model | `google/gemma-4-31B-it` |
| Checkpoint revision | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| Serving | vLLM 0.26.0; BF16; tensor/data parallelism 8/1; expert parallelism disabled |
| Temperature / top-p / top-k / seed | 1.0 / 0.95 / 64 / 42 |
| Context / maximum output / reserve | 131,072 / 120,000 / 64 tokens |
| Reasoning / tool parser | `gemma4` / `gemma4`; thinking enabled |
| Generation concurrency / maximum sequences | 256 / 256 |
| GPU memory utilization | 0.85 |
| Baseline | Frozen `gpt-5.4-mini` answers |
| Judge | `gcp/google/gemini-3.8-flash`; temperature 0; reasoning effort low; maximum output 32,768 |
| Judge concurrency | 12 tasks, two swapped positions each; 24 positional calls |

| Hindi population | Tasks | Gym score | Reference score | Gym minus reference | Gym 95% interval | Reference 95% interval |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| All | 400 | **67.40** | **69.41** | **-2.01** | 65.39–70.03 | 67.24–71.24 |
| Single-turn | 320 | 64.30 | 66.88 | -2.58 | 61.68–67.01 | 64.20–69.08 |
| Multi-turn | 80 | 80.38 | 80.69 | -0.31 | 76.44–84.81 | 76.44–85.32 |

Scores use strong-verdict weight 3 and the same Arena-Elo calculation, with
100 bootstrap rounds and seed 42. Reference intervals were recomputed in prepared
input order; the table shows absolute bounds, while JSON stores offsets. The
modest difference is consistent with independent generation/judging variation;
overlapping intervals do not establish equivalence. This is a full **Hindi**
comparison, not the 12-language Indic headline, which correctly remains absent.

All 50 scalar values across the ten Hindi slices (score, both interval offsets,
invalid-verdict count, and task count) match exactly between Gym server metrics,
repeat metrics, and the offline summary. The overall server metric is
`arena_elo/language/hi/all/score`. Independent weighted counts reproduce all ten
Gym and ten reference point estimates.

All 400 generations and judged rows completed with no missing tasks, empty policy
answers, policy truncations, or leaked thinking delimiters. Every prompt-token
count matched the archived reference, and reverification preserved the generated
answers. Of 800 positional judge replies, 13 were invalid versus 10 in the
reference. The existing parser independently reproduced all 800 saved decisions;
invalid verdicts were retained and excluded individually from weighted battles,
without selective retries. Judge finish reasons were not saved, so judge
truncation cannot be established from these rows.

The recorded checkpoint revision and serving/sampling settings match, but weight
bytes and container-image digest equality were not established. The reference
batched Indic languages together; Gym generated Hindi alone. This was the
two-stage `gym eval run` / `gym eval reverify` workflow with a separately submitted
GPU service, not a deployment validation of `gym eval submit` containers/mounts.
Artifact health reports 400 expected/completed/scored rows and zero failures;
all 400 trajectory-quality checks remain unobserved because captures were disabled.
An 18-case qualitative review found useful Hindi and instruction following, along
with arithmetic errors, incomplete code, and unsupported research claims. The
judge sometimes preferred objectively incorrect arithmetic, so the score measures
relative judge preference rather than factual accuracy.

The completed run is identified as `gemma31b-hi-full-20261125103947Z`. Prepared
Hindi input SHA-256:

```text
cd5c2b6b6d97b379e0447d33f6d4da33475aef763c53330475c6450e69fcb08e
```

Comparison command, with artifact locations represented by environment variables:

```bash
.venv/bin/python -m benchmarks.indic.arena_proxy_v3.summarize \
  "$RUN_DIR/judgments/rollouts.jsonl" \
  --inputs "$RUN_DIR/inputs/arena_proxy_v3_benchmark.jsonl" \
  --reference-verdicts "$REFERENCE_HI_VERDICTS" \
  --output "$RUN_DIR/comparison.json"
```

Raw conversations, answers, credentials, and machine-specific paths are not part
of this evidence commit. The full 12-language comparison remains outstanding;
this evidence does not establish Nemotron 4 model validation or rollout approval.

## Dataset and preparation parity

The benchmark now uses Gym's GitLab dataset downloader, following `lmarena_v3`,
with dataset `lmarena_v3_indic` pinned to version `0.0.1`.

A live `gym eval prepare` run downloaded and validated all 24 original JSONL files:
12 prompt files and 12 frozen GPT-5.4-mini reference-answer files. Their bytes and
SHA-256 hashes match the original reference source. The reference files retain
`references/gpt-5.4-mini/<lang>.jsonl`; only locally prepared Gym inputs join the
conversations and reference answers.

The resulting 4,799 tasks are byte-for-byte identical to the previous preparer
at commit `406deea245b27a7a5612dab2ea98c4c6ee7f26ec`, run against the original source
with the same 12-language selection. The combined prepared-input SHA-256 is:

```text
12963471e78a41cf3fc3357f80ec6a4d8e6dda46ba9002d1308671b54ec421f1
```

Each language has 400 eligible tasks except Tamil, which has 399. The failed Tamil
translation remains in the raw source and appears in the preparation report's
exclusions. English requires an explicit local source and is scored separately.

An initial download stalled while opening an artifact connection in the shared
registry helper. The helper now uses connect/read timeouts of 10/120 seconds and
checks HTTP status before opening the output file. Preparation completed after
this correction, reusing the already validated source cache.

## Reference protocol

The registry migration does not alter policy generation, judge prompts, verdict
parsing, or score aggregation. A direct comparison with the supplied reference
verified both parsed judge prompt templates and the ASTs of these seven methods:

- `_parse_verdict`
- `_aggregate_arena_elo`
- `_get_battles_from_judgment`
- `_compute_mle_elo`
- `_bootstrap`
- `_predict_win_rate`
- `_get_win_rate_column`

Generation request tests cover full conversation history, temperature 1.0 and 0.6,
top-p/top-k, seed 42, the 120,000-token cap, the measured context budget with a
64-token reserve, and final-answer extraction. Summary tests cover swapped
positions, strong-verdict weight 3, deterministic bootstrap, complete population
checks, English separation, and the mean across 12 Indic languages.

The official reference-comparable headline is now the server metric
`arena_elo/indic/score`, also formatted as `_avg.all` by the offline summary.
Both paths share the same reducer. Indic configuration selects the language mean
instead of the pooled `arena_elo/score`; other Arena configurations retain their
existing metrics. The reference-specific model adapter remains unchanged.

## Automated checks

195 tests passed with Python 3.13.14 and the repository's locked dependencies:

```bash
RAY_TMPDIR=/tmp .venv/bin/python -m pytest -o addopts= --no-cov -q \
  tests/unit_tests/data/test_gitlab_utils.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_generation.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_metrics.py \
  resources_servers/arena_judge/tests/test_app.py \
  benchmarks/lmarena_v3/tests/test_lmarena_v3_subsets.py
```

The registry tests cover all 12 languages, separate reference paths, pinned
versions, cache reuse, corrupt-cache replacement, HTTP error bodies, changed
artifacts, and preservation of existing prepared data. The shared downloader's
tests verify bounded timeouts and preservation of existing files on HTTP
401/404/500 responses. The existing English benchmark's subset tests pass.

The live preparation used the benchmark config plus an ignored registry settings
file and an isolated output directory. No registry endpoint, credential value, or
source dataset was added to Git. Shell syntax checks passed for all four Slurm
scripts. Scikit-learn emits its existing `penalty` deprecation warnings.

## Model rollout evidence and limits

The earlier November 20 smoke used `google/gemma-4-31B-it` at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` with vLLM 0.26.0. Five samples across
English, Hindi, Bengali, Tamil, and Telugu produced five answers and ten valid
swapped-position judgments. They were translations of one multi-turn conversation,
not five independent conversations. The selected subset completed with zero current
failures and no official Indic headline.

No new GPU rollout was needed for this data-access migration: the prepared input
bytes and generation/judging code are unchanged for the selected population.
The full Hindi run and comparison with separately generated reference judgments
are recorded above at validation head `2197986`. Dataset licensing metadata is
`NVIDIA internal, derived from arena-proxy-v3`. Full 12-language comparison and
end-to-end score equivalence have not been established.

## Server metric integration

The Indic configuration opts into `arena_elo/indic/score` as a Gym key metric,
with per-language slices in agent metrics and a matching single-repeat record for
comparison suites. Other Arena configurations retain pooled metrics. The same
server-owned reducer formats saved judgments through `summarize.py`.

The saved November 20 Gemma judgments were replayed through a local HTTP
`/aggregate_metrics` endpoint. All five tasks returned HTTP 200 aggregation;
the full offline summary matched the implementation at commit
`f847cd1579b826262a0833f8f81318ef208b69a8` exactly. Every available language score,
confidence interval, invalid-verdict count, and sample count matched the server's
agent and repeat metrics. The tracking exporter preserved those scalar values.
The five-task subset correctly omitted the official headline in all metric views.
This replay used saved policy and judge responses; no new model calls were made.

`gym env validate indic/arena_proxy_v3` validates all 4,799 prepared tasks. The
manifest uses Gym's supported `internal` licensing value; the descriptive dataset
license remains `NVIDIA internal, derived from arena-proxy-v3` in the config and
documentation.

The final targeted suite passed **303 tests**:

```bash
RAY_TMPDIR=/tmp .venv/bin/python -m pytest -o addopts= --no-cov -q \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_metrics.py \
  resources_servers/arena_judge/tests/test_app.py \
  tests/unit_tests/metrics/test_aggregate_metrics.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_generation.py \
  benchmarks/lmarena_v3/tests/test_lmarena_v3_subsets.py

.venv/bin/gym env validate indic/arena_proxy_v3
.venv/bin/pre-commit run --all-files
git diff --check
```

The tests cover the full 12-language mean, English exclusion, scalar tracking
export, single-repeat reporting, missing and masked tasks, partial preparations,
duplicate/repeated tasks, and unchanged default Arena aggregation. The complete
fixture uses 399 eligible Tamil tasks and 400 for every other language, so it
distinguishes equal language weighting from pooled task weighting. Nine verifier
and Arena-Elo method ASTs are unchanged from the previous commit.

Pre-commit passed across all files; the private guardword catalog was unavailable,
so its content scan was skipped. The full core, sandbox, and server suites were
not run locally; the complete PR requires full CI scope. Existing scikit-learn and
statistical edge-case warnings remain.

## Submission-config migration (November 25, 2026)

The benchmark now uses Gym's Slurm submission configs for policy serving and CPU
reverification. The CPU job retains `afterok` ordering, reads the same saved policy
responses and materialized inputs, and uses the original prepared population for
coverage. The Gemma profile preserves its context, top-k, parsers, and parallelism.
Both drivers install the same published Gym revision and locked dependencies.

At submission-migration commit `0210c1ce`, the generation model, preflight,
generation/judge configs, preparer, prompt templates, source checksums, summarizer,
and metric reducer were byte-identical to
`002507bb8887ec97c950cf9eca595e2770c05f01`. The judge's executable AST was unchanged;
only the thinking-stripper docstring changed. Its literal-tag behavior and use on
candidate, baseline, and judge text are now explicit.

All three submission profiles passed the real CLI's `--resolve-only --json` and
`--dry-run` paths. Rendered scripts passed `bash -n`. Command-level tests verify
the reference service/sampling settings, preflight ordering and failure handling,
CPU-only judging, saved-input paths, resume flags, dependency, and rejection of
an overly permissive key file. The token is read only by the CPU command and is
absent from the resolved configuration. Slurm's explicit
[`exclusive=exclusive` spelling](https://github.com/SchedMD/slurm/blob/slurm-24-11-0-1/src/common/slurm_opt.c#L1184)
retains the reference's exclusive allocation in Gym's key/value directives.

The five saved Gemma judgments again matched the earlier summary exactly and
returned matching language scores, intervals, and counts through the HTTP metrics
endpoint and tracking exporter. No model calls were made in this replay. A fresh
Slurm/GPU rollout with these submission configs was not run; model, verifier, and
scoring behavior are unchanged, while cluster-specific containers and mounts still
need deployment validation.

The final targeted suite passed **267 tests**, including nine submission-command
tests and the existing CLI, generation-request, preparation, metric, judge, and
English benchmark regressions:

```bash
RAY_TMPDIR=/tmp .venv/bin/python -m pytest -o addopts= --no-cov -q \
  tests/unit_tests/orchestration/test_indic_arena_submission.py \
  tests/unit_tests/orchestration/test_cli_eval_submit.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_generation.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_metrics.py \
  resources_servers/arena_judge/tests/test_app.py \
  benchmarks/lmarena_v3/tests/test_lmarena_v3_subsets.py

# With non-secret deployment variables set as in submit/README.md:
for profile in generate gemma31b judge; do
  .venv/bin/gym eval submit --config "benchmarks/indic/arena_proxy_v3/submit/$profile.yaml" --resolve-only --json
  .venv/bin/gym eval submit --config "benchmarks/indic/arena_proxy_v3/submit/$profile.yaml" --dry-run
done
.venv/bin/gym env validate indic/arena_proxy_v3
.venv/bin/pre-commit run --all-files
git diff --check
```

## Generation profile preservation (November 25, 2026)

The benchmark now forwards Gym's processed request instead of rebuilding it from
a restricted field list. Model-profile stop settings, penalties, reasoning options, direct
chat-template settings, metadata overrides, and sampling pins are preserved.
Tokenization uses Gym's prompt-field selector so it measures the same template
and reasoning settings sent to generation. The request's output-budget alias is
preserved, with the same measured 64-token context reserve.

Reference-incompatible profiles fail at startup. Conflicting per-request settings
return HTTP 422 before tokenization or generation, including budgets above
120,000, disabled thinking, truncation, continuation, and multiple completions.
The final-answer converter, data, baselines, judge, and scoring are unchanged.

Exact reference-payload tests cover both conversation shapes, both sampling
profiles, and three context-budget cases. Additional regressions check profile
and metadata precedence, direct kwargs, stop/penalty forwarding, prompt fields,
sampling pins, both budget aliases, and explicit rejection. Extra profile settings
can now change answers because they are honored; comparisons require the same
effective settings in the reference. Passing payload tests does not establish
end-to-end score equality.

Against the previous adapter, both new field-preservation regressions fail (lost
`stop` and `max_completion_tokens`), while all 12 reference-payload cases pass.
With this fix, those same reference payloads remain identical and the additional
fields survive. The standard vLLM request/template/sampling/budget suite also
passes 188 selected cases.

The benchmark/judge/submission/English regression suite passed **298 tests**;
the focused vLLM suite passed **188 tests** (486 total). The converter AST and
shared model/resources servers are unchanged. Commands against the final code:

```bash
RAY_TMPDIR=/tmp .venv/bin/python -m pytest -o addopts= --no-cov -q \
  tests/unit_tests/orchestration/test_indic_arena_submission.py \
  tests/unit_tests/orchestration/test_cli_eval_submit.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_generation.py \
  tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_metrics.py \
  resources_servers/arena_judge/tests/test_app.py \
  benchmarks/lmarena_v3/tests/test_lmarena_v3_subsets.py

RAY_TMPDIR=/tmp .venv/bin/python -m pytest -o addopts= --no-cov -q \
  responses_api_models/vllm_model/tests/test_app.py \
  -k 'chat_template_kwargs or sampling or tokenize or budget or prompt_refit'

.venv/bin/pre-commit run --all-files
git diff --check
```

Pre-commit passed; the unavailable private guardword catalog prevented that
content scan. Existing optional histogram-export, statistical, and test-client
warnings remain. The full core/sandbox/server CI suite was not run locally.

A fresh five-task smoke completed with `google/gemma-4-31B-it` revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`, vLLM 0.26.0, TP=8,
context=131,072, temperature=1.0, top-p=0.95, top-k=64, seed=42, and the Gemma
reasoning/tool parsers. The profile explicitly included zero presence and
frequency penalties. The smoke cap was **8,192 tokens**, with five concurrent
tasks and one repeat. All responses completed without truncation (224–346 output
tokens), with nonempty visible answers and no leaked thinking delimiters.

The five historical inputs cover English, Hindi, Bengali, Tamil, and Telugu;
they are translations of one multi-turn conversation. Prompt roles/content and
frozen baselines were preserved. Gym adds message type/phase fields when saving
its typed request; these do not change the conversation text. Reverification
preserved each saved policy response and produced **10 valid swapped-position
verdicts**, with zero failed tasks or invalid verdicts. Both the Gym aggregate
metrics and offline summary reported English separately and omitted the subset
headline. Coverage was complete for the five selected tasks.

The GPU model-service job completed successfully. The CPU driver used the final
code through `gym eval run`, `gym eval reverify`, and the benchmark summarizer:

```bash
# RUN_DIR contains the five prepared inputs; deployment supplies the policy endpoint.
export INPUT_JSONL="$RUN_DIR/inputs/arena_proxy_v3_benchmark.jsonl"
export CONTEXT=131072 TOP_K=64
.venv/bin/gym eval run \
  --config benchmarks/indic/arena_proxy_v3/config.yaml \
  --config benchmarks/indic/arena_proxy_v3/submit/dataset.yaml \
  --config benchmarks/indic/arena_proxy_v3/generation.yaml \
  --agent indic_arena_proxy_v3_agent --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --model-api-key EMPTY \
  --split benchmark --output "$RUN_DIR/generations/rollouts.jsonl" \
  --num-repeats 1 --concurrency 5 --temperature 1.0 --top-p 0.95 \
  --max-output-tokens 8192 +skip_verification=true +require_complete=true \
  ++policy_model.responses_api_models.vllm_model.extra_body.presence_penalty=0.0 \
  ++policy_model.responses_api_models.vllm_model.extra_body.frequency_penalty=0.0 \
  ++head_server.host=127.0.0.1 ++head_server.port=29871

# The judge credential is provided through the deployment's protected environment.
.venv/bin/gym eval reverify --config benchmarks/indic/arena_proxy_v3/judge.yaml \
  --inputs "$RUN_DIR/generations/rollouts_materialized_inputs.jsonl" \
  --rollouts "$RUN_DIR/generations/rollouts.jsonl" \
  --output "$RUN_DIR/judgments/rollouts.jsonl" --concurrency 5 \
  ++head_server.host=127.0.0.1 ++head_server.port=29872
.venv/bin/python -m benchmarks.indic.arena_proxy_v3.summarize \
  "$RUN_DIR/judgments/rollouts.jsonl" --inputs "$INPUT_JSONL" \
  --output "$RUN_DIR/metrics.json"
```

This run validates the application flow with real policy and judge calls. It
uses a reduced output cap and does not establish agreement with a separate full
reference run. It also does not validate deployment-specific `gym eval submit`
containers or mounts; the model service was submitted separately and the Gym
CLI driver ran on the CPU host. The full Hindi evaluation recorded above supersedes
this smoke's population and output-cap limits; the 12-language comparison remains
outstanding.
