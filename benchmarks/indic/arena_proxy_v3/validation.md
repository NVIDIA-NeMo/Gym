# Registry migration validation — 2026-11-24

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
A full model run compared with separately generated reference judgments is still
pending. Dataset licensing metadata is `NVIDIA internal, derived from
arena-proxy-v3`. Measured end-to-end score equivalence remains unverified.

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

The generation model, preflight, generation/judge configs, preparer, prompt
templates, source checksums, summarizer, and metric reducer are byte-identical to
`002507bb8887ec97c950cf9eca595e2770c05f01`. The judge's executable AST is unchanged;
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
