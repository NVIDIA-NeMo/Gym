# Indic Arena Proxy v3

Evaluates the next assistant reply using the complete conversation history and
pairwise comparisons against frozen GPT-5.4-mini answers. The integration uses
NeMo Gym's existing `simple_agent`, `single_agent_turn_legacy` environment server,
`vllm_model`, and `arena_judge` resources server.

## Dataset

Dataset registry: **`lmarena_v3_indic`, version `0.0.1`**. Preparation uses the same
Gym GitLab downloader as [`lmarena_v3`](../../lmarena_v3/prepare.py).

The 24 source JSONL files contain 4,800 prompts and 4,800 frozen reference answers
across Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia,
Punjabi, Tamil, Telugu, and Urdu. One failed Tamil translation is excluded during
preparation, leaving **4,799 eligible Indic tasks**. English is not in this registry
dataset.

Source files remain separate, both in the registry and in the local download cache:

```text
data/source/
  lmarena_v3_chat400_<lang>.jsonl
  references/
    gpt-5.4-mini/
      <lang>.jsonl
```

Preparation verifies source checksums, conversation identities, reference answers,
language metadata, and turn counts. It preserves every conversation message and
writes combined and per-language Gym inputs with a preparation report. Registry
downloads use only the 24 original files; the combined evaluation input is created
locally. Every source file is checked against `source-checksums.json`, including
cached downloads.

## Preparation

Install the repository and component dependencies using the standard NeMo Gym
setup, including the `mlflow` extra for registry access. Configure Gym's registry
settings using your deployment's standard `env.yaml`, as for `lmarena_v3`. Keep
credentials outside tracked files and command-line arguments. Run commands from
the repository root in the appropriate Python environment.

```bash
gym eval prepare --config benchmarks/indic/arena_proxy_v3/config.yaml
gym env validate indic/arena_proxy_v3
```

This produces `data/arena_proxy_v3_benchmark.jsonl` with 4,799 rows. For a small
selection, the preparation script also accepts `--languages hi bn ta --limit 3`
and `--output-dir`. The full source files are validated first; the limit selects
the source prefix before failed translations are excluded. Downloaded and prepared
datasets are excluded from Git.

For an English comparison, explicitly provide the original, checksum-matching
files with `--source-dir` and `--languages en`. English is scored separately.
The former Hugging Face download and `--revision` option have been replaced by the
pinned registry version. Dataset licensing: NVIDIA internal, derived from
arena-proxy-v3.

## Generation and judging

Use [`gym eval submit` configs](submit/README.md) for generation on one eight-GPU
node and CPU judging after generation succeeds. The submission guide covers
deployment settings, preparation, the `afterok` dependency, and retries.
Gym manages the vLLM service, health checks, batch scripts, and logs.

```bash
gym eval submit --config benchmarks/indic/arena_proxy_v3/submit/generate.yaml --dry-run
gym eval submit --config benchmarks/indic/arena_proxy_v3/submit/judge.yaml --dry-run
```

Set `TEMPERATURE=0.6` for that sampling experiment. Select languages and a
per-language `--limit` during preparation; the workers use those exact inputs.

Generation uses `gym eval run` with verification deferred. Judging uses
`gym eval reverify` on the saved responses, without running policy inference again.
The CPU configuration contains only the judge model server and resources server.

Generation defaults are:

| Setting | Value |
| --- | --- |
| Temperature | 1.0 |
| Top-p / top-k | 0.95 / −1 |
| Seed | 42 |
| Context window | 163,840 tokens |
| Maximum output | 120,000 tokens |
| Thinking | Enabled |

The benchmark extends Gym's vLLM model server through its request and conversion
hooks. It sends the original text messages to the chat-completions endpoint and
measures their token count before each request. The output budget is
`min(120000, context_length - prompt_tokens - 64)`. `MAX_OUTPUT_TOKENS` can lower
the cap. A preflight check verifies that reasoning and the final answer are
separated. Parser and parallelism settings must match the evaluated model.

The submission records provide each job ID and artifact directory. Outputs include:

```text
RUN_DIR/
  inputs/arena_proxy_v3_benchmark.jsonl
  inputs/preparation_report.json
  generation-submission.json
  judge-submission.json
  jobs/<generation-submission>/generation/artifacts/
    rollouts.jsonl
    rollouts_materialized_inputs.jsonl
  jobs/<judge-submission>/judging/artifacts/
    rollouts.jsonl
    rollouts_aggregate_metrics.json
    metrics.json
```

To resume, use `gym eval resubmit` with the recorded Gym job ID and benchmark name,
as described in the submission guide. Keep model, data, sampling, and judge settings
unchanged. A judge retry reuses saved policy answers. If generation is resubmitted,
submit a dependent judge job against the replacement Slurm ID.

### Five-sample Gemma smoke run

With deployment settings configured for `google/gemma-4-31B-it`, prepare:

```bash
python -m benchmarks.indic.arena_proxy_v3.prepare \
  --output-dir "$RUN_DIR/inputs" --languages hi bn ta te ur --limit 1
```

Then follow the submission guide using `submit/gemma31b.yaml` for generation and
`JUDGE_CONCURRENCY=5`. The profile preserves the previous smoke's context, top-k,
parsers, parallelism, and five concurrent tasks. This selects five translations
of the first conversation, one per language. Use a fresh run directory.

## Scoring

The default judge is Gemini 3.8 Flash with temperature 0, reasoning effort low,
and a 32,768-token output limit. Each task receives two judgments with candidate
and reference positions swapped. Creative-writing tasks use their own prompt.
Only the final answer after thinking removal is judged. The stripper recognizes
literal `<think>` / `</think>` markers and also processes the judge's reply;
other reasoning delimiters are unchanged. This retains the current Gemini judge
protocol.

Scoring uses Arena-Elo with strong-verdict weight 3, an MLE fit, and 100 bootstrap
rounds with seed 42. Each language has overall, single-turn, multi-turn, and seven
group scores. **The headline is the arithmetic mean of the 12 Indic overall
scores. English is reported separately.** Mean per-row reward is not the headline.

The resources server publishes the official headline as **`arena_elo/indic/score`**
in Gym's agent, key, and repeat metrics, on a 0–100 scale (higher is better).
Use this metric for comparisons and downstream reports. Language breakdowns use
`arena_elo/language/<lang>/<slice>/score`, where the slice is `all`, `single_turn`,
`multi_turn`, or a group name. Each slice also includes `ci_lower`, `ci_upper`,
`invalid_scores`, and `n`. English uses the `en` language prefix.

`gym eval run` and `gym eval reverify` export these metrics through Gym's normal
aggregate-metrics and tracking paths. Comparison suites should select
`arena_elo/indic/score` with bounds 0 and 100. The offline command below uses the
same server-owned reducer and formats the headline as `metrics.json`'s `_avg.all`.

Coverage is checked against the prepared inputs and their adjacent
`preparation_report.json`. When using a custom preparation directory, export
`INPUT_JSONL` pointing to its `arena_proxy_v3_benchmark.jsonl` before running Gym.
The submission configs set this automatically. Without this override, the judge uses
the benchmark's default prepared data. This protocol permits one rollout per
task and one repeat per aggregate; evaluate repeats as separate runs.

Run scoring commands in an environment with the Arena judge dependencies installed:

```bash
python -m benchmarks.indic.arena_proxy_v3.summarize "$JUDGE_DIR/artifacts/rollouts.jsonl" \
  --inputs "$RUN_DIR/inputs/arena_proxy_v3_benchmark.jsonl" \
  --output "$JUDGE_DIR/artifacts/metrics.json"
```

Both server metrics and the summary reject duplicate or mismatched tasks.
Missing, failed, or masked judgments suppress incomplete language scores and the
Indic headline. Smoke subsets do not produce an official headline. Unavailable
scores are omitted from Gym metrics and formatted as JSON null in the summary;
`arena_elo/coverage/headline_available` records availability. Successfully
recovered judgments count when their saved-answer identity matches the historical
failure. The headline requires all 4,799 eligible Indic tasks, with 399 Tamil tasks
and 400 in every other Indic language.

## Comparing setups

Use the same checkpoint, vLLM version and image, parallelism, context, sampling,
parsers, dataset version and checksums, baseline, and judge configuration in both runs.
Separate runs can still differ because of stochastic generation or hosted judging.

Compare saved reference judgments with Gym judgments using:

```bash
python -m benchmarks.indic.arena_proxy_v3.summarize "$JUDGE_DIR/artifacts/rollouts.jsonl" \
  --inputs "$RUN_DIR/inputs/arena_proxy_v3_benchmark.jsonl" \
  --reference-verdicts "$REFERENCE_INDIC_DIR"/*.jsonl \
  --output "$RUN_DIR/comparison.json"
```

Supply reference judgment files for exactly the selected languages. Include the
English reference judgments only when you explicitly prepared English inputs. The comparison
validates task coverage and metadata, records artifact hashes, and reports verdict
differences and the Indic average difference. Both sets are scored in the same
prepared-input order, so asynchronous completion order cannot change their
bootstrap comparison. Historical reference confidence intervals may use a different
row order. Judgment files alone do not establish matching generation settings.

Compare completed runs with valid provider responses and no aborted reference
generations. Gym retains its transport retries, cancellation handling, and response
validation; unresolved failures do not produce an official headline.

The existing English `benchmarks/lmarena_v3` integration is unchanged and uses
the separate `arena` resources server. An explicitly prepared chat-400 English
slice uses the Indic Arena Proxy v3 comparison protocol; report its results
separately from that existing benchmark.

## Validation

```bash
pytest tests/unit_tests/benchmarks/test_indic_arena_proxy_v3*.py \
  resources_servers/arena_judge/tests/test_app.py
```

On November 20, 2026, a live smoke run with `google/gemma-4-31B-it` at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` and vLLM 0.26.0 completed both stages:

- Five of five samples generated and judged across English, Hindi, Bengali,
  Tamil, and Telugu. That historical run used the original data source; the
  registry-only example above selects Urdu instead of English.
- Ten valid verdicts from the default judge, covering both answer positions for
  every sample, with zero current failures or invalid verdicts.
- Conversation inputs, frozen baselines, and saved policy responses preserved
  through judging and retry recovery.
- Complete coverage of the selected subset, with English separate and the
  official Indic headline suppressed.

This historical smoke used the former Slurm wrappers and validates generation,
judging, recovery, and summary behavior. Submission-config migration checks are
recorded separately in [validation.md](validation.md).
It covers one conversation across five languages and does not establish aggregate
model quality or score agreement with a separate reference run. The integration
remains experimental pending full benchmark evaluation and reference comparison.
