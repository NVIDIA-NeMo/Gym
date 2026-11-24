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
pinned registry version. Dataset licensing remains unspecified; registry access
does not establish redistribution permission.

## Generation and judging

Configure the checkpoint, container image, and cluster resources for your
deployment before submitting. The workflow runs generation on one eight-GPU
node, followed by CPU judging after generation succeeds.

```bash
export MODEL_KEY=my-model
export LANGUAGES="bn gu hi kn ml mr ne or pa ta te ur"
export TEMPERATURE=1.0 TOP_P=0.95
bash benchmarks/indic/arena_proxy_v3/slurm/submit.sh
```

Separate languages with spaces. Set `TEMPERATURE=0.6` for that sampling experiment.
`LIMIT` applies per language; the five-sample example below uses `LIMIT=1`.

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

The workflow prints the run directory and job IDs. Outputs include:

```text
RUN_DIR/
  inputs/arena_proxy_v3_benchmark.jsonl
  inputs/preparation_report.json
  generations/rollouts.jsonl
  generations/rollouts_materialized_inputs.jsonl
  judgments/rollouts.jsonl
  metrics.json
```

To resume, reuse the run directory and original model, data, sampling, and judge
settings, then resubmit the appropriate batch script. Use a fresh run directory
when those settings change. A judge retry reuses the saved policy answers.

### Five-sample Gemma smoke run

With deployment settings configured for `google/gemma-4-31B-it`, use:

```bash
export MODEL_KEY=gemma4-31b-it-smoke
export LANGUAGES="hi bn ta te ur" LIMIT=1
export CONTEXT=131072 MAX_OUTPUT_TOKENS=120000
export TEMPERATURE=1.0 TOP_P=0.95 TOP_K=64
export PARSER=gemma4 TOOL_PARSER=gemma4
export TP=8 DP=1 EP=0
export CONCURRENCY=5 JUDGE_CONCURRENCY=5
bash benchmarks/indic/arena_proxy_v3/slurm/submit.sh
```

This selects five translated versions of the first source conversation, one per
language. Use a fresh run directory. The full-run defaults remain unchanged;
the context, top-k, parsers, and expert-parallel setting above are specific to
this model configuration.

## Scoring

The default judge is Gemini 3.8 Flash with temperature 0, reasoning effort low,
and a 32,768-token output limit. Each task receives two judgments with candidate
and reference positions swapped. Creative-writing tasks use their own prompt.
Only the final answer after thinking removal is judged.

Scoring uses Arena-Elo with strong-verdict weight 3, an MLE fit, and 100 bootstrap
rounds with seed 42. Each language has overall, single-turn, multi-turn, and seven
group scores. **The headline is the arithmetic mean of the 12 Indic overall
scores. English is reported separately.** Mean per-row reward is not the headline.

The official headline is `metrics.json`'s `_avg.all` from the summary command below.
The shared resources server's `arena_elo/score` is a pooled score across tasks;
use the language average when comparing with the reference setup.

Run scoring commands in an environment with the Arena judge dependencies installed:

```bash
python -m benchmarks.indic.arena_proxy_v3.summarize "$RUN_DIR/judgments/rollouts.jsonl" \
  --inputs "$RUN_DIR/inputs/arena_proxy_v3_benchmark.jsonl" \
  --output "$RUN_DIR/metrics.json"
```

The summary rejects duplicate or mismatched tasks. Missing or failed judgments
suppress incomplete language scores and the Indic headline. Smoke subsets do not
produce an official headline. Successfully recovered judgments count when their
saved-answer identity matches the historical failure.

## Comparing setups

Use the same checkpoint, vLLM version and image, parallelism, context, sampling,
parsers, dataset version and checksums, baseline, and judge configuration in both runs.
Separate runs can still differ because of stochastic generation or hosted judging.

Compare saved reference judgments with Gym judgments using:

```bash
python -m benchmarks.indic.arena_proxy_v3.summarize "$RUN_DIR/judgments/rollouts.jsonl" \
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

This smoke validates the generation, judging, recovery, and summary workflow.
It covers one conversation across five languages and does not establish aggregate
model quality or score agreement with a separate reference run. The integration
remains experimental pending full benchmark evaluation and reference comparison.
