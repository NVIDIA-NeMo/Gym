# Indic Arena Proxy v3

Evaluates the next assistant reply using the complete conversation history and
pairwise comparisons against frozen GPT-5.4-mini answers in NeMo Gym.

## Dataset

Dataset: **`lmarena_v3_indic`, version `0.0.1`**, stored in Gym's internal GitLab
dataset registry.

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
settings using your deployment's standard `env.yaml`. Keep
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
Dataset licensing: NVIDIA internal, derived from arena-proxy-v3.

## Generation and judging

Run against a policy endpoint supplied by your deployment or EFB leaf. The
external deployment owns checkpoint serving, containers, cluster resources,
parallelism, parsers, and generation-to-judging scheduling. The benchmark uses
`gym eval run` for generation with verification deferred, then
`gym eval reverify` to judge the saved answers without policy inference.

Compose `generation.yaml` with a model profile that explicitly sets
`policy_model.responses_api_models.vllm_model.context_length` to the served
engine's actual window. There is no context-window fallback. For example, an
external Gemma 31B profile can contain:

```yaml
policy_model:
  responses_api_models:
    vllm_model:
      context_length: 131072
      extra_body:
        top_k: 64
```

Set `POLICY_CONFIG` to that profile, `POLICY_MODEL` and `POLICY_BASE_URL` to the
served model and endpoint, and `RUN_DIR` to the output directory. Configure
`policy_api_key` through the deployment's protected config. Set `INPUT_JSONL` to
the prepared `arena_proxy_v3_benchmark.jsonl`; pass the same `input_jsonl` config
value to both stages. These shell variables name explicit command arguments;
the benchmark YAML does not read them from the environment.

```bash
gym eval run \
  --config benchmarks/indic/arena_proxy_v3/config.yaml \
  --config benchmarks/indic/arena_proxy_v3/generation.yaml \
  --config "$POLICY_CONFIG" \
  --agent indic_arena_proxy_v3_agent --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --split benchmark \
  --output "$RUN_DIR/generations/rollouts.jsonl" --resume \
  --num-repeats 1 --concurrency 256 \
  +skip_verification=true +require_complete=true \
  "++input_jsonl=$INPUT_JSONL"
```

Temperature, top-p, and output-cap defaults are in `responses_create_params`.
Other generation settings come from the model-server config and composed profile:

| Setting | Value |
| --- | --- |
| Temperature | 1.0 |
| Top-p / top-k | 0.95 / −1; the model profile can override top-k |
| Seed | 42 |
| Context window | Required from the model profile; must match the served engine |
| Maximum output | 120,000 tokens |
| Thinking | Enabled |

Use `--temperature 0.6` for the 0.6 comparison, or supply the equivalent
`++responses_create_params.temperature=0.6` config override. Other sampling and
output settings can also be supplied through Gym's config or CLI flags.

Generation preserves model-profile system messages, stop settings, penalties,
reasoning options, and chat-template kwargs. Template settings follow this
precedence: config defaults, then direct request kwargs, then
`metadata.chat_template_kwargs`. Set template kwargs through these fields;
nesting them inside `extra_body` is rejected. A system prompt must also be used
in the reference run to compare that profile's scores.

The prompt token count uses the same settings and messages as generation. The
output budget is `min(requested_output_tokens, context_length - prompt_tokens - 64)`.
The requested cap defaults to 120,000; `--max-output-tokens` can lower it. A cap
above 120,000, conflicting budget aliases, disabled thinking, prompt truncation,
assistant-prefix continuation, tools, and multiple completions are rejected
explicitly. Incompatible model-server modes also fail configuration validation.
Keep `refit_output_budget_to_window` and `refit_output_budget_to_prompt` disabled
to use this measured budget and 64-token reserve.

The deployment should verify that the served model separates reasoning from the
final answer. Parser and parallelism settings must match the evaluated model.
Use identical effective model-profile settings in runs whose scores you compare.

After generation succeeds, run judging on a CPU host. Supply `judge_api_key`
through an ignored, protected `env.yaml` or deployment config. The config keys
`judge_base_url` and `judge_model_name` select the endpoint and model; they default
to the reference NVIDIA endpoint and Gemini 3.8 Flash. Keep credential values out
of tracked files and command-line arguments. The CPU configuration contains only
the judge model server and resources server.

```bash
gym eval reverify --config benchmarks/indic/arena_proxy_v3/judge.yaml \
  --inputs "$RUN_DIR/generations/rollouts_materialized_inputs.jsonl" \
  --rollouts "$RUN_DIR/generations/rollouts.jsonl" \
  --output "$RUN_DIR/judgments/rollouts.jsonl" --resume --concurrency 12 \
  "++input_jsonl=$INPUT_JSONL"
```

Outputs include:

```text
RUN_DIR/
  generations/
    rollouts.jsonl
    rollouts_materialized_inputs.jsonl
  judgments/
    rollouts.jsonl
    rollouts_aggregate_metrics.json
    metrics.json
```

For an interrupted stage, repeat its command with `--resume` and unchanged model,
input, sampling, and judge settings. A judge retry reuses the saved policy answers.
Use a fresh run directory when the checkpoint, prepared population, or evaluation
settings change.

## Scoring

The default judge is Gemini 3.8 Flash with temperature 0, reasoning effort low,
and a 32,768-token output limit. Each task receives two judgments with candidate
and reference positions swapped. Creative-writing tasks use their own prompt.
Only the final answer after thinking removal is judged. Literal `<think>` /
`</think>` markers are recognized in candidate, reference, and judge replies;
other reasoning delimiters are retained.

Scoring uses Arena-Elo with strong-verdict weight 3, an MLE fit, and 100 bootstrap
rounds with seed 42. Each language has overall, single-turn, multi-turn, and seven
group scores. **The headline is the arithmetic mean of the 12 Indic overall
scores. English is reported separately.** Mean per-row reward is not the headline.

The official headline is **`arena_elo/indic/score`**
in Gym's agent, key, and repeat metrics, on a 0–100 scale (higher is better).
Use this metric for comparisons and downstream reports. Language breakdowns use
`arena_elo/language/<lang>/<slice>/score`, where the slice is `all`, `single_turn`,
`multi_turn`, or a group name. Each slice also includes `ci_lower`, `ci_upper`,
`invalid_scores`, and `n`. English uses the `en` language prefix.

`gym eval run` and `gym eval reverify` export these metrics through Gym's normal
aggregate-metrics and tracking paths. Comparison suites should select
`arena_elo/indic/score` with bounds 0 and 100. The summary command below writes the
headline as `metrics.json`'s `_avg.all`.

Coverage is checked against the prepared inputs and their adjacent
`preparation_report.json`. The `input_jsonl` config key selects the same prepared
population for generation and judging. When using a custom preparation directory,
pass its `arena_proxy_v3_benchmark.jsonl` as shown above; otherwise the benchmark's
default prepared data is used. This protocol permits one rollout per task and one
repeat per aggregate; evaluate repeats as separate runs.

Run scoring commands in an environment with the Arena judge dependencies installed:

```bash
python -m benchmarks.indic.arena_proxy_v3.summarize "$RUN_DIR/judgments/rollouts.jsonl" \
  --inputs "$INPUT_JSONL" \
  --output "$RUN_DIR/judgments/metrics.json"
```

Both server metrics and the summary reject duplicate or mismatched tasks.
Missing, failed, or masked judgments suppress incomplete language scores and the
Indic headline. Partial datasets do not produce an official headline. Unavailable
scores are omitted from Gym metrics and formatted as JSON null in the summary;
`arena_elo/coverage/headline_available` records availability. Successfully
recovered judgments count when they refer to the same saved answer as the failed
attempt. The headline requires all 4,799 eligible Indic tasks, with 399 Tamil tasks
and 400 in every other Indic language.

## Comparing setups

Use the same checkpoint, vLLM version and image, parallelism, context, sampling,
parsers, dataset version and checksums, baseline, and judge configuration in both runs.
Separate runs can still differ because of stochastic generation or hosted judging.

Compare saved reference judgments with Gym judgments using:

```bash
python -m benchmarks.indic.arena_proxy_v3.summarize "$RUN_DIR/judgments/rollouts.jsonl" \
  --inputs "$INPUT_JSONL" \
  --reference-verdicts "$REFERENCE_INDIC_DIR"/*.jsonl \
  --output "$RUN_DIR/comparison.json"
```

Supply reference judgment files for exactly the selected languages. Include the
English reference judgments only when you explicitly prepared English inputs. The comparison
validates task coverage and metadata, records artifact hashes, and reports verdict
differences and the Indic average difference. Both sets are scored in the same
prepared-input order, so asynchronous completion order cannot change their
bootstrap comparison.

Compare completed runs with valid provider responses and no aborted reference
generations; unresolved failures do not produce an official headline.

An explicitly prepared chat-400 English slice uses the Indic Arena Proxy v3
comparison protocol. Report its results separately from the English LMArena v3
benchmark.
