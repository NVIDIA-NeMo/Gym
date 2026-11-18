# Indic Arena Proxy v3

This benchmark integrates the supplied chat400 evaluation with NeMo Gym's existing
`simple_agent`, `single_agent_turn_legacy` environment server, model servers, and
`arena_judge` resources server. Each task generates one next assistant reply from
the **complete conversation history**, including earlier assistant turns.

The dataset is [anushakamathofficial/indic-arena-proxy-v3](https://huggingface.co/datasets/anushakamathofficial/indic-arena-proxy-v3),
pinned to `da63b1db1770bc219962b196f4f93b066fa2e400`. Its 26 source JSONL files
contain 5,200 prompts and 5,200 GPT-5.4-mini references. The one Tamil translation
with status `hard_error` is recorded and excluded from evaluation: 5,199 eligible
tasks, of which 4,799 are Indic and 400 are English. Source files and frozen
references remain unchanged on the Hub.

## Prepare

Use Python 3.13.14+ and uv 0.13.4+, as required by this checkout. Run from the
repository root with the repository's Python environment installed
(`uv sync --frozen --extra dev`, then activate `.venv`). Prepare the isolated
component environments on a login node before requesting compute:

```bash
gym env prefetch --config benchmarks/indic/arena_proxy_v3/config.yaml \
  --model-type openai_model \
  +policy_base_url=http://127.0.0.1:8000/v1 \
  +policy_api_key=EMPTY +policy_model_name=prewarm
```

Prefetch installs dependencies without starting servers or models. The Arena
resources server environment includes the summary's `numpy`, `pandas`, and
`scikit-learn` dependencies. The batch workflow uses that environment for scoring;
set `SUMMARY_PYTHON_BIN` if your server environments live outside the checkout.

```bash
python benchmarks/indic/arena_proxy_v3/prepare.py
```

Or use the original files without a Hub download:

```bash
export SOURCE_DIR=/lustre/fsw/portfolios/llmservice/users/kanishks/nemotron_4_post_ablations/data/eval/lmarena_v3_chat400
python benchmarks/indic/arena_proxy_v3/prepare.py --source-dir "$SOURCE_DIR"
gym env validate indic/arena_proxy_v3
```

Preparation verifies the pinned source hashes, unique IDs, reference model/status,
conversation equality, message digests, language, groups, and turn counts. It
joins references by ID, preserves all input messages, and writes combined and
per-language Gym JSONL files plus `preparation_report.json`. Original upstream
LMArena answer fields are not used as this benchmark's reference. Use
`--languages en hi ta --limit 3` for a small smoke dataset; the full source files
are validated before limiting. Prepared data and upload staging are gitignored.

## GPU generation, then CPU judging

The submission helper prepares a unique run directory, launches one eight-GPU
generation job, and launches a CPU judge job with `afterok` dependency. It uses
`gym eval run +skip_verification=true` to save policy replies and materialized
inputs, then `gym eval reverify` to judge those exact replies without policy
inference. The CPU configuration contains only the judge model server and
resources server.

```bash
export GYM_REPO="$PWD"
export MODEL_KEY=my-model
export MODEL_PATH=/absolute/path/to/hf-checkpoint
export SLURM_ACCOUNT=your-slurm-account
export JUDGE_KEY_FILE=/absolute/path/to/your/nvidia-inference.key
export LANGUAGES="en bn gu hi kn ml mr ne or pa ta te ur"
export IMAGE=/lustre/fsw/portfolios/llmservice/users/kanishks/images/vllm-openai-v0.26.0.sqsh
export TEMPERATURE=1.0 TOP_P=0.95
bash benchmarks/indic/arena_proxy_v3/slurm/submit.sh
```

Keep `IMAGE` pointing to a readable local image. Default partitions are
`interactive` and `cpu_interactive`; override `GPU_PARTITION` and `CPU_PARTITION`
as needed. Account and logs are supplied at submission time, with no colleague's
account or output directory embedded in the batch headers. `LANGUAGES` must use
spaces because Slurm splits `--export` on commas.

Set `TEMPERATURE=0.6` for the corresponding sampling experiment. The source
generation defaults are 120,000 maximum output tokens, a 163,840-token context,
thinking enabled, and seed 42. vLLM defaults target Nemotron models:
`PARSER=nemotron_v3`, `TOOL_PARSER=qwen3_coder`, `TP=8`, `DP=1`, `EP=1`.
For other architectures, supply the appropriate parsers and parallelism;
`PARSER=none TOOL_PARSER=none EP=0` disables those model-specific options.
If you lower `CONTEXT` below 163,840, you must explicitly set
`MAX_OUTPUT_TOKENS` to leave room for the **longest tokenized input** plus the
chat template. The original standalone generator dynamically reduced output
budgets for short contexts; this Gym preset uses a fixed configured budget.
`LIMIT=3 LANGUAGES="en hi ta"` makes a small smoke submission.

The helper prints `RUN_DIR`, the generation job ID, and the dependent judge job
ID. It writes:

```text
RUN_DIR/
  inputs/arena_proxy_v3_benchmark.jsonl
  inputs/preparation_report.json
  generations/rollouts.jsonl
  generations/rollouts_materialized_inputs.jsonl
  judgments/rollouts.jsonl
  metrics.json
  logs/
```

`--resume` uses Gym's saved rollout identity. For a retry, reuse `GYM_REPO`,
`RUN_DIR`, and the original model/data/sampling/judge settings and submit the
appropriate batch script directly. Do not rerun `submit.sh` into existing
prepared inputs: it deliberately requires a fresh run directory. A judge retry
uses the saved generations; it does not request new model answers.

## Authentication

Your existing HF login is sufficient. `hf auth whoami` checks it without printing
the token. For a new machine or token rotation, use the hidden interactive prompt:

```bash
umask 077
hf auth login
hf auth whoami
```

Do not supply a token in command arguments, YAML, source files, chat, or Slurm
exports. Decline Git credential storage unless you specifically need it. Keep
the Hub token and stored-token files readable only by you (`0600`); by default
they are under `~/.cache/huggingface/`. See the official
[HF authentication guide](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-auth-login)
and [credential-path settings](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hf_token_path).

The NVIDIA judge credential is separate from the HF token. Create its file from
a terminal without putting the value in shell history:

```bash
set +x
umask 077
mkdir -p "$HOME/.secrets"
read -r -s -p 'NVIDIA inference key: ' arena_judge_secret
printf '\n'
printf '%s\n' "$arena_judge_secret" > "$HOME/.secrets/arena-judge.key"
unset arena_judge_secret
chmod 600 "$HOME/.secrets/arena-judge.key"
export JUDGE_KEY_FILE="$HOME/.secrets/arena-judge.key"
```

The batch scripts check file ownership and permissions, disable shell tracing,
and read the judge key only in the CPU job. They remove known HF and judge token
variables before `sbatch --export=ALL`; the exported key **path** is safe. Gym
masks API-key fields in persisted configuration.

## Judging and scores

Defaults match the supplied protocol: NVIDIA inference endpoint
`https://inference-api.nvidia.com/v1`, model `gcp/google/gemini-3.8-flash`,
temperature 0, reasoning effort low, and 32,768 judge output tokens. Override
`ARENA_JUDGE_BASE_URL` and `ARENA_JUDGE_MODEL` if necessary and record the change.
Endpoint/model availability must be checked by a real smoke run.

Every task receives two judgments with positions swapped. The creative template
is used only for `Creative writing / role-play`. Conversation-aware templates
are composed from Gym's existing Arena templates. `strip_thinking: true` takes
text after the last `</think>` and treats unclosed thinking as an empty answer.
This option defaults to false in the shared server, preserving existing users.

Scoring reuses Gym's Arena-Elo implementation: strong verdict weight 3, MLE,
100 bootstrap rounds, seed 42. Recompute the report with:

```bash
resources_servers/arena_judge/.venv/bin/python -m benchmarks.indic.arena_proxy_v3.summarize "$RUN_DIR/judgments/rollouts.jsonl" \
  --inputs "$RUN_DIR/inputs/arena_proxy_v3_benchmark.jsonl" \
  --output "$RUN_DIR/metrics.json"
```

Each language has `all`, `single_turn`, `multi_turn`, and seven group scores.
The Indic headline is the arithmetic mean of the 12 Indic `all.score` values;
English is separate. Do not use mean per-row reward as the benchmark headline.
The summary validates the exact expected task population and rejects duplicate
or mismatched rows. Missing or failed tasks suppress incomplete language scores
and the Indic headline. Its preparation-report hash and coverage checks also
suppress the full Indic headline for smoke or unverified input populations.

Grading is observed by the existing Arena resources server through
`observe_llm_judge_grading` and `call_judge`. Judge failures remain infrastructure
failures and can be reverified from saved policy responses. The verifier is
declared stateless. There is no new per-episode wall-clock deadline; token limits,
Gym model-request timeouts, vLLM startup timeout, and the four-hour Slurm limits
remain separate bounds. Preserve resolved configuration and data hashes when
comparing runs. This local batch workflow does not require MLflow delivery.

## Reproduce the dataset upload

The upload script stages an explicit allowlist of the 26 source JSONL files,
adds a dataset card and checksums, and uploads only with `--push`:

```bash
python benchmarks/indic/arena_proxy_v3/publish_dataset.py --source-dir "$SOURCE_DIR"
python benchmarks/indic/arena_proxy_v3/publish_dataset.py --source-dir "$SOURCE_DIR" --push
```

It excludes `.partial` files, logs, local credentials, candidate generations,
and code. Newly created repositories default to private; an existing repository
keeps its visibility. This repository already existed as public when populated.
If republishing changes the source data, update the adapter's pinned revision
and checksums together. Dataset licensing is inherited and currently unspecified
by the upstream manifest; no software license is applied to the data.

## Validation

```bash
pytest tests/unit_tests/benchmarks/test_indic_arena_proxy_v3*.py \
  resources_servers/arena_judge/tests/test_app.py
```

The integration remains experimental until a representative model smoke and
score comparison have been completed. See `validation.md` for the checks and
external validation limits recorded during integration.
