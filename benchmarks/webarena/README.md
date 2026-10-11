# WebArena

NeMo Gym runs the maintained 812-task WebArena population through the shared
`visual_browser` interaction protocol. The benchmark-specific
`webarena_browser` resource adds self-hosted-site login, mutable-site locking,
and the pinned WebArena evaluator. There is no BrowserGym runtime in this
profile.

The prepared dataset is model neutral. Nano Omni is the default policy, but a
different policy adapter can drive the same screenshots and normalized
`computer_use` actions without changing task or evaluator semantics.

## Prepare

From the repository root:

```bash
uv lock --check
uv sync --frozen --extra dev
./.venv/bin/gym eval prepare --benchmark webarena
```

Preparation downloads `webarena.jsonl` from the pinned
`jayl940712/webarena_benchmarks` revision, verifies its SHA-256 and exact
812-task denominator, then writes `benchmarks/webarena/data/webarena.jsonl`.
For an offline copy, set `WEBARENA_SOURCE_JSONL`; the same checks still apply.

## External environment

Deploy and validate the WebArena services before starting Gym, then export the
task-visible endpoints:

```bash
export WA_SHOPPING="http://host:7770"
export WA_SHOPPING_ADMIN="http://host:7780/admin"
export WA_REDDIT="http://host:9999"
export WA_GITLAB="http://host:8023"
export WA_WIKIPEDIA="http://host:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://host:3000"
export WA_HOMEPAGE="http://host:4399"
```

Some string-match tasks call a model-backed semantic judge. Provide
`WEBARENA_JUDGE_API_KEY`, and set `WEBARENA_JUDGE_BASE_URL` and
`WEBARENA_JUDGE_MODEL` when their defaults are not appropriate. Secrets stay
in the process environment and are never written to benchmark JSONL.

Reset the website deployment before every independent benchmark run. Do not
run two evaluations against the same mutable deployment: even with in-process
collision protection, external writers can change task state and invalidate
the score.

## Run

Point the Nano Omni adapter at an OpenAI-compatible policy endpoint and
generate a private, mode-0600 `env.yaml`:

```bash
export POLICY_BASE_URL="https://policy-host.example/v1"
export POLICY_MODEL_NAME="served-model-name"
read -rsp "Policy API key: " POLICY_API_KEY
export POLICY_API_KEY
printf '\n'

./.venv/bin/python benchmarks/webarena/prepare.py \
  --rollout-output "$PWD/results/webarena/full/rollouts.jsonl" \
  --force-env
```

Then start the composition:

```bash
cd benchmarks/webarena
../../.venv/bin/gym env prefetch
xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24" \
  ../../.venv/bin/gym env start
```

From a second terminal:

```bash
cd benchmarks/webarena
../../.venv/bin/gym eval run --no-serve -v --limit 1 --concurrency 1
```

After the smoke gate, omit `--limit 1` for the full 812 tasks. Stop
`gym env start` with Ctrl-C. One resource process owns one X display and one
active browser session. Scale through isolated processes or containers with
distinct displays and separately isolated WebArena deployments, not through
threads sharing a display or a mutable site stack.
