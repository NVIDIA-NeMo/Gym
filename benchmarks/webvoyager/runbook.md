# WebVoyager end-to-end runbook

This is the supported path from a clean Gym checkout to a one-task smoke run
and the maintained 552-task WebVoyager population. It applies to both policy
profiles; the browser runtime and task dataset do not change with the model.

## 1. Required inputs

| Input | Contract |
| --- | --- |
| Gym source | This checkout and its committed `uv.lock` |
| Task data | `jayl940712/webarena_benchmarks` commit `6a2977939b157b0ab9de7799bb089c721f1ac115`, root `webvoyager.jsonl` |
| Linux GUI | Xvfb 1920x1080, Chromium, PyAutoGUI, and `xclip` |
| Browser egress | US HTTP proxy in `WA_BROWSER_PROXY_SERVER` |
| CAPTCHA | Funded CapSolver key in `CAPSOLVER_API_KEY` |
| Judge | Vision-capable judge URL, key, and model |
| Policy | Nano Omni endpoint or local Qwen model assets, as selected below |

Do not run the visual browser on macOS. One process owns one X display, and
the resource server enforces `max_sessions=1`.

## 2. Install and preflight

```bash
uv lock --check
uv sync --frozen --extra dev
command -v Xvfb
command -v xvfb-run
command -v xclip
uv run --project resources_servers/visual_browser playwright install chromium
```

## 3. Prepare the pinned 552 tasks

```bash
./.venv/bin/gym eval prepare --benchmark webvoyager
```

The command downloads one immutable source file, verifies the SHA-256 recorded
in `source_lock.json`, verifies exactly 552 rows, and writes
`benchmarks/webvoyager/data/webvoyager.jsonl`. To use an offline copy, set
`WEBVOYAGER_SOURCE_JSONL`; the same hash and denominator checks still apply.

## 4. Configure proxy, CAPTCHA, and judge

Keep secrets in the process environment, never in tracked YAML:

```bash
export WA_BROWSER_PROXY_SERVER="proxy-host.example:19407"
export WA_CAPTCHA_PROVIDER="capsolver"
read -rsp "CapSolver API key: " CAPSOLVER_API_KEY
export CAPSOLVER_API_KEY
printf '\n'

export WEBARENA_JUDGE_BASE_URL="https://inference-api.nvidia.com/v1"
export WEBARENA_JUDGE_MODEL="gcp/google/gemini-3-flash-preview"
read -rsp "Judge API key: " WEBARENA_JUDGE_API_KEY
export WEBARENA_JUDGE_API_KEY
printf '\n'
```

Preflight the proxy exit and solver account:

```bash
curl -fsS -x "http://$WA_BROWSER_PROXY_SERVER" https://ifconfig.me/ip
mkdir -p results/webvoyager/preflight
./.venv/bin/python benchmarks/webvoyager/smoke_capsolver_account.py \
  --output results/webvoyager/preflight/capsolver-account.json
```

When workers reach Squid through node-local tunnels, set
`WA_CAPTCHA_PROXY_SERVER` to the public endpoint of that same proxy. CapSolver
cannot use `127.0.0.1` on a worker.

## 5. Select a policy recipe

### Qwen3.5-122B-A10B-FP8

The checked-in profile launches `Qwen/Qwen3.5-122B-A10B-FP8` with TP8,
262144-token context, Qwen3 reasoning parser, `qwen3_coder` tool parser, and
thinking enabled. Generate its composition:

```bash
./.venv/bin/python benchmarks/webvoyager/prepare.py \
  --profile qwen35_122b_a10b \
  --rollout-output "$PWD/results/webvoyager/qwen-full/rollouts.jsonl" \
  --force-env
```

The model is large; keep Hugging Face and vLLM caches on shared Lustre and use
the site-qualified image/venv recipe on the execution cluster.

### Nano Omni

Point the generic vLLM proxy at a separately managed multimodal endpoint:

```bash
export POLICY_BASE_URL="https://policy-host.example/v1"
export POLICY_MODEL_NAME="served-model-name"
read -rsp "Policy API key: " POLICY_API_KEY
export POLICY_API_KEY
printf '\n'

./.venv/bin/python benchmarks/webvoyager/prepare.py \
  --profile nano_omni \
  --rollout-output "$PWD/results/webvoyager/nano-full/rollouts.jsonl" \
  --force-env
```

The endpoint must implement the pinned Nano Omni parser/template contract in
`nano_omni_recipe_lock.json`.

## 6. Prefetch and start servers

```bash
cd benchmarks/webvoyager
../../.venv/bin/gym env prefetch
xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24" \
  ../../.venv/bin/gym env start
```

Keep this terminal open. Component logs must expose browser lease acquisition,
browser reset/action, model requests, CAPTCHA events, judge completion, and
lease release without logging secret values.

## 7. Smoke from a second terminal

```bash
cd benchmarks/webvoyager
../../.venv/bin/gym eval run --no-serve -v \
  --limit 1 \
  --concurrency 1 \
  --output ../../results/webvoyager/smoke/rollouts.jsonl
```

The smoke gate requires one rollout with a resolved reward, at least one
executable `computer_use` action, no masked infrastructure failure, and a
browser release event. A low reward is not an infrastructure failure.

## 8. Run all 552 tasks

The portable form is sequential:

```bash
../../.venv/bin/gym eval run --no-serve -v \
  --concurrency 1 \
  --output ../../results/webvoyager/full/rollouts.jsonl
```

For practical throughput, split the prepared JSONL and launch isolated Gym
processes. Each worker requires a unique DISPLAY, HOME, temporary directory,
artifact directory, component ports, and rollout output. Model-serving and
read-only caches may be shared. Do not set `--concurrency` above one against a
single PyAutoGUI resource server.

For Qwen reference parity, use four task splits, one TP8 model server per
split, and 16 isolated browser workers per split. This is process-level
parallelism; the model server can batch requests internally.

## 9. Reconcile the fixed denominator

```bash
mkdir -p results/webvoyager/full/aggregate results/webvoyager/full/cleanup
./.venv/bin/python benchmarks/webvoyager/summarize.py \
  results/webvoyager/full/workers \
  --dataset benchmarks/webvoyager/data/webvoyager.jsonl \
  --output results/webvoyager/full/aggregate/summary.json \
  --missing-output results/webvoyager/full/cleanup/retry.jsonl

jq '{expected, completed_unique, success, strict_sr, missing, duplicate_task_ids, unexpected_task_ids, invalid_or_infrastructure, comparable}' \
  results/webvoyager/full/aggregate/summary.json
```

Only explicitly superseded task IDs may be replaced by a cleanup wave. Keep
the first-wave and cleanup outputs immutable and pass both to the summarizer.

## 10. Training safety contract

Browser/provider timeouts, process loss, proxy/CAPTCHA failures, model-server
failures, and judge failures produce masked retryable samples, not reward-zero
policy examples. Browser leases are acquired and released asynchronously;
synchronous Playwright work stays on its session-affine thread. Providers must
enforce an external lease TTL because no in-process finally block can run after
SIGKILL or node loss. See [runtime architecture](runtime-architecture.md).
