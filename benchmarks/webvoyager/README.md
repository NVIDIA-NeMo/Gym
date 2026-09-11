# WebVoyager

NeMo Gym runs the maintained 552-task WebVoyager population through one
browser runtime: `visual_browser`. It uses headed Chromium under Xvfb,
Playwright for browser lifecycle and navigation, and PyAutoGUI for visible
coordinate input. The removed `browsergym_web` path and its 643-task dataset
are not part of this benchmark profile.

Model behavior is selected independently from browser execution:

| Policy profile | Model protocol | Browser runtime | Tasks |
| --- | --- | --- | ---: |
| `nano_omni` | Responses tool calls normalized to `computer_use` | `visual_browser` | 552 |
| `qwen35_122b_a10b` | Qwen XML `computer_use` calls | `visual_browser` | 552 |

Both profiles therefore see the same task rows, screenshots, proxy/CAPTCHA
behavior, action executor, and Gemini judge. Their scores can be compared as
different policies on the same benchmark runtime when their external-service
preconditions and serving recipes are also held fixed.

## Full-population validation

Full-population evidence covers both policy profiles against the same
hash-pinned 552-task population and `visual_browser` runtime. The current Nano
Omni parsing contract was repeated twice from the same frozen source and
serving recipe:

| Policy | Successful tasks | Strict SR | Completeness |
| --- | ---: | ---: | --- |
| Qwen3.5-122B-A10B-FP8 | 300/552 | 54.35% | 552 valid unique; no missing, invalid, or duplicate-valid tasks |
| Nano Omni `iter_0004622`, parser-faithful r1 | 415/552 | 75.18% | 552 valid unique; no missing, malformed, or duplicate tasks |
| Nano Omni `iter_0004622`, parser-faithful r2 | 420/552 | 76.09% | 552 valid unique; no missing, malformed, or duplicate tasks |

The Nano Omni repetitions trust the model server's configured reasoning and
tool-call parsers. Gym validates their structured calls against the declared
browser-tool contract but does not repair malformed JSON, complete missing
delimiters, infer aliases, or silently clamp action arguments. They averaged
417.5/552, or 75.63%, with a five-task run-to-run difference.

These results are rollout evidence for the runtime and fixed-denominator
reconciliation, not stable leaderboard claims. Live-site state,
proxy/CAPTCHA availability, judge behavior, and exact policy serving assets
remain part of the reproducibility contract. Direct policy comparison also
requires those inputs to be held fixed.

The [profile YAML files](configs/) and [provenance.yaml](provenance.yaml) are
the machine-readable authority for model-specific serving behavior and immutable
source identity. The complete setup, smoke, full-population execution, and
reconciliation workflow is below.

## Runtime boundary

The visual-browser service acquires browser leases asynchronously while each
live session keeps synchronous Playwright work on one session-affine thread.
Headed coordinate input remains process-isolated: one visual-browser process
owns one X display and one active PyAutoGUI session. Scale with isolated
processes or containers, not threads sharing a display.

Browser-provider, proxy/CAPTCHA, model-server, and judge failures are masked
and routed to retry rather than converted into policy reward zero. Policy
trajectories that complete and are judged unsuccessful remain valid
zero-reward samples. Providers must release leases idempotently and enforce an
external TTL as a backstop for process or node loss.

## Standard Gym flow

### Install and prepare

Use Bash on Linux. From the repository root, install the locked environment,
check the system browser prerequisites, and prepare the hash-pinned dataset:

```bash
uv lock --check
uv sync --frozen --extra dev
command -v Xvfb
command -v xvfb-run
command -v xclip
uv run --project resources_servers/visual_browser playwright install chromium
./.venv/bin/gym eval prepare --benchmark webvoyager
```

Install missing Xvfb or xclip system packages before starting the browser. The
prepare command downloads the maintained source, verifies its SHA-256 and exact
552-task denominator, and writes `benchmarks/webvoyager/data/webvoyager.jsonl`.

### Configure external services

Keep credentials in environment variables. Replace the example proxy address
with your approved proxy endpoint, and configure the judge gateway:

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

Verify that the proxy exits through the intended region and preflight the
CapSolver account before running browser tasks. Export these variables in the
terminal that will run `gym env start`.

### Select a policy

Both profiles connect to a separately managed OpenAI-compatible policy endpoint.
These benchmark configurations do not launch vLLM or allocate GPUs. Start the
selected model with a matching serving recipe, then configure its endpoint in
the terminal that will run `gym env start`:

```bash
export POLICY_BASE_URL="http://policy-host.example:8000/v1"
export POLICY_MODEL_NAME="served-model-name"
read -rsp "Policy API key: " POLICY_API_KEY
export POLICY_API_KEY
printf '\n'
```

Choose one profile and generate its composition. For Qwen3.5-122B-A10B-FP8:

```bash
./.venv/bin/python benchmarks/webvoyager/prepare.py \
  --profile qwen35_122b_a10b \
  --rollout-output "$PWD/results/webvoyager/qwen/rollouts.jsonl" \
  --force-env
```

For Nano Omni:

```bash
./.venv/bin/python benchmarks/webvoyager/prepare.py \
  --profile nano_omni \
  --rollout-output "$PWD/results/webvoyager/nano/rollouts.jsonl" \
  --force-env
```

This writes the private, mode-0600, gitignored `benchmarks/webvoyager/env.yaml`. Gym
automatically loads that file when commands run from the benchmark directory,
so the commands below do not need a separate `--config` argument. Use
`--force-env` only when intentionally replacing an existing generated file.

Tokenizer, chat template, multimodal processor, reasoning parser, and tool-call
parser are policy-server assets. They must match the selected model recipe and
are not loaded by the browser or judge. Gym validates the configured parsers'
structured output without repairing it.

### Start and smoke

Run the component servers in the foreground:

```bash
cd benchmarks/webvoyager
../../.venv/bin/gym env prefetch
xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24" \
  ../../.venv/bin/gym env start
```

From a second terminal in the same directory:

```bash
../../.venv/bin/gym eval run --no-serve -v \
  --limit 1 \
  --concurrency 1 \
  --output ../../results/webvoyager/smoke/rollouts.jsonl
```

The smoke is healthy when it resolves a reward, executes a browser action, and
releases the browser without a masked infrastructure failure. Keep the servers
running for the full evaluation below. Once evaluation is finished, stop
`gym env start` with Ctrl-C. Gym currently has no separate `env stop` command and
does not own an external proxy, judge gateway, or externally managed model server.

## Reporting and sharded runs

After a healthy smoke, run the full population from the same benchmark directory:

```bash
../../.venv/bin/gym eval run --no-serve -v \
  --concurrency 1 \
  --output ../../results/webvoyager/full/rollouts.jsonl
```

One resource process runs tasks sequentially. For throughput, launch isolated
processes or containers; every worker needs a distinct DISPLAY, HOME, temporary
directory, artifact directory, ports, and rollout file. Do not place multiple
PyAutoGUI sessions on one display.

A single `gym eval run` writes aggregate metrics and runs rollout health checks
automatically. When workers write separate rollout files, use Gym's standard
aggregation command from the same benchmark directory to merge them and compute
one global result:

```bash
../../.venv/bin/gym eval aggregate \
  --input-glob "../../results/webvoyager/shards/*/rollouts.jsonl" \
  --output ../../results/webvoyager/full/rollouts.jsonl

../../.venv/bin/gym eval health-check \
  ../../results/webvoyager/full \
  --rollouts-file rollouts.jsonl
```

`gym eval aggregate` reports scored and dropped coverage from the worker
sidecars; the health check detects malformed and duplicate rollout identities.
A reportable full result has 552 scored rows, no dropped coverage, and no
unresolved health findings. Browser, provider, proxy/CAPTCHA, model-server, and
judge failures are masked and routed to retry rather than silently counted as
policy reward zero.

Cluster images, scheduler scripts, caches, checkpoint placement, and large run
artifacts are deployment concerns and are intentionally outside this benchmark
guide.
