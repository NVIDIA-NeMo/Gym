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

The profile YAML and recipe-lock JSON files are the machine-readable authority
for model-specific serving behavior. For complete prerequisites, setup, smoke,
full-population execution, and reconciliation, use the
[Fern WebVoyager tutorial](../../fern/versions/latest/pages/evaluation-tutorials/webvoyager.mdx).

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

From the repository root, install the locked environment and prepare the
hash-pinned dataset:

```bash
uv lock --check
uv sync --frozen --extra dev
./.venv/bin/gym eval prepare --benchmark webvoyager
```

Generate a private, mode-0600 composition for one policy profile:

```bash
./.venv/bin/python benchmarks/webvoyager/prepare.py \
  --profile qwen35_122b_a10b \
  --rollout-output "$PWD/results/webvoyager/qwen/rollouts.jsonl" \
  --force-env
```

Run the component servers in the foreground:

```bash
cd benchmarks/webvoyager
xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24" \
  ../../.venv/bin/gym env start
```

From a second terminal in the same directory:

```bash
../../.venv/bin/gym eval run --no-serve --concurrency 1
```

Stop `gym env start` with Ctrl-C. Gym currently has no separate `env stop`
command and does not own an external proxy, judge gateway, or externally
managed model server.

## Fixed-denominator reporting

```bash
cd ../..
./.venv/bin/python benchmarks/webvoyager/summarize.py \
  results/webvoyager/qwen/rollouts.jsonl \
  --dataset benchmarks/webvoyager/data/webvoyager.jsonl \
  --output results/webvoyager/qwen/summary.json \
  --missing-output results/webvoyager/qwen/retry.jsonl
```

A reportable full result has all 552 task IDs exactly once and no unresolved
missing, duplicate, unexpected, masked, or infrastructure rows. Browser,
provider, proxy/CAPTCHA, model-server, and judge failures are masked and routed
to retry input rather than silently counted as policy reward zero.
