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

Start with the [end-to-end runbook](runbook.md). Model-specific details are in
[Nano Omni](nano-omni.md) and [Qwen3.5-122B-A10B](qwen35.md). Browser supply,
thread/process isolation, AgentEnv integration, and training cleanup behavior
are described in [runtime architecture](runtime-architecture.md).

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

One visual-browser resource process owns one X display and permits one active
session. Scale by launching isolated processes or containers with distinct
DISPLAY, HOME, temporary, artifact, and output paths; do not add threads that
share a display.

## Fixed-denominator reporting

```bash
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
