# WebVoyager

For a copyable path from a clean Gym checkout through a one-task smoke run and
the maintained native 552-task population, start with the
[native WebVoyager end-to-end runbook](native-v3-runbook.md). It identifies
the exact dataset and public model-serving assets, required external services,
commands, logs, and denominator checks.
The published [WebVoyager evaluation tutorial](https://docs.nvidia.com/nemo/gym/main/tutorials/evaluation-tutorials/webvoyager)
covers both supported profiles.

The first Gym profile runs the 643 official WebVoyager tasks through
`browsergym/openended`. It preserves the upstream action surface (`Click`,
`Type`, `Scroll`, `Wait`, `GoBack`, `Google`, `ANSWER`) with a safe translation
to BrowserGym high-level calls. The final answer and latest screenshots are
scored by the separate WebVoyager VLM judge.

The maintained 552-task Nano Omni screenshot/tool-call route is a separate
profile documented in [native-v3.md](native-v3.md). It uses headed Chromium,
coordinate actions, the native Gemini judge contract, a US proxy, and
CapSolver. Do not combine its 552-task score or runtime requirements with the
BrowserGym-compatible 643-task profile. The internal CLI profile name
`legacy` refers to this BrowserGym-compatible path; it does not mean that the
original Selenium runner is included.

For evaluation and RL collection, judging is per episode: the agent releases
the browser after retaining the final evidence, then obtains a binary reward
before returning that rollout. There is no 643-task evaluation barrier in the
Gym path. Rollout concurrency supplies parallelism across episodes. Judging
uses the standard Gym resources-server `/verify` endpoint; judge-call failures
are routed to the failure sidecar and can be retried with `gym eval reverify`
from retained evidence without repeating live-site actions.

Create the locked root CLI environment from the repository root. `uv lock
--check` verifies that the committed lock is current; `--frozen` prevents a
run from silently resolving a different dependency graph:

```bash
uv lock --check
uv sync --frozen --extra dev
./.venv/bin/gym eval prepare --benchmark webvoyager
```

The last command is Gym's standard dataset-only preparation API. The default
path downloads the official 643-task source from the pinned
`MinorJerry/WebVoyager` revision into the gitignored
`benchmarks/webvoyager/data/WebVoyager_data.jsonl`, verifies its SHA-256, and
then writes `webvoyager_benchmark.jsonl`. No sibling checkout is required. To
reuse an existing verified source cache instead, set
`WEBVOYAGER_SOURCE_JSONL` or pass `--source` when invoking `prepare.py`
directly.

The native profile is self-contained at the dataset layer as well. Running
`prepare.py --profile native_v3` downloads only the pinned root
`webvoyager.jsonl` file from `jayl940712/webarena_benchmarks`, verifies its
SHA-256 and 552-task denominator, and caches it under
`benchmarks/webvoyager/data`. Use `--source` or
`WEBVOYAGER_SOURCE_JSONL` only to select an existing offline copy.

For an OSWorld-style runnable composition, invoke the script directly. It
validates/prepares the selected profile and writes a private mode-`0600`,
gitignored `benchmarks/webvoyager/env.yaml`. Credentials remain process
environment variables referenced by that file:

```bash
export POLICY_BASE_URL="https://policy.example/v1"
export POLICY_API_KEY="..."
export POLICY_MODEL_NAME="policy-model"
export WEBARENA_JUDGE_BASE_URL="https://judge.example/v1"
export WEBARENA_JUDGE_API_KEY="..."
export WEBARENA_JUDGE_MODEL="judge-model"

./.venv/bin/python benchmarks/webvoyager/prepare.py --profile legacy --force-env
```

Run the complete lifecycle in one process; `gym eval run` starts the configured
servers and shuts them down when collection ends:

```bash
cd benchmarks/webvoyager
../../.venv/bin/gym eval run
```

For repeated runs, prefetch once and keep the servers foregrounded:

```bash
cd benchmarks/webvoyager
../../.venv/bin/gym env prefetch
../../.venv/bin/gym env start
```

In a second terminal, from the same directory, run
`../../.venv/bin/gym eval run --no-serve`. Stop the foreground `gym env start`
with Ctrl-C after the last run; Gym currently has no separate `env stop`
command. Externally managed vLLM, public-site proxy, and site services are not
owned or stopped by Gym.

The benchmark targets live public sites, so results are time-sensitive and
less reproducible than the self-hosted Arena benchmarks. Configure a
vision-capable judge through `webvoyager_judge_*` values in `env.yaml`, or
replace the default judge model config with another server named
`webvoyager_judge_model`.

The benchmark profile sends one current SoM screenshot plus a compact list of
labelled interactive elements to the policy, and replaces older visual
observations with an omission marker. The judge independently retains the
latest three screenshots. This matches upstream WebVoyager's context shape
without changing WebArena or VisualWebArena defaults.

For the committed ArXiv smoke row, load configs in this order. The private file
contains `policy_base_url`, `policy_api_key`, and `policy_model_name`; never
commit it. Using the policy as judge is only appropriate for integration smoke.

```bash
GYM_ROOT=/path/to/Gym
PRIVATE_CONFIG=/path/to/private/inferencehub-env.yaml

"$GYM_ROOT/.venv/bin/gym" eval run \
  --config "$PRIVATE_CONFIG" \
  --config benchmarks/webvoyager/configs/inferencehub_same_model.yaml \
  --config benchmarks/webvoyager/configs/arxiv13_smoke.yaml \
  --model-type openai_model \
  --split benchmark \
  --output /path/to/run/rollouts.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1 \
  --temperature 1.0 \
  --max-output-tokens 1000
```

Do not replace the locked `.venv/bin/gym` command with an unconstrained
root-level `uv run gym`; a newly resolved Ray version can differ from component
environments and fail before task execution. Such a run is infrastructure
failure, not zero reward.

The original Selenium runtime is intentionally not included in this first
version. It can be added later behind the same common protocol without changing
the dataset or agent contract.
