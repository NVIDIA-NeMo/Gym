# VisualWebArena

VisualWebArena extends WebArena with visually grounded tasks and reference
images. This integration runs the maintained 908-task population through the
same Gym visual-browser runtime, browser-session provider, `computer_use`
action contract, and distributed agent loop used by WebVoyager and WebArena.
Only the local-site setup, task images, and VisualWebArena evaluator are
benchmark-specific.

## Prepare

From the repository root:

```bash
gym eval prepare --benchmark visualwebarena
```

The prepare command downloads the public
[`jayl940712/webarena_benchmarks`](https://github.com/jayl940712/webarena_benchmarks)
archive at commit `6a2977939b157b0ab9de7799bb089c721f1ac115`. It verifies the
source JSONL SHA-256, exactly 908 rows, and all 346 local reference images
before producing the model-neutral
`benchmarks/visualwebarena/data/visualwebarena.jsonl`. Source identity is
recorded in `source_lock.json`.

Preparation also writes a private, gitignored
`benchmarks/visualwebarena/env.yaml`. Use `--force-env` only when replacing an
existing generated file intentionally:

```bash
python benchmarks/visualwebarena/prepare.py --force-env
```

To reuse an offline or shared checkout, set the root containing both
`visualwebarena.jsonl` and the `visualwebarena/` image tree:

```bash
export VISUALWEBARENA_SOURCE_ROOT=/shared/webarena_benchmarks
gym eval prepare --benchmark visualwebarena
```

Mount that root read-only at the same absolute path on every distributed
worker. The JSONL retains relative image paths below this root.

## Run

Deploy and reset the VisualWebArena site stack, export the deployment-specific
`WA_*` URLs and approved `WEBARENA_JUDGE_*` credentials, then run:

```bash
gym env prefetch --config benchmarks/visualwebarena/env.yaml
gym env start --config benchmarks/visualwebarena/env.yaml
```

In another terminal:

```bash
gym eval run --config benchmarks/visualwebarena/env.yaml --no-serve
```

Stop the foreground `gym env start` process with Ctrl-C after evaluation.
Each headed Chromium process owns one display and therefore permits one active
session. Scale out by launching isolated resource-server and agent processes;
do not share a display or mutable site deployment between concurrent shards.

## Runtime and evaluator boundary

The shared runtime supplies screenshots, headed Chromium, PyAutoGUI-backed
actions, lifecycle handling, artifacts, and browser-session acquisition. The
`webarena_browser` resource adds site login and URL substitution. This stacked
PR adds VisualWebArena's input-image materialization and pinned evaluator,
including fuzzy-image dependencies and before/after collision snapshots.

Prepared rows do not contain model prompts or tool schemas. The default config
selects the Nano Omni policy adapter; another compatible policy can replace
that adapter while retaining the same task population and environment. Model
and runtime results are comparable only when the source lock, site snapshot,
viewport, action profile, evaluator, and judge configuration also match.
