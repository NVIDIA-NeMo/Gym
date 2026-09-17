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
source JSONL SHA-256, exactly 908 rows, and all 341 distinct local task images
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
When using `--source` with a separately stored JSONL, also pass `--source-root`
if the images are not below the JSONL's parent directory. The generated
`env.yaml` uses that same image root for both the agent and resource server.

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
Run non-state-changing tasks before state-changing tasks, including tasks
that read across multiple sites. Reset the owned site deployment before a new
evaluation; serialization alone does not undo changes from an earlier task or
run. Phase classification and independent site replicas belong to deployment
orchestration, not to task-prompt rewriting.

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

The Nano Omni composition uses 100 steps, three recent browser screenshots,
and no additional request-level output-token cap. Context exhaustion and
terminal action errors end interaction but still invoke the evaluator on the
live state before closing the browser. Model tool arguments are not repaired.
Both task-image readers allow 32 MiB, including the maintained task image
that exceeds the previous 25 MiB default; input images are not resized to
work around that limit.

## Full-population validation

Run `2026-09-15-visualwebarena-full-908-r1` completed at 20:26 UTC on
September 16, 2026, or 04:26 Beijing time on September 17, using Gym commit
`301c577eb61457057d5cd34807073a4f4506b8fe`.

- **Model:** Nano Omni tuned checkpoint `iter_0004622`, not the public standard
  Nano Omni v3 checkpoint. The run used the reference nano-tokenizer and
  keep-history chat template.
- **Result:** 219 successful tasks out of 908, for a final SR of **24.12%**.
- **Completeness:** 908 unique tasks with valid evaluator scores; zero missing
  tasks, duplicate valid results, unresolved invalid executions, malformed
  records, or unexpected task IDs.
- **Recovery:** three infrastructure-related timeout attempts were retained
  and their tasks rerun. All three retries produced valid scores. No valid
  result was rerun or replaced; historical failed attempts remain auditable.

The run used fresh owned site replicas, executing 492 non-state-changing tasks
before 416 state-changing tasks, with per-site serialization and cross-site
barriers. It retained the original questions, input images, and evaluator
semantics, without repairing model tool arguments. The serving context was
64000 tokens, with no additional request-level output-token cap and no automatic
text-history compaction.

Final reconciliation and per-task results are archived under the run ID above;
deployment details and large artifacts are maintained separately from Gym.
This is evidence for this checkpoint and recipe, not a public-v3 baseline or
proof of score parity with another deployment.
