# SWE-Together

Interactive coding evaluation with a simulated user, prepared task repositories, and a fixed Claude Code judge. The Resources Server composes with the reusable interactive Environment Server and any candidate adapter supporting native continuation and ordered visible events. Candidate and judge adapters borrow sandboxes; Resources owns their creation and teardown.

Prepare the canonical tasks from [SWE-Together revision 891d19eb](https://github.com/Togetherbench/SWE-Together/tree/891d19eb4b3a64a47c3d49bbd066a311e0133254):

```bash
python -m benchmarks.swe_together.prepare \
  --source /path/to/SWE-Together \
  --images /path/to/image-pins.json \
  --output /path/to/tasks.jsonl
```

The image inventory maps each task ID to `image`, immutable `image_digest`, and inspected `workdir`. The importer validates all 109 task packages against the committed SHA-256 manifest. Configure `asset_root` as the checkout's `tasks/` directory. Hidden simulator and verification assets remain on the Resources host; only initial instructions, permitted repository guidance, and subsequent user messages reach the candidate.

Import `configs/swe_together.yaml` independently from the candidate, judge, models, and sandbox provider. The fixed judge definition is `responses_api_agents/claude_code_agent/configs/native.yaml`. It runs Claude Code 2.1.108 with 50 turns through a separate Gym model reference, using a fresh task image. Runtime archives can be prefetched to avoid installer egress; the observed version must match the pin. The judge adapter imports no benchmark code. `judge_sandbox_config` can override candidate sandbox options, including its separate model-route egress policy.

For images without Python, set `python_runtime_url` and `python_runtime_sha256` independently on Resources and each native adapter. The shared bootstrap installs a pinned standalone archive under `/tmp` and retains the prepared task tree. Resources and adapters each discover their own runtime; one component does not depend on another having installed it.

The reference protocol preserves upstream simulator persona, guidance, tools, conversation history, cursor, prompt truncation, four-no-op stop rule, and 15 resume limit. Completed activations produce one simulator consultation. Timed-out activations resume with the upstream interruption message after the candidate adapter confirms process cleanup. Pinned task images retain the upstream build-time history seal and intentional branch fixtures; startup does not rewrite the prepared history. Repository snapshots use independent indexes and preserve prepared edits, untracked files, commits, and nested repositories. Full binary and text artifacts are retained; the simulator and judge use upstream text filtering and main-repository patch normalization.

Correctness uses the frozen rubric and agentic judge. The host reproduces upstream weight summation and rounding, including its override of a judge-reported `gameable` score when goal results are present; both values are saved. `scoring_profile: reference` masks upstream-skipped empty/tiny patches. The explicit `judge_all` profile evaluates those submissions with the same judge and records the deviation. Exhausted simulator errors invalidate the episode; missing judge measurements are masked. Missing tagger output masks only User Correction. Zero simulator messages have a measured User Correction of zero.

`protocol_profile: reference` requires externally enforced default-deny egress. Qualification must establish the upstream allowlist, task-package protection, pinned model route, and bypass preflight evidence; a policy declaration alone is insufficient. Use the explicitly labeled `smoke` profile for incomplete provider qualification. `verified: false` remains until reviewed reference baselines.

Artifacts include image/source pins, effective execution identity, repository baselines and every incremental/cumulative patch, exact simulator input arrays and decisions, separate auxiliary-model calls, judge transcript and raw/derived verdict, canonical-test reward when produced, and close evidence. `metrics.aggregate` reports planned, returned, graded, masked, and missing counts, measured scores, complete-repeat metrics, and a separately labeled zero-filled upstream compatibility mean.

Protocol code and prompts adapted from the upstream Apache-2.0 implementation are identified in source headers; its license is retained in `UPSTREAM_LICENSE`.
