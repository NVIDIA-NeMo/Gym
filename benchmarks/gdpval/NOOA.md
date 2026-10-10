# GDPVal with a sandboxed NOOA agent

`nooa.yaml` collects the full prepared GDP task set through `nooa_single_agent_turn`.
Resources creates and owns the task sandbox; NOOA borrows it, returns its response,
and closes before Resources exports the submitted files and stops the sandbox.
The default recipe is **generation only**, so its rows have `execute_only=true`
and `mask_sample=true`. Its aggregate reports only `generation/exported` and must
not be interpreted as a benchmark score.

Supply an audited GDP image in `GDPVAL_CONTAINER_PATH` and an absolute durable
artifact directory in `PERSIST_DELIVERABLES_DIR`, or configure the corresponding
Resources fields directly. The intended Apptainer image is the audited
`python-3.13.gdpval.gym-80e4fc.sif`, built for the execution host's architecture.
It contains the published document/scientific tool stack. Do not replace it with
a generic Python image while retaining the environment description in the prompt.
The sandbox runtime still needs a normal provider/NOOA startup smoke on the target
host before collecting a full run; unit tests alone do not establish that runtime.
The supplied provider settings target the effective-UID-0 Pyxis controller on
HSG, using the non-setuid Apptainer installation. `--userns` creates a distinct
child user namespace; adding `--fakeroot` here instead caused instance joins to
fail with `Invalid argument`. The child remains mapped to the controller's user
and can write `/opt/nemo-gym-nooa`; this does not grant host root access. Other
controller identities require their own validated provider configuration. Use
the task-private configuration helper to allow at least 8192MiB of session
storage; the default 64MiB is insufficient for installing this runtime. Keep
that storage node-local and never change the host's shared configuration.

Prepare the canonical dataset with `benchmarks/gdpval/prepare.py`, then run
`python -m benchmarks.gdpval.prepare_nooa` to write `data/gdpval_nooa.jsonl`.
Preparation keeps every task's metadata, normalizes reference-file arrays, and
moves verifier-only fields into `task_input.task_data`. Rubrics, reference-model
outputs and judge credentials are never supplied to the agent.

The baseline selects temperature 1.0, top-p 1.0, 32768 maximum output tokens,
262144 model context and 250 policy calls. These are explicit NOOA baseline
choices, not a claim to reproduce Stirrup's serving/output budget.
It admits 8 concurrent episodes by default, with a 21600-second whole-episode
deadline and 180 seconds for final cleanup. The one-task canary overrides
`num_samples_in_parallel=1`, which also lowers the environment admission limit.
Slurm bounds the controller's total CPU and memory; this GDP recipe does not
request nested per-sandbox cgroup resources.

## Prompt and files

`resources_servers/gdpval/prompts/nooa_user_prompt.txt` adapts the environment
description from Gym `183ef8601aad3a3c5b933065b05c8cb87442560e`'s
`responses_api_agents/stirrup_agent/prompts/gdpval_user_prompt.txt`.
The package description is retained. Tool instructions describe NOOA's Python
execution instead of Stirrup's `code_exec`; final response plus files in
`/workspace/output` replaces Stirrup's `finish(paths=...)` call. There is no claim
that the two harnesses expose identical tools or state semantics.

NOOA's private CPython runtime is separate from the audited image's Python
package environment. The prompt explicitly directs scientific/document scripts
through the image's `python3` subprocess; those image packages are not claimed
to be importable directly in the NOOA execution tool. This runtime boundary is
an explicit harness adaptation, not a change to the audited image's packages.

Task files are uploaded under `/workspace/input/<dataset reference path>`.
Resources keeps the original bytes outside the sandbox for the judge, including
the canonical `reference_files/` layout, even if the agent modifies its copy.
The prompt asks for deliverables directly in `/workspace/output`. The exporter
preserves regular files in nested directories too, including original model-authored
ZIP files; it never flattens the tree or rebuilds archives. This keeps a nested
submission from discarding other valid outputs. It rejects traversal, links,
reserved evaluator filenames, oversized output and changed file bytes. Reference and artifact limits are explicit Resources config
fields, so a limit failure remains visible instead of silently dropping inputs.

## Saved artifacts and grading

Each Resources session uses a fresh `gdp-*` directory under the artifact root;
another attempt's artifacts are never deleted or overwritten. A completed export
contains:

- `generation.json`: schema version, typed task/episode identities and the exact
  `GDPValVerifyRequest`, including the final response and exported directory.
- `artifacts.json`: submitted filenames, sizes and SHA256 digests.
- `deliverables-*`: submitted file bytes, pristine `reference_files/`, and a
  `finish_params.json` completion marker explicitly identifying the NOOA output
  directory submission method. It does not claim a Stirrup tool was called.

Generation artifacts are saved before owner teardown. A failed stop leaves the
owner handle reachable for cleanup retry and does not trigger more inference.
In optional inline grading (`execute_only=false`), Resources calls its existing
GDP verifier and caches an identical completed request's verdict. Configure
`reward_mode=comparison`, audited reference outputs/ratings, and judge settings
explicitly before using that mode. This bridge does not change GDP scoring.

For the baseline, grade the preserved outputs separately using the chosen pinned
GDP AA-v2 `judge_only` recipe and its expected `task_<id>/repeat_<n>` layout.
Convert from `generation.json` while preserving task identities and artifact
hashes; no new policy run is required. The grader revision, reference manifest,
judge IDs/settings and exact task coverage belong in that grading run's manifest.
The legacy adaptive multistage driver expects flat rows and is intentionally not
enabled over these native generation rows. The base checkout's verifier must not
be labeled equivalent to another pinned grader without an explicit comparison.
