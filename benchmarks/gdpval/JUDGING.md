# Judge saved NOOA GDP artifacts with the canonical GDP driver

Generation and judging use separate source trees. Run `nooa.yaml` from the NOOA
integration checkout. Run `nooa_judge_only.yaml` from a separately installed Gym
checkout at **183ef8601aad3a3c5b933065b05c8cb87442560e**. Do not copy that scorer
into the NOOA checkout or label the older checkout's grader equivalent.

The judging overlay follows EFB
`3737282b3890c83cc50306955f59e326362d9c7c`'s
`configs/benchmarks/gym/gdpval/{bench,refs}.yaml`: comparison mode, four trials,
judge seed42, and the existing adaptive45-task calibration followed by220 tasks
against the nearest four reference models. Raw and normalized Elo and win rates
come from that existing driver; this integration adds no scoring implementation.
The GPT5.5 / Gemini3.1Pro / Opus4.8 panel is inherited from the pinned Gym
benchmark, preserving its separate fixed-model proxies, provider media limits,
and reasoning settings (including Claude adaptive/high). EFB's older panel-only
override must not replace this structure: all member labels would otherwise
reach the default proxy's fixed Gemini model.
The recipe's final-stage coverage floor is95%, with per-reference floors and
failure reporting retained. Report actual coverage and degradation flags beside
any score; that threshold does not imply every task was judged. Task admission is
16, document conversion concurrency is 2, and the GPT/Gemini/Claude proxy limits
are 4/2/4. These bounds preserve the prepared run's resource settings without
changing its comparisons, sampling, stages or rating calculation.

## Prepare a fresh grading bundle

Use the canonical flat JSONL that was used to prepare the NOOA generation rows.
Use the frozen reference-model JSON selected for the run, with its original
manifest, source, and hash preserved beside the preparation receipt.
`GDP_GENERATION_ROOT` is the Resources artifact root containing `gdp-*/generation.json`.
Repeat `--generation-root` for additional worker roots. These are ordinary file
operations and do not call a policy or a judge:

```bash
python "$GDP_NOOA_SOURCE/benchmarks/gdpval/prepare_nooa_judging.py" \
  --source "$GDP_CANONICAL_JSONL" \
  --generation-root "$GDP_GENERATION_ROOT" \
  --reference-models "$GDP_REFERENCE_MODELS_JSON" \
  --output "$GDP_JUDGING_BUNDLE"
```

Paths are supplied by the run owner. The default requires exactly 220 unique
canonical tasks and exactly one completed generation receipt for each. Duplicate
attempts, missing tasks, mismatched prompt/identity, changed output bytes, links,
and unexpected files fail preparation. Select receipts explicitly if a task has
multiple attempts; the converter never selects an attempt by time or score.
`--expected-tasks` exists only for explicitly scoped preparation tests; the full
canonical 45→220 recipe still requires the full dataset. For a terminal attempt
that produced no export, record its native failure and pass its canonical task ID
with `--exclude-task-id`. Exclusions preserve all canonical input rows and leave
the candidate directory absent, so the pinned driver records an omission rather
than an invented loss. A genuine export cannot be excluded. A completed export
with no submitted files still has its completion marker and is judged normally.
Freeze the selected attempts, exclusions and input hashes before the first judge
call; do not choose duplicates by score or regenerate a model failure.

The converter copies, rather than moves or links, every submitted byte and the
pristine input references into `deliverables/task_<id>/repeat_0/`. The completion
marker remains byte-identical and continues to identify NOOA's final-response
submission method. Its paths record the original submission. Original receipts
and artifact manifests are copied separately into `generation/`, where they
cannot accidentally become judged deliverables. The canonical JSONL is copied
byte-identically. `preparation.json` records the source hashes and original
physical generation episode/attempt; `judging_repeat_index=0` means one selected
candidate per task and is not a new policy execution.

## Run the pinned judge-only driver

Use a dedicated installation of the pinned Gym revision in the approved GDP
controller runtime, which includes LibreOffice and document conversion tools.
The preserved artifacts and every configured reference directory must be visible
at the same absolute paths to the agent and Resources processes. Use the same frozen 18-reference manifest for a comparable run and verify its
paths are available to your controller. Copy reference deliverables into a private
run directory before document conversion, retaining their hashes and ratings.
Conversion must not write into another run owner's reference tree. Keep the
selected reference file task-local and retain the pinned driver's availability-only
assignment repair so absent references are reassigned instead of counting as
wins. The repair uses `PERSIST_DELIVERABLES_DIR` to locate the prepared candidate
root. A separate task-local overlay enables:

```yaml
multistage:
  transport_assignment_repair:
    enabled: true
    reference_availability_only: true
```

After preparation, run from the pinned checkout. `JUDGE_API_KEY` must already be
available in the process environment; the command does not print it. The panel's
model IDs can be explicitly overridden through `JUDGE_GPT_MODEL`,
`JUDGE_GEMINI_MODEL`, and `JUDGE_CLAUDE_MODEL`; preserve the resolved IDs in the run
manifest and verify endpoint access before the full judging run. The evaluated
routes were `us/azure/openai/eccn-gpt-5.5`,
`us/aws/anthropic/eccn-claude-opus-4-8`, and
`gcp/google/gemini-3.1-pro-preview`. Endpoint and credential availability can differ
by deployment. Configure each fixed-model proxy explicitly when it needs a
different endpoint or key; `JUDGE_GEMINI_API_KEY` selects a separate Gemini key
and otherwise falls back to `JUDGE_API_KEY`. A panel label alone does not change
that proxy. Do not
substitute another judge while presenting the result as this panel. Verify real
document and audio/video comparisons through the selected routes.

```bash
cd "$GDP_JUDGE_SOURCE"
test "$(git rev-parse HEAD)" = 183ef8601aad3a3c5b933065b05c8cb87442560e
export PERSIST_DELIVERABLES_DIR="$GDP_JUDGING_BUNDLE/deliverables"
export JUDGE_GPT_MODEL=us/azure/openai/eccn-gpt-5.5
export JUDGE_CLAUDE_MODEL=us/aws/anthropic/eccn-claude-opus-4-8
export JUDGE_GEMINI_MODEL=gcp/google/gemini-3.1-pro-preview
NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=3 "$GDP_JUDGE_SOURCE/.venv/bin/gym" eval run \
  --benchmark gdpval \
  --config "$GDP_NOOA_SOURCE/benchmarks/gdpval/nooa_judge_only.yaml" \
  --config "$GDP_JUDGING_BUNDLE/judge_data.yaml" \
  --config "$GDP_REFERENCE_AVAILABILITY_CONFIG" \
  --split benchmark \
  --output "$GDP_JUDGING_BUNDLE/results/rollouts.jsonl"
```

For an immutable source archive rather than a Git checkout, verify its recorded
archive and source-file hashes instead of the `git rev-parse` check. Run the CLI
from that archive's dedicated installation, not another checkout's `gym`.

The overlay sets `judge_only=true`, `execute_only=false` and uses the existing
reference-subset-aware judgment cache. Its unused policy configuration points to
local port9 so an unexpected generation call fails instead of contacting a model.
The unused policy endpoint also makes the generic endpoint-readiness probe
inapplicable, so this recipe disables that probe; the separate real judge
preflights above remain required. No candidate sandbox is restarted. The canonical judge-only path intentionally
uses a placeholder response and scores the saved files; the real NOOA response
remains in its generation receipt. Do not treat the judge placeholder as a new
model trajectory. Do not pass `--limit` or alter stages/repeats during a run.

The converter preserves nested output trees and authored archive bytes. The
pinned candidate renderer keeps its existing recursion behavior; preservation
of a nested file does not imply the renderer consumed it. Retain the complete
artifact manifest and original generation trace alongside the score.
