# Rollout Artifact Pitfalls

Failure modes that do not announce themselves. Every one of these produces a dataset that parses,
validates against the row schema, and trains — while teaching the wrong thing.

Read this before writing a converter against raw rollout artifacts, before validating a file too
large to check in one pass, and before the first `gym eval run` against a large pivot file.

## Reconstructing One Model Call From A Flattened Output List

A rollout artifact stores the trajectory as one flat list of items — reasoning, tool calls, tool
results, assistant text, user turns, all in sequence. A pivot is one *model call*, so that structure
has to be recovered before anything else. Recover it wrong and a parallel tool-call batch gets split
across two rows, each labelled with half of what the model actually did.

Two rules, in order of preference.

**Execution runs (portable).** An environment executes every call of one response before returning
any result, so a batch can only end at a tool result or a message. This rule reads nothing but item
types, so it works whatever the producer stamps on a call id.

**Call-id indices (when available).** Some producers stamp call ids like `<tool_name>:<index>` where
the index restarts at 0 for each model call. Where that holds it is a free cross-check. Many
producers instead use opaque ids, so never *depend* on it — and never let its absence silently pass
as agreement.

**Reasoning must be transparent to both rules.** A reasoning item does not end a model call. Some
producers record reasoning *between* the calls of one response, and treating that as a turn break
splits a batch that was one generation. In one source, segmenting with reasoning as a break agreed
with the harness's own model-call count on only ~76% of trajectories; treating it as transparent
agreed on 100%.

`scripts/audit_pivot_turn_boundaries.py --source-path <file>` runs both segmentations and reports
where they disagree.

## Cross-Check The Turn Count Against The Source

If the source records a per-trajectory model-call count, use it. It is the only independent evidence
that your segmentation is right, and it settles ambiguous cases that inspection cannot.

Expect an off-by-one and understand which one you have: many harnesses seed the conversation with a
scripted opening turn that lives in the request input rather than in the output list, so the
reconstructed turns come to one fewer than the recorded count. The auditor's `--seeded-greeting`
flag accepts `auto`, `yes` or `no` so this is a stated assumption rather than a silent tolerance.

The field name is producer-specific. Point `--model-call-count-field` at whatever yours uses; if the
source has no such field the check reports `SKIPPED`, not "passed".

## Reasoning Belongs To The Prefix, Never To The Pivot

Reasoning from *earlier* turns is context and belongs in the prefix. The reasoning the pivot turn
itself emitted must be withheld — it is the thinking that produces the very action being predicted,
so replaying it hands the policy its own answer.

Two recording artifacts break this, and both are easy to miss because the data still looks ordered.

**Hoisted reasoning.** Several turns' reasoning is bunched in front of the first turn's action
instead of sitting with the calls it belongs to. The tell is content that could not have been
thought yet — an early item referring to results no tool has returned. Left alone it is a *forward*
leak: those items land in the prefix of every later pivot, handing it conclusions drawn from tool
results it has not seen. In one source this reached 44% of prefixes.

**Interleaved reasoning.** Reasoning recorded between the calls of one response. The position
carries no information — the calls are a single generation — and turns appear that "thought" only
*after* acting, with nothing before.

**Diagnose before repairing.** Run the source diagnosis over two different models through the same
harness. An anomaly that appears for one and not the other is an artifact of the recording, not
model behaviour; in one such pair, one model showed both anomalies on most trajectories and the
other showed neither. That comparison is the evidence — content inspection alone only suggests it.

**Repair.** A call thinks once before it acts, so any lead reasoning past the first belongs to a
later call: keep the first, queue the surplus, and hand it in order to following turns that have no
reasoning of their own. Normalize interleaved reasoning to the front of its own turn. Drop surplus
that finds nowhere to land — keeping it preserves the leak it was queued to fix. This is a
heuristic; report the counts so a reader can audit it, and prefer dropping the surplus outright if
your source does not support the redistribution.

**Schema trap.** A `reasoning` item replayed as *input* requires an `id`. Without it the Responses
request model rejects the row. `encrypted_content` can be dropped when null.

## `reference_output`: Carry The Call, Not The Trajectory

`expected_action` is a reduced label — it is what the verifier compares, so it drops the assistant
text of a narrate-and-call turn and never contains the turn's reasoning. Carrying the model call
verbatim alongside it, in emission order, keeps both for consumers that want the full response.

Carry **the one call**, not the whole trajectory. Attaching the full conversation to every pivot
more than doubled one dataset's size while adding nothing a consumer could not reconstruct: the
prefix is already on the row, and every other turn is already its own pivot.

## The Seed Prefix Is Not A Fixed Length

Read the seed from each source row. Do not hardcode its length or its role order.

Sources deviate. One 17k-trajectory source was overwhelmingly `system` + scripted greeting + first
user turn, but a handful of trajectories started *warm* — `system`, user, then an already-executed
tool call and its result baked into the request input — and one had its opening turns in the other
order. An audit that assumed the common shape reported dozens of false failures on the exceptions.

A pre-executed call in the seed correctly produces **no pivot row**: there is no preceding context
from which to predict it. It simply rides along in every prefix.

## Validating A File Too Big For One Pass

A single-process pass over a large pivot file is dominated by per-row model construction. One
100 GB file took ~25 minutes; the same checks over byte ranges across 32 workers took ~1.5 minutes.

Shard by **byte range**, not by copying rows out — copy-sharding a 100 GB file rewrites 100 GB
before it validates anything. Each worker seeks to its offset, skips the partial line at the
boundary because its owner is the previous worker, and reads to its own end. That is one pass in
total.

Prove the row count instead of recounting it: have each worker report the offset it actually started
and stopped at, then assert the ranges are contiguous, non-overlapping, and span the file. A second
pass to count lines costs as much as the validation did.

`scripts/validate_pivot_dataset_sharded.py` implements this and shares its row contract with
`scripts/validate_pivot_dataset.py` by importing it.

## Running A Big Pivot File Through Gym

**The serving path prepares data before it rolls anything out.** `gym eval run` runs the data
processor, which streams every dataset named in the config *twice* and then byte-copies it into a
preprocessed directory. On a large dataset that dominates the run, and `--limit` does not help
because preparation happens first. Three ways out, in order:

1. `gym eval run --no-serve --input <small file>` against already-running servers — no preparation
   pass and no copy.
2. Point the config's split at a small examples file for smoke tests. On one dataset this took a
   test from ~15 minutes to ~90 seconds.
3. `++reuse_existing_data_preparation=true` for repeat runs over the same dataset.

**Rollout records drop your extra fields.** A rollout row keeps the request, the response, the
reward and the Gym identity keys — custom top-level fields such as provenance or `reference_output`
are not carried through. They *are* preserved in the run's `*_materialized_inputs.jsonl`.

**Join on the pair, not the task index.** Rows are identified by `(_ng_task_index,
_ng_rollout_index)` together; the task index alone is not unique, and with repeats it is shared by
every rollout of the same task. `nemo_gym.reward_profile.RewardProfiler.align_rows_and_results`
already implements this join and raises on duplicate keys and unmatched rows — reuse it:

```python
from nemo_gym.reward_profile import RewardProfiler

pairs = RewardProfiler().align_rows_and_results(materialized_input_rows, rollout_rows)
```

## Checklist

Before calling a pivot dataset done:

- [ ] Segmentation agrees with the source's own model-call count, or you know why it cannot be checked.
- [ ] Both segmentation rules agree, or the call-id rule is reported as skipped rather than passed.
- [ ] No prefix ends on a `function_call`.
- [ ] Each prefix holds exactly the reasoning of the turns before it — no more, no less.
- [ ] Reasoning input items carry an `id`.
- [ ] The seed came from each row, not from an assumption.
- [ ] Every label self-scores 1.0 under the config the dataset will actually be scored with.
- [ ] Metrics report the parallel / single / chat split and a batch-size histogram.
