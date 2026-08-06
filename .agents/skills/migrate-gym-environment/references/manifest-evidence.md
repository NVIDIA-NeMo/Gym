# Manifest evidence checklist

Use evidence from the existing recipe and its accountable owner. A generated draft is a locator and composition hint, not evidence for semantic claims.

| Field | Acceptable evidence | Do not infer from |
| --- | --- | --- |
| `name`, `kind` | Canonical registry identity and how users invoke the recipe | Directory name alone when several runnable configs share it |
| `integration_profile` | Resolved driver/harness placement plus `gym env validate` classification | Task difficulty or whether tools happen to be present |
| `domain` | Typed Resources Server domain and owner confirmation | Benchmark title |
| `description` | Current task contract, README, and representative materialized row | Legacy marketing text without checking behavior |
| `modality` | Materialized input and selected model capability | File extension or README claim alone |
| `licensing` | Dataset/source license files or explicit access classification | Repository license when data or adopted code differs |
| `authors` | Explicit contributor or owning-team confirmation | `git log` alone; committers are not necessarily authors |
| `reward.range` | Verifier implementation, aggregation behavior, and endpoint fixtures | One observed rollout or a conventional `[0, 1]` assumption |
| `reward.higher_is_better` | Metric semantics and owner review | The sign of one sample reward |
| `determinism` | Re-seed behavior and fixture evidence | Presence of a seed field without a repeatability check |
| component fields | Fully resolved Hydra composition | A convenient component that is not actually selected |
| `requires`, `provides` | Declared component contracts and successful capability validation | Implementation names or informal descriptions |
| datasets | Authoritative config plus materialized row validation | A remote dataset name without local preparation provenance |
| `canonical_split` | Upstream benchmark protocol or owner-controlled dataset specification | Whichever split happens to be downloaded locally |
| `standard_prompt_config` | Canonical comparison protocol | A tuning prompt or model-specific convenience template |
| `adopted_from` | Cloneable source, exact Git ref, and reviewed reconciliation date | A project homepage or mutable branch without owner intent |
| `version` | Owner-selected first release or semantic bump over a prior published composition | Current date, latest package version, or an automatic guess |

Use `unknown` only where the schema explicitly permits it. Unknown is preferable to a false claim, but it is not a substitute for required reward or benchmark-protocol evidence.

## Scorer acceptance

The manifest's reward contract is not complete until the Resources Server has a canonical `tests/verifier_cases.jsonl` and the repository harness test. Cover:

- the preferred reward endpoint;
- the opposite endpoint;
- malformed input with an explicit failure expectation;
- a repeat re-seed case when claiming `determinism: seeded`;
- domain edge cases needed to make the declared range honest.

Regenerated expected values require human review. A passing fixture proves the implementation matches the checked expectations; it does not prove the expectations are correct.

## Exception classification

Use one of these outcomes before touching an exception:

- **component-only**: keep the Resources Server reusable; do not manufacture a catalog environment;
- **canonicalize**: choose one owner-backed runnable recipe and give variants distinct identities if they remain user-facing;
- **repair-resolution**: make static config resolution explicit without embedding credentials or production-only endpoints;
- **promote**: create a canonical environment or benchmark recipe around a reusable component;
- **retire**: remove only through the repository's normal deprecation/removal review, never as migration cleanup.

If none applies with current evidence, leave the exception visible and report the missing decision.
