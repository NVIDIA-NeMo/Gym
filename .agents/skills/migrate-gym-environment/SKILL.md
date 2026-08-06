---
name: migrate-gym-environment
description: >
  Decide whether and how to migrate one existing NeMo Gym runnable recipe to a
  complete manifest-backed entry. Use when claiming a migration-inventory
  record, changing a legacy environment or benchmark, classifying a Resources
  Server config, resolving an inventory exception, or reviewing a migration PR.
  Do not use for bulk metadata backfills.
---

# Migrate one Gym environment

Read `references/manifest-evidence.md` and `fern/versions/latest/pages/contribute/environments/migrating-an-environment.mdx` before editing.

## Decide first

Inspect one inventory record, its config, implementation, datasets, tests, documentation, and ownership evidence. Report:

```text
Decision: migrate | defer | needs-architecture
Trigger:
Canonical recipe and config:
Owner and authorship evidence:
Inventory status:
Blocking facts:
Planned file scope:
```

Choose `migrate` only for an owner-backed user-facing recipe with an unambiguous identity and config, evidenced semantic fields, and a canonical verifier fixture. Choose `defer` when ownership or evidence is missing. Choose `needs-architecture` for shared identities, noncanonical variants, component-only servers, or unresolved configs.

## Apply the decision

- Preserve unrelated worktree changes.
- Author exactly one complete manifest from evidence; never commit a `TODO_REQUIRED` draft.
- Keep runtime wiring in config and mirror only the resolved composition in the manifest.
- Add only required capability declarations, example data, verifier fixtures, tests, CODEOWNERS, and inventory updates.
- Keep score, prompt, dataset, orchestration, and runtime behavior changes out of a metadata migration.
- Require explicit authorship confirmation and a real review owner before publication.

Validate the explicit pair, then its name-based scorer fixture:

```bash
gym env validate --config CONFIG_PATH --manifest MANIFEST_PATH --json
gym env test NAME --kind KIND
```

Refresh and check inventory without creating drafts or catalog artifacts:

```bash
uv run python scripts/migrate_environment_manifests.py --inventory-only
uv run python scripts/migrate_environment_manifests.py --check
uv run python scripts/check_environment_onboarding.py \
  --changed-file MANIFEST_PATH \
  --enforce-changes \
  --run-verifier-tests
```

Preview publication with `gym env publish NAME@VERSION --kind KIND --owner OWNER --dry-run`. Run its write-producing form only when the user authorized publication.

## Stop when

- The record is component-only or its canonical identity needs an owner decision.
- Licensing, authorship, reward, determinism, or benchmark protocol lacks evidence.
- The change would create or complete manifests in bulk.
- Validation exposes a behavioral bug that belongs in a separate change.
- Publication ownership is unknown.

## Handoff

Report the decision, evidence, files changed, checks run, inventory delta, publication result if authorized, and whether runtime behavior changed.
