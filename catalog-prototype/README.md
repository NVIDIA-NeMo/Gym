# NeMo Gym catalog prototype data

This directory contains a snapshot of the unified NeMo Gym environment and benchmark catalog for UI prototyping.
The snapshot intentionally includes both manifest-backed and legacy entries so a prototype can demonstrate the
migration state.

Refresh the snapshot from the repository root:

```bash
gym list environments --json --full | jq '.' > catalog-prototype/catalog.json
```

Use `metadata_complete` and `status` in the UI:

- `metadata_complete: true` / `status: experimental` means a valid manifest supplied the full structured record.
- `metadata_complete: false` / `status: no-manifest` means Gym discovered a legacy runnable config and some fields
  are unavailable.

The full payload includes repository-relative source paths, composition, datasets, authors, reward semantics,
determinism, and benchmark protocol fields. Missing legacy metadata is represented with empty values rather than
invented values.

See `lovable-prompt.md` for a prompt that can be pasted into Lovable after importing or syncing this branch.
