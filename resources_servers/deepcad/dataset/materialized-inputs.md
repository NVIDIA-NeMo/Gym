# Offline materialized CAD tasks

Use the existing source/operation sampler offline to make a finite training pool. The output uses ordinary Gym rows with embedded prompt images and packaged verifier STEP files. Runtime training needs no CAD dataset hook or renderer. Existing lazy materialization remains available to its current callers.

Run from the Gym checkout with its installed dependencies:

```bash
python -m resources_servers.deepcad.dataset.materialized_inputs select \
  --manifest train.jsonl \
  --sampler-config sampler.json --render-overrides selection-overrides.json \
  --num-samples 25024 --batch-size 64 --seed 42 --output RELEASE/selection

python -m resources_servers.deepcad.dataset.materialized_inputs render \
  --selection RELEASE/selection --output RELEASE/shards \
  --shard-index 0 --num-shards 16 --workers 8 --chunk-size 64 \
  --render-overrides local-overrides.json

python -m resources_servers.deepcad.dataset.materialized_inputs assemble \
  --selection RELEASE/selection --shards RELEASE/shards --output RELEASE/ready

python -m resources_servers.deepcad.dataset.materialized_inputs validate RELEASE/ready
```

`sampler.json` is the existing source-operation sampling configuration. Configure the sources, operation fraction and balancing settings for the input manifest. Selection overrides must pin the matching exclusion-aware replacement catalog. The catalog checksum is frozen with the selection; every resolved replacement must be a selected entry in that catalog with matching source and source checksum.

Run selection once. Run shard indices 0–15 on CPU nodes. Work is assigned by chunk index modulo shard count. Existing complete chunks are checksum-checked and skipped; incomplete chunks can be retried using the same selection and settings. Selection and release assembly use temporary sibling directories and publish their final directories atomically after completion, so interrupted attempts can be retried without deleting final output paths. `--max-chunks 1` limits a pilot to its first assigned chunk. A separate `assemble --prefix-count 64 --output RELEASE/smoke-ready` can publish the initial complete chunk for a smoke while other chunks continue. Its receipt identifies it as a smoke prefix.

Keep active SQLite, prompt caches, sandbox state and temporary render files on node-local storage. Stage each inventory database at the relative path under the overridden `dataset_root`; staging the replacement catalog alone is insufficient. All nodes must use the same logical override paths/settings. The shared output receives aggregated chunk JSONL/TAR files and completion receipts, rather than per-reference cache locks and temporary files.

`ready` contains `train.materialized.jsonl`, `references.tar`, `references.json`, `resolutions.jsonl`, and `release.json`. Extract `references.tar` into the Gym resource server's configured `reference_root` before rollouts; rows refer to SHA256-sharded relative STEP paths. Prompt bytes remain unchanged when the reference archive is relocated. The JSONL alone is not a complete release. The exporter verifies decoded image bytes and validates every release before publishing it. Validate the ready release before requesting GPUs.

Selection preserves draw order and repeated occurrences, and rendering deduplicates work within chunks while reusing the persistent cache across chunks. Native timeout/replacement behavior is retained, including bounded fresh failure confirmation. Infrastructure errors fail the chunk. Neither source failures nor exhausted replacement budgets silently reduce the requested row count.

A 25,024-occurrence pool contains 391 complete 64-prompt batches. With 25,000 optimizer steps it is reused for about 64 passes. It freezes sampled references and prompt augmentations; it does not provide fresh full-corpus sampling on every training step. Use the ordinary loader with the runtime source/operation sampler disabled and a fresh dataloader state.
