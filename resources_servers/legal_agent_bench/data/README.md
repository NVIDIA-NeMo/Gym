# Legal Agent Bench data

`example.jsonl` contains five smoke tasks for upstream LAB commit
`f46ef86e4788545622db25dcffa3aebb7a139929`.
`example_rollouts.jsonl` contains the five corresponding completed model
trajectories, strict full-task rewards, and diagnostic criteria pass rates.
Machine-specific Harbor metadata is removed; no source document files or
credentials are committed.

Run this from the repository root to prepare the gitignored binary assets:

```bash
python resources_servers/legal_agent_bench/prepare.py
```

Preparation keeps the full 1,749-row validation index in the Harbor
task cache and atomically publishes a gitignored copy at
`data/generated/all.jsonl` for Gym dataset collation. The index and task cache
always describe the same pinned LAB snapshot. Prepared source caches live under
`data/cache/`.
`gym env start` creates a fresh credential-bearing runtime tree under
`data/runtime/`; never archive or commit that directory.
