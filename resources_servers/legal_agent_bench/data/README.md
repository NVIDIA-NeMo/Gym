# Legal Agent Bench data

`all.jsonl` is the deterministic 1,749-row index for upstream LAB commit
`f46ef86e4788545622db25dcffa3aebb7a139929`. `example.jsonl` contains five
smoke tasks from the same revision. Both files contain metadata only.
`example_rollouts.jsonl` contains the five corresponding completed model
trajectories, strict full-task rewards, and diagnostic criteria pass rates.
Machine-specific Harbor metadata is removed; no source document files or
credentials are committed.

Run this from the repository root to prepare the gitignored binary assets:

```bash
python resources_servers/legal_agent_bench/prepare.py
```

Prepared source caches live under `data/cache/`. `ng_run` creates a fresh
credential-bearing runtime tree under `data/runtime/`; never archive or commit
that directory.
