# Nemotron Analysis Artifacts

This folder contains only the Nemotron Super 49B golden analysis artifacts from
the source benchmark package:

- `nemotron-super-49b-golden-report.md`
- `nemotron-super-49b-shallow.md`
- `nemotron_super_49b_golden_report_metrics.json`
- `nemotron_super_49b_anchor_facts.json`

The large rollout JSONL is intentionally not included. Use these files only when
the user explicitly asks to inspect an example completed BLADE-style analysis
report or compare against curated anchor facts. For original CVDP rollout
examples, use the existing files under `resources_servers/cvdp/data/`.
