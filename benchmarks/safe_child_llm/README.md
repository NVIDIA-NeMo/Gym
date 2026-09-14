# safe_child_llm

Safe-Child-LLM developmental safety prompts (100 for ages 6–12, 100 for ages 13–17) collected for the paper's human
annotation protocol (binary harmfulness + 0–5 action label), with the released keyword heuristics kept as diagnostics.

- Environment, annotation, and scoring instructions: [`resources_servers/safe_child_llm/README.md`](../../resources_servers/safe_child_llm/README.md)
- Metrics, provenance, calibration, and reading guide: [`METRICS.md`](METRICS.md)
- Paper metadata and fetch script: [`paper/PAPER.md`](paper/PAPER.md), `fetch_paper.py`
- Human annotation app and label scorer: `annotation_app.py`, `score_annotations.py`
- Heuristic/rubric calibration: `calibrate.py`
- Run packages, BLADE exports, and model-card reports: `reporting/`
