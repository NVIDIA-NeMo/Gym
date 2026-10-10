# FACTS Grounding v2

This benchmark adapter prepares the 856-row public FACTS Grounding release and implements
the v2 eligibility and sentence-level grounding protocol. The private Kaggle split is not
included. The prepare script downloads the pinned public release and verifies its source
files before producing NeMo Gym JSONL.

- Environment configuration and run instructions:
  [resources_servers/facts_grounding_v2](../../resources_servers/facts_grounding_v2/README.md)
- Metric definitions and interpretation: [METRICS.md](METRICS.md)
- Source report and benchmark references: [paper/PAPER.md](paper/PAPER.md)
