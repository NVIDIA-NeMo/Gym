# injecagent

InjecAgent base setting (510 direct-harm + 544 data-stealing indirect prompt injection cases) under the upstream
fine-tuned function-calling protocol, scored by a deterministic first-tool-call verifier.

- Environment and run instructions: [`resources_servers/injecagent/README.md`](../../resources_servers/injecagent/README.md)
- Metrics, provenance, calibration, and reading guide: [`METRICS.md`](METRICS.md)
- Paper metadata and fetch script: [`paper/PAPER.md`](paper/PAPER.md), `fetch_paper.py`
- Upstream scorer (vendored verbatim, hash-checked) and replay calibration: `upstream_scorer.py`, `calibrate.py`
- Run packages, BLADE exports, and model-card reports: `reporting/`
