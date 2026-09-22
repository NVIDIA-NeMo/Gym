# facts_parametric

FACTS Parametric public set (1,052 closed-book factoid questions) graded by the official Gemini 2.5 Pro protocol.

- Environment and run instructions: [`resources_servers/facts_parametric/README.md`](../../resources_servers/facts_parametric/README.md)
- Workload declaration: [`manifest.yaml`](manifest.yaml)
- Metrics, provenance, calibration, and reading guide: [`METRICS.md`](METRICS.md)
- Technical report metadata and fetch script: [`paper/PAPER.md`](paper/PAPER.md), `fetch_paper.py`
- Upstream control (starter notebook helpers, verbatim) and calibration: `upstream_control.py`, `calibrate.py`
- Run packages, BLADE exports, and model-card reports: `reporting/`

```bash
python benchmarks/facts_parametric/prepare.py                    # pinned download + JSONL
gym env validate facts_parametric
gym env test facts_parametric
python -m benchmarks.facts_parametric.calibrate --rollouts results/full/facts_parametric.jsonl --out results/full/calibration --live-size 40
python -m benchmarks.facts_parametric.reporting.cli build --run-dir results/full --stem facts_parametric --run-id <id> --model <model> --package results/<pkg> --run-info run-info.json --calibration-dir results/full/calibration
python -m benchmarks.facts_parametric.reporting.generate_model_card_report --package results/<pkg> --base-url "$REPORT_LLM_BASE_URL" --model "$REPORT_LLM_MODEL" --api-key-env REPORT_LLM_API_KEY
```
