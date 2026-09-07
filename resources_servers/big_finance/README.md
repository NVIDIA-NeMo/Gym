# BigFinanceBench resource server

Tools-only wrapper around the Apache-2.0 `big-finance-harness` tools. The
resource-server requirements pin the tools-only fork
`wowowoxuan1994/big-finance-benchmark` to an immutable commit. Install its
isolated environment with the repository's normal resource-server setup, or
directly:

```bash
cd resources_servers/big_finance
uv venv --clear .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The fork keeps LiteLLM, Google provider, and CLI dependencies in its optional
`eval` extra. Gym installs only the base tool dependencies, avoiding the
upstream harness's model-provider dependency conflicts.

Routes are `/web_search`, `/edgar_search`, `/fetch_url`, `/python_exec`, and
`/final_answer`; each accepts the upstream JSON arguments and returns the
upstream tool string verbatim. `/seed_session`, `/verify`,
`/aggregate_metrics`, and `/reverify_mode` follow the standard Gym protocol.

`configs/big_finance.yaml` also ships a `big_finance` agent instance backed by
the shared finance loop. Its five-row `data/example.jsonl` fixture is converted
from the first five rows of the pinned public BigFinanceBench subset and retains
the evaluation-only, do-not-train, canary, prompt, license, and provenance
fields. Generated rollouts and metrics are intentionally not included.

See `benchmarks/big_finance/README.md` for credentials, scoring, safety,
preparation, and standalone parity notes.

The tool package is Apache-2.0 licensed. The public five-row fixture is derived
from the separately licensed CC BY 4.0 BigFinanceBench public subset.
