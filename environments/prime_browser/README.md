# Prime browser tasks

This environment packages 30 browser tasks for Harbor in a NeMo Gym sandbox.
The task bundle provides deterministic websites and a Chromium runtime.

## Prepare

```bash
python environments/prime_browser/prepare.py \
  --source-dir /path/to/worldsims_harbor
```

The source directory must contain `tasks/` and
`scripts/start_local_sims.py`. The prepared tasks are written to
`environments/prime_browser/data/tasks`. The command also generates
`data/validation.jsonl` from the task directories. These are the paths used by
`config.yaml`.

The task bundle is not part of Gym. Confirm its access and usage terms before
running or sharing it.

## Agent scope

The source bundle does not include a browser agent. This config uses a
Terminus-2 baseline with the `browser_open` command. The command opens one URL
in Chromium and returns rendered DOM. It is stateless and does not provide
persistent click, type, or screenshot observations. It is useful for
packaging checks and read-only tasks, but it is not a faithful computer-use
baseline for all 30 tasks. Use a computer-use agent for the full benchmark.

## Run the DOM-only baseline

Set `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`, and `PRIME_API_KEY`, then run
from the Gym repository root:

```bash
uv run gym eval run \
  --config environments/prime_browser/config.yaml \
  --model-type vllm_model \
  --model-url http://model-server/v1 \
  --model model-name \
  --agent harbor_agent \
  --input environments/prime_browser/data/validation.jsonl \
  --output browser-rollouts.jsonl \
  --split validation
```
