# Prime terminal tasks

This environment runs 30 packaged terminal tasks with Harbor in a NeMo Gym
sandbox.

## Prepare

```bash
python environments/prime_terminal/prepare.py \
  --source-dir /path/to/terminal-tasks
```

The prepared tasks are written to `environments/prime_terminal/data/tasks`.
The command also generates `data/validation.jsonl` from the task directories.
These are the paths used by `config.yaml`.

The task bundle is not part of Gym. Confirm its access and usage terms before
running or sharing it.

## Run

Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`, then run from the Gym
repository root:

```bash
uv run gym eval run \
  --config environments/prime_terminal/config.yaml \
  --model-type vllm_model \
  --model-url http://model-server/v1 \
  --model model-name \
  --agent harbor_agent \
  --input environments/prime_terminal/data/validation.jsonl \
  --output terminal-rollouts.jsonl \
  --split validation
```
