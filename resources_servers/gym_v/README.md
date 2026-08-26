# Gym-V NeMo-Gym resources server

Env-id-parametric NeMo-Gym resources server wrapping any
[`gym_v.Env`](https://github.com/rohitrango/gym-v) behind `/seed_session`,
`/step`, `/close`, and `/verify`. Env selection is driven entirely by JSONL
task rows (`env_id`, `seed`, `env_kwargs`, ...) — adding envs is data-only,
no server code change.

## Layout

```
resources_servers/gym_v/
  __init__.py
  app.py                  # GymVResourcesServer + DEFAULT_SYSTEM_PROMPT
  schemas.py              # Pydantic wire shapes
  _metadata.py            # JSON-safe Observation.metadata walker
  _observation.py         # observation_to_user_message + image_to_data_url
  _dataset_cache.py       # per-class LRU cache on reasoning-gym _make_dataset
  configs/gym_v.yaml      # server defaults
  data/example.jsonl      # smoke-test task rows
  requirements.txt        # rohitrango/gym-v @ f243cff... + pillow
  tests/test_app.py
```

## Running the helper-level tests

The observation/metadata/schemas helpers do not require `gym_v` to be
installed (they use an `ObservationLike` Protocol). Run from a venv with
`pytest`, `pydantic`, `pillow`, and `nemo_gym` on `PYTHONPATH`:

```bash
export PYTHONPATH=<repo>:<repo>/3rdparty/Gym-workspace/Gym
cd <repo>/3rdparty/Gym-workspace/Gym
python -m pytest resources_servers/gym_v/tests -v --tb=short
```

## Per-server venv (full stack)

```bash
RAY_TMPDIR=/tmp gym env test --resources-server gym_v
```

First run installs `gym-v[games,spatial,reasoning-gym]` from the
rohitrango fork pinned at `f243cff...`.

## Paired agent

This server speaks `/seed_session` + `/step` + `/close` + `/verify` and
is paired with [`responses_api_agents/gymv_agent/`](../../responses_api_agents/gymv_agent/),
which extracts `\boxed{...}` from model output and drives the step loop.
