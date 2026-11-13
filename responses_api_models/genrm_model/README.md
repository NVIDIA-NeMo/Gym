# GenRM Response API Model

GenRM (Generative Reward Model) with custom roles for pairwise comparison: **response_1**, **response_2**, **principle**. Uses a locally managed vLLM server: downloads the model and starts vLLM (e.g. via Ray).

## Layout

```
genrm_model/
  __init__.py
  app.py
  configs/
  tests/
  pyproject.toml
  setup.py
  README.md
```

## Usage

```python
from responses_api_models.genrm_model.app import GenRMModel, GenRMModelConfig
```

## Configuration

See `configs/genrm_model.yaml`. Config key under `responses_api_models` is `genrm_model` with `entrypoint: app.py`.

To use a GenRM vLLM pool that runs elsewhere, set `base_url` to that pool and `entrypoint: external_app.py`. That entrypoint launches no vLLM and doesn't use Ray, so Gym doesn't start or join a Ray cluster for it. It accepts the same keys as `app.py`, with `vllm_serve_kwargs` and `vllm_serve_env_vars` optional. `external_app.py` also supports `num_workers > 1`, which `app.py` does not. `app.py` still accepts `base_url` and skips the local launch, but it joins Ray regardless, because Gym decides whether a server uses Ray from its entrypoint class's `ray_enabled`, not from `base_url`.

If a launcher injects `base_url`, have it set the entrypoint in the same place:

```bash
++genrm_model.responses_api_models.genrm_model.base_url=http://genrm-pool:8000/v1 \
++genrm_model.responses_api_models.genrm_model.entrypoint=external_app.py
```

Keep `entrypoint: app.py` on the launcher's local-launch path, where vLLM still starts on Ray.

## Testing

Requires vllm (and optionally ray). From repo root:

```bash
cd responses_api_models/genrm_model
pytest tests/
```

## Related

- Base local vLLM: `responses_api_models/local_vllm_model/`
- GenRM Compare server: `resources_servers/genrm_compare/`
