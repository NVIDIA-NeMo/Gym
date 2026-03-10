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

## Testing

Requires vllm (and optionally ray). From repo root:

```bash
cd responses_api_models/genrm_model
pytest tests/
```

## Related

- Base local vLLM: `responses_api_models/local_vllm_model/`
- GenRM Compare server: `resources_servers/genrm_compare/`
