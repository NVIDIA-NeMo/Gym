# GenRM Response API Model

GenRM (Generative Reward Model) with custom roles for pairwise comparison: **response_1**, **response_2**, **principle**. Two subpackages follow the same template (app.py, configs/, tests/, README):

- **genrm_model.remote** – remote vLLM endpoint (no vllm/ray dependency at import time)
- **genrm_model.local** – locally managed vLLM server (download + start vLLM, e.g. via Ray)

## Layout

```
genrm_model/
  __init__.py
  pyproject.toml
  README.md           (this file)
  remote/
    app.py
    configs/
    tests/
    README.md
  local/
    app.py
    configs/
    tests/
    README.md
```

## Usage

**Remote (no vllm required to import):**
```python
from responses_api_models.genrm_model.remote.app import GenRMModel, GenRMModelConfig
```

**Local (requires vllm/local_vllm_model):**
```python
from responses_api_models.genrm_model.local.app import LocalGenRMModel, LocalGenRMModelConfig
```

## Config keys

- Remote: `genrm_model.remote` with `entrypoint: remote/app.py`
- Local: `genrm_model.local` with `entrypoint: local/app.py`

## Testing

```bash
cd responses_api_models/genrm_model
pytest remote/tests/   # no vllm
pytest local/tests/    # needs vllm
```

## Related

- Base: `responses_api_models/vllm_model/`, `responses_api_models/local_vllm_model/`
- GenRM Compare server: `resources_servers/genrm_compare/`
