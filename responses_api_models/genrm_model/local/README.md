# GenRM Local (genrm_model.local)

GenRM Response API Model with a **locally managed** vLLM server: downloads the model and starts vLLM (e.g. via Ray). Same custom roles (response_1, response_2, principle) as the remote variant.

## Configuration

See `configs/genrm_local_model.yaml`. Config key under `responses_api_models` is `genrm_model.local` with `entrypoint: local/app.py`.

## Usage

```python
from responses_api_models.genrm_model.local.app import LocalGenRMModel, LocalGenRMModelConfig
```

For a **remote** vLLM endpoint use `genrm_model.remote` (sibling subpackage).

## Testing

Requires vllm (and optionally ray). From repo root:

```bash
cd responses_api_models/genrm_model
pytest local/tests/
```

## Related

- Remote GenRM: `genrm_model.remote`
- Base local vLLM: `responses_api_models/local_vllm_model/`
- GenRM Compare server: `resources_servers/genrm_compare/`
