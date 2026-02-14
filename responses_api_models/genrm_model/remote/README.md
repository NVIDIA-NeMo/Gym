# GenRM Remote (genrm_model.remote)

GenRM Response API Model that talks to a **remote** vLLM endpoint. Custom roles: response_1, response_2, principle.

## Configuration

See `configs/genrm_remote_model.yaml`. Config key under `responses_api_models` is `genrm_model.remote` with `entrypoint: remote/app.py`.

## Usage

```python
from responses_api_models.genrm_model.remote.app import GenRMModel, GenRMModelConfig
```

For a **locally managed** vLLM server use `genrm_model.local` (sibling subpackage).

## Testing

```bash
cd responses_api_models/genrm_model
pytest remote/tests/
```

## Related

- Local GenRM: `genrm_model.local`
- Base: `responses_api_models/vllm_model/`
- GenRM Compare server: `resources_servers/genrm_compare/`
