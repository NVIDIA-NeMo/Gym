# GenRM Response API Model

A specialized Response API Model for GenRM (Generative Reward Model) that supports custom roles for pairwise comparison.

## Overview

GenRM Model extends the VLLM Model to support GenRM-specific custom roles used in pairwise response comparison:

- `response_1`: First candidate response for comparison
- `response_2`: Second candidate response for comparison
- `principle`: Optional judging principle (for principle-based comparison)

## Architecture

```
GenRMModel (inherits from VLLMModel)
    ├── GenRMConverter (extends VLLMConverter)
    │   └── Overrides _format_message() to handle custom roles
    └── Uses same infrastructure as VLLMModel for inference
```

## Key Features

- **Custom Role Support**: Handles `response_1`, `response_2`, and `principle` roles
- **Inherits VLLM Capabilities**: All VLLMModel features (reasoning parser, token IDs, etc.)
- **Principle-based Comparison**: Optional principle role for guided comparison
- **Clean Separation**: Custom role logic isolated from base VLLMModel

## Configuration

```yaml
genrm_model:
  responses_api_models:
    genrm_model:
      entrypoint: app.py
      base_url: ${genrm_base_url}
      api_key: ${genrm_api_key}
      model: ${genrm_model_name}
      return_token_id_information: false
      uses_reasoning_parser: false
      supports_principle_role: true
```

### Configuration Parameters

All VLLMModelConfig parameters are supported, plus:

- `supports_principle_role` (bool): Enable principle-based comparison mode (default: true)

## Usage

### Pairwise Comparison Request

```python
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

# Request with custom roles
request = NeMoGymResponseCreateParamsNonStreaming(
    input=[
        {
            "role": "user",
            "content": "What is the capital of France?",
            "type": "message"
        },
        {
            "role": "principle",
            "content": "Please act as an impartial judge...",
            "type": "message"
        },
        {
            "role": "response_1",
            "content": "The capital of France is Paris.",
            "type": "message"
        },
        {
            "role": "response_2",
            "content": "Paris is the capital city of France.",
            "type": "message"
        }
    ],
    model="genrm_model"
)
```

### Expected GenRM Output

GenRM models typically output structured JSON with comparison scores:

```json
{
  "score_1": 4.5,
  "score_2": 4.0,
  "ranking": 2.0
}
```

## Integration with GenRM Compare Resource Server

This model is typically used by the GenRM Compare Resource Server for pairwise comparison:

```
GenRM Compare Resource Server
    ├── Generates comparison pairs
    ├── Formats messages with response_1/response_2 roles
    └── Calls GenRM Model (/v1/responses) for each pair
```

See `resources_servers/genrm_compare/` for the comparison orchestration logic.

## Testing

Run tests with:

```bash
cd responses_api_models/genrm_model
pytest tests/
```

## Differences from VLLMModel

| Feature | VLLMModel | GenRMModel |
|---------|-----------|------------|
| Supported Roles | user, assistant, system, developer | + response_1, response_2, principle |
| Converter | VLLMConverter | GenRMConverter (extends VLLMConverter) |
| Primary Use Case | General inference | Pairwise comparison |
| Reasoning Parser | Configurable | Typically disabled |

## Implementation Notes

- **Inheritance**: GenRMModel inherits from VLLMModel for code reuse
- **Converter Override**: GenRMConverter overrides `_format_message()` to handle custom roles
- **No Base Model Modification**: VLLMModel remains completely unchanged
- **Type Safety**: Uses `NeMoGymChatCompletionCustomRoleMessageParam` for custom roles

## Related Components

- **Base Model**: `responses_api_models/vllm_model/`
- **Comparison Server**: `resources_servers/genrm_compare/`
- **Type Definitions**: `nemo_gym/openai_utils.py`
