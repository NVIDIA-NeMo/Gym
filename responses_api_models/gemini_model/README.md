# Gemini Model

Accepts OpenAI Responses API format (`NeMoGymResponseCreateParamsNonStreaming`), translates to the Google Gemini `generate_content` API, calls the Gemini backend, and translates the response back to `NeMoGymResponse`. All context management (conversation history, trimming) is handled by the agent/adapter layer — this server is a stateless translator.

Uses the native async client (`client.aio.models.generate_content`) from the `google-genai` SDK with application-level retry logic for transient errors (429, 500, 503, etc.).

## Configuration

Set up your `env.yaml` file:

```yaml
cua_gemini_api_key: <YOUR_GEMINI_API_KEY>
```

### Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `gemini_api_key` | (required) | Google AI API key |
| `gemini_model` | `gemini-2.5-computer-use-preview-10-2025` | Model name |
| `gemini_timeout` | `300.0` | Request timeout in seconds |
| `gemini_max_retries` | `4` | Max retries for transient errors |
| `thinking_level` | `MEDIUM` | Gemini thinking config level (used for Gemini 3 models) |
| `include_thoughts` | `true` | Include thought parts in the thinking config |

## Usage

### Running the server

```bash
config_paths="responses_api_models/gemini_model/configs/gemini_model.yaml,\
resources_servers/browser_gym/configs/browser_gym.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### Running tests

```bash
ng_test +entrypoint=responses_api_models/gemini_model

# Or directly:
pytest responses_api_models/gemini_model/tests/test_app.py -x -v
```

## Licensing information

- **Code**: Apache 2.0
- **Data**: N/A

## Dependencies

- `nemo_gym`: Apache 2.0
- `google-genai`: Apache 2.0
