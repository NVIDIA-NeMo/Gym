# Anthropic Model

Accepts OpenAI Responses API format (`NeMoGymResponseCreateParamsNonStreaming`), translates to the Anthropic Messages API, calls the Anthropic backend, and translates the response back to `NeMoGymResponse`. All context management (conversation history, trimming) is handled by the agent/adapter layer — this server is a stateless translator.

Uses the async Anthropic SDK (`AsyncAnthropic`) with built-in retries.

## Configuration

Set up your `env.yaml` file:

```yaml
cua_anthropic_api_key: <YOUR_ANTHROPIC_API_KEY>
```

### Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `anthropic_api_key` | (required) | Anthropic API key |
| `anthropic_model` | `claude-sonnet-4-20250514` | Model name |
| `anthropic_timeout` | `300.0` | Request timeout in seconds |
| `anthropic_max_tokens` | `4096` | Max output tokens |
| `effort_level` | `high` | Thinking effort level (sent as `output_config.effort` when not "high") |
| `computer_tool_version` | `computer_20250124` | Anthropic computer tool type version |
| `computer_betas` | `[computer-use-2025-01-24, ...]` | Beta feature flags for computer use |
| `zoom_enabled` | `false` | Enable zoom in the computer tool |

## Usage

### Running the server

```bash
config_paths="responses_api_models/anthropic_model/configs/anthropic_model.yaml,\
resources_servers/browser_gym/configs/browser_gym.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### Running tests

```bash
ng_test +entrypoint=responses_api_models/anthropic_model

# Or directly:
pytest responses_api_models/anthropic_model/tests/test_app.py -x -v
```

## Licensing information

- **Code**: Apache 2.0
- **Data**: N/A

## Dependencies

- `nemo_gym`: Apache 2.0
- `anthropic`: MIT
