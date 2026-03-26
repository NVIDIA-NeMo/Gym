# Anthropic Model

Stateless proxy that forwards requests to the Anthropic Messages API and returns raw responses. All context management (conversation history, trimming) is handled by the agent/adapter layer — this server is a pure API relay.

Uses the async Anthropic SDK (`AsyncAnthropic`) with built-in retries.

## Configuration

Set up your `env.yaml` file:

```yaml
policy_api_key: <YOUR_ANTHROPIC_API_KEY>
policy_model_name: claude-sonnet-4-20250514
```

### Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `anthropic_api_key` | (required) | Anthropic API key |
| `anthropic_model` | `claude-sonnet-4-20250514` | Model name |
| `anthropic_timeout` | `300.0` | Request timeout in seconds |

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
