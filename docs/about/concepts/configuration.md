(configuration-concepts)=

# Configuration System

NeMo Gym uses a layered configuration system with three sources resolved in priority order:

```
Server YAML Config Files  <  env.yaml  <  Command Line Arguments
    (lowest priority)                       (highest priority)
```

This separation enables:

- **Base configuration** in YAML files for shared, version-controlled settings
- **Secrets** in `env.yaml` for API keys and environment-specific values (never committed)
- **Runtime overrides** via command line for experimentation and testing

## Server Configuration Hierarchy

Every config file defines server instances following this structure:

```yaml
server_id:              # Unique name used in requests
  server_type:          # One of: responses_api_models, resources_servers, responses_api_agents
    implementation:     # Folder name inside the server type directory
      entrypoint: app.py
      # Server-specific configuration...
```

The server types map directly to the three top-level folders in the NeMo Gym repository.

## Policy Model Variables

NeMo Gym provides standard placeholders for the model being trained:

- `policy_base_url` - Model API endpoint
- `policy_api_key` - Authentication key
- `policy_model_name` - Model identifier

These enable consistent references to "the model being trained" across different resource servers and agents.

---

:::{tip}
Refer to {doc}`/reference/configuration` for complete syntax and options, or {doc}`/troubleshooting/configuration` if you encounter errors.
:::