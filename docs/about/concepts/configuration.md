(configuration-concepts)=

# Configuration System

NeMo Gym uses YAML configuration files to define which servers to run. Understanding the config structure helps you avoid common confusion when setting up training environments.

## The Three-Level Pattern

Every server definition follows this pattern:

```yaml
server_id:              # What YOU call it (your choice)
  server_type:          # What KIND of server (matches a folder)
    implementation:     # Which CODE to run (matches a subfolder)
      entrypoint: ...
```

**Why three levels?** Each level serves a different purpose:

| Level | You Control | Must Match |
|-------|-------------|------------|
| **Server ID** | ✅ Yes - name it anything | Nothing - it's your identifier |
| **Server Type** | ❌ Pick from 3 options | `responses_api_models`, `resources_servers`, or `responses_api_agents` |
| **Implementation** | ❌ Pick existing implementation | A folder inside that server type |

### Understanding the Naming Pattern

In many examples, you'll see the same name appear twice:

```yaml
example_simple_weather:        # ← Server ID
  resources_servers:
    example_simple_weather:    # ← Implementation
```

These serve different purposes:

- **Server ID** (`example_simple_weather` on line 1): Your chosen identifier for this server instance. Used in API requests and when other servers reference it. You could name it `my_weather` or `weather_prod` instead.

- **Implementation** (`example_simple_weather` on line 3): Must match the folder `resources_servers/example_simple_weather/`. This tells NeMo Gym which code to run.

Examples often use matching names for simplicity, but the two values are independent choices.

## Policy Model Variables

NeMo Gym provides three standard placeholders for "the model being trained":

- `policy_base_url` - Model API endpoint
- `policy_api_key` - Authentication key
- `policy_model_name` - Model identifier

These let you reference the training model consistently across configs without hardcoding values. Define them once in `env.yaml`, then use `${variable_name}` syntax anywhere you need them.

---

:::{seealso}
- {doc}`/reference/configuration` for complete syntax and field specifications
- {doc}`/troubleshooting/configuration` for common errors
:::

