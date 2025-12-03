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
| **Implementation** | ❌ Pick existing impl | A folder inside that server type |

## Why This Matters

A common source of confusion:

```yaml
example_simple_weather:        # ← Server ID
  resources_servers:
    example_simple_weather:    # ← Implementation (same name, different purpose!)
```

These look like duplicates, but they're not:

- **First** `example_simple_weather`: Your chosen name for this server instance. Used in API requests and when other servers reference it. You could call it `my_weather` or `weather_prod` - it's arbitrary.

- **Second** `example_simple_weather`: Must match the folder `resources_servers/example_simple_weather/`. This tells NeMo Gym which code to run.

Many examples use the same name for both (because why not?), but this obscures that they're independent choices.

## Server Types Map to Abstractions

The three server types correspond to NeMo Gym's {doc}`core abstractions </about/concepts/core-abstractions>`:

| Server Type | Abstraction | Folder |
|-------------|-------------|--------|
| `responses_api_models` | Models | `responses_api_models/` |
| `resources_servers` | Resources | `resources_servers/` |
| `responses_api_agents` | Agents | `responses_api_agents/` |

## Layered Configuration

Configuration resolves in priority order:

```text
Server YAML files  →  env.yaml  →  Command line
(base settings)      (secrets)     (overrides)
```

This lets you version-control base configs while keeping secrets separate and allowing runtime experimentation.

---

:::{seealso}
- {doc}`/reference/configuration` for complete syntax and field specifications
- {doc}`/troubleshooting/configuration` for common errors
:::

