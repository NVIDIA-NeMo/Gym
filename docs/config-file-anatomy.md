(config-file-anatomy)=

# Configuration File Anatomy

This page explains the structure of NeMo Gym configuration files—the YAML hierarchy, what each field means, and how config files relate to the three core abstractions (Model, Resource, Agent).

:::{tip}
For information about configuration priority, `env.yaml`, and command-line overrides, see {doc}`configuration-system`.
:::

---

## The Three-Level YAML Hierarchy

Every NeMo Gym config file defines **server instances** using a three-level hierarchy:

```yaml
# ┌─ LEVEL 1: Server Instance ID
# │  A unique name you choose. This is how you reference this server elsewhere.
# │
# │     ┌─ LEVEL 2: Server Type
# │     │  One of: resources_servers, responses_api_models, responses_api_agents
# │     │  Matches the top-level folders in the NeMo Gym project.
# │     │
# │     │         ┌─ LEVEL 3: Implementation Type
# │     │         │  The folder name inside the server type folder containing the code.
# │     │         │
my_weather_agent:                    # Level 1: Server Instance ID (your unique name)
  responses_api_agents:              # Level 2: Server Type (folder category)
    simple_agent:                    # Level 3: Implementation Type (specific implementation folder)
      entrypoint: app.py             # Config fields for this server...
```

### Visual Mapping to Project Structure

```yaml
Level 2                Level 3              Your Code
───────────────────────────────────────────────────────
responses_api_agents/  simple_agent/        app.py
resources_servers/     example_simple_weather/  app.py
responses_api_models/  openai_model/        app.py
```

---

## Why the "Double Nesting"?

A common point of confusion is seeing patterns like this:

```yaml
example_simple_weather:           # First occurrence
  resources_servers:
    example_simple_weather:       # Second occurrence - same name!
      entrypoint: app.py
```

**Q: Why does `example_simple_weather` appear twice?**

**A:** They serve different purposes:

| Position | Name | Purpose |
|----------|------|---------|
| Level 1 | `example_simple_weather` | **Server Instance ID** — A unique identifier you use to reference this server from other configs |
| Level 3 | `example_simple_weather` | **Implementation Type** — The actual folder name where the code lives (`resources_servers/example_simple_weather/`) |

They happen to match in many examples for clarity, but they don't have to:

```yaml
# Different names are perfectly valid:
my_weather_v2:                    # Server Instance ID (your choice - any unique name)
  resources_servers:
    example_simple_weather:       # Implementation Type (must match actual folder name)
      entrypoint: app.py
```

:::{tip}
Think of Level 1 as a **nickname** you assign, and Level 3 as the **actual folder** containing the code.
:::

---

## Server Types and Their Fields

Each server type has different required and optional fields. The server type (Level 2) determines what fields are valid.

### Resource Servers (`resources_servers`)

Resource servers provide **tools** agents can call and **verification logic** that scores agent performance.

```yaml
my_resource_server:                   # Server Instance ID
  resources_servers:                  # Server Type
    example_simple_weather:           # Implementation folder
      # ─────────────────────────────────────────────────────
      # REQUIRED FIELDS
      # ─────────────────────────────────────────────────────
      entrypoint: app.py              # Python file to run
      domain: agent                   # Task category (see table below)

      # ─────────────────────────────────────────────────────
      # OPTIONAL FIELDS
      # ─────────────────────────────────────────────────────
      verified: false                 # Has this been validated through RL training?
      verified_url: ""                # Link to training results (when verified: true)
      description: ""                 # Human-readable description for README tables

      # ─────────────────────────────────────────────────────
      # CUSTOM FIELDS
      # ─────────────────────────────────────────────────────
      # Any additional fields are passed to your server's config class
      num_processes: 8
      timeout_secs: 30
```

#### The `domain` Field

The `domain` field is **required for all resource servers**. It categorizes the type of task the server handles:

| Domain | Use Case |
|--------|----------|
| `math` | Mathematical problem-solving |
| `coding` | Code generation and programming |
| `agent` | Agent-based interactions and tool calling |
| `knowledge` | Knowledge-based question answering |
| `instruction_following` | Instruction following benchmarks |
| `long_context` | Long context handling |
| `safety` | Safety and alignment |
| `games` | Game-playing scenarios |
| `e2e` | End-to-end workflows |
| `other` | General purpose |

#### The `verified` Field
<!-- TODO: define the explicit enforcement of this in the infra -->

The `verified` field indicates whether this resource server has been **validated through actual RL training** runs:

- `verified: false` (default) — Server works but hasn't been used in a training run yet
- `verified: true` — Server has been used successfully for RL training; include `verified_url` linking to results

```yaml
my_validated_server:
  resources_servers:
    code_gen:
      entrypoint: app.py
      domain: coding
      verified: true
      verified_url: https://wandb.ai/nvidia/my-project/runs/abc123
```

### Model Servers (`responses_api_models`)

Model servers wrap LLM inference endpoints. They don't require `domain` or `verified`.

```yaml
policy_model:                         # Server Instance ID (conventionally "policy_model")
  responses_api_models:               # Server Type
    openai_model:                     # Implementation folder
      entrypoint: app.py

      # Model-specific configuration:
      openai_base_url: ${policy_base_url}   # Variable substitution from env.yaml
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

### Agent Servers (`responses_api_agents`)

Agent servers orchestrate the interaction between models and resources. They use **server references** to connect to other servers.

```yaml
my_agent:                             # Server Instance ID
  responses_api_agents:               # Server Type
    simple_agent:                     # Implementation folder
      entrypoint: app.py

      # ─────────────────────────────────────────────────────
      # SERVER REFERENCES
      # Connect this agent to other servers
      # ─────────────────────────────────────────────────────
      resources_server:
        type: resources_servers       # Server Type of the target
        name: my_resource_server      # Server Instance ID of the target
      model_server:
        type: responses_api_models
        name: policy_model

      # ─────────────────────────────────────────────────────
      # DATASETS
      # Data sources for rollout collection and training
      # ─────────────────────────────────────────────────────
      datasets:
      - name: example
        type: example                 # One of: train, validation, example
        jsonl_fpath: path/to/data.jsonl
```

---

## Server References

Agents connect to resources and models using **server references**. A reference has two fields:

```yaml
resources_server:
  type: resources_servers         # The server type (Level 2 of the target)
  name: example_simple_weather    # The Server Instance ID (Level 1 of the target)
```

The `name` must match an existing Server Instance ID from a config file loaded in the same session.

:::{warning}
A common mistake is referencing the Implementation Type (Level 3) instead of the Server Instance ID (Level 1). If your Instance ID differs from the folder name, use the Instance ID.
:::

---

## Dataset Configuration

Datasets are defined on agent servers and specify data sources for training and evaluation:

```yaml
datasets:
  # Example dataset (for testing/demonstration)
  - name: example
    type: example                       # No license/gitlab required
    jsonl_fpath: path/to/example.jsonl

  # Training dataset
  - name: train
    type: train
    jsonl_fpath: path/to/train.jsonl
    license: Apache 2.0                 # Required for train/validation
    gitlab_identifier:                  # Required for train/validation
      dataset_name: my_dataset
      version: 0.0.1
      artifact_fpath: train.jsonl
    num_repeats: 1                      # Optional: repeat dataset N times

  # Validation dataset
  - name: validation
    type: validation
    jsonl_fpath: path/to/validation.jsonl
    license: Apache 2.0
    gitlab_identifier:
      dataset_name: my_dataset
      version: 0.0.1
      artifact_fpath: validation.jsonl
```

**Dataset types:**

| Type | Purpose | Required Fields |
|------|---------|-----------------|
| `example` | Quick testing, demonstrations | Just `name`, `type`, `jsonl_fpath` |
| `train` | RL training data | + `license`, `gitlab_identifier` |
| `validation` | Evaluation during/after training | + `license`, `gitlab_identifier` |

**Valid license values:** `Apache 2.0`, `MIT`, `Creative Commons Attribution 4.0 International`, `Creative Commons Attribution-ShareAlike 4.0 International`, `TBD`

---

## How Configs Relate to Core Abstractions

NeMo Gym has three core abstractions. Each maps to a server type in configuration:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION FILE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  responses_api_models/        ←→  MODEL                                 │
│    Wraps LLM inference             Stateless text generation            │
│    (OpenAI, vLLM, etc.)            No conversation memory               │
│                                                                         │
│  resources_servers/           ←→  RESOURCE                              │
│    Provides tools                  Tools agents can call                │
│    Implements verify()             Verification/reward logic            │
│                                                                         │
│  responses_api_agents/        ←→  AGENT                                 │
│    Orchestrates interactions       Routes requests to model             │
│    Manages multi-turn              Handles tool calling                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

For more on these abstractions, see {doc}`core-abstractions`.

---

## Complete Annotated Example

Here's a full config file with detailed annotations:

```yaml
# ═══════════════════════════════════════════════════════════════════════════
# RESOURCE SERVER
# Provides the get_weather tool and verification logic
# ═══════════════════════════════════════════════════════════════════════════
example_simple_weather:               # Level 1: Server Instance ID
  resources_servers:                  # Level 2: Server Type
    example_simple_weather:           # Level 3: Implementation folder
      entrypoint: app.py              # Script to run
      domain: agent                   # Required: task category
      verified: false                 # Optional: training validation status
      description: Basic single-step tool calling  # Optional: for docs

# ═══════════════════════════════════════════════════════════════════════════
# AGENT SERVER
# Orchestrates model ↔ resource interactions
# ═══════════════════════════════════════════════════════════════════════════
simple_weather_simple_agent:          # Level 1: Server Instance ID
  responses_api_agents:               # Level 2: Server Type
    simple_agent:                     # Level 3: Implementation folder
      entrypoint: app.py

      # Connect to the resource server above
      resources_server:
        type: resources_servers
        name: example_simple_weather  # References Level 1 of the resource server

      # Connect to a model server (defined in another config file)
      model_server:
        type: responses_api_models
        name: policy_model            # Must exist in loaded configs

      # Data for rollout collection
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/example_simple_weather/data/example.jsonl
```

---

## Quick Reference

| Concept | Description |
|---------|-------------|
| **Server Instance ID** | Level 1 key — unique name you reference elsewhere |
| **Server Type** | Level 2 key — `resources_servers`, `responses_api_models`, or `responses_api_agents` |
| **Implementation Type** | Level 3 key — folder name containing the actual code |
| **`domain`** | Required for resource servers — categorizes the task |
| **`verified`** | Optional — indicates validation through training |
| **Server Reference** | `type` + `name` pattern connecting servers |

---

## Related Pages

- {doc}`configuration-system` — Configuration priority, env.yaml, and CLI overrides
- {doc}`core-abstractions` — Understanding Model, Resource, and Agent servers
- {doc}`tutorials/creating-resource-server` — Step-by-step guide to creating resource servers
