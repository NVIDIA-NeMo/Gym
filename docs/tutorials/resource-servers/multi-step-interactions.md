(tutorial-data-extraction-server)=

# Build a Data Extraction Server

In {doc}`Build a Weather API Server <simple-tool-calling>`, you built a single-tool server. Now you'll build a data extraction server where agents must call multiple tools to gather and synthesize information.

:::{card}

**Goal**: Build a multi-step extraction server where agents query multiple data sources.

^^^

**In this tutorial, you will**:

1. Create multiple related tools (`get_synonyms`, `get_value`)
2. Design tasks requiring sequential tool calls
3. Implement verification that checks all required data was gathered
4. Walk away with a server for multi-step extraction tasks

:::

:::{button-ref} /tutorials/resource-servers/index
:color: secondary
:outline:
:ref-type: doc

← Resource Server Patterns
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed {doc}`simple-tool-calling`
- ✅ Understanding of tool sequencing concepts
- ✅ NeMo Gym installed and working

**What you'll build**: A data extraction server where agents must look up synonyms for a term, then retrieve values for each synonym, and aggregate the results.

:::{tip}
**Reference implementation**: `resources_servers/example_multi_step/`
:::

---

## 1. Understand What We're Building

The extraction task requires multiple steps:

1. User asks: "Get all values for 'happiness'"
2. Agent calls `get_synonyms("happiness")` → returns `["joy", "delight", "pleasure"]`
3. Agent calls `get_value("joy")` → returns `42`
4. Agent calls `get_value("delight")` → returns `38`
5. Agent calls `get_value("pleasure")` → returns `45`
6. Agent reports: "The values are: joy=42, delight=38, pleasure=45"
7. Verification checks: Did the agent get ALL values?

```{mermaid}
sequenceDiagram
    participant Agent
    participant Server
    Agent->>Server: get_synonyms("happiness")
    Server-->>Agent: ["joy", "delight", "pleasure"]
    Agent->>Server: get_value("joy")
    Server-->>Agent: 42
    Agent->>Server: get_value("delight")
    Server-->>Agent: 38
    Agent->>Server: get_value("pleasure")
    Server-->>Agent: 45
    Agent->>Server: Final answer with all values
    Server-->>Agent: ✅ reward=1.0 (all 3 found)
```

**✅ Success Check**: You understand why this requires multiple sequential calls.

---

## 2. Create the Tools

<!-- SME: Extract and adapt from example_multi_step/app.py -->

```python
# app.py
from nemo_gym.base_resources_server import SimpleResourcesServer, BaseResourcesServerConfig

class ExtractionServerConfig(BaseResourcesServerConfig):
    pass

class ExtractionServer(SimpleResourcesServer):
    
    def __init__(self, config: ExtractionServerConfig):
        super().__init__(config)
        
        # Data: term → synonyms
        self.synonyms = {
            "happiness": ["joy", "delight", "pleasure"],
            "sadness": ["sorrow", "grief", "melancholy"],
            "anger": ["rage", "fury", "wrath"],
        }
        
        # Data: synonym → value
        self.values = {
            "joy": 42, "delight": 38, "pleasure": 45,
            "sorrow": 12, "grief": 8, "melancholy": 15,
            "rage": 88, "fury": 92, "wrath": 85,
        }
    
    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_synonyms",
                    "description": "Get synonyms for a term",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "term": {"type": "string", "description": "The term to find synonyms for"}
                        },
                        "required": ["term"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_value",
                    "description": "Get the numeric value for a specific word",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string", "description": "The word to get the value for"}
                        },
                        "required": ["word"]
                    }
                }
            }
        ]
    
    def call_tool(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "get_synonyms":
            term = tool_args.get("term", "").lower()
            synonyms = self.synonyms.get(term, [])
            return f"Synonyms: {', '.join(synonyms)}" if synonyms else "No synonyms found"
        
        elif tool_name == "get_value":
            word = tool_args.get("word", "").lower()
            value = self.values.get(word)
            return f"Value for '{word}': {value}" if value else f"No value found for '{word}'"
        
        return "Unknown tool"
```

**✅ Success Check**: You have two tools that work together.

---

## 3. Implement Multi-Step Verification

Verification must check that the agent gathered ALL required values:

```python
# Add to app.py

def verify(self, responses_create_params: dict, output: list) -> float:
    """Verify all synonym values were extracted."""
    
    # Find which term was asked about
    user_message = responses_create_params["input"][0]["content"].lower()
    
    asked_term = None
    for term in self.synonyms:
        if term in user_message:
            asked_term = term
            break
    
    if not asked_term:
        return 0.0
    
    # Get expected synonyms and their values
    expected_synonyms = self.synonyms[asked_term]
    expected_values = {syn: self.values[syn] for syn in expected_synonyms}
    
    # Extract final response
    final_response = ""
    for msg in reversed(output):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_response = msg["content"].lower()
            break
    
    # Count how many values were correctly reported
    found_count = 0
    for synonym, value in expected_values.items():
        # Check if both the synonym and its value appear
        if synonym in final_response and str(value) in final_response:
            found_count += 1
    
    # Reward based on completeness
    return found_count / len(expected_values)
```

**✅ Success Check**: Returns 1.0 only if ALL synonym values are reported.

---

## 4. Create Test Data

Create `data/example.jsonl`:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "Find all values for the synonyms of 'happiness'"}], "model": "gpt-4"}}
{"responses_create_params": {"input": [{"role": "user", "content": "Look up the values for each synonym of 'sadness'"}], "model": "gpt-4"}}
{"responses_create_params": {"input": [{"role": "user", "content": "Get the numeric values for all words related to 'anger'"}], "model": "gpt-4"}}
```

---

## 5. Run and Test

```bash
ng_collect_rollouts +agent_name=extraction_simple_agent \
    +input_jsonl_fpath=resources_servers/extraction/data/example.jsonl \
    +output_jsonl_fpath=results/extraction_rollouts.jsonl \
    +limit=3
```

**✅ Success Check**: Reward=1.0 means agent called all necessary tools and reported all values.

---

## Adapt for Your Use Case

This pattern works for any multi-step data gathering:

- **Research assistant**: `search(query)` → `get_details(result_id)` for each result
- **Inventory check**: `list_warehouses()` → `check_stock(warehouse_id)` for each
- **Document analysis**: `list_sections(doc)` → `extract_entities(section)` for each

---

## Learn More

- {doc}`/training/verification/index` — Designing reward functions for multi-step tasks
- {doc}`/training/rollout-collection/configure-sampling` — Sampling for multi-step rollouts

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Stateful Counter
:link: counter-game-server
:link-type: doc

Add state that persists across tool calls.
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` Math Verifier
:link: math-verifier-server
:link-type: doc

Use an LLM to verify open-ended outputs.
:::

::::
