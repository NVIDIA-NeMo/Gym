(langgraph-agent)=

# LangGraph Agent

The LangGraph agent adapter lets you run [LangGraph](https://github.com/langchain-ai/langgraph) agents as NeMo Gym agent servers. It enables bringing existing LangGraph agents into NeMo Gym, for use with resources servers to construct new environments and train agents on diverse tasks. 

`reflection_agent` is used as an example here. Additional example implementations (`orchestrator_agent`, `rewoo_agent`, `parallel_thinking_agent`) are included in the source, but may not produce monotonic trajectories, and therefore will not work with NeMo RL by default. They work for evaluations, serve as examples, and can sometimes be adapted for training depending on the RL framework and use case.

---

## Base Adapter

`LangGraphAgentAdapter` (`app.py`) is an abstract base class. Subclass it and implement three methods:

```python
class LangGraphAgentAdapter(SimpleResponsesAPIAgent):

    @abstractmethod
    def build_graph(self) -> Any:
        # Return a compiled LangGraph graph
        ...

    @abstractmethod
    async def get_initial_state(
        self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict
    ) -> dict:
        # Build the initial graph state from the incoming request
        ...

    @abstractmethod
    def extract_outputs(self, final_state: dict) -> list:
        # Extract the output messages from the final graph state
        ...
```

The adapter's `responses()` method invokes the graph (`graph.ainvoke`), propagates session cookies, and returns a `NeMoGymResponse`. The `run()` method handles the full rollout: `seed_session`, `responses`, `verify`.

State must include `last_policy_response` (a `NeMoGymResponse`) or you must override `extract_model_response()`.

---

## Reflection Agent

**File:** `reflection_agent.py`
**Config:** `configs/reflection_agent.yaml`

After each generation step, the agent checks for an `<answer>` tag or whether `max_reflections` has been reached. If so it stops, otherwise it critiques the answer and generates again. With `max_reflections=2` you get at most 3 generate calls and 2 critique calls.

When using the agent with a resources server, the convention is to put the config in the resources server's `configs/` folder. Recall that an environment is composed of agent + resources server + datasets together. For agent-only environments with no resources server, the config lives in the agent's `configs/` folder.

**Example config:**

```yaml
reflection_agent:
  responses_api_agents:
    reflection_agent:
      entrypoint: reflection_agent.py
      resources_server:
        type: resources_servers
        name: ???          # set to your resources server name
      model_server:
        type: responses_api_models
        name: policy_model
      max_reflections: 2
```

---

## Quick Start

Start the servers. This example uses the `reasoning_gym` resources server:

```bash
ng_run "+config_paths=[resources_servers/reasoning_gym/configs/reflection_agent.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Collect rollouts:

```bash
ng_collect_rollouts \
    +agent_name=reasoning_gym_reflection_agent \
    +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=example_rollouts.jsonl \
    +limit=1
```

---

## Implementing a Custom LangGraph Agent

1. Subclass `LangGraphAgentAdapter` from `app.py`.
2. Implement `build_graph()`: return a compiled `StateGraph`.
3. Implement `get_initial_state()`: convert the input data `NeMoGymResponseCreateParamsNonStreaming` body into the graph's initial state dict.
4. Implement `extract_outputs()`: return `final_state["policy_outputs"]`, the list of all model outputs.
5. Store the most recent model call response as `last_policy_response` in state. The adapter replaces the `.output` field with the full `policy_outputs` trajectory before returning, the same pattern as `simple_agent`. The full trajectory across all turns is what is returned for training and verification.
6. Propagate `cookies` through all `server_client.post()` calls and carry `resp.cookies` forward in state.
7. Create a YAML config in `configs/` pointing to your entrypoint file.

```python
from app import LangGraphAgentAdapter, LangGraphAgentConfig
from langgraph.graph import END, StateGraph

class MyAgentConfig(LangGraphAgentConfig):
    my_param: int = 3

class MyAgent(LangGraphAgentAdapter):
    config: MyAgentConfig

    def build_graph(self):
        graph = StateGraph(...)
        # add nodes, edges
        return graph.compile()

    async def get_initial_state(self, body, cookies):
        return {"messages": [...], "policy_outputs": [], "cookies": cookies, ...}

    def extract_outputs(self, final_state):
        return final_state["policy_outputs"]

if __name__ == "__main__":
    MyAgent.run_webserver()
```
