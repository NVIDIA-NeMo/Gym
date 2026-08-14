# Haystack Agent with Local Tools

This example shows how to use a [Haystack Agent](https://docs.haystack.deepset.ai/docs/agent)
as the agent loop in NeMo Gym. It builds a small Haystack pipeline, adds two
local tools, serializes the pipeline to YAML, and selects that pipeline through
the NeMo Gym Haystack Agent configuration for rollout collection and
evaluation.

GPQA Diamond is used as an example workload to compare two configurations:

- **Baseline:** the minimal Haystack pipeline, with no Haystack-local tools.
- **Tool-enabled:** a pipeline with a basic calculator and Tavily web search,
  exposed to the model as `calculator` and `wiki_search`.

The same approach works with other NeMo Gym Resources Servers and datasets.

## How the integration works

The NeMo Gym `HaystackAgent` loads a serialized Haystack
[`Pipeline`](https://docs.haystack.deepset.ai/docs/pipelines). The pipeline must
contain a Haystack `Agent` backed by `NeMoGymResponsesChatGenerator`.

During a rollout:

1. NeMo Gym sends the task input to `HaystackAgent`.
2. `NeMoGymResponsesChatGenerator` converts between the Responses API items
   used by NeMo Gym and Haystack `ChatMessage` objects. It sends each model
   turn to the configured NeMo Gym model server through `/v1/responses`.
3. Haystack runs the model-and-tool loop until the model returns a text answer
   or the Agent reaches `max_agent_steps`.
4. NeMo Gym reconstructs the Responses API trajectory and sends the final
   response to the Resources Server for verification.

The example tools run in the Haystack Agent process. They are separate from
request-scoped HTTP tools and Resources Server tools exposed over MCP.

## Example files

- `build_pipeline.py` constructs and serializes the tool-enabled pipeline.
- `example_tools.py` defines the calculator and configures Tavily search.
- `../configs/pipeline.yaml` is the committed minimal pipeline used as the
  baseline.
- `../configs/example_pipeline_with_tools.yaml` is generated locally by
  `build_pipeline.py` and is not committed.
- `../configs/haystack_agent.yaml` shows the generic NeMo Gym agent
  configuration and its `pipeline_yaml` field.

## Prerequisites

Install the Haystack Agent dependencies in the environment used to launch the
agent. The tool-enabled example also requires the optional `tavily-haystack`
integration:

```bash
cd responses_api_agents/haystack_agent
uv pip install -r requirements.txt tavily-haystack
```

Set `TAVILY_API_KEY` in the terminal that starts the tool-enabled environment.
See the [Tavily quickstart](https://docs.tavily.com/documentation/quickstart)
for account and API-key setup.

```bash
export TAVILY_API_KEY=<your-key>
```

Your model endpoint must support tool calling through the Responses API.
Configure it using the usual NeMo Gym settings, such as `policy_base_url`,
`policy_api_key`, and `policy_model_name`.

## Build the tool-enabled pipeline

The pipeline is defined in Python so that ordinary Haystack tools and
components can be added directly:

```python
agent = Agent(
    chat_generator=NeMoGymResponsesChatGenerator(server_name="policy_model"),
    tools=[wiki_search_tool, calculator],
    system_prompt="...",
    exit_conditions=["text"],
    max_agent_steps=20,
)

pipeline = Pipeline()
pipeline.add_component("agent", agent)
```

The calculator uses Haystack's [`@tool`](https://docs.haystack.deepset.ai/docs/tool)
decorator. `TavilyWebSearchTool` comes from the optional Tavily integration; in
this example it is named `wiki_search`, returns one result, and restricts
searches to `wikipedia.org`.

Generate the pipeline YAML from the `haystack_agent` directory:

```bash
uv run python example/build_pipeline.py
```

The script calls `Pipeline.dumps()` and writes
`configs/example_pipeline_with_tools.yaml`. See Haystack's
[serialization guide](https://docs.haystack.deepset.ai/docs/serialization)
for details about saving and loading pipelines.

Regenerate the YAML whenever you change the Agent, its prompt, or its local
tools.

## Select the pipeline in NeMo Gym

In your local Gym configuration, wire the Haystack Agent to a Resources Server
and model server, then set `pipeline_yaml`. Paths are resolved relative to the
`responses_api_agents/haystack_agent` directory.

Use the committed minimal pipeline for a baseline:

```yaml
responses_api_agents:
  haystack_agent:
    resources_server:
      type: resources_servers
      name: <resources_server_name>
    model_server:
      type: responses_api_models
      name: policy_model
    pipeline_yaml: configs/pipeline.yaml
```

To enable the example tools, change only the selected pipeline:

```yaml
responses_api_agents:
  haystack_agent:
    resources_server:
      type: resources_servers
      name: <resources_server_name>
    model_server:
      type: responses_api_models
      name: policy_model
    pipeline_yaml: configs/example_pipeline_with_tools.yaml
```

Start the environment using your local configuration.

## Collect a rollout

After the environment is running, run `gym eval run --no-serve` in a separate
terminal window. Replace `<agent_id>` with the name assigned to the Haystack
Agent in your local Gym configuration.

```bash
gym eval run --no-serve \
  --agent <agent_id> \
  --input <dataset.jsonl> \
  --output results/haystack_agent_rollouts.jsonl
```

Gym writes the rollout JSONL and aggregate metrics beside the output file. The
rollout preserves the model's reasoning, function calls, tool outputs, and
final message as Responses API output items.

## Optional: reproduce the GPQA Diamond comparison

GPQA Diamond provides a convenient multiple-choice workload for comparing the
minimal and tool-enabled pipelines.

The full GPQA Diamond data is not committed. Generate the Gym-formatted split
from the repository root:

```bash
python3 resources_servers/gpqa_diamond/dataset_preprocess.py
```

This writes `resources_servers/gpqa_diamond/data/train.jsonl`. For a quick
smoke test, use the five examples in
`resources_servers/gpqa_diamond/data/example.jsonl`.

Run the same model, sampling parameters, input rows, and rollout count for both
pipelines. Start each environment with the corresponding `pipeline_yaml`, then
collect against the already-running environment:

```bash
gym eval run --no-serve \
  --agent <agent_id> \
  --input resources_servers/gpqa_diamond/data/train.jsonl \
  --output results/gpqa_diamond_haystack_<baseline-or-tools>.jsonl \
  --limit 64
```

Compare `mean/reward` or `pass@1/accuracy` in the generated aggregate-metrics
files.

### Example subset results

The example used `nvidia/nemotron-3-nano-30b-a3b` through the hosted
prototype API in the
[NVIDIA API Catalog](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b).
The catalog page provides an API reference, API-key setup, and a free endpoint
for experimentation.

One 64-question run produced:

| Configuration | Correct / 64 | Accuracy | No-answer rate | Tool calls |
| --- | ---: | ---: | ---: | ---: |
| Minimal pipeline | 47 | 73.44% | 3.13% | 0 |
| Calculator + Tavily search | 51 | 79.69% | 1.56% | 65 |
| Difference | **+4** | **+6.25 pp** | **-1.57 pp** | — |

Treat this as an example run, not a controlled estimate of tool impact. The
tool-enabled pipeline also changes the system prompt and `max_agent_steps`.
The prompt encourages the model to use the provided tools to verify its answer,
so the result does not isolate the effect of web search or the calculator.

## Extending the example

Haystack supports additional tool forms, including
[`ComponentTool`](https://docs.haystack.deepset.ai/docs/componenttool) and
[`PipelineTool`](https://docs.haystack.deepset.ai/docs/pipelinetool), as well
as a set of [ready-made tools](https://docs.haystack.deepset.ai/docs/ready-made-tools).
The NeMo Gym Haystack Agent can also combine local tools with request-scoped
HTTP tools or a `ContextAwareMCPToolset` when the Resources Server exposes tools
over MCP.
See the [Haystack Agents overview](https://docs.haystack.deepset.ai/docs/agents)
for the broader agent and tool model.
