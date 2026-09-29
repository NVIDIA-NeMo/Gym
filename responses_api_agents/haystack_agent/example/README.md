# Haystack Agent with Local Tools

This example runs a serialized Haystack `Agent` as a NeMo Gym agent loop. The
tool-enabled pipeline includes two Haystack-local tools:

- `wiki_search` — Tavily search restricted to `wikipedia.org`.
- `calculator` — basic arithmetic.

The Agent uses `NeMoGymResponsesChatGenerator` with the `policy_model` model
server, exits on text, and permits at most 20 agent steps. Its
`system_prompt` is unset, so task context comes from the incoming Responses API
input rather than an example-specific prompt.

Local tools run in the Haystack Agent process. They are distinct from
request-scoped HTTP tools supplied by a Resources Server and MCP tools. For
example, a Workplace Assistant rollout can use its CRM, email, calendar,
project-management, and analytics tools in addition to these local tools.

## Files

- `build_pipeline.py` builds the tool-enabled pipeline.
- `example_pipeline_with_tools.yaml` is its committed serialized output.
- `example_tools.py` defines `calculator` and `wiki_search`.
- `../configs/pipeline.yaml` is the minimal, no-local-tools baseline.

## Setup

Install the Haystack Agent requirements and the optional Tavily integration:

```bash
cd responses_api_agents/haystack_agent
uv pip install -r requirements.txt tavily-haystack
export TAVILY_API_KEY=<your-key>
```

The model endpoint must support Responses API tool calls. Configure it through
the usual Gym settings, including `policy_base_url`, `policy_api_key`, and
`policy_model_name`.

## Regenerate the pipeline

From `responses_api_agents/haystack_agent`, run:

```bash
uv run python example/build_pipeline.py
```

This overwrites `example/example_pipeline_with_tools.yaml`. Regenerate it after
changing the Agent or either local tool.

## Select a pipeline

Set `pipeline_yaml` in a Gym configuration. Paths are relative to
`responses_api_agents/haystack_agent`.

```yaml
responses_api_agents:
  haystack_agent:
    resources_server:
      type: resources_servers
      name: <resources_server_name>
    model_server:
      type: responses_api_models
      name: policy_model
    pipeline_yaml: example/example_pipeline_with_tools.yaml
```

Use `configs/pipeline.yaml` instead for the baseline without local tools.

Run a configured environment, then collect rollouts:

```bash
gym eval run --no-serve \
  --agent <agent_id> \
  --input <dataset.jsonl> \
  --output results/haystack_agent_rollouts.jsonl
```

The output JSONL preserves model messages, function calls, tool outputs, and
the final response for verifier inspection.
