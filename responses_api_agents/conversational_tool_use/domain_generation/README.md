# Conversational Tool-Use Domain Generation

This package exposes domain sampling as a Gym `SimpleResponsesAPIAgent`. With the default `followup_count: 1`, one
`/run` request:

1. send the input prompt as one user message to `/v1/chat/completions`;
2. parse the response with the package's non-validating JSON parser;
3. append the first response's domain names to the follow-up prompt;
4. make one follow-up completion and return both candidate batches.

Additional follow-ups use the original prompt plus all names collected so far. Set `followup_count: 0` to make only the
initial call.

Internal chat requests contain only `messages`; response sampling fields are not forwarded. Model-server transport
errors remain rollout failures. JSON parsing errors become an empty batch, later configured follow-ups still run, and a
normal return has reward `1.0`. Candidate values are not normalized or schema-validated.

## Input and result

The input row has the initial prompt in `responses_create_params.input`:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "<domain prompt>"}]
  },
  "agent_ref": {
    "type": "responses_api_agents",
    "name": "conversational_tool_use_domain_generation"
  }
}
```

Use one rollout for each sampler. Collect as many independent rollouts as the dataset requires.
`data/example.jsonl` contains the active prompt.

The result extends Gym's normal verify response with:

```text
result.candidates
generation_trace.protocol_version
generation_trace.request_index
generation_trace.followup_count
generation_trace.phases[initial|followup].request
generation_trace.phases[initial|followup].response
generation_trace.phases[initial|followup].parsed_value
generation_trace.phases[initial|followup].parse_error
```

The model calls use Gym's rollout-prefixed URL when observability is enabled, so the standard model-call capture is
correlated with the sampler rollout.

The default one-follow-up trace retains `domain-generation/v1`. Configurations with zero or multiple follow-ups emit
`domain-generation/v2`; the trace model accepts both versions.

## Prompts

`prompts/domain_generation.txt` is the active domain-generation prompt.

## Configuration

Start the sampler and any supported model server. This example uses the generic OpenAI-compatible model server:

```bash
gym env start \
  --config responses_api_agents/conversational_tool_use/domain_generation/configs/conversational_tool_use_domain_generation.yaml \
  --model-type openai_model \
  --model-url "$MODEL_BASE_URL" \
  --model "$MODEL_NAME" \
  '+policy_api_key=${oc.env:MODEL_API_KEY}'
```

`gym env start` stays in the foreground. In a separate terminal, collect from an explicit input JSONL:

```bash
gym eval run --no-serve \
  --agent conversational_tool_use_domain_generation \
  --input responses_api_agents/conversational_tool_use/domain_generation/data/example.jsonl \
  --output /tmp/conversational_tool_use/domain_rollouts.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```

The config creates `domain_generation_model` as an independent copy of Gym's standard `policy_model` instance supplied
by `--model-type`. Override that copy when domain sampling should use a different model from later stages.

Agent controls:

| Setting | Default | Meaning |
|---|---:|---|
| `model_server` | `domain_generation_model` | Model-server instance used for internal chat completions |
| `followup_count` | `1` | Number of sequential follow-up calls after the initial call; `0` disables follow-ups |

The input row owns the initial prompt. The copied model server owns model and sampling defaults such as temperature,
top-p, output tokens, and provider-level concurrency. `gym eval run --num-repeats` controls the number of independent
samplers, while `--concurrency` controls how many sampler rollouts run concurrently. Unknown agent settings are rejected
so misspelled controls cannot silently fall back to defaults.

## Policy/tool materialization

`materialize.py` converts the domain rollout JSONL into policy/tool agent inputs:

```bash
python -m responses_api_agents.conversational_tool_use.domain_generation.materialize \
  --input-file /path/to/domain-rollouts.jsonl \
  --output-file /path/to/policy-tool-inputs.jsonl \
  --profile general
```

Each candidate is preserved unchanged under `domain`. The output adds an empty Gym `responses_create_params.input`, a
stable candidate ID, source lineage, and the `conversational_tool_use_policy_tool_generation` agent reference. IDs and
lineage include available Gym task, rollout, and retry-attempt coordinates.
`--profile` is required and must be `general` or `proactive`; its value is stored on every output row so the same domain
rollouts can materialize inputs for both policy profiles. Candidates are deduplicated by
`candidate["name"].casefold()` with the first occurrence winning. Punctuation and whitespace are significant, and
default output order follows the rollout file and candidate-list order.

Shuffling is opt-in and requires an explicit integer:

```bash
python -m responses_api_agents.conversational_tool_use.domain_generation.materialize \
  --input-file /path/to/domain-rollouts.jsonl \
  --output-file /path/to/policy-tool-inputs.jsonl \
  --profile proactive \
  --shuffle-seed 1
```

The shuffle uses an isolated `random.Random(seed)` instance.

## Tests

```bash
gym env test \
  +entrypoint=responses_api_agents/conversational_tool_use/domain_generation \
  +should_validate_data=false
```
