# Conversational Tool-Use Domain Generation

This package exposes domain sampling as a Gym `SimpleResponsesAPIAgent`. One `/run` request is one sampler:

1. send the input prompt as one user message to `/v1/chat/completions`;
2. parse the response with the package's non-validating JSON parser;
3. append the first response's domain names to the follow-up prompt;
4. make the second chat completion and return both candidate batches.

The two chat requests contain only `messages`; response sampling fields are not forwarded. Model-server transport errors
remain rollout failures. JSON parsing errors become an empty batch, the follow-up still runs, and a normal return has
reward `1.0`. Candidate values are not normalized or schema-validated.

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
generation_trace.phases[initial|followup].request
generation_trace.phases[initial|followup].response
generation_trace.phases[initial|followup].parsed_value
generation_trace.phases[initial|followup].parse_error
```

The model calls use Gym's rollout-prefixed URL when observability is enabled, so the standard model-call capture is
correlated with the sampler rollout.

## Prompts

`prompts/domain_generation.txt` is the active domain-generation prompt. Additional package-owned prompt variants are
kept under `prompts/archive/`; every prompt filename is whitespace-free.

## Configuration

Combine `configs/conversational_tool_use_domain_generation.yaml` with a model-server configuration defining
`policy_model`, then collect the desired number of rollouts from `data/example.jsonl`.

## Policy/tool materialization

`materialize.py` converts the domain rollout JSONL into policy/tool agent inputs:

```bash
python -m responses_api_agents.conversational_tool_use_domain_generation.materialize \
  --input-file /path/to/domain-rollouts.jsonl \
  --output-file /path/to/policy-tool-inputs.jsonl \
  --profile general
```

Each candidate is preserved unchanged under `domain`. The output adds an empty Gym `responses_create_params.input` and
the `conversational_tool_use_policy_tool_generation` agent reference. `--profile` is required and must be `general` or
`proactive`; its value is stored on every output row so the same domain rollouts can materialize inputs for both policy
profiles. Candidates are deduplicated by `candidate["name"].casefold()` with the first occurrence winning. Punctuation
and whitespace are significant, and default output order follows the rollout file and candidate-list order.

Shuffling is opt-in and requires an explicit integer:

```bash
python -m responses_api_agents.conversational_tool_use_domain_generation.materialize \
  --input-file /path/to/domain-rollouts.jsonl \
  --output-file /path/to/policy-tool-inputs.jsonl \
  --profile proactive \
  --shuffle-seed 1
```

The shuffle uses an isolated `random.Random(seed)` instance.

## Tests

```bash
python -m pytest -p no:cacheprovider \
  responses_api_agents/conversational_tool_use_domain_generation/tests
```
