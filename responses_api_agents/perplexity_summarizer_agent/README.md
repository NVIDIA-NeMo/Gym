# Perplexity Summarizer Agent

Search-augmented QA evaluation agent for Perplexity summarizer models. Implements a custom agent loop with tool-call counting that forces text-only responses once the configured limit is reached.

## Datasets

| Dataset | Eval Focus | Rollout Type | max_tool_calls | Judge |
|---------|-----------|-------------|----------------|-------|
| perplexity_user_if | Instruction following | Pre-baked trajectory | 0 | LLM (followed: yes/no) |
| perplexity_search | Search quality | Pre-baked trajectory | 0 | Reward model (not yet implemented) |
| perplexity_chat | Chat quality (no tools) | Pre-baked trajectory | 0 | Reward model (not yet implemented) |
| perplexity_abstention | Abstention behavior | Pre-baked trajectory | 0 | LLM (followed: yes/no) |
| perplexity_frames | Multi-hop reasoning | Fresh rollout | 3 | LLM (correct: yes/no) |
| perplexity_facts_grounding | Factual grounding | Fresh rollout | 3 | LLM (correct: yes/no) |

## How It Works

### Pre-baked Trajectories

Datasets like `perplexity_user_if`, `perplexity_search`, `perplexity_chat`, and `perplexity_abstention` use pre-baked trajectories. The model receives a full conversation including prior tool calls and results, then generates only the final summary. `max_tool_calls=0` forces `tool_choice="none"` from the start.

### Fresh Rollouts

Datasets like `perplexity_frames` and `perplexity_facts_grounding` use fresh rollouts. The model searches the web via the Perplexity Search API and builds its own trajectory, up to `max_tool_calls=3`.

### Tool-Call Suppression

Once the tool-call limit is reached, three layers prevent the model from making further tool calls:

1. **`tool_choice="none"`** -- removes tools from the API request entirely.
2. **`bad_words` filter** -- the vLLM server blocks `<tool_call>` / `</tool_call>` tokens from being generated. This is a vLLM-native parameter passed via `extra_body` in the request. It is not part of the OpenAI API spec and requires a vLLM-compatible endpoint.
3. **Prompt hint** -- the string `<system>No more tool calls allowed.</system>` is appended to the output of the last tool result. This gives the model an explicit signal to generate a text summary instead of attempting more tool calls.

### Default Generation Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| temperature | 1.0 | Set per-dataset in YAML |
| top_p | 1.0 | Set per-dataset in YAML |
| max_output_tokens | 32768 | Set per-dataset in YAML |
| bad_words | `["<tool_call>", "</tool_call>"]` | vLLM-specific, applied when tool_choice="none" |
| enable_thinking | true | Set in agent config (policy model) |

## Configuration

The agent config (policy model settings) is separate from per-dataset resource server configs. Compose them together via `config_paths`:

```
config_paths="responses_api_agents/perplexity_summarizer_agent/configs/perplexity_summarizer_agent.yaml,\
resources_servers/perplexity_summarizer/configs/perplexity_summarizer_user_if.yaml"
```

## Running with W&B

```bash
WANDB_PROJECT=my-project
EXPERIMENT_NAME=perplexity_summarizer/model-name
config_paths="responses_api_agents/perplexity_summarizer_agent/configs/perplexity_summarizer_agent.yaml,\
resources_servers/perplexity_summarizer/configs/perplexity_summarizer_user_if.yaml"
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    +wandb_project=$WANDB_PROJECT \
    +wandb_name=$EXPERIMENT_NAME \
    ++output_jsonl_fpath=results/$EXPERIMENT_NAME.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++resume_from_cache=true \
    ++split=validation \
    '++policy_model=${swap_key:model-name}'
```

## License

NVIDIA Internal Use Only. Do Not Distribute.
