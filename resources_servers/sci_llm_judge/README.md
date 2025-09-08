# Scientific Knowledge LLM-as-judge Resources Server

### Overview
Uses an LLM as a judge to compare a model’s generated answer against the expected answer.

Prompt and labels are configurable via config.

### Key config fields
- `judge_system_message`: optional system message. If omitted, no system message is added.
- `judge_prompt_template` (required): user prompt template. Placeholders: `{question}`, `{expected_answer}`, `{generated_answer}`.
- `judge_equal_label` / `judge_not_equal_label`: labels the judge must output. Defaults to `[[A=B]]` and `[[A!=B]]`.
- `check_twice_swap` (bool, default false): if true, after an initial equal verdict, performs a second judge call swapping expected and generated answers to reduce bias.

### Input schema
Accepts the same outer request structure as other resources servers:
- `responses_create_params`: the original model query (used here to extract a question/context string from user messages for the judge prompt).
- `response`: the model output in OpenAI Responses schema. The last assistant message text is used as the generated answer.
- `expected_answer`: gold answer to compare against.

### Example config
```yaml
sci_llm_judge:
  resources_servers:
    sci_llm_judge:
      entrypoint: app.py

sci_llm_judge_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: sci_llm_judge
      model_server:
        type: responses_api_models
        name: openai_model
      datasets:
      - name: sciq_validation
        type: example
        jsonl_fpath: resources_servers/sci_llm_judge/data/sciq_validation.jsonl
```

### Usage
Spin up with a judge model and prompt:
```bash
config_paths="resources_servers/sci_llm_judge/configs/sci_llm_judge.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[$config_paths]" \
  +sci_llm_judge.resources_servers.sci_llm_judge.judge_model_server.name=openai_model \
  +sci_llm_judge.resources_servers.sci_llm_judge.judge_responses_create_params.max_output_tokens=256 \
  +sci_llm_judge.resources_servers.sci_llm_judge.judge_system_message="You are a careful arbiter." \
  +sci_llm_judge.resources_servers.sci_llm_judge.judge_prompt_template="""
<|Problem|>\n{question}\n\n<|Gold|>\n{expected_answer}\n\n<|Prediction|>\n{generated_answer}\n"""
```

Then query via any agent; verification happens with `/verify` on this server when evaluating rollouts.

### Notes
- By default (`check_twice_swap=false`), the server performs a single judge pass. If the verdict is equal, reward is 1 and one evaluation is returned; if not equal, reward is 0 and one evaluation is returned.
- If `check_twice_swap=true` and the first pass is equal, a second pass is performed with expected and generated answers swapped. Reward is 1 only if the second pass is also equal; in this case, two evaluations are returned.
- If the judge output doesn’t include either label, it defaults to not-equal.

## Licensing
Code: Apache 2.0   
Data: CC-BY-NC-3.0 (Examples from https://huggingface.co/datasets/allenai/sciq)


