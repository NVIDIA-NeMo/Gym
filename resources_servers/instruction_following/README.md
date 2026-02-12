# Instruction Following Resource Server

### Overview

Evaluates language model responses against instruction-following criteria using Open-Instruct and IFEval constraints.

### Input schema
- `responses_create_params`: OpenAI Responses create params.
- `instruction_id_list` (required): List of instruction types.
- `prompt` (required): The prompt with the instructions in natural text.
- `kwargs` (required): Details of the instruction.
- `grading_mode`: Either binary (reward is whether all instructions must be followed) or fraction (reward is fraction of instructions followed).

The dataset can be found at https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following 

### Example dataset row (dataset format)
```json
{
  "id": 8654,
  "instruction_id_list": [
    "paragraphs:paragraphs",
    "detectable_format:title",
    "length_constraints:number_words"
  ],
  "prompt": "how can i install jayhorn step by step\n\nYour response should comply with these constraints:\n1. There should be 2 paragraphs. Paragraphs are separated with the markdown divider: ***\n2. Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.\n3. Answer with less than 204 words.",
  "kwargs": [
    {},
    null,
    {
      "num_words": 204,
      "relation": "less than"
    }
  ],
  "responses_create_params": {
    "input": [
      {
        "role": "user",
        "content": "how can i install jayhorn step by step\n\nYour response should comply with these constraints:\n1. There should be 2 paragraphs. Paragraphs are separated with the markdown divider: ***\n2. Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.\n3. Answer with less than 204 words."
      }
    ],
    "tools": [],
    "parallel_tool_calls": false
  }
}
```

### Usage

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/instruction_following/configs/instruction_following.yaml"
ng_run "+config_paths=[$config_paths]" "+simple_agent.responses_api_agents.simple_agent.resources_server.name=instruction_following"

ng_download_dataset_from_gitlab \
    +dataset_name=instruction_following \
    +version=0.0.1 \
    +artifact_fpath=instruction_following.jsonl \
    +output_fpath=data/instruction_following.jsonl

ng_collect_rollouts \
    +agent_name=simple_agent \
    +input_jsonl_fpath=data/instruction_following.jsonl \
    +output_jsonl_fpath=data/instruction_following_output.jsonl +limit=5
```

### Rollout example

```json
{"responses_create_params": {"input": [{"content": "how can i install jayhorn step by step\n\nYour response should comply with these constraints:\n1. There should be 2 paragraphs. Paragraphs are separated with the markdown divider: ***\n2. Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.\n3. Answer with less than 204 words.", "role": "user"}], "parallel_tool_calls": false, "tools": []}, "response": {"id": "resp_68a636ef31dc8197b9ac160155bc709800bce3df79b195bd", "created_at": 1755723503.0, "error": null, "incomplete_details": null, "instructions": null, "metadata": {}, "model": "gpt-4.1-2025-04-14", "object": "response", "output": [{"id": "msg_68a636efd4fc8197a14689b070111cef00bce3df79b195bd", "content": [{"annotations": [], "text": "<<Installing JayHorn: Step-by-Step Guide>>\n\nTo install JayHorn, first ensure you have Java (Java 8 or later) and Maven installed on your system. Download or clone the JayHorn repository from GitHub using the command `git clone https://github.com/jayhorn/jayhorn.git` and navigate to the project directory with `cd jayhorn`. Next, build JayHorn using Maven by running `mvn package`, which compiles the source code and creates a runnable JAR file in the `target` folder.\n\n***\nAfter building, you can run JayHorn with the command `java -jar target/jayhorn-*-jar-with-dependencies.jar <your-java-file>`. Replace `<your-java-file>` with the path to the Java file you wish to verify. For additional usage instructions or configuration, refer to the `README.md` in the JayHorn repo. If you encounter dependency or build errors, ensure Maven and Java versions are correctly set and consult JayHorn\u2019s GitHub issues for troubleshooting support.", "type": "output_text", "logprobs": []}], "role": "assistant", "status": "completed", "type": "message"}], "parallel_tool_calls": false, "temperature": 1.0, "tool_choice": "auto", "tools": [], "top_p": 1.0, "background": false, "max_output_tokens": null, "max_tool_calls": null, "previous_response_id": null, "prompt": null, "reasoning": {"effort": null, "generate_summary": null, "summary": null}, "service_tier": "default", "status": "completed", "text": {"format": {"type": "text"}, "verbosity": "medium"}, "top_logprobs": 0, "truncation": "disabled", "usage": {"input_tokens": 78, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 208, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 286}, "user": null, "prompt_cache_key": null, "safety_identifier": null, "store": true}, "reward": 1.0, "follow_all_instructions": true, "follow_instruction_list": [true, true, true], "kwargs": [{}, null, {"num_words": 204, "relation": "less than"}], "instruction_id_list": ["paragraphs:paragraphs", "detectable_format:title", "length_constraints:number_words"], "prompt": "how can i install jayhorn step by step\n\nYour response should comply with these constraints:\n1. There should be 2 paragraphs. Paragraphs are separated with the markdown divider: ***\n2. Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.\n3. Answer with less than 204 words."}
```

### Reward Profiling

We run reward profiling with Qwen/Qwen3-30B-A3B-Instruct-2507. We evaluate the model in binary mode, and generate 16 rollouts for 500 prompts in the dataset. We have the following statistics:
- Percent of prompts with all zero rewards: 54.6
- Percent of prompts with all one rewards: 26.8
- Percent of prompts with mixed rewards: 18.6
- Average rollout reward: 37.09

## Testing

```bash
ng_test +entrypoint=resources_servers/instruction_following/
```


# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
