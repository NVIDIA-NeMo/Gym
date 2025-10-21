# MCQA Resources Server

### Overview
Verifies multiple-choice QA (MCQA) model outputs. 
It consumes agent trajectories and returns a reward based on whether the assistant’s final output matches the gold answer.

### Input schema
Required fields:
- `responses_create_params`: OpenAI Responses create params
  - Use only a user message with the question and options (e.g., "A: … B: …").
- `options` (required): List of dicts mapping a single letter to option text, e.g. `[{"A": "Option_Text"}, {"B": "..."}]`
- `expected_answer` (required): The gold letter (single character). Must be one of the letters present in `options`

Optional fields:
- `grading_mode`: Answer extraction method (default: `"strict_single_letter_boxed"`)
- `template_metadata`: Custom regex pattern for answer extraction (see below)
- `uuid`: Unique identifier for the question
- `metadata`: Optional arbitrary metadata (not used for grading)

Notes:
- Letters are validated against the keys present in `options`
- While most datasets use A–D, any letter set is supported as long as it matches the provided options

### Grading modes
- `strict_single_letter_boxed` (default)
  - Requires exactly one uppercase letter inside a LaTeX box: `\boxed{...}`
  - Example it accepts: `Final: \boxed{ [C] }`
- `lenient_boxed`
  - First tries strict boxed letter.
  - If none, it looks ONLY inside the first `\boxed{...}` and tries to match the option text (normalized, case/whitespace-insensitive). Returns the corresponding letter only if exactly one option matches.
  - Example: `options = [{"A": "Circle"}, {"B": "Square}]`. This will match either `\boxed{Square}` or `\boxed{B}`
- `lenient_answer_colon`
  - Extracts content after `Answer:` (case-insensitive).
  - If it is a single allowed letter, use it.
  - Otherwise, if it exactly equals (after normalization) one option's text, use that letter.
  - Example: `options = [{"A": "Circle"}, {"B": "Square}]`. This will match `Answer: B` or `Answer: Square`.
    - Legacy from NeMo-RL

### Custom answer extraction (template_metadata) - OPTIONAL
For datasets with custom prompt formats, you can optionally use `template_metadata` with a custom regex pattern.

**Note:** If you don't need custom formats, see `data/example.jsonl` for standard usage with `grading_mode` only.

- `template_metadata.output_regex`: Custom regex pattern to extract the answer letter
  - **Optional field** - use only if you need custom answer formats
  - Takes **priority** over `grading_mode` when present
  - Case-insensitive matching (IGNORECASE flag)
  - Uses rightmost (last) match if multiple matches exist
  - Gracefully falls back to `grading_mode` if regex is invalid

**Example formats supported:**
- `"Option Selected: B"` → regex: `Option Selected:\s*([A-Za-z])`
- `"Final Choice: C"` → regex: `Final Choice:\s*([A-Za-z])`
- `"ANSWER IS D"` → regex: `ANSWER IS\s*([A-Za-z])`
- `"Answer: B"` (plain) → regex: `Answer\s*:\s*(?!Answer)\s*([A-Za-z])`

**Priority order:** `template_metadata.output_regex` (if present) → `grading_mode` (default)

### Example dataset row (standard format)
```json
{
    "responses_create_params":
    {
        "input":
        [
            {
                "role": "user",
                "content": "You should output your final response letter inside \\boxed{} and nothing else You can first think step-by-step. Which of the following genetic tests is used to identify the presence of a specific mutation associated with cystic fibrosis?\nA: Karyotyping\nB: Polymerase Chain Reaction (PCR)\n..."
            }
        ]
    },
    "options": [{"A": "Karyotyping"}, {"B": "Polymerase Chain Reaction (PCR)"}, ...],
    "expected_answer": "B",
    "grading_mode": "strict_single_letter_boxed",
    "uuid": "3c26f339-4b88-54be-b72a-e9c438ca6335"
}
```

### Example with template_metadata (custom format)
```json
{
    "responses_create_params":
    {
        "input":
        [
            {
                "role": "user",
                "content": "Which genetic test identifies cystic fibrosis mutations?\nA: Karyotyping\nB: PCR\n...\n\nChoose the correct option.\nConclude with \"ANSWER IS X\" on the final line."
            }
        ]
    },
    "options": [{"A": "Karyotyping"}, {"B": "PCR"}, ...],
    "expected_answer": "B",
    "grading_mode": "strict_single_letter_boxed",
    "template_metadata":
    {
        "output_regex": "ANSWER IS\\s*([A-Za-z])\\s*",
        "template_id": "mcqa_generated_019",
        "prompt_type": "generated",
        "format_type": "mcqa"
    },
    "uuid": "eb07c826-fed5-57f8-bee6-bb29e099069d"
}
```

**Note:** Example files in `data/example_with_template_metadata.jsonl` use simulated `reward_profiles` for demonstration purposes.

### Example of rollouts and usage

**Standard format (with `grading_mode`):**
```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/mcqa/configs/mcqa.yaml"

# This dataset already contains the instruction to add response inside \boxed{}
ng_download_dataset_from_gitlab \
    +dataset_name=syn_gpqa_v1.1 \
    +version=1.1.2 \
    +artifact_fpath=filtered_decontaminated.jsonl \
    +output_fpath=data/MCQA_filtered_decontaminated.jsonl

ng_run "+config_paths=[$config_paths]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=mcqa

ng_collect_rollouts \
    +agent_name=simple_agent \
    +input_jsonl_fpath=data/MCQA_filtered_decontaminated.jsonl \
    +output_jsonl_fpath=data/MCQA_filtered_decontaminated_samples_rollouts.jsonl +limit=5
```

**With template_metadata (custom regex):**
```bash
# Using example file with 5 different custom prompt formats
ng_collect_rollouts \
    +agent_name=simple_agent \
    +input_jsonl_fpath=resources_servers/mcqa/data/example_with_template_metadata.jsonl \
    +output_jsonl_fpath=resources_servers/mcqa/data/example_rollouts_with_template_metadata.jsonl \
    +limit=5
```

Rollout example

```json
{
    "responses_create_params":
    {
        "background": null,
        "include": null,
        "input":
        [
            {
                "content": "You should output your final response letter inside \\boxed{} and nothing else You can first think step-by-step. Which of the following statements about genetic screening and testing is true in the context of a complex polygenic disorder like Type 2 Diabetes Mellitus?\nA: A single gene test can definitively diagnose Type 2 Diabetes Mellitus.\nB: Genetic screening can predict the exact age of onset for Type 2 Diabetes Mellitus.\nC: Genetic testing can identify all individuals who will develop Type 2 Diabetes Mellitus.\nD: Environmental factors play a more significant role than genetic factors in the development of Type 2 Diabetes Mellitus.\nE: A polygenic risk score can provide a probability of developing Type 2 Diabetes Mellitus.\nF: Genetic screening for Type 2 Diabetes Mellitus is currently more reliable than traditional clinical screening methods.\nG: Type 2 Diabetes Mellitus is caused by a single gene mutation.\nH: Genetic testing can definitively rule out the possibility of developing Type 2 Diabetes Mellitus.\nI: Genetic screening can identify all genetic mutations responsible for Type 2 Diabetes Mellitus.\nJ: Current genetic screening methods can replace clinical assessments for diagnosing Type 2 Diabetes Mellitus.",
                "role": "user"
            }
        ],
        "instructions": null,
        "max_output_tokens": null,
        "max_tool_calls": null,
        "metadata": null,
        "model": null,
        "parallel_tool_calls": null,
        "previous_response_id": null,
        "prompt": null,
        "reasoning": null,
        "service_tier": null,
        "store": null,
        "temperature": null,
        "text": null,
        "tool_choice": null,
        "tools":
        [],
        "top_logprobs": null,
        "top_p": null,
        "truncation": null,
        "user": "",
        "stream": null
    },
    "response":
    {
        "id": "resp_68a87c3d2dd081909f5f72e6d562428d0972ba00fed2a9e0",
        "created_at": 1755872317.0,
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "metadata":
        {},
        "model": "gpt-4.1-2025-04-14",
        "object": "response",
        "output":
        [
            {
                "id": "msg_68a87c3e2e00819085a173137c75d47b0972ba00fed2a9e0",
                "content":
                [
                    {
                        "annotations":
                        [],
                        "text": "Let's analyze each option step-by-step:\n\nA: A single gene test can definitively diagnose Type 2 Diabetes Mellitus.  \n- Type 2 Diabetes is a complex polygenic disorder, not caused by a single gene. Therefore, this is false.\n\nB: Genetic screening can predict the exact age of onset for Type 2 Diabetes Mellitus.  \n- No genetic test can currently predict the exact age of onset for such a complex disorder. False.\n\nC: Genetic testing can identify all individuals who will develop Type 2 Diabetes Mellitus.  \n- Due to gene-environment interactions and incomplete penetrance, this is false.\n\nD: Environmental factors play a more significant role than genetic factors in the development of Type 2 Diabetes Mellitus.  \n- While environment is important, this is a debated point and too strong of a statement without further context.\n\nE: A polygenic risk score can provide a probability of developing Type 2 Diabetes Mellitus.  \n- This is a true statement! Polygenic risk scores combine information from multiple genetic loci to estimate the probability (risk) of disease.\n\nF: Genetic screening for Type 2 Diabetes Mellitus is currently more reliable than traditional clinical screening methods.  \n- This is false; clinical screening remains more reliable.\n\nG: Type 2 Diabetes Mellitus is caused by a single gene mutation.  \n- This is false; it's polygenic.\n\nH: Genetic testing can definitively rule out the possibility of developing Type 2 Diabetes Mellitus.  \n- False. There are many genes and environmental factors.\n\nI: Genetic screening can identify all genetic mutations responsible for Type 2 Diabetes Mellitus.  \n- False; many risk loci are still unknown.\n\nJ: Current genetic screening methods can replace clinical assessments for diagnosing Type 2 Diabetes Mellitus.  \n- False. Genetic methods are supplemental, not replacements.\n\nThe answer is:\n\n\\boxed{E}",
                        "type": "output_text",
                        "logprobs":
                        []
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message"
            }
        ],
        "parallel_tool_calls": true,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools":
        [],
        "top_p": 1.0,
        "background": false,
        "max_output_tokens": null,
        "max_tool_calls": null,
        "previous_response_id": null,
        "prompt": null,
        "reasoning":
        {
            "effort": null,
            "generate_summary": null,
            "summary": null
        },
        "service_tier": "default",
        "status": "completed",
        "text":
        {
            "format":
            {
                "type": "text"
            },
            "verbosity": "medium"
        },
        "top_logprobs": 0,
        "truncation": "disabled",
        "usage":
        {
            "input_tokens": 249,
            "input_tokens_details":
            {
                "cached_tokens": 0
            },
            "output_tokens": 380,
            "output_tokens_details":
            {
                "reasoning_tokens": 0
            },
            "total_tokens": 629
        },
        "user": null,
        "prompt_cache_key": null,
        "safety_identifier": null,
        "store": true
    },
    "reward": 1.0,
    "expected_answer": "E",
    "extracted_answer": "E"
}

```

### Implementation notes
- The server extracts the last assistant message's text from the Responses output.
- Letters are validated against the provided `options` keys.
- For `lenient_boxed`, only boxed content is considered; it must match exactly one option's text after normalization.
- **template_metadata priority**: When `template_metadata.output_regex` is present, it takes priority over `grading_mode` for answer extraction.
- **Backward compatibility**: Existing datasets without `template_metadata` continue to work using `grading_mode`.


## Licensing information
Code: Apache 2.0
