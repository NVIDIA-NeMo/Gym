# Turing VIF Resource Server

A NeMo Gym resource server that integrates **Turing VIF** (Verifiable Instruction Following) validators for comprehensive instruction-following evaluation in reinforcement learning training.

## Overview

This resource server provides two types of validators:

1. **Fast Rule-Based Validators (~50+)**: CPU-efficient validators for structural constraints like word count, keyword presence, formatting, punctuation, etc.

2. **LLM Judge Validators (~27)**: Async LLM-as-a-judge validators for semantic/stylistic constraints like tone, formality, politeness, and linguistic patterns.

3. **Custom LLM Judge Questions**: Free-form yes/no questions evaluated by an LLM judge.

## Quick Start

### 1. Set up environment variables

Create `env.yaml` in your Gym root:

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-5-2025-08-07  # or gpt-4.1-2025-04-14
```

### 2. Start the servers

```bash
cd /path/to/Gym
source .venv/bin/activate
ng_run "+config_paths=[resources_servers/turing_vif/configs/turing_vif.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

### 3. Run a test

```bash
ng_collect_rollouts \
    +agent_name=turing_vif_simple_agent \
    +input_jsonl_fpath=resources_servers/turing_vif/data/example.jsonl \
    +output_jsonl_fpath=results.jsonl
```

## Architecture

```
turing_vif/
├── app.py                    # Main resource server (TuringVIFResourcesServer)
├── vif_validators/           # Validation logic
│   ├── __init__.py
│   ├── validator.py          # Fast rule-based validators
│   ├── data_loader.py        # Instruction definitions & prompts
│   ├── instruction_definition.csv
│   ├── subinstruction_definition.csv
│   └── evaluation_modes.csv
├── configs/
│   └── turing_vif.yaml       # Server configuration
├── data/
│   └── example.jsonl         # Example dataset
├── tests/
│   ├── __init__.py
│   └── test_app.py           # Unit tests (47 tests)
├── requirements.txt
└── README.md
```

## Supported Instructions

### Fast Validators (Rule-Based)

| Category | Instructions |
|----------|-------------|
| **Length Constraints** | `number_words`, `number_characters`, `sentence_length`, `word_repetition`, `unique_words`, `word_length`, `avg_word_length`, `paragraph_length` |
| **Keywords** | `existence`, `frequency`, `forbidden_words`, `letter_frequency`, `alliteration`, `palindrome_word`, `positioning`, `vowel_count`, `consonant_count` |
| **Format** | `json_format`, `numbered_list`, `bullet_lists`, `title`, `multiple_sections`, `number_paragraphs`, `sentences_per_paragraph`, `nested_list`, `table`, `heading_depth`, `sentence_count`, `sentence_endings` |
| **Case** | `all_caps`, `lowercase`, `alternating`, `first_letter_cap`, `capital_word_frequency`, `lowercase_word_frequency`, `*_target` variants, `case_ratio`, `vowel_consonant_balance` |
| **Punctuation** | `no_comma`, `no_period`, `question_exclaim`, `end_rule` |
| **Start/End** | `start_checker`, `end_checker`, `wrap_checker`, `quotation` |
| **Detectable Content** | `postscript`, `number_placeholders`, `numeric_inclusion` |

### LLM Judge Validators

| Category | Instructions |
|----------|-------------|
| **Stylistic** | `tone_formality`, `emotional_tone`, `politeness`, `descriptive_level`, `literary_style`, `sentence_tone_consistency`, `voice`, `figurative_language`, `tone_transition`, `emotive_adjectives`, `sensory_detail`, `rhythm_pattern` |
| **Linguistic** | `pragmatic_context`, `speech_act`, `syntactic_pattern`, `grammatical_mood`, `morphological_form`, `phonological_pattern`, `sound_symbolism` |
| **Situational** | `role_based`, `task_specific`, `audience_alignment`, `contextual_scenario`, `perspective`, `emotional_alignment`, `cultural_context`, `temporal_context`, `environment_setting` |

## Configuration

### Server Config (`configs/turing_vif.yaml`)

```yaml
turing_vif:
  resources_servers:
    turing_vif:
      entrypoint: app.py
      domain: instruction_following
      # LLM Judge configuration - uses policy model by default
      judge_base_url: ${policy_base_url}
      judge_api_key: ${policy_api_key}
      judge_model: ${policy_model_name}  # Supports GPT-5 and GPT-4.1
```

### Model Support

| Model | API Used | Notes |
|-------|----------|-------|
| **GPT-5** (`gpt-5-*`) | Responses API (`/v1/responses`) | Full reasoning model support |
| **O1/O3** (`o1-*`, `o3-*`) | Responses API | Reasoning models |
| **GPT-4.1** | Chat Completions API (`/v1/chat/completions`) | Standard chat model |
| **GPT-4** | Chat Completions API | Standard chat model |

The server automatically detects the model type and uses the appropriate API.

## Dataset Format

Each entry in your JSONL dataset should have:

```json
{
  "id": 1,
  "instructions": [
    {"instruction_id": "length_constraints:number_words", "relation": "at least", "num_words": 50},
    {"instruction_id": "keywords:existence", "keywords": ["research", "methodology"]},
    {"instruction_id": "stylistic:tone_formality", "tone_level": "formal"}
  ],
  "llm_judge": [
    {"uid": 1, "content": "Does the response contain a clear call-to-action?"}
  ],
  "responses_create_params": {
    "input": [{"role": "user", "content": "Write about research methodology..."}]
  }
}
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Optional | Request identifier |
| `instructions` | Yes | List of instruction objects with `instruction_id` and parameters |
| `llm_judge` | Optional | List of custom yes/no questions for LLM judge evaluation |
| `responses_create_params` | Yes | Parameters for the LLM response generation |

## Running Tests

```bash
cd /path/to/Gym
source .venv/bin/activate
pytest resources_servers/turing_vif/tests/ -v
```

## API Endpoints

### POST /verify

Validates a response against instructions.

**Request Body:**
```json
{
  "id": 1,
  "instructions": [...],
  "llm_judge": [{"uid": 1, "content": "Is the response professional?"}],
  "response": {
    "output": [{"content": [{"text": "...response text..."}]}]
  }
}
```

**Response:**
```json
{
  "reward": 1.0,
  "follow_all_instructions": true,
  "follow_instruction_list": [true, true, false],
  "validation_results": [
    {"instruction": "length_constraints:number_words", "status": "Passed", "message": "Word count: 75 (at least 50)"},
    {"instruction": "stylistic:tone_formality", "status": "Passed", "message": "The response maintains formal tone..."},
    {"instruction": "llm_judge_1", "status": "Passed", "message": "Yes, the response includes..."}
  ]
}
```

## Integration Notes

### Async Design

- All LLM judge calls use `NeMoGymAsyncOpenAI` for non-blocking I/O
- LLM validators run in parallel via `asyncio.gather`
- Fast validators run synchronously (CPU-bound, ~1-5ms each)
- Follows NVIDIA NeMo Gym integration guidelines (no extra threads/processes)

### Performance

| Validator Type | Typical Latency |
|----------------|-----------------|
| Fast (rule-based) | 1-5ms |
| LLM Judge (GPT-4.1) | 500-2000ms |
| LLM Judge (GPT-5) | 1000-5000ms |

### Scaling

For high-throughput training:
1. Use rate limiting for LLM judge API calls
2. Consider micro-batching LLM evaluations (future enhancement)
3. Cache definition lookups (already implemented)
4. Use GPT-4.1 for faster judge evaluations when reasoning is not critical

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `401 Unauthorized` | Check `policy_api_key` in `env.yaml` |
| `400 Bad Request` with GPT-5 | Ensure you're using the latest `app.py` with Responses API support |
| `ModuleNotFoundError` | Run `ray stop --force` and restart servers |
| Server won't start | Delete `.venv` in `resources_servers/turing_vif/` and restart |

### Debugging

View server logs:
```bash
# Check terminal output from ng_run
# Or view Ray dashboard at http://127.0.0.1:8265
```

## Contributing

1. Add new fast validators to `vif_validators/validator.py`
2. Add instruction definitions to the CSV files
3. Write unit tests in `tests/test_app.py`
4. Update this README with new instructions

## License

Apache-2.0
