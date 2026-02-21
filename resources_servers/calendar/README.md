# Description
This is a resource server for training and evaluating calendar scheduling assistants. The server verifies whether an assistant correctly schedules events with time constraints in a conversational multi-turn setting.

The assistant must:
- Parse user requests to add events to a calendar
- Track time constraints (before/after/between/at specific times)
- Ensure events don't conflict with each other
- Schedule events within specified time windows
- Return the calendar state as a JSON list

The conversations in the dataset are generated using personas from the [nvidia/Nemotron-Personas-USA dataset](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) on Hugging Face.

# Example usage

## Running servers
The following is an example command for running this resource server along with an OpenAI model:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml, \
resources_servers/calendar/configs/calendar.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

Rollouts can be collected using the example dataset as follows:

```bash
ng_collect_rollouts \
    +agent_name=calendar_agent \
    +input_jsonl_fpath=data/example.jsonl \
    +output_jsonl_fpath=results/example_rollouts.jsonl \
    +limit=5
```

The input JSONL file should contain entries with:
- `responses_create_params`: Dictionary with `input` field containing the conversation history
- `exp_cal_state`: Dictionary mapping event IDs to expected event details (with `event_id`, `event_name`, `duration`, `constraint`, `min_time`, `max_time`)

**Example input format:**
```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a scheduling assistant..."},
      {"role": "user", "content": "Schedule a team meeting at 10am for 1 hour"}
    ]
  },
  "exp_cal_state": {
    "1": {
      "event_id": 1,
      "event_name": "Team Meeting",
      "duration": 60,
      "constraint": "at 10am",
      "min_time": "10am",
      "max_time": "4pm"
    }
  }
}
```

# Verification Logic

The server grades assistant responses based on multiple criteria:

**Reward = 0 (failure) if:**
- Response contains `<think>` tags
- No JSON list extracted from response (when events expected)
- Wrong number of events scheduled
- Events have time conflicts (overlapping times)
- Events violate time constraints:
  - "before X": event must end at or before time X
  - "after X": event must start at or after time X
  - "between X and Y": event must start at/after X and end at/before Y
  - "at X": event must start exactly at time X
- Events outside min/max time window
- Wrong event duration

**Reward = 1 (success) if:**
- All events correctly scheduled
- No time conflicts
- All constraints satisfied
- Or when no events are expected and response is valid


# Data Generation Pipeline

The dataset is created through a three-step pipeline:

## Step 1: Generate Synthetic Conversations
Uses `create_synth_conversations.py` to create synthetic multi-turn conversations between a user and assistant about scheduling calendar events.

**Features:**
- Uses personas from `nvidia/Nemotron-Personas-USA`
- Generates events with realistic durations (30-90 minutes in 15-min increments)
- Creates time constraints: "before", "after", "between", "at"
- Includes natural conversation flow with small talk
- Ensures non-overlapping, valid schedules

**Example command:**
```bash
python create_synth_conversations.py \
    --n-samples 2000 \
    --n-workers 100 \
    --n-events 7 \
    --min-time 600 \
    --max-time 960 \
    --model "openai/gpt-oss-120b" \
    --endpoint vllm \
    --ds-name "nvidia/Nemotron-Personas-USA" \
    --output ./data/train.json
```

**Key parameters:**
- `--n-samples`: Number of conversation samples to generate
- `--n-events`: Number of events per conversation (default: 7)
- `--min-time`: Minimum time in minutes from midnight (default: 600 = 10am)
- `--max-time`: Maximum time in minutes from midnight (default: 960 = 4pm)

**Output format:**
```json
{
  "sample_id": 0,
  "persona": "PERSONA: ...",
  "events": [...],
  "expected_calendar_states": [...],
  "user_prompts": [...],
  "user_intents": [...],
  "smalltalk_factor": 0.75,
  "constraint_eagerness": 0.85
}
```

## Step 2: Generate Model Rollouts
Uses `generate_rollouts.py` to have a model generate actual responses to the conversations and grade them.

**Features:**
- Alternates between "easy" mode (allows responses without JSON when the user turn does not require changes to calendar state) and "hard" mode (requires JSON list)
- Grades responses using the verification logic
- Stops conversation at first failure
- Retries on errors
- Supports VLLM and NIMS endpoint

**Example command:**
```bash
python generate_rollouts.py \
    --input ./data/train.json \
    --output ./data/rollouts.json \
    --model "Qwen/Qwen3-8B" \
    --min-time "10am" \
    --max-time "4pm" \
    --n-workers 100 \
```

**Key parameters:**
- `--model`: Model to use for generating responses
- `--min-cal-entries`: Minimum number of calendar entries to keep sample
- `--n-samples`: Limit number of samples to process
- `--offset`: Skip first N samples

**Output format:**
```json
{
  "conversation": [...],
  "grade": 1,
  "grade_reason": "...",
  "exp_cal_state": {...},
  "mode": "hard"
}
```

## Step 3: Preprocess for Training
Uses `dataset_preprocess.py` to convert rollouts into the final training format.

**Features:**
- Removes last message (assistant response to be predicted)
- Removes reasoning content from messages
- Converts to JSONL format
- Splits into train/validation sets
- Optionally filters to only failed rollouts for training

**Example command:**
```bash
python dataset_preprocess.py \
    --input ./data/rollouts.json \
    --output_train ./data/train.jsonl \
    --output_val ./data/validation.jsonl \
    --n_val 128 \
    --exclude_success
```

**Key parameters:**
- `--n_val`: Number of validation samples (taken from end)
- `--exclude_success`: Only include failed rollouts (grade=0) for training

**Output format (JSONL):**
```json
{"responses_create_params": {"input": [...]}, "exp_cal_state": {...}}
```


# Licensing information
Code: Apache 2.0

Data:
- nvidia/Nemotron-Personas-USA: Creative Commons Attribution 4.0 International (CC-BY-4.0)

Dependencies:
- nemo_gym: Apache 2.0

