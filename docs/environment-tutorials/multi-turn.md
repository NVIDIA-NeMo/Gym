(env-multi-turn)=
# Multi-Turn Environments

Train models on extended conversations where context accumulates across user/assistant exchanges.

:::{card}

**Goal**: Build a multi-turn environment for conversational RL training.

^^^

**In this tutorial, you will**:

1. Understand the difference between multi-turn and multi-step
2. Run the calendar scheduling environment
3. Implement custom verification for conversations
4. Generate synthetic training data

:::

:::{button-ref} multi-step
:color: secondary
:outline:
:ref-type: doc

← Previous: Multi-Step Environments
:::

---

## Why Multi-Turn RL?

Standard fine-tuning trains models on static conversation transcripts. Multi-turn RL goes further:

| Approach | Training Signal | Learns From |
|----------|-----------------|-------------|
| **SFT** | Token prediction loss | What a good response looks like |
| **Multi-turn RL** | Task completion reward | Whether the task was actually solved |

With RL, the model learns that a well-formatted response that schedules conflicting events gets reward=0, while a response that satisfies all constraints gets reward=1. This drives the model toward task success, not just fluent output.

---

## What You'll Build

In this tutorial, you'll run the **calendar scheduling** environment—a multi-turn resources server that trains models to:

- Parse user requests across conversation turns
- Track state (calendar events) as the conversation progresses
- Satisfy time constraints (before, after, between, at)
- Avoid event conflicts

These patterns apply to customer support, tutoring, booking systems, and other conversational tasks.

---

## Multi-Turn vs Multi-Step

| Aspect | Multi-Turn | Multi-Step |
|--------|------------|------------|
| **Interaction** | User ↔ Assistant dialogue | Tool calling loops |
| **Context growth** | Conversation history accumulates | Tool results accumulate |
| **Control flow** | User drives the conversation | Model drives the workflow |
| **Example** | Calendar scheduling | Trip planning with APIs |
| **Termination** | User goal achieved | Task complete or max steps |

**Multi-turn**: The model responds to user messages, building a conversation over time.

**Multi-step**: The model calls tools within a single turn to complete a task autonomously. See {doc}`multi-step` for that pattern.

---

## Run the Calendar Environment

All commands run from the **repository root** using the NeMo Gym CLI tools.

### 1. Start the Servers

The `ng_run` command starts the resources server and model server defined in the config files:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/calendar/configs/calendar.yaml"

ng_run "+config_paths=[$config_paths]"
```

:::{note}
The `+` prefix is Hydra syntax for appending to config. This is standard NeMo Gym convention.
:::

**Expected output**:
```text
[INFO] Starting resources server: calendar
[INFO] Starting model server: openai_model
[INFO] Servers ready at http://localhost:8000
```

### 2. Collect Rollouts

The `ng_collect_rollouts` command generates model responses and verifies them. A **rollout** is a conversation transcript with verification results.

```bash
ng_collect_rollouts \
    +agent_name=calendar_simple_agent \
    +input_jsonl_fpath=resources_servers/calendar/data/example.jsonl \
    +output_jsonl_fpath=results/calendar_rollouts.jsonl \
    +limit=5
```

**Expected output**:
```text
[INFO] Processing 5 samples...
[INFO] Sample 1: reward=1.0 (pass)
[INFO] Sample 2: reward=0.0 (constraint_violated)
...
[INFO] Wrote 5 rollouts to results/calendar_rollouts.jsonl
```

The output file contains conversation history plus verification results:
- **reward=1**: All constraints satisfied, no conflicts
- **reward=0**: Verification failed (reason logged)

---

## Data Format

Training data includes conversation history and expected final state (`resources_servers/calendar/data/example.jsonl`):

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a helpful assistant..."},
      {"role": "user", "content": "Schedule a team meeting at 10am for 1 hour"},
      {"role": "assistant", "content": "[{\"event_id\": 0, ...}]"},
      {"role": "user", "content": "Could you slot the meeting before noon?"}
    ]
  },
  "exp_cal_state": {
    "0": {
      "event_id": 0,
      "duration": 60,
      "constraint": "before 12pm",
      "min_time": "10:00",
      "max_time": "16:00"
    }
  }
}
```

**Key fields**:

- `responses_create_params.input`: Conversation history (system, user, assistant messages)
- `exp_cal_state`: Expected state after the model's response—used for verification

:::{note}
The field name for expected state varies by resources server. The calendar example uses `exp_cal_state`. Your custom server will use a different field name.
:::

---

## Implementation

The calendar server demonstrates key multi-turn patterns.

:::{tip}
**Full source**: See `resources_servers/calendar/` for the complete implementation:
- `app.py` — Server and verification logic
- `utils.py` — Grading functions
- `prompts.py` — System prompts
- `configs/calendar.yaml` — Configuration
:::

### Custom Request Schema

A **resources server** is a FastAPI service that verifies model outputs against expected outcomes. Define a custom request class that extends `BaseRunRequest` with your expected state field:

```python
# From resources_servers/calendar/app.py

from typing import Any

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class CalendarRunRequest(BaseRunRequest):
    exp_cal_state: dict[str, Any]


class CalendarVerifyRequest(CalendarRunRequest, BaseVerifyRequest):
    pass
```

`CalendarRunRequest` adds the expected state field. `CalendarVerifyRequest` combines the request with the model's response for verification.

### Verification Logic

The `verify` method extracts the assistant's response and grades it against the expected state:

```python
# From resources_servers/calendar/app.py

from utils import grade_assistant_response


class CalendarResourcesServer(SimpleResourcesServer):
    config: CalendarResourcesServerConfig

    async def verify(self, body: CalendarVerifyRequest) -> BaseVerifyResponse:
        assistant_response = body.response.output[-1].content[0].text
        exp_cal_state = body.exp_cal_state
        try:
            reward, reason = grade_assistant_response(assistant_response, exp_cal_state)
        except Exception:
            reward = 0

        return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

:::{tip}
The `reason` value is useful for debugging. For production, log it to diagnose failures.
:::

### Grading Function

The `grade_assistant_response` function performs verification, returning a reward (0 or 1) and a reason string:

```python
# From resources_servers/calendar/utils.py

def grade_assistant_response(assistant_response, exp_cal_state, allow_no_json_list=False):
    # Invalid response: contains reasoning tags
    if "<think>" in assistant_response:
        return 0, "think_found"

    # No events expected
    elif len(exp_cal_state) == 0:
        return 1, "pass"

    else:
        # Events were expected to be scheduled/changed
        cal_state = extract_json_list(assistant_response)
        if cal_state is None or len(cal_state) == 0:
            if allow_no_json_list:
                return 1, "pass"
            else:
                return 0, "no_json_list"

        events_dict = {}
        for event in cal_state:
            events_dict[str(event["event_id"])] = event

        # Wrong number of events
        if len(events_dict) != len(exp_cal_state):
            return 0, "different_number_of_events"

        # Check for time conflicts
        for event in cal_state:
            if is_event_conflicting(cal_state, event, exclude_event=event):
                return 0, "conflicting_events"

        # Check all constraints are satisfied
        for event_id in exp_cal_state.keys():
            if not is_constraint_satisfied(events_dict[event_id], exp_cal_state[event_id]):
                return 0, "constraint_violated"

    return 1, "pass"
```

**Failure reasons** (reward = 0):

| Reason | Description |
|--------|-------------|
| `think_found` | Response contains `<think>` tags |
| `no_json_list` | No JSON list found in response |
| `different_number_of_events` | Event count doesn't match expected |
| `conflicting_events` | Events have overlapping times |
| `constraint_violated` | Time constraint not satisfied |
| `error_in_grading` | Exception during verification (malformed response) |

**Success** (reward = 1): `pass` — all constraints satisfied, no conflicts.

### Constraint Validation

The `is_constraint_satisfied` function checks four constraint types:

```python
# From resources_servers/calendar/utils.py

def is_constraint_satisfied(event, exp_event):
    # Duration must match
    if event["duration"] != exp_event["duration"]:
        return False

    # Event must be within the time window
    min_time = time_to_minutes(exp_event["min_time"])
    max_time = time_to_minutes(exp_event["max_time"])
    event_start = time_to_minutes(event["start_time"])
    event_end = event_start + event["duration"]
    if event_start < min_time or event_end > max_time:
        return False

    # Check constraint type
    constraint = exp_event["constraint"]
    if constraint is None:
        return True
    elif constraint.startswith("before "):
        # Event must end at or before the specified time
        constraint_time = time_to_minutes(constraint.replace("before ", ""))
        return event_end <= constraint_time
    elif constraint.startswith("after "):
        # Event must start at or after the specified time
        constraint_time = time_to_minutes(constraint.replace("after ", ""))
        return event_start >= constraint_time
    elif constraint.startswith("between "):
        # Event must start at/after X and end at/before Y
        parts = constraint.replace("between ", "").split(" and ")
        time_x = time_to_minutes(parts[0])
        time_y = time_to_minutes(parts[1])
        return event_start >= time_x and event_end <= time_y
    elif constraint.startswith("at "):
        # Event must start exactly at the specified time
        constraint_time = time_to_minutes(constraint.replace("at ", ""))
        return event_start == constraint_time
    return True
```

---

## System Prompt

The system prompt defines the task and output format. From `resources_servers/calendar/prompts.py`:

```python
SYSTEM_PROMPT = """You are a helpful assistant. If asked to help with organizing the user's calendar, adhere to the following rules

1. Calendar Format: Only print the calendar in a json list format with each entry containing the following fields: "event_id" (unique integer id), "event_name" (string), "start_time"(string in 24-hour HH:MM format), "duration" (integer - in minutes)
2. Initial State: Assume the calendar is empty at the start of the conversation.
3. User Constraints: Honor all time-based constraints specified by the user for an event (e.g., "before 11 am," "after 2 pm", "at 11:15am"). These constraints are permanent and **must be maintained during any rescheduling**.
4. Conflict Resolution: Ensure that there are no conflicts (overlapping events) in the calendar. If a new event conflicts with an existing one, automatically reschedule events to the next available slot that respects all active constraints. Do not ask for user confirmation.
5. Interpret constraints like "before 11 am" as "ends at or before 11 am", "after 2 pm" as "starts at or after 2 pm".
6. General Queries: For any requests not related to the calendar, respond as a helpful assistant.
7. Ensure that all events are between {start_time_str} and {end_time_str}.
8. Display the calendar in the json list format in every responses."""
```

Key design choices:

- **Structured output**: JSON list format enables programmatic verification
- **Explicit constraint semantics**: "before X" means "ends at or before X"
- **No user confirmation**: Model must act autonomously on conflicts
- **Output on every turn**: Calendar state in each response enables per-turn verification

:::{note}
The template uses `{start_time_str}` and `{end_time_str}` placeholders. Generated training data may include additional rules (e.g., "Do not ask follow-up questions") based on the data generation pipeline settings.
:::

---

## Configuration

The calendar config defines the resources server and agent:

```yaml
# From resources_servers/calendar/configs/calendar.yaml

calendar:
  resources_servers:
    calendar:
      entrypoint: app.py
      domain: agent
      verified: false

calendar_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: calendar
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: resources_servers/calendar/data/train.jsonl
        huggingface_identifier:
          repo_id: nvidia/Nemotron-RL-agent-calendar_scheduling
          artifact_fpath: train.jsonl
        license: Apache 2.0
```

The agent uses `simple_agent` for the conversation loop. Set `max_steps` to limit turns:

```yaml
calendar_simple_agent:
  responses_api_agents:
    simple_agent:
      max_steps: 20  # Maximum conversation turns
```

---

## Data Generation Pipeline

:::{dropdown} Advanced: Generate Training Data
:icon: beaker

Generate large-scale training datasets using the calendar data pipeline.

### Step 1: Generate Synthetic Conversations

```bash
python resources_servers/calendar/create_synth_conversations.py \
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

**Features**:
- Personas from `nvidia/Nemotron-Personas-USA` for diverse styles
- Events with durations 30-90 minutes
- Constraints: before, after, between, at
- Natural conversation flow with small talk

### Step 2: Generate Model Rollouts

```bash
python resources_servers/calendar/generate_rollouts.py \
    --input ./data/train.json \
    --output ./data/rollouts.json \
    --model "Qwen/Qwen3-8B" \
    --min-time "10am" \
    --max-time "4pm" \
    --n-workers 100
```

Generates model responses and grades them.

### Step 3: Preprocess for Training

```bash
python resources_servers/calendar/dataset_preprocess.py \
    --input ./data/rollouts.json \
    --output_train ./data/train.jsonl \
    --output_val ./data/validation.jsonl \
    --n_val 128 \
    --exclude_success
```

Converts rollouts to JSONL format. Use `--exclude_success` to keep only failed rollouts for harder training examples.

:::

---

## Adapt for Your Use Case

To create your own multi-turn environment:

1. **Define expected state**: Create a custom request class with fields for expected outcomes
2. **Design verification**: Implement logic to compare model output against expected state
3. **Structure output format**: Use structured output (JSON, XML) for reliable parsing
4. **Create training data**: Include conversation history and expected state per sample

:::{tip}
**Quick start**: Copy `resources_servers/calendar/` and modify the verification logic for your domain.
:::

---

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Environments
:link: multi-step
:link-type: doc
Build autonomous tool-calling workflows.
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-Judge
:link: llm-as-judge
:link-type: doc
Use an LLM to evaluate subjective quality.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Start training models on your environment.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Prepare Data
:link: /data/prepare-validate
:link-type: doc
Learn more about data formats and validation.
:::

::::
