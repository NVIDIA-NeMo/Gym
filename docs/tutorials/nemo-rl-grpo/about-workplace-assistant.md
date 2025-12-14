(training-nemo-rl-grpo-about-workplace-assistant)=

# About Workplace Assistant

Workplace Assistant is a **{term}`multi-step <Multi-step>` agentic {term}`tool-use <Tool Use / Function Calling>` {term}`training environment <Training environment>`** that tests a model's ability to execute business tasks in a simulated workplace setting.

:::{card}

**Goal**: Understand the training environment and how tasks are structured and verified.

^^^

**In this section, you will learn**:

1. How tasks are structured for multi-step tool calling
2. The available databases and tools
3. How the environment verifies task completion

:::

:::{button-ref} training-nemo-rl-grpo-index
:color: secondary
:outline:
:ref-type: ref

← Back to Tutorial Overview
:::

---

## How the Model Completes Tasks

For each task, the model must:

1. Understand the user's intent from natural language
2. Determine which tools to call and in what order
3. Infer correct parameters (for example, look up email addresses or find matching customer records)
4. Execute all necessary steps to complete the task

The model has up to **6 tool calling steps** to accomplish each task.

---

## Available Databases and Tools

Each task is a natural language request that the model must complete using the available tools. All tasks share the same set of tools that allow the model to retrieve more information or perform actions. Each {term}`task instance <Task Instance>` uses isolated database instances so actions from different rollouts don't interfere.

- **Databases**: Email, Calendar, Analytics, Project Management, Customer Relationship Manager (CRM)
- **Tools**: Distributed across these databases
- **Tasks**: Common business activities (such as sending emails, scheduling meetings, and managing projects)

All tasks are available in the [Workplace Assistant HuggingFace dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant).

---

## Task Examples

::::{tab-set}

:::{tab-item} Single-Step Task

**User query**: "Send an email to john.smith@atlas.com with the subject 'Team Meeting' and body 'Let's meet tomorrow at 2pm to discuss the project.'"

**Expected tool call**:
```python
email_send_email(
    recipient="john.smith@atlas.com",
    subject="Team Meeting",
    body="Let's meet tomorrow at 2pm to discuss the project."
)
```

The tool adds a new email to the emails database.

:::

:::{tab-item} Multi-Step Task

**User query**: "John is taking over all of Akira's leads that are interested in software. Can you reassign them in the CRM?"

**Expected output sequence**:

1. `company_directory_find_email_address(name="Akira")` → Returns `"akira.tanaka@atlas.com"`
2. `company_directory_find_email_address(name="John")` → Returns `"john.smith@atlas.com"`
3. `customer_relationship_manager_search_customers(assigned_to_email="akira.tanaka@atlas.com", product_interest="software", status="lead")` → Returns 3 matching leads
4. `customer_relationship_manager_update_customer(customer_id="00000095", field="assigned_to_email", new_value="john.smith@atlas.com")`
5. `customer_relationship_manager_update_customer(customer_id="00000080", field="assigned_to_email", new_value="john.smith@atlas.com")`
6. `customer_relationship_manager_update_customer(customer_id="00000035", field="assigned_to_email", new_value="john.smith@atlas.com")`

:::

:::{tab-item} Input Format

Each task is a `responses_create_params` object:

```json
{
  "responses_create_params": {
    "input": [
      {
        "role": "system",
        "content": "Today's date is Thursday, 2023-11-30 and the current time is 23:59:00. Remember the current date and time when answering queries. Meetings must not start before 9am or end after 6pm."
      },
      {
        "role": "user",
        "content": "John is taking over all of Akira's leads that are interested in software. Can you reassign them in the CRM?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "name": "email_send_email",
        "description": "Sends an email to the specified recipient.",
        "parameters": {
          "type": "object",
          "properties": {
            "recipient": {
              "type": "string",
              "description": "Email address of the recipient"
            },
            "subject": {
              "type": "string",
              "description": "Subject line of the email"
            },
            "body": {
              "type": "string",
              "description": "Body content of the email"
            }
          },
          "required": ["recipient", "subject", "body"],
          "additionalProperties": false
        },
        "strict": false
      }
    ]
  }
}
```

The full task includes all 27 tools across the 5 databases.

:::

::::

---

## How Verification Works

The environment is implemented as a FastAPI-based resource server that executes tools and verification. It uses **state-matching verification**: instead of requiring exact tool sequences, it compares final database states.

::::{tab-set}

:::{tab-item} Why State-Matching?

- **Flexibility**: Multiple valid solution paths exist for the same task
- **Robustness**: Model can recover from mistakes mid-trajectory
- **Goal-oriented**: Focuses on outcomes, not specific procedures

:::

:::{tab-item} Verification Process

```python
async def verify(self, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
    ground_truth = body.ground_truth
    response = body.response.output

    total_score = 0.0

    # Convert list of ResponseFunctionToolCall objects into list of dictionaries
    predicted_function_calls = []

    for message in response:
        if message.type == "function_call":
            predicted_function_calls.append(message.model_dump())

    predicted_chat_content = []

    for message in response:
        if message.type == "output_text":
            predicted_chat_content.append(message.model_dump())

    total_score += is_correct(predicted_function_calls, ground_truth, None) * 1.0
    return WorkbenchVerifyResponse(**body.model_dump(), reward=total_score)
```

The `is_correct` function implements the state-matching logic:

```python
def is_correct(predicted_actions, ground_truth_actions, error):
    ...
    
    # Execute both sequences in fresh environments
    predict_env = execute_actions_and_reset_state(predicted_actions)
    ground_truth_env = execute_actions_and_reset_state(ground_truth_actions)
    
    ... # Extract specific state info

    # Compare final states of all 5 databases
    return (
        predicted_calendar_state.equals(ground_truth_calendar_state)
        and predicted_email_state.equals(ground_truth_email_state)
        and predicted_analytics_state.equals(ground_truth_analytics_state)
        and predicted_project_management_state.equals(ground_truth_project_management_state)
        and predicted_customer_relationship_manager_state.equals(ground_truth_customer_relationship_manager_state)
    )
```

:::

:::{tab-item} Error Handling

Tool execution errors are returned to the model (not terminating the rollout), allowing self-correction:

```python
async def route_to_python_function(self, path, body, request):
    ...
    tool_env = self.session_id_to_tool_env[session_id]
    args = body.model_dump(exclude_unset=True)

    try:
        function = tool_env["functions"][path]
        result = function(**args)
        return WorkbenchResponse(output=result)
    except Exception as e:
        # Return error to model so it can self-correct
        return WorkbenchResponse(output=f"Error executing tool '{path}': {str(e)}")
```

:::

::::
