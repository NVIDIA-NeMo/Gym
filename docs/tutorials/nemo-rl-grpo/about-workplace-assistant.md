(training-nemo-rl-grpo-about-workplace-assistant)=

# About Workplace Assistant

The Workplace Assistant is a **{term}`multi-step <Multi-step>` agentic {term}`tool-use <Tool Use / Function Calling>` {term}`training environment <Training environment>`** that tests a model's ability to execute business tasks in a simulated workplace setting.

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

The model has **6 steps** to accomplish each task.

---

## Available Databases and Tools

Each {term}`task instance <Task Instance>` uses isolated database instances so actions from different rollouts don't interfere.

| Database | Example Tools |
|----------|---------------|
| Email | `email_send_email`, `email_search_emails` |
| Calendar | `calendar_create_event`, `calendar_get_events` |
| Analytics | `analytics_get_report`, `analytics_run_query` |
| Project Management | `project_create_task`, `project_update_status` |
| CRM | `customer_relationship_manager_search_customers`, `customer_relationship_manager_update_customer` |

All tasks are available in the [Workplace Assistant HuggingFace dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant).

---

## Task Examples

::::{tab-set}

:::{tab-item} Single-Step Task

**User query**: "Send an email to john.smith@atlas.com with the subject 'Team Meeting' and body 'Let's meet tomorrow at 2pm to discuss the project.'"

**Expected tool call**:
```python
email_send_email(
    recipient="alex.martinez@atlas.com",
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
3. `customer_relationship_manager_search_customers(...)` → Returns 3 matching leads
4. `customer_relationship_manager_update_customer(customer_id="00000095", ...)` 
5. `customer_relationship_manager_update_customer(customer_id="00000080", ...)`
6. `customer_relationship_manager_update_customer(customer_id="00000035", ...)`

:::

:::{tab-item} Input Format

Each task is a `responses_create_params` object:

```json
{
  "responses_create_params": {
    "input": [
      {
        "role": "system",
        "content": "Today's date is Thursday, 2023-11-30..."
      },
      {
        "role": "user", 
        "content": "Send an email to john.smith@atlas.com..."
      }
    ],
    "tools": [
      {"type": "function", "name": "email_send_email", ...},
      {"type": "function", "name": "calendar_create_event", ...}
    ]
  }
}
```

:::

::::

---

## How Verification Works

The environment uses **state-matching verification**: instead of requiring exact tool sequences, it compares final database states.

::::{tab-set}

:::{tab-item} Why State-Matching?

- **Flexibility**: Multiple valid solution paths exist for the same task
- **Robustness**: Model can recover from mistakes mid-trajectory
- **Goal-oriented**: Focuses on outcomes, not specific procedures

:::

:::{tab-item} Verification Process

```python
def is_correct(predicted_actions, ground_truth_actions, error):
    # Execute both sequences in fresh environments
    predict_env = execute_actions_and_reset_state(predicted_actions)
    ground_truth_env = execute_actions_and_reset_state(ground_truth_actions)
    
    # Compare final states of all 5 databases
    return (
        predicted_calendar_state.equals(ground_truth_calendar_state) and
        predicted_email_state.equals(ground_truth_email_state) and
        # ... (all 5 databases)
    )
```

:::

:::{tab-item} Error Handling

Tool execution errors are returned to the model (not terminating the rollout), allowing self-correction:

```python
def route_to_python_function(tool_name, arguments):
    try:
        result = tool_env["functions"][tool_name](**arguments)
        return WorkbenchResponse(output=result)
    except Exception as e:
        # Return error to model so it can self-correct
        return WorkbenchResponse(output=f"Error executing tool: {str(e)}")
```

:::

::::

---

:::{button-ref} training-nemo-rl-grpo-gym-configuration
:color: primary
:ref-type: ref

Next: Gym Configuration →
:::
