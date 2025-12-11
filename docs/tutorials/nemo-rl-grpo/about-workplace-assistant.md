(training-nemo-rl-grpo-about-workplace-assistant)=

# About the Workplace Assistant Training Environment

The Workplace Assistant is a **{term}`multi-step <Multi-step>` agentic {term}`tool-use <Tool Use / Function Calling>` {term}`training environment <Training environment>`** that tests an model's ability to execute business tasks in a simulated workplace setting.

**Generally, the model must:**
1. Understand the user's intent from natural language
2. Determine which tools to call and in what order
3. Infer correct parameters (e.g., look up email addresses, find matching customer records)
4. Execute all necessary steps to complete the task


## Task examples

Each task is a natural language request that the model must complete using the available tools. All tasks share the same set of tools that allow to model to retrieve more information or perform actions. Each {term}`task instance <Task Instance>` uses a separate set of database instances so that information retrieved or actions performed are specific to that task instance only. The model needs to accomplish the task within 6 {term}`steps <Multi-step>`.

- **Databases**: Email, Calendar, Analytics, Project Management, Customer Relationship Manager (CRM)
- **Tools**: distributed across these databases
- **Tasks**: common business activities (e.g., sending emails, scheduling meetings, managing projects)

In the examples below, you will see the exact {term}`Responses create params <Responses API>` that will be provided to the model. The model is provided with:
1. Surrounding contextual information in the system prompt (first input with role `system`) like what the current day is.
2. The user query (second input with role `user`) that contains the task for the model to perform.
3. Available tools (`tools` parameter) with descriptions that inform the model how and when to use that tool.

All tasks can be found in the [Workplace Assistant Huggingface dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant).

### Single-Step Task (1 tool call needed)
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
        "content": "Send an email to john.smith@atlas.com with the subject 'Team Meeting' and body 'Let's meet tomorrow at 2pm to discuss the project.'"
      }
    ],
    "tools": [
      {"type": "function", "name": "email_send_email", "description": "Sends an email to a recipient.", "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["recipient", "subject", "body"]}},
      {"type": "function", "name": "email_search_emails", "description": "Searches for emails matching the given query...", "parameters": {"...": "..."}},
      {"type": "function", "name": "calendar_create_event", "...": "..."},
      // ... other tools (calendar, analytics, project_management, CRM)
    ],
  }
}
```

**Expected tool call:** `email_send_email(recipient="alex.martinez@atlas.com", subject="Team Meeting", body="Let's meet tomorrow at 2pm to discuss the project.")`

The `email_send_email` tool will be executed with the above parameters. Within the training environment, this action will add a new email to the emails database with this information.


### Multi-Step Task (requires > 1 tool call)
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
        "content": "John is taking over all of Akira's leads that are interested in software. Can you reassign them in the crm?"
      }
    ],
    "tools": [
        // Same tools as above
    ],
  }
}
```

**Expected output sequence:**
1. `company_directory_find_email_address(name="Akira")` → Returns `"akira.tanaka@atlas.com"`
2. `company_directory_find_email_address(name="John")` → Returns `"john.smith@atlas.com"`
3. `customer_relationship_manager_search_customers(assigned_to_email="akira.tanaka@atlas.com", product_interest="software", status="lead")` → Returns 3 matching leads
4. `customer_relationship_manager_update_customer(customer_id="00000095", field="assigned_to_email", new_value="john.smith@atlas.com")`
5. `customer_relationship_manager_update_customer(customer_id="00000080", field="assigned_to_email", new_value="john.smith@atlas.com")`
6. `customer_relationship_manager_update_customer(customer_id="00000035", field="assigned_to_email", new_value="john.smith@atlas.com")`


## Key implementation details
The environment is implemented as a FastAPI-based resource server that executes tools and verification. Here's how it works:

### Session Management

Each rollout gets its own isolated session with fresh tool environments:

```python
async def seed_session(self, request: Request, body: BaseSeedSessionRequest):
    session_id = request.session[SESSION_ID_KEY]
    toolkits = [
        "email",
        "calendar",
        "analytics",
        "project_management",
        "customer_relationship_manager",
    ]
    self.session_id_to_tool_env[session_id] = get_tools(toolkits)
    return BaseSeedSessionResponse()
```

This ensures each task starts with a clean slate and tool calls from different rollouts don't interfere.

### Tool implementations

Tool calls are routed to Python functions:

```python
def route_to_python_function(tool_name, arguments):
    try:
        result = tool_env["functions"][tool_name](**arguments)
        return WorkbenchResponse(output=result)
    except Exception as e:
        # Return error to model so it can self-correct (don't terminate)
        return WorkbenchResponse(output=f"Error executing tool: {str(e)}")
```

**Key feature**: Tool execution errors are returned to the model as part of the response (rather than terminating the rollout), allowing the agent to self-correct and retry during execution.

### State Matching for Verification

The environment uses **state-matching verification**: instead of requiring exact tool sequences, it compares final database states.

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
    ..
    
    # Execute both sequences in fresh environments
    predict_env = execute_actions_and_reset_state(predicted_actions)
    ground_truth_env = execute_actions_and_reset_state(ground_truth_actions)
    
    .. # Extract specific state info

    # Compare final states of all 5 databases
    return (
        predicted_calendar_state.equals(ground_truth_calendar_state) and
        predicted_email_state.equals(ground_truth_email_state) and
        predicted_analytics_state.equals(ground_truth_analytics_state) and
        predicted_project_management_state.equals(ground_truth_project_management_state) and
        predicted_customer_relationship_manager_state.equals(ground_truth_customer_relationship_manager_state)
    )
```

**Why State-matching verification?**:
- **Flexibility**: Multiple valid solution paths exist for the same task
- **Robustness**: Model can recover from mistakes mid-trajectory
- **Goal-oriented**: Focuses on outcomes, not specific procedures
