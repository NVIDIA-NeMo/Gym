# Setup and Installation

**Goal**: Get NeMo Gym installed and servers running with your first successful agent interaction

## Prerequisites

- **Python 3.12+** (check with `python3 --version`)
- **Git** (for cloning the repository)
- **OpenAI API key** (for the tutorial agent)

## Step 1: Clone and Install

```bash
# Clone the repository
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym

# Install UV (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install NeMo Gym
uv sync --extra dev --group docs
```

**✅ Success Check**: You should see something that indicates a newly activated environment e.g. `(.venv)` or `(NeMo-Gym)` in your terminal prompt.

## Step 2: Configure Your API Key

Create an `env.yaml` file in the project root:

```bash
# Create env.yaml with your OpenAI credentials
cat > env.yaml << EOF
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-openai-api-key-here
policy_model_name: gpt-4.1-2025-04-14
EOF
```

> [!IMPORTANT]
> Replace `sk-your-actual-openai-api-key-here` with your real OpenAI API key. This file keeps secrets out of version control while making them available to NeMo Gym.

## Step 3: Start the Servers

```bash
# Define which servers to start
config_paths="resources_servers/example_simple_weather/configs/simple_weather.yaml,responses_api_models/openai_model/configs/openai_model.yaml"

# Start all servers
ng_run "+config_paths=[${config_paths}]"
```

**✅ Success Check**: You should see output like:
```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://127.0.0.1:11000 (Press CTRL+C to quit)
INFO:     Started server process [12346]  
INFO:     Uvicorn running on http://127.0.0.1:62920 (Press CTRL+C to quit)
...
```

This means **4 servers are now running**:
1. **Head server** (coordinates everything)
2. 3 Gym servers. These 3 servers and their high level config should be printed to terminal!
   1. **Simple weather resource** (provides weather tool)
   2. **OpenAI model server** (connects to GPT-4)
   3. **Simple agent** (orchestrates model + resources)

## Step 4: Test the Setup

Open a **new terminal** (keep servers running in the first one):

```bash
# Navigate to project directory
cd /path/to/Gym

# Activate virtual environment
source .venv/bin/activate

# Test the agent
python responses_api_agents/simple_agent/client.py
```

**✅ Success Check**: You should see JSON output showing:
1. Agent calling the weather tool
2. Weather tool returning data  
3. Agent responding to the user

Example output:
```json
[
    {
        "arguments": "{\"city\":\"San Francisco\"}",
        "call_id": "call_OnWAk719Jr3tte4OmCJtJOB4",
        "name": "get_weather",
        "type": "function_call",
        "id": "fc_68a3739f2f0081a1aae4b93d5df07c100cb216b5cc4adbc4",
        "status": "completed"
    },
    {
        "call_id": "call_OnWAk719Jr3tte4OmCJtJOB4",
        "output": "{\"city\": \"San Francisco\", \"weather_description\": \"The weather in San Francisco is cold.\"}",
        "type": "function_call_output",
        "id": null,
        "status": null
    },
    {
        "id": "msg_68a373a1099081a1bb265ecf3b26c0dc0cb216b5cc4adbc4",
        "content": [
            {
                "annotations": [],
                "text": "The weather in San Francisco tonight is cold. You might want to wear layers or bring a jacket to stay comfortable while you're out. Let me know if you want outfit advice or tips on where to go!",
                "type": "output_text",
                "logprobs": []
            }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message"
    }
]
```

## Troubleshooting

### Problem: "command not found: ng_run"
Make sure you activated the virtual environment:
```bash
source .venv/bin/activate
```

### Problem: "Missing mandatory value: policy_api_key"
Check your `env.yaml` file has the correct API key format. Do not surround your API key with quotes.

### Problem: "python: command not found"
Try `python3` instead of `python`, or check your virtual environment.

### Problem: No output from client script
Make sure the servers are still running in the other terminal.

### Problem: OpenAI API errors
- Verify your API key is valid
- Check you have sufficient credits
- Ensure the model name is correct

## File Structure After Setup

Your directory should look like this:
```
Gym/
├── env.yaml                    # Your API credentials (git-ignored)
├── .venv/                      # Virtual environment (git-ignored)
├── nemo_gym/                   # Core framework code
├── resources_servers/          # Tools and environments
├── responses_api_models/       # Model integrations  
├── responses_api_agents/       # Agent implementations
└── tutorials/                  # These tutorial files
```

## What's Running?

When you ran `ng_run`, you started a complete AI agent system:

- **Web servers** handling HTTP requests
- **Agent logic** coordinating between components
- **Weather tool** ready to be called
- **OpenAI integration** ready to think and respond

## Next Steps

Now that everything is working, let's understand what just happened and how to interact with your agent.

→ **[Next: Your First Agent]**
