(gs-setup-installation)=

# Setup and Installation

Welcome to NeMo Gym! In this tutorial, you will install NeMo Gym, configure your training environment—using a simple weather example to verify all components work together. This allows you to test your environment before collecting rollouts at scale.

:::{card}

**Goal**: Get NeMo Gym installed and servers running, then verify all components work together.

^^^

**In this tutorial, you will**:

1. Clone the repository and install dependencies
2. Configure your OpenAI API key
3. Start the NeMo Gym servers
4. Test the setup

:::

## Before You Start

Make sure you have these prerequisites ready before beginning:

- **Python 3.12+** (check with `python3 --version`)
- **Git** (for cloning the repository)
- **OpenAI API key with available credits** (for the tutorial agent)

---

## 1. Clone and Install

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

**✅ Success Check**: Verify that you can see something that indicates a newly activated environment such as `(.venv)` or `(NeMo-Gym)` in your terminal prompt.

## 2. Configure Your API Key

Create an `env.yaml` file in the project root:

```bash
# Create env.yaml with your OpenAI credentials
cat > env.yaml << EOF
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-openai-api-key-here
policy_model_name: gpt-4.1-2025-04-14
EOF
```

:::{important}
Replace `sk-your-actual-openai-api-key-here` with your real OpenAI API key. This file keeps secrets out of version control while making them available to NeMo Gym.

**Requirements**:

- Your API key must have available credits (check [OpenAI billing](https://platform.openai.com/account/billing) 🔗)
- The model must support function calling (most GPT-4 models do)
- Refer to [OpenAI's models documentation](https://platform.openai.com/docs/models) 🔗 for available models

:::

:::{dropdown} Optional: Validate your API key before proceeding

Want to catch configuration issues early? Test your API key before starting servers:

```bash
python -c "
import openai
import yaml

# Load your configuration
with open('env.yaml') as f:
    config = yaml.safe_load(f)

# Test API access
client = openai.OpenAI(
    api_key=config['policy_api_key'],
    base_url=config['policy_base_url']
)

# Try a simple request
response = client.chat.completions.create(
    model=config['policy_model_name'],
    messages=[{'role': 'user', 'content': 'Say hello'}],
    max_tokens=10
)
print('✅ API key validated successfully!')
print(f'Model: {config[\"policy_model_name\"]}')
print(f'Response: {response.choices[0].message.content}')
"
```

**✅ Success Check**: Verify that you can see "API key validated successfully!" and a response from the model.

If this step fails, you will see a clear error message (like quota exceeded or invalid key) before investing time in server setup.

:::

:::{dropdown} Troubleshooting: "Missing mandatory value: policy_api_key"
Check your `env.yaml` file has the correct API key format. Do not surround your API key with quotes.
:::

## 3. Start the Servers

```bash
# Define which servers to start
config_paths="resources_servers/example_simple_weather/configs/simple_weather.yaml,responses_api_models/openai_model/configs/openai_model.yaml"

# Start all servers
ng_run "+config_paths=[${config_paths}]"
```

**✅ Success Check**: Verify that you can see output like:
```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://127.0.0.1:11000 (Press CTRL+C to quit)
INFO:     Started server process [12346]  
INFO:     Uvicorn running on http://127.0.0.1:62920 (Press CTRL+C to quit)
...
```

:::{note}
The head server always uses port **11000**. Other servers get automatically assigned ports (like 62920, 52341, etc.) - your port numbers will differ from the example above.
:::

When you ran `ng_run`, it started all the servers you configured:

- **Head server:** coordinating all components
- **Resources server:** defining tools and verification
- **Model server:** providing LLM inference
- **Agent server:** orchestrating how the model interacts with the resources

:::{dropdown} Troubleshooting: "command not found: ng_run"
Make sure you activated the virtual environment:

```bash
source .venv/bin/activate
```

:::

## 4. Test the Setup

Open a **new terminal** (keep servers running in the first one):

```bash
# Navigate to project directory
cd /path/to/Gym

# Activate virtual environment
source .venv/bin/activate

# Test the agent
python responses_api_agents/simple_agent/client.py
```

**✅ Success Check**: Verify that you can see JSON output showing:
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

:::{dropdown} Troubleshooting: "python: command not found"
Try `python3` instead of `python`, or check your virtual environment.
:::

:::{dropdown} Troubleshooting: No output from client script
Make sure the servers are still running in the other terminal.
:::

:::{dropdown} Troubleshooting: OpenAI API errors or "500 Internal Server Error"

If you encounter errors when running the client, check these common causes:

**Quota/billing errors** (most common):

```text
Error code: 429 - You exceeded your current quota
```

- **Solution**: Add credits to your OpenAI account at [platform.openai.com/account/billing](https://platform.openai.com/account/billing) 🔗
- The tutorial requires minimal credits (~$0.01-0.05 per run)

**Invalid API key**:

```text
Error code: 401 - Incorrect API key provided
```

- **Solution**: Verify your API key in `env.yaml` matches your [OpenAI API keys](https://platform.openai.com/api-keys) 🔗
- Ensure no extra quotes or spaces around the key

**Model access errors**:

```text
Error code: 404 - Model not found
```

- **Solution**: Ensure your account has access to the model specified in `policy_model_name`
- Try using `gpt-4o` or `gpt-4-turbo` if `gpt-4.1-2025-04-14` isn't available

**Testing your API key**:

```bash
# Quick test to verify API access
python -c "
import openai
client = openai.OpenAI()
print(client.models.list().data[0])
"
```

:::

## File Structure After Setup

Your directory should look like this:

```bash
Gym/
├── env.yaml                    # Your API credentials (git-ignored)
├── .venv/                      # Virtual environment (git-ignored)
├── nemo_gym/                   # Core framework code
├── resources_servers/          # Tools and environments
├── responses_api_models/       # Model integrations  
├── responses_api_agents/       # Agent implementations
└── docs/                       # Documentation files
```

