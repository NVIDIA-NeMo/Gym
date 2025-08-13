# Table of Contents
- [Table of Contents](#table-of-contents)
- [NeMo-Gym](#nemo-gym)
- [Setup](#setup)
  - [Helpful development commands](#helpful-development-commands)
- [How To: Run a simple agent](#how-to-run-a-simple-agent)
  - [Introduction](#introduction)
  - [Configs](#configs)
  - [Running servers](#running-servers)
  - [Reasoning in the Response API](#reasoning-in-the-response-api)
  - [Run tests for simple agent](#run-tests-for-simple-agent)
- [How To: Add a resource server](#how-to-add-a-resource-server)
- [How To: Upload and download a dataset from Gitlab](#how-to-upload-and-download-a-dataset-from-gitlab)
- [How To: Offline trajectory collection](#how-to-offline-trajectory-collection)

# NeMo-Gym
# Setup
Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Initialize environment
```bash
uv venv --python 3.13
source .venv/bin/activate
```

Install NeMo Gym
```bash
uv sync
```


## Helpful development commands
Lint
```bash
ruff check --fix
```

Format
```bash
ruff format
```

Run Nemo Gym tests
```bash
ng_dev_test
```

View test coverage
```bash
coverage html
```

Run tests for a single server e.g. `responses_api_agents/simple_agent`
```bash
ng_test +entrypoint=responses_api_agents/simple_agent
```

Run all server tests
```bash
ng_test_all
```


# How To: Run a simple agent
Reading time: 10 mins
Date: Mon Aug 04, 2025

## Introduction
In this example, we will run a simple agent that uses the GPT 4.1 model and has access to a very simple dummy get_weather tool. NeMo Gym has three core abstractions: models, resources, and agents.

1. Models - found under `responses_api_models`, NeMo Gym's model abstraction contains OpenAI Chat Completions and Responses compatible interfaces. Models are intended to abstract out any model quirks, e.g. pointing to an OpenAI endpoint or a local VLLM instance, using a reasoning model or a non-reasoning model, using a model with different chat templating, etc, so that Agents can freely point to any model instance.
   1. Think “gpt 4.1”, “o3”, “claude sonnet”, “nano v2”, etc.
2. Resources - found under `resources_servers`, NeMo Gym's resource abstraction contains the environment including tool implementations or "step" functions like in OpenAI Gym, as well as any verification or reward logic. Resource servers are intended to abstract out any heavy processing that needs to be done, so that Agents can efficiently async and await on model and resource server calls.
   1. Think "FastAPI server" or "verifier".
3. Agents - found under `responses_api_agents`, NeMo Gym's agent abstraction contains an OpenAI Responses compatible interface. Agents are intended to abstract out any major system designs that sit on top of model and resource servers.
   1. Think “deep research agent”, “search agent”, “customer service agent”, “Claude code”, “math agent”, etc.

The diagram below shows the rough mental model of these three core abstractions, as well as the positioning of NeMo Gym.

![](resources/rl_verifiers_system_design.png)


## Configs
NeMo Gym operates using YAML configuration files and command line arguments via Hydra and OmegaConf. The rough skeleton of a config is annotated and shown below, using the simple agent config as an example `responses_api_agents/simple_agent/configs/simple_agent.yaml`.
```yaml
# `simple_agent` here is the name or ID of this server and will be used to identify it in subsequent requests.
# If you spin up multiple servers, you must ensure that each name/ID is unique.
simple_agent:
  # This is the server type. There are 3 server types: responses_api_models, resources_servers, and responses_api_agents.
  # These server types are all held in the three folders in the top-level directory of NeMo-Gym, parallel to the nemo_gym folder.
  responses_api_agents:
    # This is the model/resource/agent type. This is custom and written by you.
    # This must be the name of the folder inside the server type folder.
    simple_agent:
      # This is the server entrypoint path, relative to the agent type folder. When your server is run, it will be run through here.
      entrypoint: app.py
      # Everything below here is a server-specific variable. In this case (as we will see in a second), there are two top-level variables `resources_server` and `model_server`.
      resources_server:
        type: resources_servers
        # This `???` is Hydra syntax for a required but missing field
        name: ???
      model_server:
        type: responses_api_models
        name: openai_model
```

This is how this YAML config translates to the simple agent config as defined in Python in `responses_api_agents/simple_agent/app.py`.
```python
from nemo_gym.server_utils import ResourcesServerRef, ModelServerRef

class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
```

You can define your server configs to require or accept any arbitrary structures or values. In this case, we require two variables `resources_server` and `model_server` that are server reference objects. These server reference objects are how you can refer to one server from another server, in a server-instance agnostic way. For example, this SimpleAgentConfig doesn't need any `model_server` in particular, just __a__ `model_server`.

If your config contains a server reference that doesn't exist, NeMo Gym will let you know e.g.:
```bash
AssertionError: Could not find type='responses_api_models' name='simple_model_server' in the list of available servers: [AgentServerRef(type='responses_api_agents', name='simple_agent'), ModelServerRef(type='responses_api_models', name='openai_model'), ResourcesServerRef(type='resources_servers', name='simple_weather')]
```

If your config is missing an argument or argument value, NeMo Gym will let you know e.g.:
```bash
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: openai_model.responses_api_models.openai_model.openai_api_key
    full_key: openai_model.responses_api_models.openai_model.openai_api_key
    object_type=dict
```


## Running servers
In NeMo Gym, you run servers using the `ng_run` or `nemo_gym_run` bash commands. You can pass in configurations in three ways: as YAML config paths, as part of a local `env.yaml` file, or as part of command line args. For example, a run command might look like:
```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"
ng_run "+config_paths=[$config_paths]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=simple_weather
```
We provide our Yaml config files using the `config_paths` command line argument. We specify 3 configs, one for our simple agent, which relies on our simple model server and simple weather servers. By default, the simple agent doesn't point to any specific resources server (see the `resources_server... name: ???` above), so we provide this pointer via command line using Hydra syntax `simple_agent.responses_api_agents.simple_agent.resources_server.name=simple_weather`.

Our example relies on an OpenAI model server that uses GPT 4.1 by default. We also need to provide our OpenAI API key in order to properly run this example. At runtime, NeMo Gym will read from a local and git-ignored file at `env.yaml`. This `env.yaml` file is intended to hold sensitive information that should not be checked in, like API keys or other secrets. Create your `env.yaml` file in this directory, copy in the following information, and add your OpenAI API key.
```yaml
openai_model:
  responses_api_models:
    openai_model:
      openai_api_key: {your OpenAI API key}
```

You can also use env.yaml to store config values for convenience e.g. in `env.yaml`:
```yaml
simple_weather_config_paths:
- responses_api_agents/simple_agent/configs/simple_agent.yaml
- responses_api_models/openai_model/configs/openai_model.yaml
- resources_servers/simple_weather/configs/simple_weather.yaml
```
Then you can run NeMo Gym like:
```bash
ng_run '+config_paths=${simple_weather_config_paths}' \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=simple_weather
```


Config values will be resolved in the following order: Earlier config paths < later config paths < env.yaml < command line args.

After filling in your OpenAI API key, run the `ng_run` command above. You should see an output that looks like this:
```bash
INFO:     Started server process [49744]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:11000 (Press CTRL+C to quit)
Audited 1 package in 6ms
Activate with: source .venv/bin/activate
Audited 1 package in 8ms
Audited 1 package in 248ms
INFO:     Started server process [49762]
INFO:     Uvicorn running on http://127.0.0.1:62922 (Press CTRL+C to quit)
INFO:     Started server process [49761]
INFO:     Uvicorn running on http://127.0.0.1:62920 (Press CTRL+C to quit)
INFO:     Started server process [49768]
INFO:     Uvicorn running on http://127.0.0.1:62921 (Press CTRL+C to quit)
```

Now we can query our agent.
```bash
python responses_api_agents/simple_agent/client.py
```
Inside the client.py file, we import the `ServerClient` class and instantiate a `server_client`. The server client is immediately usable to query our Responses API-compatible agent. This is also how you query servers from inside other servers at runtime.
```python
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient

server_client = ServerClient.load_from_global_config()
server_client.post(
    server_name="simple_agent",  # This is your server name or ID
    url_path="/v1/responses",
    json=NeMoGymResponseCreateParamsNonStreaming(...),
)
...
```
You should see an output like this:
```bash
[2025-08-04 20:35:19,983][httpx][INFO] - HTTP Request: POST http://127.0.0.1:62920/v1/responses "HTTP/1.1 200 OK"
[
    {
        "arguments": "{\"city\":\"San Francisco\"}",
        "call_id": "call_LUyS0YUp4TMk7IjHR1ZQPDr7",
        "name": "get_weather",
        "type": "function_call",
        "id": "fc_68917bf6d228819ca904b692df8406fc0f318621c04a7037",
        "status": "completed"
    },
    {
        "call_id": "call_LUyS0YUp4TMk7IjHR1ZQPDr7",
        "output": "{\"city\": \"San Francisco\", \"weather_description\": \"The weather in San Francisco is cold.\"}",
        "type": "function_call_output"
    },
    {
        "id": "msg_68917bf79324819c9a004816a1b2f47e0f318621c04a7037",
        "content": [
            {
                "annotations": [],
                "text": "The weather in San Francisco is currently cold. If you need more details, such as temperature or a specific forecast, let me know!",
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


When you run NeMo Gym, a head server will spin up that contains the single source of truth configuration for all of its servers. This header server is what the `ServerClient.load_from_global_config()` reads from in order to get information about how to query each individual server. This way, all hostnames and ports are abstracted away from any consumers of NeMo Gym. However, a host and port can still be specified for any server if the orchestrator wishes so. If no port is specified, a random one will be chosen by the operating system.


## Reasoning in the Response API
Agents and verifiers work with responses in a standardized format based on the OpenAI Responses API schema. The verifier receives an object where the `output` field conforms to the Response object output [documented here]("https://platform.openai.com/docs/api-reference/responses/object#responses/object-output").

The `output` list may contain multiple item types, such as:
- `ResponseOutputMessage` - The main user-facing message content returned by the model.
- `ResponseOutputItemReasoning` - Internal reasoning or "thinking" traces that explain the model’s thought process.
- `ResponseFunctionToolCall` - A request from the model to invoke an external function or tool.

**Example**
If a chat completion contains both thinking traces and user-facing text:
```python
ChatCompletion(
    Choices=[
        Choice(
            message=ChatCompletionMessage(
                content="<think>I'm thinking</think>Hi there!",
                tool_calls=[{...}, {...}],
                ...
            )
        )
    ],
    ...
)
```
In the Responses schema, this would be represented as:
```python
Response(
    output=[
        ResponseOutputItemReasoning(
            type="reasoning",
            summary=[
                Summary(
                    type="summary_text",
                    text="I'm thinking",
                )
            ]
        ),
        ResponseOutputMessage(
            role="assistant",
            type="message",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text="Hi there!",
                )
            ]
        ),
        ResponseFunctionToolCall(
            type="function_call",
            ...

        ),
        ResponseFunctionToolCall(
            type="function_call",
            ...

        ),
        ...
    ]
)
```

Reasoning traces (`Reasoning` items) are parsed before the verifier processes the output. The parsing is **model-specific**, and the verifier does not need to worry about the extracting or interpreting reasoning traces. The verifier receives these items already separated and clearly typed.


## Run tests for simple agent
Run the Simple Chat Agent tests. `ng_test` or `nemo_gym_test` stands for `Nemo Gym Test`.
```bash
ng_test +entrypoint=responses_api_agents/simple_agent
```

Tests are strongly encouraged and you must have at least one test for every server you make. Test coverage is not explicitly required which means that **YOU ARE RESPONSIBLE FOR YOUR OWN SERVER CORRECTNESS AND FUNCTION**.


# How To: Add a resource server
Reading time: 5 mins
Date: Tue Aug 05, 2025

Resource servers are used to abstract out any business logic of tool implementations and verifiers. Each resource server must implement a `verify` function.

Resource servers live in the `resources_servers` folder. Initialize a resource server now. For this example, we will be writing a dummy test weather server.
```bash
ng_init_resources_server +entrypoint=resources_servers/test_weather
```

For the purposes of this example, we don't have any external dependencies, but if you want to add server-specific requirements, you would do so in the `requirements.txt` file. You can add requirements for external PyPI packages or Github repos.
```txt
-e nemo-gym @ ../../
{additional dependencies here}
```


Implement a tool for your agent to use in `app.py`. Start by adding your request and response schemas
```python
...
class TestWeatherResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GetWeatherRequest(BaseModel):
    city: str


class GetWeatherResponse(BaseModel):
    city: str
    weather_description: str


class TestWeatherResourcesServer(SimpleResourcesServer):
    config: TestWeatherResourcesServerConfig

...
```
Implement a `get_weather` function under the `TestWeatherResourcesServer` class. For now we will just always say it is cold.
```python
...
        # app.post("/get_weather")(self.get_weather)

        return app

    async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
        return GetWeatherResponse(
            city=body.city, weather_description=f"The weather in {body.city} is cold."
        )

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)
...
```
Register your new `get_weather` function as a FastAPI route.
```python
...
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        app.post("/get_weather")(self.get_weather)

        return app
...
```

You can see a complete example of `app.py` in `resources_servers/simple_weather/app.py`!

Run an agent with your new server!
```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"
ng_run "+config_paths=[$config_paths]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=test_weather
```

Run a query with your new resources server! Your agent should say that it's cold in SF :)
```bash
python responses_api_agents/simple_agent/client.py
```

After you implement your server, please make sure to update the README.md with appropriate licensing information! Your PR will not be merged unless licensing information is present and accurate.


# How To: Upload and download a dataset from Gitlab
We want to track and version golden versions of our datasets so that we always know what data is being trained on and that the data we are training on is high quality. Major versions of all training datasets should be tracked in NeMo Gym. For example, the HelpSteer dataset https://huggingface.co/datasets/nvidia/HelpSteer3 has 3 major versions 1, 2, and 3. Each of these major versions would be uploaded and tracked in NeMo Gym.

Right now, NeMo Gym is hosted in Nvidia Gitlab and we use Gitlab's model artifact registry to store datasets. https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models?first=30&orderBy=created_at&sort=desc#/

Gitlab uses MLFlow to interface with its model artifact registry. You will need:
1. The NeMo Gym repository Gitlab URI.
   1. Go to the Model Registry page, click the "..." next to "Create model", then click "Using the MLFlow client".
   2. The URI will look something like `https://gitlab-master.nvidia.com/api/v4/projects/191584/ml/mlflow/`
2. Your Gitlab token. Your Gitlab token must have the `api` and `read_api` scopes.

Provide your MLFlow credentials in `env.yaml`. 
```yaml
mlflow_tracking_uri: {your NeMo Gym Gitlab URI}
mlflow_tracking_token: {your Gitlab PAT}
```

Upload a dataset to Gitlab model artifact registry. Dataset name will be your model artifact name. Version must be a str in the format `x.x.x`.
```bash
ng_upload_dataset_to_gitlab \
    +dataset_name=multineedle \
    +version=0.0.1 \
    +input_jsonl_fpath=data/multineedle_benchmark.jsonl
```

Download a dataset from Gitlab model artifact registry.
```bash
ng_download_dataset_to_gitlab \
    +dataset_name=multineedle \
    +version=0.0.1 \
    +artifact_fpath=multineedle_benchmark.jsonl \
    +output_fpath=data/multineedle_benchmark.jsonl
```


# How To: Offline trajectory collection
Reading time: 5 mins
Date: Tue Aug 05, 2025

Spin up your agent. For this example, we will use the multineedle resources server!
```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"
ng_run "+config_paths=[$config_paths]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=multineedle
```

Download the MultiNeedle data
```bash
ng_download_dataset_to_gitlab \
    +dataset_name=multineedle \
    +version=0.0.1 \
    +artifact_fpath=multineedle_benchmark.jsonl \
    +output_fpath=data/multineedle_benchmark.jsonl
```

Run trajectory collection.
```bash
ng_collect_traj +agent_name=simple_agent \
    +input_jsonl_fpath=data/multineedle_benchmark.jsonl \
    +output_jsonl_fpath=results/multineedle_trajectory_collection.jsonl \
    +limit=null
```

View the trajectories just collected!
```
ng_viewer +jsonl_fpath=results/multineedle_trajectory_collection.jsonl
```
