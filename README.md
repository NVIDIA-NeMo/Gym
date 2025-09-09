# Table of Contents
- [Table of Contents](#table-of-contents)
- [NeMo-Gym](#nemo-gym)
- [Setup](#setup)
  - [Helpful development commands](#helpful-development-commands)
- [How To: Run a simple agent](#how-to-run-a-simple-agent)
  - [TL;DR](#tldr)
  - [Introduction](#introduction)
  - [Configs](#configs)
    - [Special policy model placeholders](#special-policy-model-placeholders)
  - [Running servers](#running-servers)
  - [OpenAI Responses vs Chat Completions API](#openai-responses-vs-chat-completions-api)
  - [Run tests for simple agent](#run-tests-for-simple-agent)
- [How To: Add a resource server](#how-to-add-a-resource-server)
  - [TLDR final expected artifacts](#tldr-final-expected-artifacts)
- [How To: Upload and download a dataset from Gitlab](#how-to-upload-and-download-a-dataset-from-gitlab)
- [How To: Offline rollout collection or synthetic data generation](#how-to-offline-rollout-collection-or-synthetic-data-generation)
- [How To: Prepare and validate data for PR submission or RL training](#how-to-prepare-and-validate-data-for-pr-submission-or-rl-training)
- [How To: ng\_dump\_config - Dump a YAML config as exactly as NeMo Gym sees it](#how-to-ng_dump_config---dump-a-yaml-config-as-exactly-as-nemo-gym-sees-it)
- [How To: Use NeMo Gym with a non-Responses compatible API endpoint like vLLM](#how-to-use-nemo-gym-with-a-non-responses-compatible-api-endpoint-like-vllm)
- [FAQ: VSCode and Git setup](#faq-vscode-and-git-setup)
- [FAQ: SFT and RL](#faq-sft-and-rl)
- [FAQ: Error: Found files with missing copyright](#faq-error-found-files-with-missing-copyright)
- [FAQ: build-docs / Build docs CI failures](#faq-build-docs--build-docs-ci-failures)
- [FAQ: NeMo Gym, training frameworks, and token IDs](#faq-nemo-gym-training-frameworks-and-token-ids)
- [FAQ: NeMo Gym what CI/CD do I need to pass?](#faq-nemo-gym-what-cicd-do-i-need-to-pass)


# NeMo-Gym
# Setup
Clone NeMo-Gym. It's recommended to clone via SSH if you are a developer.
```bash
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym
```

Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Initialize environment
```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install NeMo Gym
```bash
uv sync --extra dev --group docs
```

If you are a developer, install pre-commit hooks
```bash
pre-commit install
```


## Helpful development commands
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

## TL;DR
After setup above:
```bash
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: {your OpenAI API key}
policy_model_name: gpt-4.1-2025-04-14" > env.yaml

config_paths="resources_servers/simple_weather/configs/simple_weather.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"

python responses_api_agents/simple_agent/client.py
```


## Introduction
In this example, we will run a simple agent that uses the GPT 4.1 model and has access to a very simple dummy get_weather tool. NeMo Gym has three core abstractions: models, resources, and agents.

1. Models - found under `responses_api_models`, NeMo Gym's model abstraction contains OpenAI Chat Completions and Responses compatible interfaces. Models are intended to abstract out any model quirks, e.g. pointing to an OpenAI endpoint or a local VLLM instance, using a reasoning model or a non-reasoning model, using a model with different chat templating, etc, so that Agents can freely point to any model instance.
   1. Think “gpt 4.1”, “o3”, “claude sonnet”, “nano v2”, etc.
2. Resources - found under `resources_servers`, NeMo Gym's resource abstraction contains the environment including tool implementations or "step" functions like in OpenAI Gym, as well as any verification or reward logic. Resource servers are intended to abstract out any heavy processing that needs to be done, so that Agents can efficiently async and await on model and resource server calls.
   1. Think "FastAPI server" or "verifier".
3. Agents - found under `responses_api_agents`, NeMo Gym's agent abstraction contains an OpenAI Responses compatible interface. Agents are intended to abstract out any major system designs that sit on top of model and resource servers.
   1. Think “deep research agent”, “search agent”, “customer service agent”, “Claude code”, “math agent”, etc.


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
        name: policy_model
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
AssertionError: Could not find type='responses_api_models' name='simple_model_server' in the list of available servers: [AgentServerRef(type='responses_api_agents', name='simple_agent'), ModelServerRef(type='responses_api_models', name='policy_model'), ResourcesServerRef(type='resources_servers', name='simple_weather')]
```

If your config is missing an argument or argument value, NeMo Gym will let you know e.g.:
```bash
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: policy_model.responses_api_models.openai_model.openai_api_key
    full_key: policy_model.responses_api_models.openai_model.openai_api_key
    object_type=dict
```


### Special policy model placeholders
There is one set of special NeMo Gym variables relating to the agent policy model. These are the `policy_base_url`, `policy_api_key`, `policy_model_name` variables. When you go to train a model, these are the information that will be used to query the model server endpoint you are trying to train. By default, every agent will refer to this shared `policy_model` model server.
```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model_name: ${policy_model_name}
```


## Running servers
In NeMo Gym, you run servers using the `ng_run` or `nemo_gym_run` bash commands. You can pass in configurations in three ways: as YAML config paths, as part of a local `env.yaml` file, or as part of command line args. For example, a run command might look like:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"
ng_run "+config_paths=[$config_paths]" \
    +simple_weather_simple_agent.responses_api_agents.simple_agent.resources_server.name=simple_weather
```
We provide our Yaml config files using the `config_paths` command line argument. We specify 3 configs, one for our simple agent, which relies on our simple model server and simple weather servers. By default, the simple agent doesn't point to any specific resources server (see the `resources_server... name: ???` above), so we provide this pointer via command line using Hydra syntax `simple_agent.responses_api_agents.simple_agent.resources_server.name=simple_weather`.

Our example relies on an OpenAI model server. We need to provide our OpenAI API key and other model information in order to properly run this example. At runtime, NeMo Gym will read from a local and git-ignored file at `env.yaml`. This `env.yaml` file is intended to hold sensitive information that should not be checked in, like API keys or other secrets. Create your `env.yaml` file in this directory, copy in the following information, and add your OpenAI API key.
```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: {your OpenAI API key}
policy_model_name: gpt-4.1-2025-04-14
```
Please never commit any secrets in your config files! We explicitly provide a way to avoid this using the `env.yaml`. You should run `touch env.yaml` and your NeMo Gym folder should look like this i.e. if you run `ls .` you should see something like:
```
...
cache
data
nemo_gym
...
env.yaml
...
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


**Config values will be resolved in the following order: Earlier config paths < later config paths < env.yaml < command line args.**


After filling in your OpenAI API key, run the `ng_run` command below.
```bash
config_paths="resources_servers/simple_weather/configs/simple_weather.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```
You should see an output that looks like this:
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
    server_name="simple_weather_agent",  # This is your server name or ID
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


When you run NeMo Gym, a head server will spin up that contains the single source of truth configuration for all of its servers. This header server is what the `ServerClient.load_from_global_config()` reads from in order to get information about how to query each individual server. This way, all hostnames and ports are abstracted away from any consumers of NeMo Gym. However, a host and port can still be specified for any server if the orchestrator wishes so. If no port is specified, a random one will be chosen by the operating system.


## OpenAI Responses vs Chat Completions API
Agents and verifiers work with responses in a standardized format based on the OpenAI Responses API schema. The verifier receives an object where the `output` field conforms to the Response object output [documented here](https://platform.openai.com/docs/api-reference/responses/object#responses/object-output).

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
```
-e nemo-gym[dev] @ ../../
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


Run the tests for your server
```bash
ng_test +entrypoint=resources_servers/simple_weather
```


You can also run detailed tests after running tests the first time
```bash
cd resources_servers/simple_weather
source .venv/bin/activate
pytest
```

At some point, you will want to actually add data that can be used to query your server. Please follow the instructions for [How To: Prepare and validate data for PR submission or RL training](#how-to-prepare-and-validate-data-for-pr-submission-or-rl-training).


If you need some dataset preprocessing or formatting scripts, please place them your resources server directory e.g. `resources_servers/simple_weather/my_preprocess_script.py`.


You are required to have the following 3 files in your resources server data folder:
1. example.jsonl - contains 5 example inputs to an agent server that uses your resources server. These examples need to be created on your own using whatever data processing script you want. It's highly suggested to store the data processing scripts in each folder if possible.
2. example_metrics.json - the metrics for the examples above, as output by `ng_prepare_data` in the data validation flow above.
3. example_rollouts.jsonl - rollouts through your resources server for the 5 example inputs in example.jsonl.


## TLDR final expected artifacts
1. All the artifacts produced by `ng_init_resources_server +entrypoint=resources_servers/test_weather`. Your agent and resources server must be runnable.
```bash
multineedle_config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/multineedle/configs/multineedle.yaml"
ng_run "+config_paths=[${multineedle_config_paths}]"
```
2. At least 1 test at `resources_servers/test_weather/tests/test_app.py`.
3. 5 examples found at `resources_servers/test_weather/data/examples.jsonl`
4. Example metrics as output by `ng_prepare_data` found at `resources_servers/test_weather/data/example_metrics.json`
```bash
multineedle_config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/multineedle/configs/multineedle.yaml"
ng_prepare_data "+config_paths=[${multineedle_config_paths}]" \
    +output_dirpath=data/multineedle \
    +mode=example_validation
```
5. Example rollouts as output by `ng_collect_rollouts` found at `resources_servers/test_weather/data/example_rollouts.jsonl`
```bash
ng_collect_rollouts +agent_name=multineedle_simple_agent \
    +input_jsonl_fpath=resources_servers/multineedle/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/multineedle/data/example_rollouts.jsonl \
    +limit=null \
    +num_repeats=null \
    +num_samples_in_parallel=null
```


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
ng_download_dataset_from_gitlab \
    +dataset_name=multineedle \
    +version=0.0.1 \
    +artifact_fpath=multineedle_benchmark.jsonl \
    +output_fpath=data/multineedle_benchmark.jsonl
```


# How To: Offline rollout collection or synthetic data generation
Reading time: 5 mins
Date: Tue Aug 05, 2025

NeMo Gym can be used for rollout collection e.g. for DPO or for synthetic data generation e.g. for SFT!

Spin up your agent. For this example, we will use the multineedle resources server!
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/multineedle/configs/multineedle.yaml"
ng_run "+config_paths=[$config_paths]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=multineedle
```

Download the MultiNeedle data
```bash
ng_download_dataset_from_gitlab \
    +dataset_name=multineedle \
    +version=0.0.1 \
    +artifact_fpath=multineedle_benchmark.jsonl \
    +output_fpath=data/multineedle_benchmark.jsonl
```

Run rollout collection.
```bash
ng_collect_rollouts +agent_name=multineedle_simple_agent \
    +input_jsonl_fpath=data/multineedle_benchmark.jsonl \
    +output_jsonl_fpath=results/multineedle_rollout_collection.jsonl \
    +limit=null \
    +num_repeats=null \
    +num_samples_in_parallel=null
```

View the rollouts just collected!
```
ng_viewer +jsonl_fpath=results/multineedle_rollout_collection.jsonl
```

# How To: Prepare and validate data for PR submission or RL training
When you use `ng_init_resources_server +entrypoint=resources_servers/multineedle` to initialize a resources server, you will get a config.yaml that looks like the below code block. The dataset information for training, validation, and example will be inside the scope of your agent config (e.g. under simple_agent) and is a list of dataset objects.

```yaml
multineedle_resources_server:
  resources_servers:
    multineedle:
      entrypoint: app.py
multineedle_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: multineedle_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        license: Apache 2.0
        jsonl_fpath: resources_servers/multineedle/data/train.jsonl
        gitlab_identifier:
          dataset_name: multineedle
          version: 0.0.1
          artifact_fpath: multineedle/train.jsonl
        license: Apache 2.0
      - name: validation
        type: validation
        license: Apache 2.0
        jsonl_fpath: resources_servers/multineedle/data/validation.jsonl
        gitlab_identifier:
          dataset_name: multineedle
          version: 0.0.1
          artifact_fpath: multineedle/validation.jsonl
        license: Apache 2.0
      - name: example
        type: example
        jsonl_fpath: resources_servers/multineedle/data/example.jsonl
```

A dataset object consists of:
- Name: An identifier for you
- Type: train, validation, or example. Train and validation are as used in NeMo RL or other train frameworks. More information about the example type is in the next section.
- Jsonl fpath: the local file path to your jsonl file for this dataset.
- Gitlab identifier: The remote path to the dataset as held in the Gitlab dataset registry. This field is required for train and validation datasets. (Not required for example datasets since those are required to be committed to Git).
- License: The license of that dataset. Required for train and validation datasets and not required for example datasets, similar in principle to the Gitlab identifier.
- Start idx, end idx: used for slicing your dataset.
```yaml
- name: train
  type: train
  jsonl_fpath: resources_servers/multineedle/data/train.jsonl
  gitlab_identifier:
    dataset_name: multineedle
    version: 0.0.1
    artifact_fpath: multineedle/validation.jsonl
  license: Apache 2.0
```

Each config.yaml in the resources server requires at least one agent with one example dataset. This example dataset is the first 5 rows of your train dataset that is used for sanity checks on the format for your dataset and the format of each individual example and for others to quickly understand your data.

For every PR that contributes data, we require common dataset statistics and sanity checks on the data itself. This process is also helpful to catch any simple issues before you ever train with NeMo RL. NeMo Gym provides a helper command ng_prepare_data to do so.
```bash
config_paths="resources_servers/multineedle/configs/multineedle.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/multineedle \
    +mode=example_validation

# Run NeMo Gym servers the exact same way with the same configs!
ng_run "+config_paths=[$config_paths]"
```

The `ng_prepare_data` command will:
1. Attempt to load all the datasets you specified from disk. Missing datasets will be reported before any processing is done.
2. For each dataset, read example by example. Check the format and report the filepaths and indices/ranges of offending examples if any.
   1. We only require that the dataset has one key responses_create_params which is valid Responses API schema.
3. Compute aggregate statistics, print them to terminal, and save them next to the jsonl fpaths.
   1. Number of examples
   2. Avg/max/min number of tools
   3. Input length in terms of OpenAI tokens
   4. Avg/max/min number of turns
   5. Number of unique create params
   6. Avg/max/min temperature and other sampling params
   7. Number of unique user messages
4. Check that the aggregate statistics of individual datasets match those of existing aggregate statistics.
5. Collate all the examples into one final train and validation dataset jsonl files at the output dirpath specified for downstream NeMo RL or other train framework consumption.
6. The final aggregate statistics are reported and saved next to the train and validation datasets.
7. [NeMo RL train] Use the exact same config paths to ng_prepare_data and the train/validation dataset paths output in step 5. There is no special pre or post processing done in the NeMo Gym/RL integration other than shuffling and distributed data loading. What you see is what you get.


The `ng_prepare_data` command has 2 modes, one for actual train and validation set preparation, and one for example validation intended to sanity check your data format. You would typically run `+mode=example_validation` when first contributing a resources server, and then run with `+mode=train_preparation` when you actually go to train.
```bash
config_paths="resources_servers/multineedle/configs/multineedle.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/multineedle \
    +mode=example_validation
```


# How To: ng_dump_config - Dump a YAML config as exactly as NeMo Gym sees it
```bash
# Example ng_run command
config_paths="resources_servers/multineedle/configs/multineedle.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[$config_paths]"


# Dump the exact yaml config that NeMo gym sees, just by swapping ng_run -> ng_dump_config
ng_dump_config "+config_paths=[$config_paths]"
```


# How To: Use NeMo Gym with a non-Responses compatible API endpoint like vLLM
As of Sep 05, 2025, not many models have been trained with middlewares or chat templates that are easily parseable to OpenAI Responses API schema, with the notable exception of OpenAI's own open source model GPT-OSS. Since Gym is first-party Responses API, this makes Gym very difficult to use with basically any model.

As a result, we provide a Responses API to Chat Completions mapping middleware layer in the form of `responses_api_models/vllm_model`. VLLMModel assumes that you are pointing to a vLLM instance (since it relies on vLLM-specific endpoints like `/tokenize` and vLLM-specific arguments like `return_tokens_as_token_ids`).

**To use VLLMModel, just change the `responses_api_models/openai_model/configs/openai_model.yaml` in your config paths to `responses_api_models/vllm_model/configs/vllm_model.yaml`!**
```bash
config_paths="resources_servers/multineedle/configs/multineedle.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```

Here is an e2e example of how to spin up a NeMo Gym compatible vLLM Chat Completions OpenAI server.
- If you want to use tools, please find the appropriate vLLM arguments regarding the tool call parser to use. In this example, we use Qwen3-30B-A3B, which is suggested to use the `hermes` tool call parser.
- **Important note**: Please do NOT use a reasoning parser argument to vLLM here. The Responses to Chat Completions middleware logic needs to parse to and from Responses Reasoning items and Chat Completion Message content. **Do NOT use things like `--reasoning-parser qwen3`**.
```bash
uv venv --python 3.12 --seed 
source .venv/bin/activate
# hf_transfer for faster model download. datasets for downloading data from HF
uv pip install hf_transfer datasets vllm --torch-backend=auto

# Qwen/Qwen3-30B-A3B, usable in Nemo RL!
HF_HOME=.cache/ \
HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download Qwen/Qwen3-30B-A3B

HF_HOME=.cache/ \
HOME=. \
vllm serve \
    Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240
```


# FAQ: VSCode and Git setup
Here are some suggestions for easier development using the VSCode code editor.

VSCode workspace settings at `.vscode/settings.json`
```
{
    "git.enableCommitSigning": true,
    "git.alwaysSignOff": true
}
```

Set up your Github signing keys! https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification#ssh-commit-signature-verification

Specifically, if you visit https://github.com/settings/keys while logged into your account, you should see the following:
1. Under the "SSH keys" major section, there are 2 subsections
   1. Authentication keys
   2. Signing key

More often than node, the SHA256 displayed by Github (SHA256:xxxx) should be the same for the two keys above since you probably want to just use the same SSH key for both purposes. If you do not see the following, please following the signing keys link above!


For developers that sign commits via SSH keys, this is configuration so that VSCode source control is able to sign commits properly!
```bash
git config gpg.format ssh
git config user.signingkey ~/.ssh/id_ed25519.pub
```


# FAQ: SFT and RL
Reading time: 5 mins
Date: Fri Aug 15, 2025

SFT (supervised fine tuning) and RL (reinforcement learning) are two different ways of optimizing your model for different tasks and each have their own use cases.

Let's say you wanted to train your model to be really good at math.
- For SFT, you would take some input math questions and either ask human annotators to provide a gold response, or run it through a stronger teacher model and get your SFT target. And then you would SFT on these input + gold response pairs.
- For RL, you would take some input math questions and implement a way to score model answers. During RL training, you would ask the model you are trying to train these math questions, score the model responses using your scorer, and use the scores as a signal on how to optimize your model. Model responses with higher scores would be encouraged.


One way I like to think about these things is:
- You can do RL on SFT data, where your input is your SFT input, and the model answer scorer is just an exact match on the SFT gold label.
- You can also do SFT on RL data via synthetic data generation, where you run your inputs into some strong teacher model, score the responses, and use the scores to pick your SFT gold label.

Tying back to NeMo Gym, NeMo gym can be used to create synthetic data for SFT training by running strong teacher models on the different environments. Critically, it will also be used as the source of data during RL training.


# FAQ: Error: Found files with missing copyright
If you get an error like this on your PR:
```
Error: Found files with missing copyright:
path= ./resources_servers/comp_coding/scripts/validate_dataset.py
path= ./resources_servers/comp_coding/scripts/build_examples.py
path= ./resources_servers/comp_coding/app.py
```

Please add the following copyright snippet to the top of the files listed:
```python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```


# FAQ: build-docs / Build docs CI failures
If you see some docs building related errors that are kind of cryptic regarding .rst files like
```
updating environment: [config changed ('toc_object_entries_show_parents')] 16 added, 0 changed, 0 removed
reading sources... [100%] index
/Users/bxyu/Documents/nemo-gym/nemo_gym/server_utils.py.rst:3: WARNING: Document headings start at H2, not H1 [myst.header]
/Users/bxyu/Documents/nemo-gym/nemo_gym/server_utils.py.rst:3: WARNING: Document headings start at H2, not H1 [myst.header]
/Users/bxyu/Documents/nemo-gym/README.md:: WARNING: image file not readable: resources/rl_verifiers_system_design.png [image.not_readable]
looking for now-outdated files... none found
pickling environment... done
checking consistency... done
```
You may need to reformat some of your docstrings to Napoleon format docstrings https://sphinxcontrib-napoleon.readthedocs.io/en/latest/


# FAQ: NeMo Gym, training frameworks, and token IDs
One of the goals of NeMo Gym is to act as a rollout tool for LLM post-training, either as synthetic data generation for SFT or as training environments for RL.

RL training frameworks don't typically operate in OpenAI schema; they operate in tokens IDs. It is especially critical to always have the correct token IDs during training so that we stay on-policy and to make sure that what we think the model sees is what the model actually sees. However, when providing this OpenAI schema compatible interface to training environment developers, we lose track of the token IDs in Gym.

For example, say we are training a Qwen 3 family model. During rollouts, the model may sample from the entire token distribution. The token IDs are then decoded into text and subsequently converted to OpenAI schema and returned to the training environment developer. At some point for multi-step and multi-turn scenarios, the training environment developer will call the model again with the previously output OpenAI schema. This re-tokenization causes problems since a single string may map to multiple possible sequences of token IDs. So if the model generations token ID sequence 1 and the re-tokenization outputs token ID sequence 2, suddenly things may become off policy when the Gym result is consumed by the RL training framework.

So, the OpenAI compatible model server in a training framework needs to be able to handle this discrepancy. In order to do that, Gym needs a handle on the ground truth token IDs and it needs to provide that information back to the training frameworks' OpenAI compatible server.

TODO @bxyu-nvidia: expand on this later.


# FAQ: NeMo Gym what CI/CD do I need to pass?

NeMo Gym has an E2E suite of CI/CD in the form of Github actions workflows. Some of these are critical to PR merge and some of the mare not.

For the majority of PRs, there are 5 checks that need to pass:
1. DCO
2. Code linting / Lint check (pull_request)
3. Copyright check / copyright-check / main (pull_request)
4. Secrets detector / secrets-detector / secrets-detector (pull_request)
5. Unit tests / Test (pull_request)

Examples of PR checks that most PRs do not need to wait for to pass:
1. CICD NeMo / cicd-container-build / build / main (push)
2. CICD NeMo / Nemo_CICD_Test (push)
...
