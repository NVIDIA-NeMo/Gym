# Table of Contents
- [Table of Contents](#table-of-contents)
- [NeMo-Gym](#nemo-gym)
- [Setup](#setup)
  - [Helpful development commands](#helpful-development-commands)
- [How To: Run a simple agent](#how-to-run-a-simple-agent)
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
- [FAQ: VSCode and Git setup](#faq-vscode-and-git-setup)
- [FAQ: SFT and RL](#faq-sft-and-rl)
- [FAQ: Why NeMo Gym?](#faq-why-nemo-gym)
- [FAQ: Error: Found files with missing copyright](#faq-error-found-files-with-missing-copyright)
- [FAQ: build-docs / Build docs CI failures](#faq-build-docs--build-docs-ci-failures)

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


### Special policy model placeholders
There is one set of special NeMo Gym variables relating to the target agent model. These are the `policy_base_url`, `policy_api_key`, `policy_model_name` variables. When you go to train a model, these are the information that will be used to query the model server endpoint you are trying to train. By default, every agent will refer to this shared `openai_model` model server.
```yaml
openai_model:
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
ng_collect_rollouts +agent_name=simple_agent \
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
        name: openai_model
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

# FAQ: Why NeMo Gym?

NeMo Gym is a large-scale collection of high-quality verifier environments for multi-verifier RL training.  
To enable this, NeMo Gym provides infra support for the rollout server that runs 100+ verifiers in parallel.

The document below details why we designed NeMo Gym the way we did. It also includes a direct comparative study that clearly differentiates NeMo Gym from other environment frameworks.

\[Banghua\] As of Thu Aug 21:

1. Gym is completely different from any of the alternatives above in terms of data **coverage, quantity and quality.** For example, for math only, gym contains 1M+ high-quality math verifiable dataset curated by our internal team, with great math verify \+ LLM-as-a-judge support. In contrast, SkyRL and verifiers above only have a small train subset of GSM8K and AIME. We also have close to 10k SWE development, which require both high quality data curation efforts and good infra support. In contrast, Aviary only focuses on scientific knowledge environment. **None of the existing frameworks support general multi-turn tool-use agent, with tools like search, code execution, and other synthetic tools.**  
2. We will be a **superset** of all existing gym environments. We are already a super-set of Sky RL Lab Gym and verifiers. We have integrated all GEM environments. We’re working with Aviary to incorporate them as well.  
3. As is shown from Brian’s comparison below, we have much **better infra support for scaling**. And the plan is to use NeMo Gym for 500B+ model training for quality improvement. This will make nemo gym battle tested in frontier model training, while the other gyms are mostly for smaller-scale experiments.

Key use case requirements to avoid training environment scale, complexity, and diversity limitations:

1. Can I easily build my environment without worrying about a training framework?  
2. Can I easily call my model using OpenAI Responses and not worry about reasoning parsing?  
3. Can I easily use your environment framework to build an agent application product?  
4. Can I easily use your environment framework to build a simple multi-agent system?  
5. Can I easily run individual SWE-bench task Docker containers?  
6. Can I easily add an agent built with any agent framework?  
7. Can I easily add any environment framework?  
8. Can I easily simultaneously use math-verify==0.7.0 and math-verify==0.8.0 in 2 different environments?  
9. Can I easily spin up multiple environments at once?

Key principles

1. \[Reqs 1, 2\] Decoupled from training framework  
2. \[Reqs 2, 3, 4, 6, 7\] Standardized behind OpenAI Responses  
3. \[Reqs 3, 4, 6\] Explicit Agent vs model abstraction  
4. \[Reqs 3, 4, 5, 6, 7\] REST environment servers and container compatible  
5. \[Reqs 8, 9\] Separate Python env per server at runtime

\[Brian note\] There are some rows yet to be filled in here.

| Environment framework | Decoupled from training framework | Standardized behind OpenAI Responses | Explicit Agent vs model abstraction | REST environment servers and container compatible | Separate Python env per server at runtime |
| :---- | :---- | :---- | :---- | :---- | :---- |
| NeMo Gym | ✅ | ✅ | ✅ | ✅ | ✅ |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | ❌Environment abstraction only has a step function and is fully orchestrated by training framework ([link](https://github.com/NVIDIA-NeMo/RL/blob/bc24887c72a6e1b2699a228bc87c588546dfe6b7/nemo_rl/environments/interfaces.py#L52)) | ❌Uses OpenAI Chat Completions message-like interface ([link](https://github.com/NVIDIA-NeMo/RL/blob/bc24887c72a6e1b2699a228bc87c588546dfe6b7/nemo_rl/data/llm_message_utils.py#L56)) | ❌Only has policy abstraction | ❌Environments are used via a step function ([link](https://github.com/NVIDIA-NeMo/RL/blob/bc24887c72a6e1b2699a228bc87c588546dfe6b7/nemo_rl/environments/interfaces.py#L52)) | ✅Each environment worker can be run with a separate Python executable ([link](https://github.com/NVIDIA-NeMo/RL/blob/bc24887c72a6e1b2699a228bc87c588546dfe6b7/nemo_rl/environments/math_environment.py#L238)) |
| [SkyRL Lab Gym](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym) | ❌Environment abstraction only has a step function and is fully orchestrated by training framework ([link](https://github.com/NovaSky-AI/SkyRL/blob/825f2e82e289d6011d80957c48132618fed3d460/skyrl-gym/skyrl_gym/core.py#L35)) | ❌Environment is passed a raw string LLM response ([link](https://github.com/NovaSky-AI/SkyRL/blob/825f2e82e289d6011d80957c48132618fed3d460/skyrl-gym/skyrl_gym/core.py#L40)) | ❌Environment is passed a raw string LLM response ([link](https://github.com/NovaSky-AI/SkyRL/blob/825f2e82e289d6011d80957c48132618fed3d460/skyrl-gym/skyrl_gym/core.py#L40)) | ❌Environments are used via a step function ([link](https://github.com/NovaSky-AI/SkyRL/blob/825f2e82e289d6011d80957c48132618fed3d460/skyrl-gym/skyrl_gym/core.py#L35)) | ❌All the tools are in one folder and share dependencies ([link](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym/skyrl_gym/tools%20)) |
| [Axon RL GEM](https://github.com/axon-rl/gem) | ❌Environment abstraction only has a step function and is fully orchestrated by training framework ([link](https://github.com/axon-rl/gem/blob/3bc7695fd60e7196d5c64f5ed6fbc9edf5fff65a/gem/core.py#L29)) | ❌Environment is passed a raw string LLM response ([link](https://github.com/axon-rl/gem/blob/3bc7695fd60e7196d5c64f5ed6fbc9edf5fff65a/gem/envs/code_env.py#L81)) | ❌Environment is passed a raw string LLM response ([link](https://github.com/axon-rl/gem/blob/3bc7695fd60e7196d5c64f5ed6fbc9edf5fff65a/gem/envs/code_env.py#L81)) | ❌Environment abstraction only has a step function and is fully orchestrated by training framework ([link](https://github.com/axon-rl/gem/blob/3bc7695fd60e7196d5c64f5ed6fbc9edf5fff65a/gem/core.py#L29)) | ❌All the [envs](https://github.com/axon-rl/gem/tree/main/gem/envs) and [tools](https://github.com/axon-rl/gem/tree/main/gem/tools) use the same dependencies |
| [willccbb Verifiers](https://github.com/willccbb/verifiers) | ❌Environment abstraction requires a train or validation dataset ([link](https://github.com/willccbb/verifiers/blob/828125a3ade0216e7d0a51a9b362d264f6ab68e0/verifiers/envs/environment.py#L159)) | ❌Uses OpenAI Chat Completions ([link](https://github.com/willccbb/verifiers/blob/828125a3ade0216e7d0a51a9b362d264f6ab68e0/verifiers/types.py#L31)) | ✅Environments are provided an OpenAI client ([link](https://github.com/willccbb/verifiers/blob/9f197f7ececcc1367bde504e881dc4d938a019e1/verifiers/envs/environment.py#L239)) | ❌Environments are directly imported during training ([link](https://github.com/willccbb/verifiers/blob/9f197f7ececcc1367bde504e881dc4d938a019e1/README.md?plain=1#L77)) | ❌Even though each verifier has a separate pyproject.toml ([link](https://github.com/willccbb/verifiers/blob/828125a3ade0216e7d0a51a9b362d264f6ab68e0/environments/vf_aime2024/pyproject.toml#L7)), they are still imported at the same time at runtime ([link](https://github.com/willccbb/verifiers/blob/9f197f7ececcc1367bde504e881dc4d938a019e1/README.md?plain=1#L77)) |
| [Future House Aviary](https://github.com/Future-House/aviary) | ✅ | ❌OpenAI Chat Completions-like interface ([link](https://github.com/Future-House/aviary/blob/70d9c1fcf756425591504c3bd8f618b41ce8b8ee/src/aviary/tools/base.py#L125)) | ❌Environments are used via a step function ([link](https://github.com/Future-House/aviary/blob/70d9c1fcf756425591504c3bd8f618b41ce8b8ee/src/aviary/env.py#L107)) | ❌Environments are used via a step function ([link](https://github.com/Future-House/aviary/blob/70d9c1fcf756425591504c3bd8f618b41ce8b8ee/src/aviary/env.py#L107)) | ❌Even though each verifier has a separate pyproject.toml ([link](https://github.com/Future-House/aviary/blob/70d9c1fcf756425591504c3bd8f618b41ce8b8ee/packages/gsm8k/pyproject.toml)), they are still imported at the same time at runtime ([link](https://github.com/Future-House/aviary/blob/70d9c1fcf756425591504c3bd8f618b41ce8b8ee/README.md?plain=1#L180)) |
| [OpenThought Reasoning Gym](https://github.com/open-thought/reasoning-gym) | ✅Just a data generator | ❌Environment is passed a raw string LLM response ([link](https://github.com/open-thought/reasoning-gym/blob/02b7fac86358f7ef6239608b0b738a5a03ecfe9e/reasoning_gym/algebra/complex_arithmetic.py#L168)) | ❌Environment is passed a raw string LLM response ([link](https://github.com/open-thought/reasoning-gym/blob/02b7fac86358f7ef6239608b0b738a5a03ecfe9e/reasoning_gym/algebra/complex_arithmetic.py#L168)) | ❌Environments are directly imported ([link](https://github.com/open-thought/reasoning-gym/blob/02b7fac86358f7ef6239608b0b738a5a03ecfe9e/README.md?plain=1#L49)) | ❌All the datasets are in one folder and share dependencies ([link](https://github.com/open-thought/reasoning-gym/tree/02b7fac86358f7ef6239608b0b738a5a03ecfe9e/reasoning_gym)) |
| [NousResearch Atropos](https://github.com/NousResearch/atropos) |  |  |  | ✅Has servers and container compatible ([link](https://github.com/NousResearch/atropos/blob/ee8094d697428f378445840308cb788d02af7120/environments/code_execution_server/coding_server.py#L42)) | ❌All environments share the same Python env ([link](https://github.com/NousResearch/atropos/blob/ee8094d697428f378445840308cb788d02af7120/README.md?plain=1#L159)) |
| [VerL](https://github.com/volcengine/verl) |  |  |  |  |  |
| [RAGEN](https://github.com/RAGEN-AI/RAGEN) |  |  |  |  |  |
| [rLLM](https://github.com/rllm-org/rllm/tree/main) |  |  |  |  |  |
| [LlamaGym](https://github.com/KhoomeiK/LlamaGym) |  |  |  |  |  |
| [AReal](https://github.com/inclusionAI/AReaL) |  |  |  |  |  |
| [OpenPipe ART](https://github.com/OpenPipe/ART) |  |  |  |  |  |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) |  |  |  |  |  |


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
