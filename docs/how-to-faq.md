This document is a smattering of How-To's and FAQs that have not made their way into an official tutorial yet!

- [How To: Run tests for simple agent](#how-to-run-tests-for-simple-agent)
- [How To: Add a resource server](#how-to-add-a-resource-server)
  - [TLDR final expected artifacts](#tldr-final-expected-artifacts)
- [How To: Upload and download a dataset from Gitlab](#how-to-upload-and-download-a-dataset-from-gitlab)
- [How To: Upload and download a dataset from HuggingFace](#how-to-upload-and-download-a-dataset-from-huggingface)
- [How To: Prepare and validate data for PR submission or RL training](#how-to-prepare-and-validate-data-for-pr-submission-or-rl-training)
- [How To: ng\_dump\_config - Dump a YAML config as exactly as NeMo Gym sees it](#how-to-ng_dump_config---dump-a-yaml-config-as-exactly-as-nemo-gym-sees-it)
- [How To: Use NeMo Gym with a non-Responses compatible API endpoint like vLLM](#how-to-use-nemo-gym-with-a-non-responses-compatible-api-endpoint-like-vllm)
- [How To: Multi-verifier usage](#how-to-multi-verifier-usage)
- [How To: Profile your resources server](#how-to-profile-your-resources-server)
- [How To: Use a custom client to call Gym Responses API model endpoints during training](#how-to-use-a-custom-client-to-call-gym-responses-api-model-endpoints-during-training)
- [How To: Detailed anatony of a Gym config](#how-to-detailed-anatony-of-a-gym-config)
- [How To: Use Ray for parallelizing CPU-intensive tasks](#how-to-use-ray-for-parallelizing-cpu-intensive-tasks)
  - [Ray Setup in NeMo Gym](#ray-setup-in-nemo-gym)
    - [Automatic Initialization](#automatic-initialization)
    - [Ray Configuration](#ray-configuration)
  - [Using Ray for CPU-Intensive Tasks](#using-ray-for-cpu-intensive-tasks)
- [FAQ: OpenAI Responses vs Chat Completions API](#faq-openai-responses-vs-chat-completions-api)
- [FAQ: DCO and commit signing VSCode and Git setup](#faq-dco-and-commit-signing-vscode-and-git-setup)
- [FAQ: SFT and RL](#faq-sft-and-rl)
- [FAQ: Error: Found files with missing copyright](#faq-error-found-files-with-missing-copyright)
- [FAQ: PermissionError when starting NeMo Gym in sandboxed environments](#faq-permissionerror-when-starting-nemo-gym-in-sandboxed-environments)
- [FAQ: build-docs / Build docs CI failures](#faq-build-docs--build-docs-ci-failures)
- [FAQ: NeMo Gym, training frameworks, and token IDs](#faq-nemo-gym-training-frameworks-and-token-ids)
- [FAQ: NeMo Gym what CI/CD do I need to pass?](#faq-nemo-gym-what-cicd-do-i-need-to-pass)
- [FAQ: Why aiohttp backend and not httpx/httpcore for async http?](#faq-why-aiohttp-backend-and-not-httpxhttpcore-for-async-http)


# How To: Run tests for simple agent
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

You can see a complete example of `app.py` in `resources_servers/example_simple_weather/app.py`!

Run an agent with your new server!
```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_simple_weather/configs/simple_weather.yaml"
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
ng_test +entrypoint=resources_servers/example_simple_weather
```


You can also run detailed tests after running tests the first time
```bash
cd resources_servers/example_simple_weather
source .venv/bin/activate
pytest
```

At some point, you will want to actually add data that can be used to query your server. Please follow the instructions for [How To: Prepare and validate data for PR submission or RL training](#how-to-prepare-and-validate-data-for-pr-submission-or-rl-training).


If you need some dataset preprocessing or formatting scripts, please place them your resources server directory e.g. `resources_servers/example_simple_weather/my_preprocess_script.py`.


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


# How To: Upload and download a dataset from HuggingFace
The huggingface client requires that your credentials are in `env.yaml`, along with some other pertinent details needed to upload to the designated place. 
```yaml
hf_token: {your huggingface token}
hf_organization: {your huggingface org}
hf_collection_name: {your collection}
hf_collection_slug: {your collection slug}  # alphanumeric string found at the end of a collection URI

# optional:
hf_dataset_prefix: str  # field to override the default value "NeMo-Gym" prepended to the dataset name
```

Naming convention for Huggingface datasets is as follows.

`{hf_organization}/{hf_dataset_prefix}-{domain}–{resource_server_name}-{your dataset name}`

E.g.:

`Nvidia/Nemo-Gym-Math-library_judge_math-dapo17k`


You will only need to manually input the `{your dataset name}` portion of the above when inputting the `dataset_name` flag in the upload command (see below). Everything preceding it will be automatically populated using your config prior to upload.

To upload to Huggingface, use the below command:
```bash
resource_config_path="resources_servers/multineedle/configs/multineedle.yaml"
ng_upload_dataset_to_hf \
    +dataset_name={your dataset name} \
    +input_jsonl_fpath=data/multineedle_benchmark.jsonl \
    +resource_config_path=${resource_config_path}
```

Because of the required dataset nomenclature, the resource server config path is required when uploading. Specifically, `domain` is used in the naming of a dataset in Huggingface.

You can optionally pass a `+delete_from_gitlab=true` flag to the above command, which will delete the model and all of its artifacts from Gitlab. By default, this is set to `False`.
```bash
resource_config_path="resources_servers/multineedle/configs/multineedle.yaml"
ng_upload_dataset_to_hf \
    +dataset_name={your dataset name} \
    +input_jsonl_fpath=data/multineedle_benchmark.jsonl \
    +resource_config_path=${resource_config_path} \
    +delete_from_gitlab=true
```

There will be a confirmation dialog to confirm the deletion:
```bash
[Nemo-Gym] - Dataset uploaded successful
[Nemo-Gym] - Found model 'fs-test' in the registry. Are you sure you want to delete it from Gitlab? [y/N]:
```

You can also run the below command which does the same thing without the need for a `+delete_from_gitlab` flag:

```bash
resource_config_path="resources_servers/multineedle/configs/multineedle.yaml"
ng_gitlab_to_hf_dataset \
    +dataset_name={your dataset name} \
    +input_jsonl_fpath=data/multineedle_benchmark.jsonl \
    +resource_config_path=${resource_config_path}
```

If you've already uploaded to Huggingface and just want to do a standalone delete from Gitlab:
```bash
ng_delete_dataset_from_gitlab \
    +dataset_name={your dataset name}
```

**Important note**: Gitlab model names are case sensitive. There can be models named 'My_Model' and 'my_model' living simultaneously in the registry. When uploading to Huggingface with the intention of deleting Gitlab artifacts, be sure the casing of your Huggingface dataset name matches that of Gitlab's.

Downloading a dataset from Huggingface is straightforward:
```bash
ng_download_dataset_from_hf \
    +repo_id=Nvidia/NeMo-Gym-Instruction_Following-multineedle-{your dataset name} \
    +artifact_fpath=multineedle_benchmark.jsonl \
    +output_fpath=data/multineedle_benchmark_hf.jsonl
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
        num_repeats: 1
        gitlab_identifier:
          dataset_name: multineedle
          version: 0.0.1
          artifact_fpath: multineedle/train.jsonl
        license: Apache 2.0
      - name: validation
        type: validation
        license: Apache 2.0
        jsonl_fpath: resources_servers/multineedle/data/validation.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: multineedle
          version: 0.0.1
          artifact_fpath: multineedle/validation.jsonl
        license: Apache 2.0
      - name: example
        type: example
        jsonl_fpath: resources_servers/multineedle/data/example.jsonl
        num_repeats: 1
```

A dataset object consists of:
- Name: An identifier for you
- Type: train, validation, or example. Train and validation are as used in NeMo RL or other train frameworks. More information about the example type is in the next section.
- Jsonl fpath: the local file path to your jsonl file for this dataset.
- Num repeats: optionally repeat each row when preparing or collating data. Defaults to 1 if unspecified.
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


# How To: Multi-verifier usage
Gym is explicitly designed to support multi-verifier training.

Let's say you want to use both math and search verifiers. Normally how you spin up the servers individually is:
For math:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"
ng_run "+config_paths=[${config_paths}]"
```
For search:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/google_search/configs/google_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

If you want to use them both you would just add the yamls together like:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

The same process goes for data preparation and downstream training framework Gym configuration, you would just add additional server configs.


# How To: Profile your resources server
For large scale verifier training, it's critical that your resources server is as efficient as possible. It may be slammed with 16k concurrent requests or more. Gym provides easy tools to profile and understand the efficiency of your servers.

In one terminal, start your agent, model, and resources servers, with profiling enabled.
- `profiling_enabled` (bool): whether profiling is enabled or not. By default this is disabled since it incurs some slight overhead we don't want at runtime.
- `profiling_results_dirpath` (str): The directory to save all server profiling results in. Previous logs for the same will be overwritten in the same directory.
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"
ng_run "+config_paths=[${config_paths}]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/library_judge_math
```

In another terminal, run some large number of rollouts against your servers. Use the `limit` and `num_repeats` flags to adjust the number of samples you want to run.
```bash
ng_collect_rollouts +agent_name=library_judge_math_simple_agent \
    +input_jsonl_fpath=resources_servers/library_judge_math/data/dapo17k_bytedtsinghua_train.jsonl \
    +output_jsonl_fpath=temp/library_judge_math_rollouts.jsonl \
    +limit=1024 \
    +num_repeats=1
```

After `ng_collect_rollouts` finishes, ctrl+c to quit your servers. You should see some output in the terminal like this:
```bash
```

The log file content for a server will look something like the following:
```
name                                                                                                                      ncall       tsub      ttot      tavg      
.../nemo-gym/resources_servers/library_judge_math/app.py:118 LibraryJudgeMathResourcesServer.verify                       1024        0.009755  17.98387  0.017562
.../nemo-gym/resources_servers/library_judge_math/app.py:145 LibraryJudgeMathResourcesServer._verify_answer               1024        0.002933  17.87998  0.017461
.../nemo-gym/resources_servers/library_judge_math/app.py:173 LibraryJudgeMathResourcesServer._verify_answer_with_library  1024        0.007851  17.87704  0.017458
.../nemo-gym/resources_servers/library_judge_math/app.py:191 <genexpr>                                                    2339        0.001695  0.029082  0.000012
.../nemo-gym/resources_servers/library_judge_math/app.py:163 _mute_output                                                 2048        0.007473  0.016538  0.000008
```

- `ncall`: number of calls (how many times the function/subroutine was invoked).
  - The `LibraryJudgeMathResourcesServer.verify` function was invoked 1024 times.
- `tsub`: time spent inside the subroutine itself, excluding calls to other functions (sometimes called "self time").
  - The `LibraryJudgeMathResourcesServer.verify` function __itself__ accounted for only 0.009755s of time.
- `ttot`: total time spent in the subroutine, including all the functions it called.
  - The `LibraryJudgeMathResourcesServer.verify` function and all functions it called including `_verify_answer`, etc accounted for a total of 17.98387s.
- `tavg`: average time per call (often ttot / ncall).
  - The `LibraryJudgeMathResourcesServer.verify` function took 0.017562s per call on average.


# How To: Use a custom client to call Gym Responses API model endpoints during training
During training time, Gym keeps track of the ground truth prompt token ids, generation token ids, and generation log probs for downstream consumption by the RL framework. As a result, we need to add a few fields to request and response schemas in order to properly facilitate this. This usually doesn't matter if you are using 100% Gym, but in certain situations you may need or want to use a separate client (e.g. LiteLLM, your own OpenAI client, etc) to call model endpoints.

For Chat Completions, outside of training, an Assistant message will look like:
```python
ChatCompletionMessage(
    content="<think>I'm thinking</think>Hi there!",
    tool_calls=[{...}, {...}],
    ...
)
```
During training, a Chat Completions Assistant message will look like:
```python
ChatCompletionMessage(
    content="<think>I'm thinking</think>Hi there!",
    tool_calls=[{...}, {...}],
    prompt_token_ids=[...],  # List[int]
    generation_token_ids=[...],  # List[int]
    generation_log_probs=[...],  # List[float]
    ...
)
```
And you have to ensure that when you make a request with your custom client that these three extra fields (prompt_token_ids, generation_token_ids, and generation_log_probs) are passed through correctly on a message level. And this also applies to the response i.e. you need to ensure that your custom client will correctly return these three extra fields.


It's an analogous story for Responses-compatible APIs.


# How To: Detailed anatony of a Gym config
Let's break down the anatomy of a Gym config further and help clarify some things.

TODO: bxyu-nvidia

```yaml
# `library_judge_math` here at the top most level is the unique name of your resources server. This must be unique across your config.
# When you or other servers call this server, they will do so using the ServerClient and its name.
library_judge_math:
  # `resources_servers` here at the second level is the server type. There are 3 server types in gym: agent, model, or resources.
  resources_servers:
    # This is the resources server type. This is not unique at runtime, and you can spin up multiple instances of this with different configs if you wish!
    library_judge_math:
      entrypoint: app.py
      judge_model_server:
        type: responses_api_models
        name: ???
      judge_responses_create_params: {
        input: []
      }
      should_use_judge: false
library_judge_math_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: library_judge_math
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: resources_servers/library_judge_math/data/dapo17k_bytedtsinghua_train.jsonl
        gitlab_identifier:
          dataset_name: bytedtsinghua_dapo17k
          version: 0.0.1
          artifact_fpath: dapo17k_bytedtsinghua_train.jsonl
        license: Apache 2.0
      - name: validation
        type: validation
        jsonl_fpath: resources_servers/library_judge_math/data/aime24_bytedtsinghua_validation.jsonl
        gitlab_identifier:
          dataset_name: bytedtsinghua_dapo17k
          version: 0.0.1
          artifact_fpath: aime24_bytedtsinghua_validation.jsonl
        license: Apache 2.0
```



# How To: Use Ray for parallelizing CPU-intensive tasks

NeMo Gym automatically sets up Ray for distributed computing for CPU-intensive tasks.

## Ray Setup in NeMo Gym

### Automatic Initialization
Ray is initialized when you start NeMo Gym servers:

```bash
ng_run "+config_paths=[$config_paths]"
```

The initialization happens in two places:
1. **Main Process** (`cli.py`): Ray is initialized in the main process when `RunHelper.start()` is called
2. **Server Process** (`server_utils.py`): Each server invokes `initialize_ray()` during its startup and connects to the same Ray cluster initialized by the main process.

### Ray Configuration
You can also specify a custom Ray cluster address in your config:
```yaml
ray_head_node_address: "ray://your-cluster-address:10001"
```
Training frameworks like [Nemo-RL](https://github.com/NVIDIA-NeMo/RL) will configure the Ray head node address, allowing remote tasks to run across all nodes in the cluster.

If not specified, NeMo Gym will start a local Ray cluster and store the address in the global config for child processes to connect to.

## Using Ray for CPU-Intensive Tasks

Here's how to parallelize CPU-intensive functions using Ray's `@ray.remote` decorator. Please refer to [Ray documentation](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html) for more options.

```python
import ray

# Decorate your CPU-intensive function
# Spread tasks across different nodes for better parallelization
@ray.remote(scheduling_strategy="SPREAD")
def cpu_intensive_task(data):
    # Your expensive computation here
    result = expensive_computation(data)
    return result

# Use it in your code
def process_data_parallel(data_list):
    # Submit all tasks to Ray
    futures = [cpu_intensive_task.remote(data) for data in data_list]
    
    # Get results
    results = ray.get(futures)
    return results
```


# FAQ: OpenAI Responses vs Chat Completions API
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


# FAQ: DCO and commit signing VSCode and Git setup
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
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```


# FAQ: PermissionError when starting NeMo Gym in sandboxed environments

If you see an error like this when running `ng_run`:

```python
PermissionError: [Errno 1] Operation not permitted (originated from sysctl() malloc 1/3)

Traceback:
  File "ray/thirdparty_files/psutil/_psosx.py", line 337, in pids
    ls = cext.pids()
```

**What's happening:**

Ray (NeMo Gym's distributed computing dependency) uses `psutil` to enumerate and monitor processes, which requires calling system calls like `sysctl()` on macOS. In sandboxed execution environments, these system calls are blocked for security reasons.

**Who is affected:**

This is an **edge case** that only affects users in restricted environments:
- Sandboxed command execution tools (like Cursor's AI sandbox)
- Docker containers with restricted capabilities
- CI/CD runners with security restrictions

**Normal users running in their own terminal will NOT encounter this** - they have full system permissions by default.

**Solution (if you're affected):**

If you're running NeMo Gym in a sandboxed environment and hit this error, you'll need to either:

1. **Disable the sandbox** for the command (if your environment supports it)
2. **Grant additional permissions** to allow Ray to access process information
3. **Run outside the sandbox** in a normal terminal environment

For most development and production use cases, simply running `ng_run` in your regular terminal will work without any issues.

**Specific workaround for Cursor AI:**

If you're using Cursor's AI assistant to run NeMo Gym commands and encounter this error, the AI will need to run commands with elevated permissions. This is not something you configure - the AI assistant will automatically request additional permissions (specifically `required_permissions: ["all"]`) when it detects this error. If you see the error persist, try asking the AI to restart the servers or run the command in your own terminal instead.

**Why NeMo Gym can't fix this:**

This is a fundamental incompatibility between:
- Ray legitimately needing system access to manage distributed workers
- Sandboxed environments intentionally restricting system access for security

There's no practical workaround that NeMo Gym can implement - the solution is to run with appropriate permissions for your environment.


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

# FAQ: Why aiohttp backend and not httpx/httpcore for async http?

TL;DR: httpx is O(n^2) runtime where n is the number of queued requests (i.e. for each request, we check all other queued requests). This is terribly inefficient and results in major slowdowns.

On Wed Sep 17, 2025, inspired by the Deepseek R1 Nature paper, we tried launching a larger rollout batch run with up to 16 off policy steps in NeMo RL. Our setting resulted in Gym being slammed with 16k concurrent requests. At the time, we were using a single Gym instance with multiple data-parallel vLLM workers, and that setup hung for 40 minutes before the first request was processed. Something was wrong.

Before that time, we had also gotten reports that the rollout collection in Gym couldn't be used with high concurrency i.e. in some cases people had to set the concurrency to 32 requests in parallel. Putting these two data points together, we figured something was wrong with the concurrency setup in Gym.

For some context, Gym is a set of servers that end up calling a model endpoint server at some point. And it's really important that we never artificially restrict the concurrency in the Gym side since technically we are always clients of that model endpoint server, since the model endpoint server could handle many more requests than we're restricting the concurrency to. So we always want Gym to be as efficient as possible and not have e.g. max parallel requests or smth parameter in Gym.

Eventually, we isolated the issue to our async http backend -- httpx and httpcore. We originally decided to use httpx for the async http backend in Gym because the OpenAI client uses it by default so we can share the same backend http client. Unfortunately, the httpcore connection pool subroutine for pooling connections over requests is O(n^2) where n is the number of queued requests.

Networking mental model:
1. A request is sent by Gym to the model endpoint server.
2. This request requires a connection from our client side to the server side.
   1. This connection is a socket (identified by a port) and a socket is an open file (managed by the operating system).
   2. If we are sending 100 requests, in the worst case we could open 100 connections == 100 open files. This quickly becomes very expensive.
   3. So, async http backends will pool requests across connections to a single endpoint, where multiple requests can leverage the same file if they are going to the same endpoint origin.
   4. This is called connection pooling. And it's possible that all 100 requests share a single connection.
3. But this connection pooling now needs some management logic. When the client sends a new request, it needs to determine if that request can reuse an existing connection.
   1. And this is where the httpcore connection pool logic is very inefficient.

Here are the key calls in the stack trace:
1. OpenAI client at some point calls httpx client
2. httpx client calls into the transport [here](https://github.com/encode/httpx/blob/4b23574cf83307ce27d3b14b4a425dc58c57d28d/httpx/_client.py#L1014)
3. Transport calls into httpcore connection pool [here](https://github.com/encode/httpx/blob/4b23574cf83307ce27d3b14b4a425dc58c57d28d/httpx/_transports/default.py#L250)
4. For each request, the httpcore connection pool calls this `_assign_requests_to_connections` subroutine [here](https://github.com/encode/httpcore/blob/5974b03c7df89d3ee4e23779900d5349d550753c/httpcore/_async/connection_pool.py#L228)
   1. This subroutine loops through connections [here](https://github.com/encode/httpcore/blob/5974b03c7df89d3ee4e23779900d5349d550753c/httpcore/_async/connection_pool.py#L284)
   2. and loops through queued requests [here](https://github.com/encode/httpcore/blob/5974b03c7df89d3ee4e23779900d5349d550753c/httpcore/_async/connection_pool.py#L303)
   3. Which results in a total of O(n^2) runtime if the number of queued requests is large. Which is always the case if we slam with some larger number of requests.

In the end, we decided to swap our http backend from httpx to aiohttp since we had good prior experience working with aiohttp in production infra.

Here are some Github issues related to this problem. They didn't help too much, but they did validate our solution (kind of) to use aiohttp as as async http backend instead.
- https://github.com/openai/openai-python/issues/1596
- https://github.com/encode/httpx/issues/3215#issuecomment-2220795088

If you are using AsyncOpenAI client with a parallelism > 32, you may also want to check if this kind of inefficiency also affects your setup.
