(reference-faq)=
# FAQ

:::{note}
This page provides quick answers to commonly asked questions. Over time, these topics will be integrated into the structured product documentation (tutorials, guides, and reference sections) as we expand coverage. We've documented them here to provide immediate help while more comprehensive documentation is in progress.
:::

## Key Terms

Before diving in, here are some NeMo Gym-specific terms:

- **Resources server**: A server that provides task environments, tools, and verification logic for RL training. Examples include math verifiers, code execution sandboxes, and tool-calling environments.
- **Agent server**: Orchestrates interactions between the model and resources server, managing conversation flow and tool calls.
- **Model server**: Wraps an LLM endpoint (vLLM, OpenAI, etc.) to provide a consistent API for NeMo Gym.
- **Responses API**: OpenAI's structured output format that separates reasoning, messages, and tool calls. NeMo Gym uses this as its native schema.

# How To: Run tests for simple agent
Run the Simple Chat Agent tests. `ng_test` or `nemo_gym_test` stands for `Nemo Gym Test`.
```bash
ng_test +entrypoint=responses_api_agents/simple_agent
```

Tests are strongly encouraged and you must have at least one test for every server you make. Test coverage is not explicitly required which means that **YOU ARE RESPONSIBLE FOR YOUR OWN SERVER CORRECTNESS AND FUNCTION**.


# How To: Upload and download a dataset from HuggingFace
The huggingface client requires that your credentials are in `env.yaml`, along with some other pertinent details needed to upload to the designated place.

:::{warning}
**Security**: The `env.yaml` file contains sensitive credentials. Ensure it is listed in your `.gitignore` to prevent accidental commits. Never commit API tokens to version control.
:::

```yaml
hf_token: {your huggingface token}
hf_organization: {your huggingface org}
hf_collection_name: {your collection}
hf_collection_slug: {your collection slug}  # alphanumeric string found at the end of a collection URI

# optional:
hf_dataset_prefix: str  # field to override the default value "Nemotron-RL" prepended to the dataset name
```

Naming convention for Huggingface datasets is as follows.

`{hf_organization}/{hf_dataset_prefix}-{domain}–{resource_server OR dataset_name}`

E.g.:

`nvidia/Nemotron-RL-math-OpenMathReasoning`


You will only need to manually input the `{dataset_name}` portion of the above when inputting the `dataset_name` flag in the upload command (refer to the command below). Everything preceding it will be automatically populated using your config prior to upload. Note that it is optional, and overrides `resource_server` if used.

To upload to Huggingface, use the below command:
```bash
resource_config_path="resources_servers/multineedle/configs/multineedle.yaml"
ng_upload_dataset_to_hf \
    +dataset_name={your dataset name} \
    +input_jsonl_fpath=data/multineedle_benchmark.jsonl \
    +resource_config_path=${resource_config_path}
```

Because of the required dataset nomenclature, the resource server config path is required when uploading. Specifically, `domain` is used in the naming of a dataset in Huggingface.

By default, the `split` parameter for uploading is set to `train`, which will run a check on the required fields `{"responses_create_params"}`. Specifying `validation` or `test` bypasses this check:

```bash
resource_config_path="resources_servers/multineedle/configs/multineedle.yaml"
ng_gitlab_to_hf_dataset \
    +dataset_name={your dataset name} \
    +input_jsonl_fpath=data/multineedle_benchmark_validation.jsonl \
    +resource_config_path=${resource_config_path} \
    +split=validation
```

## Uploading with Pull Request workflow
When uploading to an organization repository where you don't have direct write access (e.g., nvidia/), use the `+create_pr=true` flag to create a Pull Request instead of pushing directly. You can also customize the commit message and description.

If you want to specify the revision (branch name), you can add the `+revision={your branch name}` flag. Excluding `create_pr` (or setting it to `false`) assumes you are committing to an existing branch. Including it assumes it will be a brand new branch.

```bash
ng_upload_dataset_to_hf \
    +dataset_name=OpenMathReasoning \
    +input_jsonl_fpath=data/validation.jsonl \
    +resource_config_path=${resource_config_path} \
    +split=validation \
    +create_pr=true \
    +revision=my-branch-name \
    +commit_message="Add validation set" \
    +commit_description="Includes 545 examples"
```

The command will output a link to the created Pull Request:
```bash
[Nemo-Gym] - Pull Request created: https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning/discussions/1
```

:::{note}
The commit_message and commit_description parameters work for both direct pushes and Pull Requests. If not provided, HuggingFace auto-generates a commit message based on the filename.
:::


## Deleting Datasets from Gitlab
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
[Nemo-Gym] - Dataset upload successful
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

:::{important}
Gitlab model names are case sensitive. There can be models named 'My_Model' and 'my_model' living simultaneously in the registry. When uploading to Huggingface with the intention of deleting Gitlab artifacts, be sure the casing of your Huggingface dataset name matches that of Gitlab's.
:::


## Downloading Datasets from Huggingface
Downloading a dataset from Huggingface is straightforward:

**For structured datasets (with train/validation/test splits):**
```bash
ng_download_dataset_from_hf \
    +repo_id=nvidia/Nemotron-RL-knowledge-mcqa \
    +output_dirpath=data/mcqa \
    +split=train
```
The `split` parameter is optional. If omitted, all available splits will be downloaded as separate JSONL files.


**For raw file repositories (with specific JSONL files):**
```bash
ng_download_dataset_from_hf \
    +repo_id=nvidia/Nemotron-RL-instruction_following \
    +output_dirpath=data/instruction_following \
    +artifact_fpath=instruction_following.jsonl
```
Use `artifact_fpath` when the HuggingFace repo contains raw/arbitrary JSONL files rather than structured dataset splits. You cannot specify both `split` and `artifact_fpath`.


# How To: Prepare and validate data for PR submission or RL training
When you use `ng_init_resources_server +entrypoint=resources_servers/example_multi_step` to initialize a resources server, you will get a config.yaml that looks like the below code block. The dataset information for training, validation, and example will be inside the scope of your agent config (e.g. under simple_agent) and is a list of dataset objects.

```yaml
example_multi_step_resources_server:
  resources_servers:
    example_multi_step:
      entrypoint: app.py
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_multi_step_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        license: Apache 2.0
        jsonl_fpath: resources_servers/example_multi_step/data/train.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: example_multi_step
          version: 0.0.1
          artifact_fpath: example_multi_step/train.jsonl
        huggingface_identifier:
          repo_id: nvidia/Nemotron-RL-instruction_following
          artifact_fpath: instruction_following.jsonl
        license: Apache 2.0
      - name: validation
        type: validation
        license: Apache 2.0
        jsonl_fpath: resources_servers/example_multi_step/data/validation.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: example_multi_step
          version: 0.0.1
          artifact_fpath: example_multi_step/validation.jsonl
        huggingface_identifier:
          repo_id: nvidia/Nemotron-RL-instruction_following
          artifact_fpath: if_validation.jsonl
        license: Apache 2.0
      - name: example
        type: example
        jsonl_fpath: resources_servers/example_multi_step/data/example.jsonl
        num_repeats: 1
```

A dataset object consists of:
- Name: An identifier for you
- Type: train, validation, or example. Train and validation are as used in NeMo RL or other train frameworks. More information about the example type is in the next section.
- Jsonl fpath: the local file path to your jsonl file for this dataset.
- Num repeats: optionally repeat each row when preparing or collating data. Defaults to 1 if unspecified.
- Gitlab identifier: (NVIDIA internal) The remote path to the dataset as held in the Gitlab dataset registry. This field is required for train and validation datasets. (Not required for example datasets since those are required to be committed to Git).
- HuggingFace identifier: (Public) The remote path to the dataset on HuggingFace. Contains `repo_id` (required) and optionally `artifact_fpath` for raw file repos. If `artifact_fpath` is omitted, the datasets library will infer the `split` from the dataset `type`.
- License: The license of that dataset. Required for train and validation datasets and not required for example datasets, similar in principle to the Gitlab identifier.
- Start idx, end idx: used for slicing your dataset.
```yaml
- name: train
  type: train
  jsonl_fpath: resources_servers/example_multi_step/data/train.jsonl
  gitlab_identifier:
    dataset_name: example_multi_step
    version: 0.0.1
    artifact_fpath: example_multi_step/validation.jsonl
  huggingface_identifier:
    repo_id: nvidia/example_multi_step
    artifact_fpath: example_validation.jsonl
  license: Apache 2.0
```

Each config.yaml in the resources server requires at least one agent with one example dataset. This example dataset is the first 5 rows of your train dataset that is used for sanity checks on the format for your dataset and the format of each individual example and for others to quickly understand your data.

For every PR that contributes data, we require common dataset statistics and sanity checks on the data itself. This process is also helpful to catch any simple issues before you ever train with NeMo RL. NeMo Gym provides a helper command ng_prepare_data to do so.
```bash
config_paths="resources_servers/example_multi_step/configs/example_multi_step.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/example_multi_step \
    +mode=example_validation
```

To download missing datasets automatically, add +should_download=true. By default, datasets are downloaded from HuggingFace:
```bash
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/example_multi_step \
    +mode=train_preparation \
    +should_download=true
```

For NVIDIA internal users, you can download from GitLab instead:

```bash
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/example_multi_step \
    +mode=train_preparation \
    +should_download=true \
    +data_source=gitlab
```

Run NeMo Gym servers the exact same way with the same configs!
```bash
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
config_paths="resources_servers/example_multi_step/configs/example_multi_step.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/example_multi_step \
    +mode=example_validation
```


# How To: ng_dump_config - Dump a YAML config as exactly as NeMo Gym sees it
```bash
# Example ng_run command
config_paths="resources_servers/example_multi_step/configs/example_multi_step.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[$config_paths]"


# Dump the exact yaml config that NeMo gym sees, just by swapping ng_run -> ng_dump_config
ng_dump_config "+config_paths=[$config_paths]"
```


# How To: Use NeMo Gym with a non-Responses compatible API endpoint like vLLM
Most models use Chat Completions format rather than the OpenAI Responses API schema that NeMo Gym uses natively. To bridge this gap, NeMo Gym provides a conversion layer.

As a result, we provide a Responses API to Chat Completions mapping middleware layer in the form of `responses_api_models/vllm_model`. VLLMModel assumes that you are pointing to a vLLM instance (since it relies on vLLM-specific endpoints like `/tokenize` and vLLM-specific arguments like `return_tokens_as_token_ids`).

**To use VLLMModel, just change the `responses_api_models/openai_model/configs/openai_model.yaml` in your config paths to `responses_api_models/vllm_model/configs/vllm_model.yaml`!**
```bash
config_paths="resources_servers/example_multi_step/configs/example_multi_step.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```

Here is an e2e example of how to spin up a NeMo Gym compatible vLLM Chat Completions OpenAI server.
- If you want to use tools, find the appropriate vLLM arguments regarding the tool call parser to use. In this example, we use Qwen3-30B-A3B, which is suggested to use the `hermes` tool call parser.
- If you are using a reasoning model, find the appropriate vLLM arguments regarding reasoning parser to use. In this example, we use Qwen3-30B-A3B, which is suggested to use the `qwen3` reasoning parser.

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
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 10240
```


# How To: Multi-verifier usage
Gym is explicitly designed to support multi-verifier training.

Let's say you want to use both math and search verifiers. Normally how you spin up the servers individually is:
For math:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"
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
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

The same process goes for data preparation and downstream training framework Gym configuration, you would just add additional server configs.


# How To: Profile your resources server
For large scale verifier training, it's critical that your resources server is as efficient as possible. It can be slammed with 16k concurrent requests or more. Gym provides easy tools to profile and understand the efficiency of your servers.

In one terminal, start your agent, model, and resources servers, with profiling enabled.
- `profiling_enabled` (bool): whether profiling is enabled or not. By default this is disabled since it incurs some slight overhead we don't want at runtime.
- `profiling_results_dirpath` (str): The directory to save all server profiling results in. Previous logs for the same will be overwritten in the same directory.
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"
ng_run "+config_paths=[${config_paths}]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/math_with_judge
```

In another terminal, run some large number of rollouts against your servers. Use the `limit` and `num_repeats` flags to adjust the number of samples you want to run.
```bash
ng_collect_rollouts +agent_name=math_with_judge_simple_agent \
    +input_jsonl_fpath=resources_servers/math_with_judge/data/dapo17k_bytedtsinghua_train.jsonl \
    +output_jsonl_fpath=temp/math_with_judge_rollouts.jsonl \
    +limit=1024 \
    +num_repeats=1
```

After `ng_collect_rollouts` finishes, ctrl+c to quit your servers. You should see some output in the terminal like the following:
```bash
```

The log file content for a server will look something like the following:
```
name                                                                                                                      ncall       tsub      ttot      tavg      
.../nemo-gym/resources_servers/math_with_judge/app.py:118 LibraryJudgeMathResourcesServer.verify                       1024        0.009755  17.98387  0.017562
.../nemo-gym/resources_servers/math_with_judge/app.py:145 LibraryJudgeMathResourcesServer._verify_answer               1024        0.002933  17.87998  0.017461
.../nemo-gym/resources_servers/math_with_judge/app.py:173 LibraryJudgeMathResourcesServer._verify_answer_with_library  1024        0.007851  17.87704  0.017458
.../nemo-gym/resources_servers/math_with_judge/app.py:191 <genexpr>                                                    2339        0.001695  0.029082  0.000012
.../nemo-gym/resources_servers/math_with_judge/app.py:163 _mute_output                                                 2048        0.007473  0.016538  0.000008
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
During training time, Gym keeps track of the ground truth prompt token ids, generation token ids, and generation log probs for downstream consumption by the RL framework. As a result, we need to add a few fields to request and response schemas in order to properly facilitate this. This usually doesn't matter if you are using 100% Gym, but in certain situations you may need or want to use a separate client (such as LiteLLM, your own OpenAI client, and so on) to call model endpoints.

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

Here's how to parallelize CPU-intensive functions using Ray's `@ray.remote` decorator. Refer to [Ray documentation](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html) for more options.

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

The `output` list can contain multiple item types, such as:
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


# FAQ: SFT and RL
Reading time: 5 mins
Date: Fri Aug 15, 2025

SFT (supervised fine tuning) and RL (reinforcement learning) are two different ways of optimizing your model for different tasks and each have their own use cases.

Let's say you wanted to train your model to be really good at math.
- For SFT, you would take some input math questions and either ask human annotators to provide a gold response, or run it through a stronger teacher model and get your SFT target. And then you would SFT on these input + gold response pairs.
- For RL, you would take some input math questions and implement a way to score model answers. During RL training, you would ask the model you are trying to train these math questions, score the model responses using your scorer, and use the scores as a signal on how to optimize your model. Model responses with higher scores would be encouraged.


One way I like to think about these things is:
- You can do RL on SFT data, where your input is your SFT input, and the model answer scorer is just an exact match on the SFT gold label.
- You can also do SFT on RL data using synthetic data generation, where you run your inputs into some strong teacher model, score the responses, and use the scores to pick your SFT gold label.

Tying back to NeMo Gym, NeMo gym can be used to create synthetic data for SFT training by running strong teacher models on the different environments. Critically, it will also be used as the source of data during RL training.



# FAQ: PermissionError when starting NeMo Gym in sandboxed environments

If you see an error like the following when running `ng_run`:

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
If you see some docs building related errors that are kind of cryptic regarding .rst files like the following
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

For example, say we are training a Qwen 3 family model. During rollouts, the model can sample from the entire token distribution. The token IDs are then decoded into text and subsequently converted to OpenAI schema and returned to the training environment developer. At some point for multi-step and multi-turn scenarios, the training environment developer will call the model again with the previously output OpenAI schema. This re-tokenization causes problems since a single string can map to multiple possible sequences of token IDs. So if the model generations token ID sequence 1 and the re-tokenization outputs token ID sequence 2, suddenly things can become off policy when the Gym result is consumed by the RL training framework.

So, the OpenAI compatible model server in a training framework needs to be able to handle this discrepancy. In order to do that, Gym needs a handle on the ground truth token IDs and it needs to provide that information back to the training frameworks' OpenAI compatible server.

See the "How To: Use a custom client to call Gym Responses API model endpoints during training" section above for related details on token ID handling.


# FAQ: Why use aiohttp backend instead of httpx/httpcore for async http?

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
