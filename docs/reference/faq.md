(reference-faq)=
# FAQ

:::{note}
This page provides quick answers to commonly asked questions. Over time, these topics will be integrated into the structured product documentation (tutorials, guides, and reference sections) as we expand coverage. We've documented them here to provide immediate help while more comprehensive documentation is in progress.
:::


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


# How To: Profile your resources server
For large scale verifier training, it's critical that your resources server is as efficient as possible. It can be slammed with 16k concurrent requests or more. NeMo Gym provides easy tools to profile and understand the efficiency of your servers.

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

Tying back to NeMo Gym, NeMo Gym can be used to create synthetic data for SFT training by running strong teacher models on the different environments. Critically, it will also be used as the source of data during RL training.



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
