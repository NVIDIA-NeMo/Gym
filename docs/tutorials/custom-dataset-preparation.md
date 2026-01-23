---
orphan: true
---

(custom-dataset-preparation)=

# Preparing a Custom Dataset for NeMo Gym

Learn the dataset requirements for NeMo Gym and how to prepare a custom dataset for integration with post-training.

:::{card}

**Goal**: Download and preprocess a custom math-focused dataset for post-training tasks with NeMo Gym.

^^^

**In this tutorial, you will**:

1. Download a dataset from Hugging Face
2. Write a script to lightly preprocess the data
3. Setup a config file to process the dataset during training
4. Prepare the data using the `ng_prepare_data` tool

:::

:::{button-ref} ../get-started/rollout-collection
:color: secondary
:outline:
:ref-type: doc

← Previous: Rollout Collection
:::

:::{important}
Run all commands from the **repository root** directory (where `pyproject.toml` is located).
:::

## What You'll Build

By the end of this tutorial, you'll have:

- [ ] A training dataset with 999k rows at `data/train.jsonl`
- [ ] A validation dataset with 1k rows at `data/validation.jsonl`
- [ ] A resource server config to process your dataset

---

## Dataset Requirements

NeMo Gym has a few requirements for the structure of the dataset for it to understand how to process the data. Datasets must be in `.jsonl` format where each line in the dataset is a complete JSON object. An example of valid jsonl data is as follows:

```json
{"input": [{"role": "user", "content": "What is 2+2?"}], "expected_answer": "4"}
{"input": [{"role": "user", "content": "What is 3*5?"}], "expected_answer": "15"}
{"input": [{"role": "user", "content": "What is 10/2?"}], "expected_answer": "5"}
```

Additionally, NeMo Gym expects both a `responses_create_params` and `agent_ref` field for each row in the dataset.

The `responses_create_params` field serves as the input field to the policy during training. This is sent to the policy and used to generate responses to each row in the dataset.

The `agent_ref` field maps each row in the dataset to a specific resource server. A training dataset can blend multiple resource servers in a single file. The `agent_ref` tag tells NeMo Gym which resource server to use for each row in the dataset.

Examples of modifying datasets to include these fields are shown later in this tutorial.

:::{tip}
The `responses_create_params` field is required as NeMo Gym uses the [Responses API](https://platform.openai.com/docs/api-reference/responses) for sending requests to models. At the time of writing, the Responses API format is only supported by a limited number of open-source models, such as those published by OpenAI. To overcome this, a translation layer is used by NeMo Gym to mimic the Responses API format when sending requests to the local models.
:::

---

## 1. Downloading a Dataset from Hugging Face

First, we need to acquire a dataset. For the purposes of this tutorial, we will use a public dataset available on Hugging Face, though the same high-level process can be applied to private datasets on Hugging Face, or datasets located on other repositories and/or local storage.

We will use the [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) dataset which is a public dataset containing millions of math problems and their corresponding solutions.

:::{note}
While the *OpenMathInstruct-2* dataset is a good example for this tutorial on how to take an existing dataset and make it compatible with NeMo Gym, NVIDIA has published several math-focused datasets specifically for NeMo Gym, such as [nvidia/Nemotron-RL-math-OpenMathReasoning](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning) and [nvidia/Nemotron-3-Nano-RL-Training-Blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend). If you are looking to begin post-training quickly with an existing dataset, it is recommended to use those instead.
:::

Create a new Python script named `download.py` on a system where you will be running NeMo Gym. Add the following contents to the script:

```python
import json
from datasets import load_dataset

output_file = "openmathinstruct2.jsonl"
dataset_name = "nvidia/OpenMathInstruct-2"

with open(output_file, "w", encoding="utf-8") as f:
    for line in load_dataset(dataset_name, split="train_1M", streaming=True):
        f.write(json.dumps(line)+"\n")
```

Save and close the file after copying the contents. The script can be modified to use a different dataset from Hugging Face by replacing `dataset_name` with the name of your dataset. Additionally, replace `train_1M` with a different split name if necessary, such as `train` or `validation`. This will depend on the specific split names on Hugging Face.

This script will download the `nvidia/OpenMathInstruct-2` dataset from Hugging Face and save the data from the `train_1M` split locally as `openmathinstruct2.jsonl`. The `train_1M` split contains a smaller subset of one million rows which is easier to process for this tutorial.

To download and save the dataset locally, run the following in the repository root directory:

```bash
uv sync
uv run download.py
```

This should take a couple minutes depending on your network speed. After it completes, you should see a new file named `openmathinstruct2.jsonl` in your local directory. Verify the file has one million items with:

```bash
$ wc -l openmathinstruct2.jsonl 
1000000 openmathinstruct2.jsonl
```

Your first line should look similar to the following:

```bash
$ head -n 1 openmathinstruct2.jsonl
{"problem": "Solve for $y$:\n\n$$\\frac{y^2 - 3y + 2}{y - 2} = y + 1$$", "generated_solution": "Start by multiplying both sides by $y - 2$ to eliminate the denominator:\n\\[ (y^2 - 3y + 2) = (y + 1)(y - 2) \\]\n\nExpand both sides:\n\\[ y^2 - 3y + 2 = y^2 - y - 2 \\]\n\nSubtract $y^2$ from both sides to get:\n\\[ -3y + 2 = -y - 2 \\]\n\nAdd $3y$ to both sides:\n\\[ 2 = 2y - 2 \\]\n\nAdd $2$ to both sides:\n\\[ 4 = 2y \\]\n\nDivide by $2$ to solve for $y$:\n\\[ y = \\frac{4}{2} \\]\n\n\\[ y = \\boxed{2} \\]", "expected_answer": "2", "problem_source": "augmented_math"}
```

The dataset is now ready for preprocessing.

---

## 2. Preprocess the Dataset

With the dataset downloaded locally, we need to add the `responses_create_params` column to each row in the dataset with the appropriate data. As a reminder, the `responses_create_params` field translates inputs into the Responses API format which is required by NeMo Gym. We will also remove extraneous fields from the dataset to reduce the file size and improve the efficiency of the dataloaders during post-training.

The `responses_create_params` field expects a dictionary object with an `input` field containing the input prompts. This will typically include a *system prompt* which informs the model how to handle the particular request, plus a *user prompt* which includes the actual prompt.

As an example, here's a line from the dataset in `responses_create_params` format:

```json
"responses_create_params": {
    "input": [
        {
            "role": "system",
            "content": "Your task is to solve a math problem. Make sure to put the answer (and only the answer) inside \\boxed{}."
        },
        {
            "role": "user",
            "content": "Solve for $y$:\n\n$$\\frac{y^2 - 3y + 2}{y - 2} = y + 1$$"
        }
    ]
}
```

The system prompt is `Your task is to solve a math problem. Make sure to put the answer (and only the answer) inside \\boxed{}.` This instructs the model to solve the problem and answer in a particular format for consistency. Depending on the format or structure of your dataset, this system prompt should be modified to properly reflect your goals.

The actual math problem is `Solve for $y$:\n\n$$\\frac{y^2 - 3y + 2}{y - 2} = y + 1$$`, found in the `user` sub-field. This will be unique for each row in the dataset.

We will write another script to add the `responses_create_params` field to each row in the dataset. This script opens the `openmathinstruct2.jsonl` file locally, reads the data line-by-line, adds the `responses_create_params` field based on that line's question field, and splits the data into train and validation files, locally.

In the OpenMathInstruct-2 dataset, the input field I want to use is named `problem`. For custom datasets, replace the `INPUT_FIELD` in the script below with the name of the existing input field to use with your prompts.

Save the following in a Python script named `preprocess.py`:

```python
import json, os

INPUT_FIELD = "problem"
FILENAME = "openmathinstruct2.jsonl"
SYSTEM_PROMPT = "Your task is to solve a math problem. Make sure to put the answer (and only the answer) inside \\boxed{}."

dirpath = os.path.dirname(FILENAME) or "."
with open(FILENAME, "r", encoding="utf-8") as fin, \
    open(os.path.join(dirpath, "train.jsonl"), "w", encoding="utf-8") as ftrain, \
    open(os.path.join(dirpath, "validation.jsonl"), "w", encoding="utf-8") as fval:
    for i, line in enumerate(fin):
        if not line.strip():
            continue
        row = json.loads(line)
        row.pop("generated_solution", None)
        row.pop("problem_source", None)
        row["responses_create_params"] = {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row.get(INPUT_FIELD, "")},
            ]
        }
        # Rename the input field to "question"
        row["question"] = row.pop(INPUT_FIELD, None)
        out = json.dumps(row) + "\n"
        # Save the first 99% of the data to train.jsonl
        (ftrain if i < 999000 else fval).write(out)
```

Run the script with:

```bash
uv run preprocess.py
```

This will add the `responses_create_params` field to each row in your dataset and remove the `generated_solution` and `problem_source` fields as they are not used for training and add extra bloat to the files.

This will create a `train.jsonl` and `validation.jsonl` file locally with the former containing 999,000 lines and the latter containing 1,000 lines of data. Verify the contents with:

```bash
$ wc -l train.jsonl validation.jsonl
   999000 train.jsonl
     1000 validation.jsonl
  1000000 total
```

Your first line of each file should look similar to the following:

```bash
$ head -n 1 train.jsonl 
{"problem": "Solve for $y$:\n\n$$\\frac{y^2 - 3y + 2}{y - 2} = y + 1$$", "expected_answer": "2", "responses_create_params": {"input": [{"role": "system", "content": "Your task is to solve a math problem. Make sure to put the answer (and only the answer) inside \\boxed{}."}, {"role": "user", "content": "Solve for $y$:\n\n$$\\frac{y^2 - 3y + 2}{y - 2} = y + 1$$"}]}}
$ head -n 1 validation.jsonl
{"problem": "Let $n$ be the number of positive integers less than 100 that are either a multiple of 4 or a multiple of 5, but not both. Let $m$ be the number of positive integers less than 100 that are multiples of 20. Compute the value of $(n-m)^2$.", "expected_answer": "961", "responses_create_params": {"input": [{"role": "system", "content": "Your task is to solve a math problem. Make sure to put the answer (and only the answer) inside \\boxed{}."}, {"role": "user", "content": "Let $n$ be the number of positive integers less than 100 that are either a multiple of 4 or a multiple of 5, but not both. Let $m$ be the number of positive integers less than 100 that are multiples of 20. Compute the value of $(n-m)^2$."}]}}
```

The dataset is now compatible with NeMo Gym and we can begin creating our configuration files.

:::{note}
We skipped the `agent_ref` field here as that gets added automatically by the `ng_prepare_data` tool as shown later in this tutorial.
:::

:::{note}
Certain deterministic datasets may require an `expected_answer` field, such as math and science-based datasets that expect a particular answer. This will depend on the resource server that you use. The `expected_answer` is used to match the generated response with the expected value for accuracy.
:::

---

## 3. Setup the Config File

With the dataset processed locally, we need to create a config file to instruct NeMo Gym how to process the data during post-training. This is done by creating a new resource server config.

Since this tutorial is focused on math, we will use one of the existing math-based resource server configs as a template. For other domains, use an existing resource server's config as a template that best matches your use-case. For custom resource servers, refer to the [Creating Resource Server](creating-resource-server.md) guide. We will use the `resources_servers/math_with_judge/configs/math_with_judge.yaml` config as a template for this example.

Create a new YAML file called `openmathinstruct2.yaml` and save it in `resources_servers/math_with_judge/configs/`. Copy the following contents to the config file:

```yaml
math_with_judge:
  resources_servers:
    math_with_judge:
      entrypoint: app.py
      judge_model_server:
        type: responses_api_models
        name: policy_model
      judge_responses_create_params: {
        input: []
      }
      should_use_judge: false
      domain: math
      verified: false
      description: Open math dataset with math-verify
      value: Improve math capabilities
math_with_judge_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: math_with_judge
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: train.jsonl
        license: Creative Commons Attribution 4.0 International
      - name: validation
        type: validation
        jsonl_fpath: validation.jsonl
        license: Creative Commons Attribution 4.0 International
```

This script builds on the `math_with_judge` resource server config while pointing to the `train.jsonl` and `validation.jsonl` files that were processed locally. The key difference is the `jsonl_fpath` value points directly to the processed files and it doesn't try to download a dataset directly.

---

## 4. Prepare the Dataset Using NeMo Gym

With the config file saved, the `ng_prepare_data` tool will process and validate the dataset and add the `agent_ref` field to each row in the dataset. To recap, the `agent_ref` tag informs NeMo gym of which resource server to use to process each row in the dataset. Since a dataset can contain a blend of data pointing at different resource servers, the `agent_ref` does the routing for NeMo Gym.

The `ng_prepare_data` tool matches the agent type in the config file with the agent name. In our case, the agent name is `math_with_judge_simple_agent` and the agent type is `responses_api_agents`. To use a different agent, modify the config file in the previous section to the desired agent.

To prepare the dataset, run `ng_prepare_data` as follows:

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/math_with_judge/configs/openmathinstruct2.yaml"
uv run ng_prepare_data "+config_paths=[${config_paths}]" +mode=train_preparation +output_dirpath=data
```

In the above command, we indicate we want to use the `vllm_model_for_training.yaml` config for our API model and the new `openmathinstruct2.yaml` config we created for processing the math data.

The command takes about a minute to run and validates the datasets, adds the `agent_ref` tag, and collates datasets into the output directory in a single training file and a single validation file. If you used multiple resource servers with multiple configs, it will append all of the data from the individual input files into a single output file.

After the command completes, you should see your train and validation files in the `data/` directory, each containing the same line counts as before:

```bash
$ wc -l data/*.jsonl
   999000 data/train.jsonl
     1000 data/validation.jsonl
  1000000 total
```

You should also see the new `agent_ref` field in the dataset:

```bash
$ head -n 1 data/train.jsonl 
{"problem": "Solve for $y$:\n\n$$\\frac{y^2 - 3y + 2}{y - 2} = y + 1$$", "expected_answer": "2", "responses_create_params": {"input": [{"role": "system", "content": "Your task is to solve a math problem. Make sure to put the answer (and only the answer) inside \\boxed{}."}, {"role": "user", "content": "Solve for $y$:\n\n$$\\frac{y^2 - 3y + 2}{y - 2} = y + 1$$"}]}, "agent_ref": {"type": "responses_api_agents", "name": "math_with_judge_simple_agent"}}
```

Your custom dataset is now ready for post-training with NeMo Gym!

---

## Next Steps

Now that you have a valid dataset:

1. **Create Resource Server**: Create a custom resource server for processing novel data
2. **Integrate with RL**: Use {ref}`RL Training with NeMo RL using GRPO <training-nemo-rl-grpo-index>` to train models on your tasks

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Collect Rollouts
:link: offline-training-w-rollouts
:link-type: doc
Learn how to collect and process rollouts for your data.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Train models using your custom dataset with NeMo RL.
:::

::::

---

## Troubleshooting

### Import errors

Ensure you are running commands from the repository root directory and have installed dependencies:

```bash
uv sync
```

### 403 error while pulling Hugging Face dataset

For private and/or gated datsets on Hugging Face, an authorization token is required. Set your HF token with:

```bash
HF_TOKEN=<insert token here>
```

Re-run any failed commands again after setting the variable.

---

## Summary

You've learned how to:

✅ Download a dataset from Hugging Face
✅ Add the `responses_create_params` field to each row  
✅ Write a resource server config file to process your data
✅ Prepare the dataset with `ng_prepare_data`

You are now ready to use your processed custom dataset for a variety of downstream tasks in NeMo Gym!
