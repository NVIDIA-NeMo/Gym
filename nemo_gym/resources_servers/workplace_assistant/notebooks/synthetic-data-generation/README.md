# Generating Multi-Step Tool-Calling Datasets with Data Designer

Generate synthetic training data for multi-step tool-calling agents using [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner). This notebook produces user queries and simulated agent trajectories for the Workplace Assistant environment, applies two levels of LLM judge filtering for quality control, and exports training data in NeMo Gym JSONL format.

## Prerequisites

- **NVIDIA API Key** from [build.nvidia.com](https://build.nvidia.com) (or your own LLM endpoint). If you need one, go to [API Keys](https://build.nvidia.com/settings/api-keys), sign in, and create a key before opening the notebook.
- **Python 3.11+**

## Setup

```bash
# From this directory
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

## Usage

Open the notebook in Jupyter or Colab and run the cells sequentially:

```bash
cd nemo_gym/resources_servers/workplace_assistant/notebooks/synthetic-data-generation/
jupyter notebook multistep-toolcalling-sdg.ipynb
```

> **Important:** Run the notebook from this directory (`nemo_gym/resources_servers/workplace_assistant/notebooks/synthetic-data-generation/`) so that relative imports for `tools/` and `utils/` resolve correctly.

## Related Resources

- [Workplace Assistant Resource Server](https://github.com/NVIDIA-NeMo/Gym/tree/main/nemo_gym/resources_servers/workplace_assistant)
- [NeMo Gym Rollout Collection](https://docs.nvidia.com/nemo/gym/latest/get-started/rollout-collection.html)
- [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)