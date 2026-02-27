# Generating Multi-Step Tool-Calling Datasets with Data Designer

Generate synthetic training data for multi-step tool-calling agents using [NVIDIA Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner). This notebook produces user queries and simulated agent trajectories for the Workplace Assistant environment, applies dual-level LLM judge filtering for quality control, and exports training data in NeMo Gym JSONL format.

## Prerequisites

- **NVIDIA API Key** from [build.nvidia.com](https://build.nvidia.com) (or your own LLM endpoint)
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
cd examples/multistep_toolcalling_datagen/
jupyter notebook multistep-toolcalling.ipynb
```

> **Important:** Run the notebook from this directory (`examples/multistep_toolcalling_datagen/`) so that relative imports for `tools/` and `utils/` resolve correctly.

## Related Resources

- [Workplace Assistant Resource Server](../../resources_servers/workplace_assistant/)
- [NeMo Gym Rollout Collection](https://docs.nvidia.com/nemo/gym/latest/get-started/rollout-collection.html)
- [NVIDIA Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)

## Licensing

Code: Apache 2.0
Data: Apache 2.0
