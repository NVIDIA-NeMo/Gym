(env-real-world-data-generation)=

# Generating Training Data

Generate synthetic task data (user queries) for the {doc}`Workplace Assistant <real-world-environment>` environment using [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner). 

This pipeline focuses on generating tasks for use with the environment. It also simulates agent trajectories, but these are used for quality filtering and validation --- the environment itself produces the actual model responses during rollout collection. The Workplace Assistant uses 27 tools across 6 databases, and NeMo Data Designer can produce realistic multi-step user queries at scale.

:::{button-ref} real-world-environment
:color: secondary
:outline:
:ref-type: doc

< Back to Workplace Assistant
:::

---

## Pipeline Overview

The data generation pipeline:

1. Load tool schemas for the Workplace Assistant environment
2. Use NeMo Data Designer to generate realistic multi-step user queries
3. Simulate agent trajectories (step-by-step tool-call solutions)
4. Apply dual-level LLM judge filtering to ensure data quality
5. Export task data in NeMo Gym JSONL format

---

## Prerequisites

- **NVIDIA API Key** from [build.nvidia.com](https://build.nvidia.com) (or your own LLM endpoint)
- **Python 3.11+**
- Install tutorial dependencies (paths relative to the [NeMo Gym repository root](https://github.com/NVIDIA-NeMo/Gym)):

  ```bash
  cd resources_servers/workplace_assistant/notebooks/synthetic-data-generation/
  uv pip install -r requirements.txt
  ```

---

## Running the Notebook

The tutorial is provided as a Jupyter notebook at `resources_servers/workplace_assistant/notebooks/synthetic-data-generation/multistep-toolcalling-sdg.ipynb`.

> **Important:** Run the notebook from the `resources_servers/workplace_assistant/notebooks/synthetic-data-generation/` directory so that relative imports for `tools/` and `utils/` resolve correctly.

:::{button-link} https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/workplace_assistant/notebooks/synthetic-data-generation/multistep-toolcalling-sdg.ipynb
:color: primary
:class: sd-rounded-pill

View Notebook on GitHub
:::

---

## What's Next?

After generating your task data, use it with the Workplace Assistant resources server to {doc}`collect rollouts </get-started/rollout-collection>` (where the environment produces model responses) and then proceed to {doc}`GRPO training </training-tutorials/nemo-rl-grpo/index>`.

:::{button-ref} real-world-implementation
:color: primary
:outline:
:ref-type: doc

Next: Resources Server Implementation >
:::
