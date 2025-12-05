(training-datasets-overview)=

# Datasets

Prepare, validate, and format training datasets for RL, SFT, and DPO workflows.

---

## How It Works

In NeMo Gym, datasets are **outputs** of the rollout collection pipeline, not inputs. After collecting and verifying rollouts, you format them for your training objective.

```{mermaid}
flowchart LR
    subgraph Input["Inputs"]
        P["Prompts"]
        RS["Resource<br/>Servers"]
    end
    
    subgraph Pipeline["Gym Pipeline"]
        C["Collect"]
        V["Verify"]
    end
    
    subgraph Output["Training Datasets"]
        RL["RL Format<br/>(GRPO)"]
        SFT["SFT Format"]
        DPO["DPO Format"]
    end
    
    Input --> Pipeline
    Pipeline --> Output
    
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef pipeline fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000
    
    class P,RS input
    class C,V pipeline
    class RL,SFT,DPO output
```

The `ng_prepare_data` command adds agent routing metadata and formats rollouts for specific training frameworks.

---

## Prepare Data

Format and validate rollouts for training.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Prepare Training Data
:link: prepare-data
:link-type: doc

Use `ng_prepare_data` to validate, format, and add agent routing to datasets.
+++
{bdg-secondary}`ng_prepare_data`
:::

:::{grid-item-card} {octicon}`upload;1.5em;sd-mr-1` HuggingFace Integration
:link: huggingface-integration
:link-type: doc

Upload and download datasets from HuggingFace Hub.
+++
{bdg-secondary}`sharing`
:::

::::

---

## Reference

Look up format specifications and schema requirements.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` Format Specification
:link: format-specification
:link-type: doc

Schema definitions for SFT, DPO, and RL training data formats.
+++
{bdg-secondary}`reference`
:::

::::

---

```{toctree}
:maxdepth: 1
:hidden:

prepare-data
huggingface-integration
format-specification
```

