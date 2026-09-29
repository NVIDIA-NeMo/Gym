# BrowseComp Advanced Harness

RL environment for the [BrowseComp](https://openai.com/index/browsecomp/) benchmark — a challenging web-browsing research task
requiring multi-hop investigation through search and page-level browsing.

## Overview

This resources server provides two browsing tools backed by [Tavily](https://tavily.com/):

- **`search`** — parallel web search returning top results with content snippets
- **`browse`** — full-page content extraction from given URLs

Reward is computed via an LLM judge that compares the model's final answer against the ground truth.

## Key Config Parameters

Add to your `env.yaml` (or pass via CLI):

```yaml
tavily_api_key: <YOUR_TAVILY_KEY>   # or a list of keys for rotation
exclude_domains_file_path: /path/to/excluded_domains.json

judge_model_base_url: <YOUR_JUDGE_MODEL_URL>
judge_model_api_key: ""
judge_model_name: Qwen/Qwen3-235B-A22B-Instruct-2507
```

## Running

```bash
# If you want to run with browsecomp benchmark instead of the example samples, need to change the datasets part to the one like `benchmarks/browsecomp/config.yaml` in `resources_servers/browsecomp_advanced_harness/configs/browsecomp_advanced_harness.yaml`
config_paths="resources_servers/browsecomp_advanced_harness/configs/browsecomp_advanced_harness.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_run "+config_paths=[${config_paths}]"
```


# Licensing information
Code: ?
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
