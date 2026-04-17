# Description

The train data can be found at `example_multi_step/data/train.jsonl` and the validation data can be found at `example_multi_step/data/validation.jsonl`.

This is a demonstration example of a multi-step extraction agent. The LLM will be provided a user query and need to use the tools provided to extract list of synonym values. The agent will be provided with a bunch of synonyms for each, and it must get and extract the values for every synonym that appears in this list.

## Tutorial

For an annotated walkthrough of this server's tools and verification logic, see the [Multi-Step Environment](https://docs.nvidia.com/nemo/gym/latest/environment-tutorials/multi-step-environment.html) tutorial.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
