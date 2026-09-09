# Description
RL enviroment which allows access to web search (Search Provider: Tavily)

## Prerequisites and setup

Follow [Tavily search access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/tavily-search) before starting this environment. It covers the required domain-exclusion policy, portable local configuration, separate Tavily/model/GitLab credentials, and dataset paths.

The documented internal workflow requires approved NVIDIA service, policy-artifact, and dataset access. NVIDIA users can start with the existing access contact, @rgala. The repository does not document a public distribution route for the approved exclusion policy; external users must confirm an authorized route with the maintainers before running this recipe. Do not omit or replace the required policy with an empty/test file to make startup succeed.


### Performance Metrics
100*16 samples:
- Acc: 0.3212
- Time in `gym eval run`: 44 mins


# Licensing information
Code: ?
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
