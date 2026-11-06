# AgentVista with OpenCode

This external-benchmark adapter supports both Eval Factory evaluation and NeMo RL
training through actual Gym agent/model APIs. The separately provisioned
[AgentVista source](https://gitlab-master.nvidia.com/pritamb/agentvista/-/tree/06f6a8f1532766d876415757107a11731b6148ae)
owns the four original tools, MCP server, native OpenCode worker, dataset
materialization, answer judge and run preparation. No SaaS runtime dependency is added.

Set `AGENTVISTA_ROOT`, `AGENTVISTA_DATASET`, `AGENTVISTA_IMAGES`,
`AGENTVISTA_OUTPUT_ROOT`, `AGENTVISTA_WORKER_PYTHON`, and `AGENTVISTA_OPENCODE`.
Use a separate worker environment with `integrations/requirements.txt` from the
source checkout and OpenCode CLI 1.17.11. Configure Serper, optional Jina, the
fixed visit summarizer and the original answer judge as documented by that source.
Judge credentials are excluded from the actor's environment.

`config.yaml` selects the evaluation agent and the legacy environment relay.
Factory should pass job-specific `agentvista_dataset_path` and
`agentvista_output_root` values to both preparation and collection. The dataset
preparer hash-binds every source record/image without any model requests.
Evaluation policy calls use `agentvista_opencode.py`; training uses
`agentvista_training.py` and `responses_api_agents/agentvista_agent/training.py`.

The training adapters preserve trainer sampling and witness the terminal served
completion identity. Gym owns sampled token IDs, log probabilities and media;
no text retokenization is used to reconstruct policy training data. Missing or
ambiguous captures and infrastructure failures are masked. The answer-judge
reward is model-judged, not deterministic RLVR verification.

Defaults are 150 OpenCode steps and a 5400-second actor timeout. MCP tool
namespacing and OpenCode's history/compaction make this a new harness protocol;
no exact original-context or published-score parity is claimed.

Local source validation covers real OpenCode/MCP/Python image feedback and Gym
HTTP/schema/correlation contracts. Run the source-backed contracts with
`AGENTVISTA_ROOT=/path/to/agentvista pytest responses_api_agents/agentvista_agent/tests`.
A hosted-model benchmark run and GPU optimizer/refit smoke remain outstanding;
this integration is not baselined or certified.
