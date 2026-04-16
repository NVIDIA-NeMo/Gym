# IFBench

[IFBench](https://github.com/allenai/IFBench) is an instruction following benchmark from AllenAI that evaluates how well a model follows explicit constraints embedded in a prompt. It covers 57 instruction types across categories like word counts, keyword placement, formatting, and custom puzzle-like constraints.

## Configuration

- **Grading mode**: `fraction` — reward is the fraction of instructions followed per response
- **Resources server**: `ifbench` (dedicated, uses AllenAI's `instructions_registry` directly)

## Prerequisites

The `ifbench` resources server clones the AllenAI IFBench repo from GitHub on first startup. **This requires outbound internet access from wherever the server process runs.** See [No internet access](#no-internet-access) below if you are running inside a container that restricts outbound network access.

## Prepare data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/ifbench/config.yaml]"
```

## Run

```bash
ng_e2e_collect_rollouts \
    "+config_paths=[benchmarks/ifbench/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    ++output_jsonl_fpath=results/benchmarks/ifbench.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++resume_from_cache=true \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<>
```

## No internet access

If the container restricts outbound network access, the server will crash on startup with:

```
subprocess.CalledProcessError: Command '['git', 'clone', ..., 'https://github.com/allenai/IFBench.git', ...]' died with <Signals.SIGABRT: 6>
```

This happens because `setup_ifbench.py` tries to clone the IFBench repo at server startup, but the outgoing connection is blocked inside the container. Note that the cluster node itself may have internet access, but the container environment may block it.

**Fix:** run `setup_ifbench.py` outside the container from the repo root. It handles the clone, patch, and marker automatically:

```bash
python resources_servers/ifbench/setup_ifbench.py
```

Make sure `.ifbench/` is included when you copy or sync the repo to the cluster. Once the `.installed` marker is present, the server skips the clone entirely on startup.
