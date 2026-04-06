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

**Fix:** pre-clone the repo outside the container (e.g. on a login node), then include it when syncing the repo to the cluster.

```bash
# Run this outside the container, on a machine with internet access
cd nemo-gym
git clone https://github.com/allenai/IFBench.git resources_servers/ifbench/.ifbench
cd resources_servers/ifbench/.ifbench
git checkout c6767a19bd82ac0536cab950f2f8f6bcc6fabe7c
cd ..

# Patch out the spaCy auto-download (fires at import time in the unpatched code)
sed -i "s/^download('en_core_web_sm')$/# download('en_core_web_sm')  # pre-installed via requirements.txt/" .ifbench/instructions.py

# Mark the clone as complete so setup_ifbench.py skips it on startup
touch .ifbench/.installed
```

Make sure `.ifbench/` is included when you copy or sync the repo to the cluster. Once the `.installed` marker is present, `setup_ifbench.py` skips the clone entirely and goes straight to adding the repo to `sys.path`.
