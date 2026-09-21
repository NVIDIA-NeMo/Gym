# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch the ASB campaign on Modal -- one container per model, in parallel.

    modal run --env=FDR scripts/modal_harness/asb.py                  # every incomplete model
    modal run --env=FDR scripts/modal_harness/asb.py --models supervl,qwen

Models run concurrently because each gets its own container: no shared ports, no shared Ray
cluster, no shared memory. That is the difference from running them on one machine, where
three stacks produced twenty server processes and one working eval.

Each is pinned to its endpoint's region so the harness is not crossing the country on every
call: three deployments are us-west, Super-VL is us-east.
"""

from __future__ import annotations

from scripts.modal_harness.campaign import app, campaign_state, run_campaign, skip_reason


#: key -> (slug, base_url, model id, token env var, concurrency)
#:
#: Concurrency is per-deployment rather than global. Ultra and Kimi sustain high ceilings;
#: Qwen and Super-VL emit 2-3x the tokens per rollout and were single-replica until scaled,
#: so their right ceiling is a property of the deployment and not a constant.
MODELS: dict[str, tuple[str, str, str, str, int]] = {
    "ultra": (
        "nemotron-3-ultra-550b",
        "https://snorkelai-fdr--ep-nvidia-nemotron-3-ultra-550b-a55b-nvfp-63eebc.us-west.modal.direct/v1",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        "MODAL_PROXY_TOKEN",
        96,
    ),
    "kimi": (
        "kimi-k3",
        "https://snorkelai-fdr--ep-kimi-k3-server.us-west.modal.direct/v1",
        "moonshotai/Kimi-K3",
        "MODAL_PROXY_TOKEN",
        96,
    ),
    "qwen": (
        "qwen3.5-122b-a10b",
        "https://snorkelai-fdr--ep-qwen3-5-122b-a10b-fp8-server.us-west.modal.direct/v1",
        "Qwen/Qwen3.5-122B-A10B-FP8",
        "MODAL_PROXY_TOKEN",
        96,
    ),
    "supervl": (
        "nemotron-3.5-super-vl",
        "https://snorkelai-fdr--nemotron-3-5-super-vl-ea-nemotronvision.us-east.modal.direct/v1",
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        "SUPER_VL_MODAL_TOKEN",
        96,
    ),
}

CONFIG_PATHS = ["resources_servers/asb/configs/asb.yaml"]
AGENT = "asb_agent"
INPUT_PATH = "resources_servers/asb/data/all.jsonl"
PREPARE_MODULE = "benchmarks.asb.prepare"
EXPECTED_ROWS = 10_800
GIT_REF = "claude-rundgren/asb-adapter-nemogym-318586"


@app.local_entrypoint()
def main(models: str = "supervl,qwen", concurrency: int = 0, git_ref: str = GIT_REF) -> None:
    keys = [key.strip() for key in models.split(",") if key.strip()]
    unknown = [key for key in keys if key not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model keys: {', '.join(unknown)} (have {', '.join(MODELS)})")

    state = campaign_state.remote("asb")

    handles = []
    for key in keys:
        slug, base_url, model_id, token_var, default_concurrency = MODELS[key]
        reason = skip_reason(state, slug, EXPECTED_ROWS)
        if reason:
            print(f"skip {key} -> {slug}: {reason}")
            continue
        print(f"launching {key} -> {slug}")
        handles.append(
            run_campaign.spawn(
                slug=slug,
                namespace="asb",
                config_paths=CONFIG_PATHS,
                agent=AGENT,
                input_path=INPUT_PATH,
                policy_base_url=base_url,
                policy_model=model_id,
                policy_token_var=token_var,
                expected_rows=EXPECTED_ROWS,
                concurrency=concurrency or default_concurrency,
                judge_server_name="asb_judge_model",
                prepare_module=PREPARE_MODULE,
                git_ref=git_ref,
            )
        )

    print(f"\n{len(handles)} container(s) running in parallel; waiting...\n")
    for handle in handles:
        result = handle.get()
        short = f" ({result['missing']} missing)" if result["missing"] else ""
        print(f"  {result['slug']:24s} {result['landed']}/{result['expected']}{short}")
