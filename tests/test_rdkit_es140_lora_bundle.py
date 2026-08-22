from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "cluster/rdkit_no_tool_grpo"


def load_script(name: str):
    path = BUNDLE / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_materializes_explicit_four_sample_validation(tmp_path: Path) -> None:
    prepare = load_script("prepare_direct_data.py")
    source = tmp_path / "source"
    source.mkdir()
    for split, count in (("train", 3), ("test", 2)):
        rows = [
            json.dumps({"method": "direct", "prompt": f"{split} prompt {index}"})
            for index in range(count)
        ]
        (source / f"{split}.jsonl").write_text("\n".join(rows) + "\n")

    assert prepare.convert_split(tmp_path, source, "train", 123) == 3
    assert prepare.convert_split(tmp_path, source, "test", 123) == 2
    assert prepare.materialize_repeated_validation(tmp_path) == 8

    data = tmp_path / prepare.OUTPUT_REL
    test_rows = (data / "test.jsonl").read_text().splitlines()
    repeated = (data / "test_eval4.jsonl").read_text().splitlines()
    assert repeated == [row for row in test_rows for _ in range(4)]
    first = json.loads(repeated[0])
    assert first["responses_create_params"]["max_output_tokens"] == 123
    assert first["responses_create_params"]["tools"] == []


def test_production_config_preserves_batch_and_lora_contract() -> None:
    config = yaml.safe_load((BUNDLE / "rdkit_no_tool_grpo.yaml").read_text())
    grpo = config["grpo"]
    policy = config["policy"]
    peft = policy["megatron_cfg"]["peft"]

    assert grpo["num_prompts_per_step"] == 64
    assert grpo["num_generations_per_prompt"] == 16
    assert grpo["max_num_steps"] == 200
    assert grpo["val_period"] == 5
    assert grpo["val_at_start"] is False
    assert policy["train_global_batch_size"] == 1024
    assert policy["generation"]["max_new_tokens"] == 32768
    assert policy["max_total_sequence_length"] == 65536
    assert peft == {
        "enabled": True,
        "target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        "exclude_modules": [],
        "dim": 8,
        "alpha": 8,
        "dropout": 0.0,
        "dropout_position": "post",
        "lora_A_init_method": "xavier",
        "lora_B_init_method": "zero",
        "a2a_experimental": False,
        "lora_dtype": None,
    }
    assert config["data"]["validation"]["data_path"].endswith("test_eval4.jsonl")


def test_production_launcher_requires_baseline_dependency() -> None:
    launcher = (BUNDLE / "submit_chain.sh").read_text()
    assert "START_DEPENDENCY must identify" in launcher
    assert "CHAIN_JOBS=${CHAIN_JOBS:-3}" in launcher
    assert "JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-12:00:00}" in launcher
    assert "--dependency=\"${START_DEPENDENCY}\"" in launcher
    assert "REQUIRE_BASELINE_GATE=1" in launcher


def test_wandb_internal_id_leaves_room_for_rollout_table_names() -> None:
    config = yaml.safe_load((BUNDLE / "rdkit_no_tool_grpo.yaml").read_text())
    launcher = (BUNDLE / "submit_chain.sh").read_text()
    run_id = "rdkit-es140-grpo-v06-r8"
    run_name = (
        "rdkit-nemotron3-nano-grpo-lora-r8-a8-es140-64p16g-i200-"
        "lr3e-6-32k-iad-p0-64g"
    )
    artifact_name = f"run-{run_id}-trainrdkit_chemistry_direct_agentfull_result"

    assert f"WANDB_RUN_ID=${{WANDB_RUN_ID:-{run_id}}}" in launcher
    assert 'if (( ${#WANDB_RUN_ID} > 32 )); then' in launcher
    assert config["logger"]["wandb"]["id"] == f"${{oc.env:WANDB_RUN_ID,{run_id}}}"
    assert config["logger"]["wandb"]["name"] == f"${{oc.env:WANDB_RUN_NAME,{run_name}}}"
    assert run_id != run_name
    assert len(run_id) <= 32
    assert len(artifact_name) <= 128


def test_cluster_launch_stages_code_and_caches_node_local() -> None:
    ray_sub = (BUNDLE / "nemo_rl_assets/ray.sub").read_text()
    assert "/raid/scratch/${USER}/rdkit-nemo-rl-${SLURM_JOB_ID}" in ray_sub
    assert 'export GYM_DIR="${STAGED_WORKDIR}"' in ray_sub
    assert 'export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-$NODE_LOCAL_ROOT/enroot/cache}' in ray_sub
    assert 'export HF_HOME="${NODE_LOCAL_ROOT}/hf-home"' in ray_sub
    assert 'export WANDB_DIR="${NODE_LOCAL_ROOT}/wandb"' in ray_sub
    assert "UV_CACHE_DIR_OVERRIDE:/root/.cache/uv" not in ray_sub
    assert "unset UV_CACHE_DIR UV_CACHE_DIR_OVERRIDE" in ray_sub
    assert ray_sub.count("import ray.scripts.scripts") == 2
    assert ray_sub.count("torch.cuda.get_device_capability()") == 2
    assert "setup_integration_venv.sbatch" not in (BUNDLE / "README.md").read_text()
    assert (BUNDLE / "build_integration_sqsh.sbatch").is_file()
