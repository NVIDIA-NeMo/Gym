#!/usr/bin/env python3
"""Validate the RDKit no-tool GRPO bundle before Slurm submission."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

EXPECTED = {"train": 1024, "test": 1000}
SMOKE_EXPECTED = {"train_smoke": 64, "test_smoke": 128}
VALIDATION_REPEATS = 4
EXPECTED_DATA_HASHES = {
    "train": "7acc2e5b9909ee3279fa26599ef1cba1388e328a2f248c152137bae2652c00cf",
    "test": "b02ee40add79325edbc5f41c785a1c1288dd471de9d633046d063525f59a3303",
    "train_smoke": "852cc228799ea8317f555f164ba78a753985fc78a1e2f2616f09192368de0383",
    "test_smoke": "9bb195bf7de05a1c5c437b46966ecd1e1acf44efdf2d12ed52eba13d653f74fb",
}
AGENT = "rdkit_chemistry_direct_agent"
DEFAULT_MAX_OUTPUT_TOKENS = 32768
PINNED_GYM_COMMIT = "1a4912e231bb2795b062f7de97496caaf382c7f6"
ES_ADAPTER_SHA256 = "e281afb74b78d5ef337c233c1b1faeb42ac72208ca76ed4b5563e1ca4adb3de4"
ES_ADAPTER_TENSOR_COUNT = 11916
RDKIT_SERVER_COMMIT = "a9fb7a13cf5492737945124f09a8c20e563f8ef0"
RDKIT_SERVER_HASHES = {
    "app.py": "1e18cb15c51abdddad129b27fa812f27e73cc4bde4893fd072fd43edcdb40312",
    "requirements.txt": "4427b1e36fdc2b7ad1c8b811e9ff64877fcde44457cdb34425ad04fdeef13f23",
    "sandbox_launcher.py": "9f576bf8d7a885dea2d1b8867cc4db6777f9f12503f0f0807131d10ea15cddc7",
}
SERVER_COMPONENTS = (
    ("resources_servers", "rdkit_chemistry", "app.py"),
    ("responses_api_agents", "simple_agent", "app.py"),
    ("responses_api_models", "vllm_model", "app.py"),
)


def default_max_output_tokens() -> int:
    raw = os.environ.get("MAX_OUTPUT_TOKENS") or os.environ.get("MAX_NEW_TOKENS")
    if raw is None:
        return DEFAULT_MAX_OUTPUT_TOKENS
    value = int(raw)
    if value <= 0:
        raise ValueError("max output tokens must be positive")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(
    data_dir: Path, split: str, expected_count: int, expected_max_output_tokens: int
) -> None:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    count = 0
    for line_number, line in enumerate(path.open(), start=1):
        row = json.loads(line)
        if row.get("method") != "direct":
            raise ValueError(f"{path}:{line_number} method is not direct")
        rcp = row.get("responses_create_params")
        if not isinstance(rcp, dict):
            raise ValueError(f"{path}:{line_number} missing responses_create_params")
        if rcp.get("tools") != []:
            raise ValueError(f"{path}:{line_number} tools must be []")
        if rcp.get("max_output_tokens") != expected_max_output_tokens:
            raise ValueError(
                f"{path}:{line_number} max_output_tokens is {rcp.get('max_output_tokens')}, "
                f"expected {expected_max_output_tokens}"
            )
        if not rcp.get("input"):
            raise ValueError(f"{path}:{line_number} missing responses_create_params.input")
        if row.get("agent_ref", {}).get("name") != AGENT:
            raise ValueError(f"{path}:{line_number} wrong agent_ref")
        count += 1

    if count != expected_count:
        raise ValueError(f"{path} has {count} rows, expected {expected_count}")
    expected_hash = EXPECTED_DATA_HASHES[split]
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{path} has SHA-256 {actual_hash}, expected {expected_hash}")


def validate_repeated_validation(data_dir: Path, expected_max_output_tokens: int) -> None:
    source_path = data_dir / "test.jsonl"
    repeated_path = data_dir / "test_eval4.jsonl"
    if not repeated_path.is_file():
        raise FileNotFoundError(repeated_path)
    source_rows = source_path.read_text().splitlines()
    repeated_rows = repeated_path.read_text().splitlines()
    expected_count = len(source_rows) * VALIDATION_REPEATS
    if len(repeated_rows) != expected_count:
        raise ValueError(f"{repeated_path} has {len(repeated_rows)} rows, expected {expected_count}")
    for index, source_row in enumerate(source_rows):
        start = index * VALIDATION_REPEATS
        if repeated_rows[start : start + VALIDATION_REPEATS] != [
            source_row
        ] * VALIDATION_REPEATS:
            raise ValueError(f"{repeated_path} does not repeat test row {index} four times")
    first = json.loads(repeated_rows[0])
    if first["responses_create_params"]["max_output_tokens"] != expected_max_output_tokens:
        raise ValueError(f"{repeated_path} has the wrong output-token budget")


def validate_repository_version(root: Path, source_gym_commit: str | None) -> str:
    if source_gym_commit:
        if len(source_gym_commit) != 40 or any(
            character not in "0123456789abcdef" for character in source_gym_commit
        ):
            raise ValueError("--source-gym-commit must be a full lowercase Git SHA")
        return source_gym_commit
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PINNED_GYM_COMMIT, "HEAD"],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        actual_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        raise ValueError(
            f"Gym checkout {actual_commit} does not descend from pinned commit "
            f"{PINNED_GYM_COMMIT}"
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_lora_config(bundle: Path) -> None:
    config = OmegaConf.load(bundle / "rdkit_no_tool_grpo.yaml")
    peft = config.policy.megatron_cfg.peft
    expected_targets = {"linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"}
    if not peft.enabled:
        raise ValueError("Megatron PEFT must be enabled")
    if peft.dim != 8 or peft.alpha != 8 or peft.dropout != 0.0:
        raise ValueError("Expected rank-8, alpha-8, zero-dropout Megatron LoRA")
    if set(peft.target_modules) != expected_targets or list(peft.exclude_modules):
        raise ValueError("Unexpected Megatron LoRA target or exclusion modules")
    if peft.lora_A_init_method != "xavier" or peft.lora_B_init_method != "zero":
        raise ValueError("The continuation adapter must start with a zero output")
    if config.policy.megatron_cfg.optimizer.lr != 3.0e-6:
        raise ValueError("Expected GRPO LoRA learning rate 3e-6")
    if config.policy.max_total_sequence_length != 65536:
        raise ValueError("Expected total sequence length 65536")
    if config.policy.generation.max_new_tokens != 32768:
        raise ValueError("Expected generated-token limit 32768")
    if config.grpo.num_prompts_per_step != 64 or config.grpo.num_generations_per_prompt != 16:
        raise ValueError("Expected 64 prompts x 16 generations per optimizer step")
    if config.grpo.num_val_generations_per_prompt != 4:
        raise ValueError("Expected four generations per validation prompt")
    if not str(config.data.validation.data_path).endswith("/test_eval4.jsonl"):
        raise ValueError("Production validation must use the explicit 4,000-row repeat split")
    if config.cluster.num_nodes != 8 or config.cluster.gpus_per_node != 8:
        raise ValueError("Expected the 64-GPU IAD production layout")


def validate_es_adapter(adapter_dir: Path) -> None:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    provenance_path = adapter_dir / "provenance.json"
    for path in (config_path, weights_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(weights_path) != ES_ADAPTER_SHA256:
        raise ValueError("Unexpected ES Step 140 adapter hash")
    adapter_config = json.loads(config_path.read_text())
    if adapter_config.get("r") != 8 or adapter_config.get("lora_alpha") != 8.0:
        raise ValueError("Unexpected ES adapter rank or alpha")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("selection", {}).get("inferno_update_count") != 140:
        raise ValueError("Adapter provenance is not post-Step 140")
    if provenance.get("lora", {}).get("tensor_count") != ES_ADAPTER_TENSOR_COUNT:
        raise ValueError("Unexpected ES adapter tensor count")


def validate_merged_model(merged_model_dir: Path, *, deep: bool = False) -> None:
    manifest_path = merged_model_dir / "es140_merge_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported ES merge manifest schema")
    if manifest.get("merge_method") != "direct_safetensors_lora_b_matmul_a":
        raise ValueError("Merged model was not produced by the validated direct merge")
    if manifest.get("source_adapter", {}).get("sha256") != ES_ADAPTER_SHA256:
        raise ValueError("Merged model was not built from the expected ES adapter")
    if manifest.get("source_adapter", {}).get("inferno_update_count") != 140:
        raise ValueError("Merged model manifest does not identify Step 140")
    validation = manifest.get("validation", {})
    if validation.get("passed") is not True:
        raise ValueError("Direct LoRA weight-merge validation did not pass")
    if validation.get("adapter_tensor_count") != ES_ADAPTER_TENSOR_COUNT:
        raise ValueError("Direct merge used an unexpected number of adapter tensors")
    if validation.get("merged_module_count") != ES_ADAPTER_TENSOR_COUNT // 2:
        raise ValueError("Direct merge did not consume every LoRA A/B tensor pair")
    for item in manifest.get("output_files", []):
        path = merged_model_dir / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Merged model output hash mismatch: {path}")
        if deep and sha256(path) != item["sha256"]:
            raise ValueError(f"Merged model output hash mismatch: {path}")


def validate_baseline_summary(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text())
    if summary.get("schema_version") != 1 or summary.get("validation_only") is not True:
        raise ValueError(f"Unsupported validation-only summary: {path}")
    if summary.get("optimizer_updates") != 0 or summary.get("passed") is not True:
        raise ValueError(f"Step-zero validation gate did not pass: {path}")
    if summary.get("expected_rollouts") != 4000 or summary.get("observed_rollouts") != 4000:
        raise ValueError(f"Step-zero validation did not score exactly 4,000 rollouts: {path}")


def validate_merge_parity_summary(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text())
    if summary.get("schema_version") != 2 or summary.get("passed") is not True:
        raise ValueError(f"Direct merged-weight validation did not pass: {path}")
    if summary.get("validation_type") != "direct_lora_weight_merge":
        raise ValueError(f"Unexpected merged-weight validation type: {path}")
    if summary.get("adapter_tensor_count") != ES_ADAPTER_TENSOR_COUNT:
        raise ValueError(f"Merged-weight validation used the wrong adapter tensor count: {path}")
    if summary.get("merged_module_count") != ES_ADAPTER_TENSOR_COUNT // 2:
        raise ValueError(f"Merged-weight validation did not consume all LoRA pairs: {path}")
    if summary.get("unmapped_adapter_tensor_count") != 0:
        raise ValueError(f"Merged-weight validation left adapter tensors unmapped: {path}")
    if summary.get("nonfinite_delta_count") != 0:
        raise ValueError(f"Merged-weight validation found non-finite deltas: {path}")


def validate_server_sources(root: Path) -> None:
    rdkit_server_dir = root / "resources_servers/rdkit_chemistry"
    for filename, expected_hash in RDKIT_SERVER_HASHES.items():
        path = rdkit_server_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing RDKit server file from {RDKIT_SERVER_COMMIT}: {path}"
            )
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"RDKit server file {path} has SHA-256 {actual_hash}, expected "
                f"{expected_hash} from {RDKIT_SERVER_COMMIT}"
            )

    for component_type, component_name, entrypoint in SERVER_COMPONENTS:
        server_dir = root / component_type / component_name
        entrypoint_path = server_dir / entrypoint
        if not entrypoint_path.is_file():
            raise FileNotFoundError(f"Missing configured server entrypoint: {entrypoint_path}")

        manifests = [
            path
            for path in (server_dir / "pyproject.toml", server_dir / "requirements.txt")
            if path.is_file()
        ]
        if len(manifests) != 1:
            raise ValueError(
                f"{server_dir} must contain exactly one of pyproject.toml or "
                f"requirements.txt; found {[path.name for path in manifests]}"
            )


def validate_pinned_gym_setup(root: Path, bundle: Path) -> None:
    sys.path.insert(0, str(root))
    from nemo_gym.cli_setup_command import setup_env_command
    from nemo_gym.global_config import (
        HEAD_SERVER_DEPS_KEY_NAME,
        PIP_INSTALL_VERBOSE_KEY_NAME,
        PYTHON_VERSION_KEY_NAME,
        SKIP_VENV_IF_PRESENT_KEY_NAME,
        UV_PIP_SET_PYTHON_KEY_NAME,
        UV_VENV_DIR_KEY_NAME,
    )

    setup_config = OmegaConf.create(
        {
            HEAD_SERVER_DEPS_KEY_NAME: [],
            PIP_INSTALL_VERBOSE_KEY_NAME: False,
            PYTHON_VERSION_KEY_NAME: f"{sys.version_info.major}.{sys.version_info.minor}",
            SKIP_VENV_IF_PRESENT_KEY_NAME: False,
            UV_PIP_SET_PYTHON_KEY_NAME: True,
            UV_VENV_DIR_KEY_NAME: str(bundle / "venvs"),
        }
    )
    for component_type, component_name, _ in SERVER_COMPONENTS:
        server_dir = root / component_type / component_name
        command = setup_env_command(server_dir, setup_config, component_name)
        if not command.startswith(f"cd {server_dir} && "):
            raise ValueError(f"Unexpected setup command for {server_dir}: {command}")

    rdkit_app = importlib.import_module("resources_servers.rdkit_chemistry.app")
    simple_agent = importlib.import_module("responses_api_agents.simple_agent.app")
    if rdkit_app.extract_predicted_value(
        "Final Answer = 42", "count", answer_format="fmt_28"
    ) != 42.0:
        raise ValueError("RDKit answer parser compatibility check failed")
    if rdkit_app.compute_reward(42.0, 42.0, property_type="count") != 1.0:
        raise ValueError("RDKit reward compatibility check failed")
    if not hasattr(simple_agent, "SimpleAgent"):
        raise ValueError("Simple agent import compatibility check failed")


def validate_container_preflight(
    bundle: Path, *, source_gym_commit: str, container_path: Path
) -> None:
    stamp_path = bundle / "preflight/resource_server_container.json"
    if not stamp_path.is_file():
        raise FileNotFoundError(
            f"Missing successful container preflight stamp: {stamp_path}. Submit "
            f"{bundle / 'preflight_resource_server.sbatch'} before production."
        )

    stamp = json.loads(stamp_path.read_text())
    if stamp.get("schema_version") != 1:
        raise ValueError(f"Unsupported container preflight stamp: {stamp_path}")
    if stamp.get("gym_commit") != source_gym_commit:
        raise ValueError("Container preflight Gym commit does not match the current checkout")
    if stamp.get("resource_files") != RDKIT_SERVER_HASHES:
        raise ValueError("Container preflight RDKit sources do not match the current pinned sources")

    container_stat = container_path.stat()
    expected_container = {
        "path": str(container_path),
        "size": container_stat.st_size,
        "mtime_ns": container_stat.st_mtime_ns,
    }
    if stamp.get("container") != expected_container:
        raise ValueError("Container preflight does not match the current Nemo-RL squashfs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-gym-commit")
    parser.add_argument("--container-path", type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=default_max_output_tokens())
    parser.add_argument("--require-container-preflight", action="store_true")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--merged-model-dir", type=Path)
    parser.add_argument("--require-adapter", action="store_true")
    parser.add_argument("--require-merged-model", action="store_true")
    parser.add_argument("--deep-verify-merged-model", action="store_true")
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--require-baseline-gate", action="store_true")
    parser.add_argument("--merge-parity-summary", type=Path)
    parser.add_argument("--require-merge-parity", action="store_true")
    args = parser.parse_args()
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")

    root = args.root.absolute()
    bundle = root / "cluster/rdkit_no_tool_grpo"
    data = bundle / "data"

    source_gym_commit = validate_repository_version(root, args.source_gym_commit)
    validate_server_sources(root)
    validate_pinned_gym_setup(root, bundle)
    validate_lora_config(bundle)
    if args.require_container_preflight:
        if args.container_path is None:
            raise ValueError("--container-path is required with --require-container-preflight")
        validate_container_preflight(
            bundle,
            source_gym_commit=source_gym_commit,
            container_path=args.container_path.absolute(),
        )
    if args.require_adapter:
        if args.adapter_dir is None:
            raise ValueError("--adapter-dir is required with --require-adapter")
        validate_es_adapter(args.adapter_dir.absolute())
    if args.require_merged_model:
        if args.merged_model_dir is None:
            raise ValueError("--merged-model-dir is required with --require-merged-model")
        validate_merged_model(
            args.merged_model_dir.absolute(), deep=args.deep_verify_merged_model
        )
    if args.require_baseline_gate:
        if args.baseline_summary is None:
            raise ValueError("--baseline-summary is required with --require-baseline-gate")
        validate_baseline_summary(args.baseline_summary.absolute())
    if args.require_merge_parity:
        if args.merge_parity_summary is None:
            raise ValueError("--merge-parity-summary is required with --require-merge-parity")
        validate_merge_parity_summary(args.merge_parity_summary.absolute())

    for split, count in EXPECTED.items():
        validate_split(data, split, count, args.max_output_tokens)
    for split, count in SMOKE_EXPECTED.items():
        validate_split(data, split, count, args.max_output_tokens)
    validate_repeated_validation(data, args.max_output_tokens)

    direct_config = (bundle / "rdkit_chemistry_direct.yaml").read_text()
    forbidden = ["ns_tools", "sandbox_launcher", "sandbox_venv_path", "NEMO_SKILLS_SANDBOX"]
    for marker in forbidden:
        if marker in direct_config:
            raise ValueError(f"direct config contains forbidden marker: {marker}")

    required = [
        bundle / "rdkit_no_tool_grpo.yaml",
        bundle / "vllm_model_for_training_nano_v3.yaml",
        bundle / "reasoning_parsers/nano_v3_reasoning_parser.py",
        bundle / "nemo_rl_assets/ray.sub",
        bundle / "nemo_rl_assets/run_grpo_nemo_gym.py",
        bundle / "preflight_resource_server.py",
        bundle / "preflight_resource_server.sbatch",
        bundle / "preflight_submission.sbatch",
        bundle / "submit_smoke.sh",
        bundle / "submit_chain.sh",
        bundle / "merge_es_adapter.py",
        bundle / "merge_es_adapter.sbatch",
        bundle / "dependency_preflight.sbatch",
        bundle / "build_integration_sqsh.sbatch",
        bundle / "submit_baseline.sh",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))

    print("RDKit no-tool GRPO bundle validation passed")


if __name__ == "__main__":
    main()
