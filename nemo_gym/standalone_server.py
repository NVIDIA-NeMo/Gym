#!/usr/bin/env python3
"""Standalone NeMo Gym server for disaggregated RL-Gym deployments.

Starts the Gym servers (head, agent, model, resource) as a long-running
HTTP service that can be deployed on a separate Kubernetes cluster/pod.

The RL cluster connects to this service via the head server URL.

Usage:
    python -m nemo_gym.standalone_server \
        --port 8080 \
        --vllm-base-urls http://vllm-svc:8000 \
        --config-yaml /path/to/gym_config.yaml \
        --model-name Qwen/Qwen3-0.6B
"""
import argparse
import signal
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME


def main():
    parser = argparse.ArgumentParser(description="Standalone NeMo Gym server")
    parser.add_argument("--port", type=int, default=8080, help="Head server port")
    parser.add_argument(
        "--vllm-base-urls",
        type=str,
        nargs="+",
        required=True,
        help="vLLM HTTP server base URLs from the RL cluster",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Policy model name (e.g., Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="Path to a Gym config YAML (e.g., nemo_gym_env.yaml)",
    )
    parser.add_argument(
        "--dotenv-path",
        type=str,
        default=None,
        help="Path to the nemo_gym_env.yaml dotenv file",
    )
    args = parser.parse_args()

    # Build the global config dict
    initial_global_config_dict = {}
    if args.config_yaml:
        initial_global_config_dict = OmegaConf.to_container(
            OmegaConf.load(args.config_yaml), resolve=True
        )

    initial_global_config_dict["policy_model_name"] = args.model_name
    initial_global_config_dict["policy_api_key"] = "dummy_key"
    initial_global_config_dict["policy_base_url"] = args.vllm_base_urls
    initial_global_config_dict.setdefault("global_aiohttp_connector_limit_per_host", 16_384)
    initial_global_config_dict.setdefault("global_aiohttp_connector_limit", 65_536)

    # Head server config
    initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
        "host": "0.0.0.0",
        "port": args.port,
    }

    # Determine dotenv path
    dotenv_path = Path(args.dotenv_path) if args.dotenv_path else None
    if dotenv_path is None:
        # Try default location relative to nemo-rl
        candidate = Path(__file__).parent.parent.parent.parent / "nemo_rl" / "environments" / "nemo_gym_env.yaml"
        if candidate.exists():
            dotenv_path = candidate

    print(f"Starting standalone NeMo Gym server on port {args.port}")
    print(f"vLLM base URLs: {args.vllm_base_urls}")
    print(f"Model: {args.model_name}")

    rh = RunHelper()
    rh.start(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            dotenv_path=dotenv_path,
            initial_global_config_dict=DictConfig(initial_global_config_dict),
            skip_load_from_cli=True,
        )
    )

    rh.display_server_instance_info()
    print(f"\nStandalone Gym server is running on port {args.port}")
    print("Press Ctrl+C to stop.")

    # Block until SIGTERM/SIGINT
    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        rh.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Wait forever
    signal.pause()


if __name__ == "__main__":
    main()
