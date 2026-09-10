# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU control: grade the reference patch with Gym's existing Pro verifier.

Uses a completed smoke run's config and inputs. Does not run Hermes or call a model.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


async def check(args):
    config = json.loads((args.smoke_results / "config.json").read_text())
    # Scratch is allocated per job; a previous run's scratch directory has been removed.
    if os.environ.get("HERMES_OVERLAY_ROOT"):
        config["sandbox"]["apptainer"]["create"]["overlay_root"] = os.environ["HERMES_OVERLAY_ROOT"]
    os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(config)

    from omegaconf import OmegaConf

    from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
    from resources_servers.swebench_pro.app import (
        SWEBenchProResourcesServer,
        SWEBenchProResourcesServerConfig,
        SWEBenchProVerifyRequest,
    )
    from resources_servers.swebench_pro.client import EMPTY_RESPONSE

    block = config["swebench_pro_resources_server"]["resources_servers"]["swebench_pro"]
    server = SWEBenchProResourcesServer(
        config=SWEBenchProResourcesServerConfig(
            name="swebench_pro_resources_server",
            **(block | {"is_verifying_golden_patch": True}),
        ),
        server_client=ServerClient(
            head_server_config=config["head_server"],
            global_config_dict=OmegaConf.create(config),
        ),
    )
    results = []
    try:
        for row in map(json.loads, (args.smoke_results / "inputs.jsonl").read_text().splitlines()):
            response = await server.verify(
                SimpleNamespace(session={SESSION_ID_KEY: uuid4().hex}),
                SWEBenchProVerifyRequest.model_validate(row | {"response": EMPTY_RESPONSE}),
            )
            result = response.model_dump(mode="json") | {"validation_kind": "reference_patch_control"}
            results.append(result)
            print(json.dumps({k: result[k] for k in ["instance_id", "resolved", "patch_applied", "error"]}))
    finally:
        await server.shutdown()
        args.output.write_text("".join(json.dumps(result) + "\n" for result in results))
    return 0 if results and all(result["resolved"] for result in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(asyncio.run(check(parser.parse_args())))
