# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke an original VS-Bench scorer inside an externally isolated process.

Source: https://github.com/zelaix/VS-Bench, MIT, revision
2c3531cdf91d1b620b4a9015126a242eb305a4d0. Three original scorers execute
model text with eval. This worker is NOT a sandbox. A caller handling model
output must first establish a credential-free, network-denied isolation boundary.
"""

import importlib
import json
import math
import os
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

from reference_reasoning import GAMES


def score(author_root: Path, dataset_root: Path, request: dict) -> float:
    game = request["game"]
    row_index = request["row_index"]
    if game not in GAMES or not isinstance(row_index, int) or isinstance(row_index, bool) or not 0 <= row_index < 400:
        raise ValueError("Invalid source dataset identity")
    if not isinstance(request["answer"], str):
        raise ValueError("Answer must be a string")
    for name in ("envs",):
        package = types.ModuleType(name)
        package.__path__ = [str(author_root / name)]
        sys.modules[name] = package
    sys.path.insert(0, str(author_root))
    import pandas as pd
    import yaml
    from envs.base_env import BaseEnv

    importlib.import_module("envs." + game)
    with (author_root / "configs/env_configs" / f"{game}.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config.setdefault("params", {})["image_dir"] = ""
    os.chdir(author_root)
    env = BaseEnv.from_config(config)
    # The author loads parquet through pandas and calls to_dict(orient="index").
    # Preserve these label types; the converted JSON label is only metadata.
    rows = pd.read_parquet(dataset_root / "perception" / f"{game}.parquet", columns=["label"]).to_dict(orient="index")
    result = float(env.get_perception_reward(request["answer"], rows[str(row_index)]["label"]))
    if not math.isfinite(result):
        raise ValueError("Non-finite author reward")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: reference_perception_score.py <upstream-root> <dataset-root>")
    with redirect_stdout(sys.stderr):
        result = score(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), json.load(sys.stdin))
    print(json.dumps({"reward": result}, allow_nan=False))
