# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay the math_with_judge README trajectory collation through the real CLI and collator, offline."""

import json
import re
import shlex
import sys
from pathlib import Path

from omegaconf import OmegaConf
from pytest import MonkeyPatch

import nemo_gym.cli.main as cli_main
from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.data.dataset_preparation import TrainDataProcessor


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "resources_servers/math_with_judge"
README = SERVER_ROOT / "README.md"
TRAJECTORY_REPEATS = 16


def _readme_collate_overrides(monkeypatch: MonkeyPatch) -> list[str]:
    """Return the Hydra overrides the README's `gym dataset collate` block hands to dispatch."""
    blocks = re.findall(r"```bash\n(.*?)```", README.read_text(), flags=re.DOTALL)
    commands = [block for block in blocks if block.lstrip().startswith("gym dataset collate")]
    assert len(commands) == 1, "Expected one `gym dataset collate` block in the README"
    captured: dict = {}

    def fake_dispatch(target: str, overrides: list[str]) -> None:
        captured.update(target=target, overrides=overrides)

    monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(cli_main, "dispatch", fake_dispatch)
    monkeypatch.setattr(sys, "argv", shlex.split(commands[0].replace("\\\n", " ")))
    cli_main.main()
    assert captured["target"] == "nemo_gym.cli.dataset:prepare_data"
    return captured["overrides"]


def test_readme_trajectory_collation_repeats_dapo17k_train_as_validation(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    # NeMo RL's `is_trajectory_collection` reads only the validation split, so the README step must put the
    # DAPO17k train problems there, each repeated. The dataset file is a local fixture, so nothing is
    # downloaded, and env.yaml is skipped, so no developer credentials are read.
    monkeypatch.chdir(REPO_ROOT)
    overrides = _readme_collate_overrides(monkeypatch)
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        OmegaConf.from_dotlist([override.lstrip("+") for override in overrides]),
    )
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True
        )
    )

    datasets = config.math_with_judge_simple_agent.responses_api_agents.simple_agent.datasets
    assert [(d.type, d.get("num_repeats", 1)) for d in datasets] == [("validation", TRAJECTORY_REPEATS)]
    (dataset,) = datasets
    maintained = OmegaConf.load(SERVER_ROOT / "configs/dapo17k.yaml")
    (maintained_train,) = [
        d
        for d in maintained.math_with_judge_simple_agent.responses_api_agents.simple_agent.datasets
        if d.type == "train"
    ]
    assert dataset.gitlab_identifier == maintained_train.gitlab_identifier
    # Collate checks the real data against `<jsonl stem>_metrics.json`, so the committed sidecar must describe
    # this split. Sharing dapo17k.yaml's local file would compare against its train sidecar and fail.
    data_fpath = Path(dataset.jsonl_fpath)
    sidecar = json.loads((REPO_ROOT / data_fpath.with_name(f"{data_fpath.stem}_metrics.json")).read_text())
    assert (sidecar["type"], sidecar["num_repeats"]) == ("validation", TRAJECTORY_REPEATS)
    assert sidecar["gitlab_identifier"] == OmegaConf.to_container(dataset.gitlab_identifier)

    source_rows = [json.loads(line) for line in (SERVER_ROOT / "data/example.jsonl").read_text().splitlines()[:3]]
    fixture = tmp_path / dataset.jsonl_fpath
    fixture.parent.mkdir(parents=True)
    fixture.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    monkeypatch.chdir(tmp_path)
    config.output_dirpath = str(tmp_path / "collated")

    TrainDataProcessor().run(config)

    collated = tmp_path / "collated"
    assert sorted(p.name for p in collated.iterdir()) == ["validation.jsonl", "validation_metrics.json"]
    rows = [json.loads(line) for line in (collated / "validation.jsonl").read_text().splitlines()]
    expected_questions = [row["question"] for row in source_rows for _ in range(TRAJECTORY_REPEATS)]
    assert [row["question"] for row in rows] == expected_questions
