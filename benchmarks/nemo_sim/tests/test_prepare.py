# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.nemo_sim import prepare as prepare_module


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{"first_name": "Morgan", "age": 42}]), path)


def test_prepare_downloads_pinned_source_with_gym_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "nemo_sim.jsonl"
    example_path = tmp_path / "example.jsonl"
    example_path.write_text('{"nemo_sim_sampling":{"locale":"en_US","seed":1042}}\n')
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)
    monkeypatch.setattr(prepare_module, "EXAMPLE_FPATH", example_path)
    monkeypatch.setattr(prepare_module.shutil, "which", lambda executable: f"/bin/{executable}")
    monkeypatch.setattr(
        prepare_module,
        "get_global_config_dict",
        lambda: {
            "ngc_cli_api_key": "configured-ngc-key",  # pragma: allowlist secret
            "ngc_cli_org": "configured-org",
        },
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_download(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((command, kwargs))
        _write_parquet(Path(command[-1]) / "download" / "en_US.parquet")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare_module.subprocess, "run", fake_download)

    result = prepare_module.prepare(personas_cache_dir=tmp_path / "personas")

    assert result == output_path.absolute()
    assert output_path.read_text() == example_path.read_text()
    command, kwargs = calls[0]
    assert command[4] == "nvidia/nemotron-personas/nemotron-personas-dataset-en_us:0.0.2"
    assert "configured-ngc-key" not in command
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["NGC_CLI_API_KEY"] == "configured-ngc-key"  # pragma: allowlist secret
    assert environment["NGC_CLI_ORG"] == "configured-org"
    source_path = tmp_path / "personas" / "0.0.2" / "source" / "en_US.parquet"
    manifest = json.loads(source_path.with_suffix(".manifest.json").read_text())
    assert manifest["rows"] == 1
    assert len(manifest["sha256"]) == 64


def test_prepare_reuses_existing_source_without_ngc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "nemo_sim.jsonl"
    example_path = tmp_path / "example.jsonl"
    example_path.write_text("{}\n")
    source_path = tmp_path / "personas" / "0.0.2" / "source" / "en_US.parquet"
    _write_parquet(source_path)
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)
    monkeypatch.setattr(prepare_module, "EXAMPLE_FPATH", example_path)
    monkeypatch.setattr(
        prepare_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("prepared source must not invoke NGC"),
    )

    prepare_module.prepare(personas_cache_dir=tmp_path / "personas")

    assert output_path.is_file()
    assert source_path.with_suffix(".manifest.json").is_file()
