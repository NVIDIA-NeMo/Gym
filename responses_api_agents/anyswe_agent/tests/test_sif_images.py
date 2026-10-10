# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local Apptainer .sif task images."""

import json
from pathlib import Path

import pytest

from responses_api_agents.anyswe_agent.app import AnySweAgent


class TestSifImages:
    def test_image_prefers_local_sif_over_row_image(self, tmp_path: Path) -> None:
        sif = tmp_path / "biolab__orange3-91ad02f1.sif"
        sif.write_bytes(b"sif")
        image = AnySweAgent._sandbox_image(
            {
                "instance_id": "biolab__orange3-91ad02f1",
                "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                "container_formatter": str(tmp_path / "{instance_id}.sif"),
                "instance_dict": json.dumps({"docker_image": "namanjain12/orange3_final:91ad02f1"}),
            }
        )
        assert image == str(sif)

    def test_image_sif_resolves_r2e_naming_across_template_lists(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        sif = tmp_path / "b" / "namanjain12_orange3_final_91ad02f1.sif"
        sif.write_bytes(b"sif")
        image = AnySweAgent._sandbox_image(
            {
                "instance_id": "biolab__orange3-91ad02f1",
                "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                "container_formatter": [
                    str(tmp_path / "a" / "{instance_id}.sif"),
                    str(tmp_path / "b" / "namanjain12_{instance_id}.sif"),
                ],
            }
        )
        assert image == str(sif)

    def test_image_sif_tries_the_swebench_rewrites(self, tmp_path: Path) -> None:
        sif = tmp_path / "sweb.eval.x86_64.astropy_1776_astropy-12907.sif"
        sif.write_bytes(b"sif")
        info = {
            "instance_id": "astropy__astropy-12907",
            "container_formatter": str(tmp_path / "sweb.eval.x86_64.{instance_id}.sif"),
        }
        assert AnySweAgent._sandbox_image(info) == str(sif)

    def test_image_sif_requires_an_exact_name(self, tmp_path: Path) -> None:
        (tmp_path / "psf__requests-11420.sif").write_bytes(b"sif")
        info = {"instance_id": "psf__requests-1142", "container_formatter": str(tmp_path / "{instance_id}.sif")}
        with pytest.raises(FileNotFoundError, match="no .sif image found"):
            AnySweAgent._sandbox_image(info)

    def test_image_uses_an_explicit_sif_path(self, tmp_path: Path) -> None:
        sif = tmp_path / "x.sif"
        sif.write_bytes(b"sif")
        info = {
            "instance_id": "dask__dask-9213",
            "container_formatter": "docker://swebench/sweb.eval.x86_64.{instance_id}",
            "sif_path": str(sif),
        }
        assert AnySweAgent._sandbox_image(info) == str(sif)
        with pytest.raises(FileNotFoundError, match="sif_path does not exist"):
            AnySweAgent._sandbox_image({**info, "sif_path": str(tmp_path / "missing.sif")})
