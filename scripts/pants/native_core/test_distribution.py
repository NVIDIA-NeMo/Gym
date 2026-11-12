# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard original-project packaging and portable build-directory inputs."""

import tempfile
import unittest
from importlib.metadata import Distribution
from pathlib import Path

from distribution import distribution_targets, validate_distribution


class DistributionTargetsTest(unittest.TestCase):
    def test_wheel_metadata_and_backend_must_match_real_editable_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = "Name: nemo-gym\nVersion: 1\nRequires-Python: >=3.13\nRequires-Dist: dependency==1\n"
            wheel = "Wheel-Version: 1.0\nGenerator: setuptools (84.2.1)\n"
            entries = "[console_scripts]\ngym = sampled.cli:main\n"
            expected = root / "expected.dist-info"
            actual = root / "actual.dist-info"
            for directory in (expected, actual):
                directory.mkdir()
                (directory / "METADATA").write_text(metadata)
                (directory / "WHEEL").write_text(wheel)
                (directory / "entry_points.txt").write_text(entries)
            self.assertEqual(validate_distribution(expected, Distribution.at(actual))["version"], "1")
            for filename, changed in (
                ("METADATA", metadata.replace("dependency==1", "dependency==2")),
                ("METADATA", metadata.replace("Version: 1", "Version: 2")),
                ("entry_points.txt", entries.replace("sampled.cli:main", "other.cli:main")),
                ("WHEEL", wheel.replace("84.2.1", "84.3.0")),
            ):
                with self.subTest(filename=filename):
                    original = (actual / filename).read_text()
                    (actual / filename).write_text(changed)
                    with self.assertRaises(ValueError):
                        validate_distribution(expected, Distribution.at(actual))
                    (actual / filename).write_text(original)

    def test_original_project_backend_and_scripts_remain_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / "scripts/pants/native_core").mkdir(parents=True)
            source = '[project]\nname="nemo-gym"\n[tool.distutils.egg_info]\negg_base="build metadata"\n'
            (repo / "pyproject.toml").write_text(source)
            captured = {}

            def record(kind):
                return lambda **values: captured.setdefault(values["name"], (kind, values))

            exec(
                distribution_targets(repo),
                {
                    "files": record("files"),
                    "relocated_files": record("relocated_files"),
                    "python_distribution": record("python_distribution"),
                    "python_artifact": lambda **values: values,
                },
            )
            target = captured["native-project"][1]
            self.assertFalse(target["generate_setup"])
            self.assertFalse(target["sdist"])
            self.assertEqual(target["provides"], {"name": "nemo-gym"})
            self.assertNotIn("entry_points", target)
            self.assertIn(":assets", target["dependencies"])
            self.assertEqual(captured["native-project-build-dir"][1]["dest"], "build metadata")
            placeholder = captured["native-build-directory"][1]["sources"][0]
            self.assertTrue((repo / placeholder).is_file())
            self.assertEqual((repo / "pyproject.toml").read_text(), source)

    def test_build_directory_cannot_escape_checkout(self):
        for egg_base in ("/outside", "../outside", "cache/../../outside"):
            with self.subTest(egg_base=egg_base), tempfile.TemporaryDirectory() as temporary:
                repo = Path(temporary)
                (repo / "pyproject.toml").write_text(
                    f'[project]\nname="nemo-gym"\n[tool.distutils.egg_info]\negg_base="{egg_base}"\n'
                )
                with self.assertRaisesRegex(ValueError, "within its checkout"):
                    distribution_targets(repo)


if __name__ == "__main__":
    unittest.main()
