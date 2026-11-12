# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise repository coverage scope with external dependencies and runtime helpers."""

import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from coverage import CoverageData
from prepare import native_coverage_config


class NativeCoverageTest(unittest.TestCase):
    def test_native_scope_preserves_source_arcs_contexts_and_excludes_only_instrumentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkout = root / "source"
            checkout.mkdir()
            external = root / "pex-site-packages"
            external.mkdir()
            (external / "dependency.py").write_text("VALUE = 41\n")
            (checkout / "app.py").write_text("def choose(value):\n    if value:\n        return 42\n    return 0\n")
            tests = checkout / "tests"
            tests.mkdir()
            (tests / "test_scope.py").write_text(
                "import importlib.util, os\nfrom pathlib import Path\nimport app, dependency\n"
                "def test_scope():\n"
                "    assert app.choose(True) == dependency.VALUE + 1\n"
                "    if os.environ.get('NATIVE_HELPERS'):\n"
                "        for path in ('.native-child-site/sitecustomize.py', '__editable___fixture_finder.py'):\n"
                "            spec = importlib.util.spec_from_file_location('owned_helper', Path(path))\n"
                "            spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
            )
            project = (
                '[tool.coverage.run]\ndata_file = "results/.coverage"\nbranch = true\n'
                'context = "sampled-context"\nomit = ["tests/*"]\n'
                "[tool.coverage.report]\nfail_under = 95.0\nprecision = 2\n"
            )
            (checkout / "pyproject.toml").write_text(project)
            (checkout / "native.toml").write_text(
                native_coverage_config(project, finder="__editable___fixture_finder.py")
            )
            child = checkout / ".native-child-site"
            child.mkdir()
            environment = dict(
                os.environ,
                PYTHONPATH=str(external),
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            )
            environment.pop("PYTEST_ADDOPTS", None)
            outputs = {}
            for label, config, source in (
                ("baseline", "pyproject.toml", "."),
                ("unscoped", "pyproject.toml", None),
                ("native", "native.toml", "."),
            ):
                data_path = checkout / (".coverage" if label == "native" else "results/.coverage")
                env = dict(environment, COVERAGE_FILE=str(data_path))
                if label != "baseline":
                    env["NATIVE_HELPERS"] = "1"
                    (child / "sitecustomize.py").write_text("VALUE = 123\n")
                    (checkout / "__editable___fixture_finder.py").write_text("VALUE = 456\n")
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-p",
                        "pytest_cov",
                        "-q",
                        "tests",
                        "--cov",
                        *(["--cov-reset", f"--cov={source}"] if source is not None else []),
                        f"--cov-config={config}",
                        "--cov-report=",
                        "--cov-fail-under=0",
                    ],
                    cwd=checkout,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(data_path.exists(), result.stdout + result.stderr)
                data = CoverageData(basename=str(data_path))
                data.read()
                self.assertTrue(data.measured_files(), f"{label}: {result.stdout} {result.stderr}")
                outputs[label] = data
            # Reproduce the hosted failure before asserting the repair.
            self.assertIn(str(external / "dependency.py"), outputs["unscoped"].measured_files())
            self.assertIn(str(child / "sitecustomize.py"), outputs["unscoped"].measured_files())
            native = outputs["native"]
            baseline = outputs["baseline"]
            self.assertEqual(native.measured_files(), {str(checkout / "app.py")})
            self.assertEqual(baseline.measured_files(), native.measured_files())
            self.assertTrue(native.has_arcs())
            self.assertEqual(native.arcs(str(checkout / "app.py")), baseline.arcs(str(checkout / "app.py")))
            self.assertEqual(native.measured_contexts(), {"sampled-context"})
            self.assertEqual(
                native.contexts_by_lineno(str(checkout / "app.py")),
                baseline.contexts_by_lineno(str(checkout / "app.py")),
            )

    def test_configuration_preserves_report_threshold_and_handles_missing_omit(self):
        project = (
            '[tool.coverage.run]\ndata_file = "results/.coverage"\n'
            'concurrency = ["thread", "multiprocessing"]\n'
            '[tool.coverage.report]\nfail_under = 95.0\nexclude_also = ["if TYPE_CHECKING:"]\n'
        )
        result = tomllib.loads(native_coverage_config(project, finder=None))["tool"]["coverage"]
        self.assertEqual(result["report"], tomllib.loads(project)["tool"]["coverage"]["report"])
        self.assertEqual(result["run"]["omit"], [".native-child-site/*"])
        self.assertEqual(result["run"]["concurrency"], ["thread", "multiprocessing"])


if __name__ == "__main__":
    unittest.main()
