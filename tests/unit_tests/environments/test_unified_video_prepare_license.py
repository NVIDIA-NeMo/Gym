# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train-license metadata must satisfy the real native descriptor parser."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from environments.unified_video.prepare import CLAW, component_config
from nemo_gym.config.config_types import maybe_get_server_instance_config
from nemo_gym.config.global_config import GlobalConfigDictParser


class NativeDatasetLicenseRegression(unittest.TestCase):
    def generated(self, source, folder):
        rows = folder / "rows.jsonl"
        rows.write_text("")
        result = component_config(
            source,
            prepared=SimpleNamespace(tasks=[], folder=folder),
            rows=rows,
            context_tokens=262144,
            output_tokens=8192,
            max_steps=32,
        )
        result["policy_model_name"] = "unit-test-not-a-service"
        return result

    def test_claw_train_descriptor_validates_with_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for source, expected in [(CLAW, "NVIDIA Internal Use Only, Do Not Distribute")]:
                with self.subTest(source=source):
                    parsed = OmegaConf.create(self.generated(source, Path(temporary)))
                    instance, error = maybe_get_server_instance_config(source, parsed[source])
                    self.assertIsNone(error)
                    self.assertIsNotNone(instance)
                    self.assertEqual(instance.datasets[0].license, expected)
                    self.assertEqual(GlobalConfigDictParser().detect_and_report_almost_servers(parsed), [])

    def test_missing_license_is_rejected_by_actual_native_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            for source in [CLAW]:
                with self.subTest(source=source):
                    config = self.generated(source, Path(temporary))
                    del config[source]["responses_api_agents"]["opencode_agent"]["datasets"][0]["license"]
                    parsed = OmegaConf.create(config)
                    instance, error = maybe_get_server_instance_config(source, parsed[source])
                    self.assertIsNone(instance)
                    self.assertIsNotNone(error)
                    self.assertIn("license", str(error))
                    almost = GlobalConfigDictParser().detect_and_report_almost_servers(parsed)
                    self.assertEqual([name for name, _ in almost], [source])


if __name__ == "__main__":
    unittest.main()
