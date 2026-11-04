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
import re
from pathlib import Path

import pytest
import yaml


@pytest.mark.docs
def test_opensandbox_provider_config_excerpt_matches_shipped_values():
    """Omitted settings are intentional; every copied provider value must stay current."""
    root = Path(__file__).resolve().parents[2]
    page = (root / "fern/versions/latest/pages/infrastructure/sandbox/opensandbox.mdx").read_text()
    section = page.split("## Provider Config\n", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"^```yaml\n(.*?)^```", section, flags=re.MULTILINE | re.DOTALL)
    assert len(blocks) == 1, "Expected one canonical provider-config excerpt"
    # Parse, but do not resolve environment interpolations or require service credentials.
    documented = yaml.safe_load(blocks[0])["sandbox"]["opensandbox"]
    assert {"timeout_s", "skip_health_check"} <= documented["create"].keys()
    canonical = yaml.safe_load((root / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml").read_text())[
        "sandbox"
    ]["opensandbox"]
    for name, settings in documented.items():
        assert settings == {key: canonical[name][key] for key in settings}, f"{name} has drifted from shipped config"
