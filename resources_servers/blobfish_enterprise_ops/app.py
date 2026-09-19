# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
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
"""NeMo Gym entrypoint for Blobfish enterprise-operations suites."""

from pathlib import Path

from resources_servers.blobfish_enterprise_ops.blobfish_nemo_gym.fixture import verifier_fixture
from resources_servers.blobfish_enterprise_ops.blobfish_nemo_gym.server import BlobfishSuiteResourcesServer


# `gym env test <environment>` scores the cases in tests/verifier_cases.jsonl on the self-contained
# fixture suite (tests/fixtures/mini_suite) through the same code path as /verify.
VERIFIER_FIXTURE = verifier_fixture(Path(__file__).resolve().parent / "tests")

if __name__ == "__main__":
    BlobfishSuiteResourcesServer.run_webserver()
