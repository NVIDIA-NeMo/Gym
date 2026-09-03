# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from nemo_gym.sandbox.providers import SandboxImagePrepareRequest, SandboxImagePrepareResult
from responses_api_agents.anyterminal_agent import prepare


class FakeImagePrepareProvider:
    def __init__(self) -> None:
        self.requests: list[SandboxImagePrepareRequest] = []

    def prepare_image(self, request: SandboxImagePrepareRequest) -> SandboxImagePrepareResult:
        self.requests.append(request)
        return SandboxImagePrepareResult(
            image=str(request.target_path),
            ok=True,
            prepared=True,
            detail="built",
        )


def test_build_one_image_delegates_to_provider_prepare_image(tmp_path: Path) -> None:
    provider = FakeImagePrepareProvider()

    name, ok, detail = prepare._build_one_image(
        provider,
        "task",
        "example/image:tag",
        tmp_path,
        force=True,
    )

    assert (name, ok, detail) == ("task", True, "built")
    assert provider.requests == [
        SandboxImagePrepareRequest(
            image="example/image:tag",
            target_dir=tmp_path,
            target_name="task",
            force=True,
            attempts=prepare.IMAGE_BUILD_ATTEMPTS,
            retry_delay_s=prepare.IMAGE_BUILD_RETRY_DELAY_SECONDS,
        )
    ]
