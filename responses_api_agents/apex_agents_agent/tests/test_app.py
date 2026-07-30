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
from nemo_gym.sandbox import SandboxSpec
from responses_api_agents.apex_agents_agent.app import (
    DEFAULT_ENV_IMAGE,
    _read_entrypoint_source,
    build_spec,
)


def test_build_spec_sets_image_env_and_model_url():
    spec = build_spec(
        {"task_id": "t", "world_id": "w", "prompt": "do it"},
        model_url="http://gym-model:8000/v1",
    )
    assert isinstance(spec, SandboxSpec)
    assert spec.image  # non-empty image
    assert spec.env["NV_MODEL_URL"] == "http://gym-model:8000/v1"
    assert spec.env["APEX_TASK_ID"] == "t"
    assert spec.env["APEX_WORLD_ID"] == "w"


def test_build_spec_honors_overrides():
    spec = build_spec(
        {"task_id": "t", "world_id": "w"},
        model_url="http://m/v1",
        image="custom-image:1",
        orchestrator_model="openai/gpt-5",
        provider_options={"binds": ["/a:/b"]},
    )
    assert spec.image == "custom-image:1"
    assert spec.env["APEX_ORCHESTRATOR_MODEL"] == "openai/gpt-5"
    assert spec.provider_options == {"binds": ["/a:/b"]}


def test_build_spec_defaults_to_env_image():
    spec = build_spec({"task_id": "t", "world_id": "w"}, model_url="http://m/v1")
    assert spec.image == DEFAULT_ENV_IMAGE


def test_read_entrypoint_source_returns_guest_script():
    src = _read_entrypoint_source()
    assert "sandbox_entrypoint" in src
    assert "NV_MODEL_URL" in src
